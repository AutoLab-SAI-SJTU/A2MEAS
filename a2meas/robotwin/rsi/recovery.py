"""Observation-based RSI recovery."""
import re
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from collections import deque
import os
import cv2
from scipy.optimize import linear_sum_assignment
from .actor import Actor
from .geometry import BASE_P, BASE_R, HEAD_C, HEAD_RCW, RightArmKinematics, camera_k, ee_world, project, projection_matrices, rot, triangulate, wrist_projection
from .vision import MugRGBPerception

def object_prompt(phrase):
    text = phrase.lower().replace('-', ' ')
    for words, prompt in [(['tea box', 'box for tea', 'leafy patterns', 'green leafy'], 'tea box'), (['coffee box', 'coffee packaging'], 'coffee box'), (['rubik', 'rubiks', 'rubikscube'], 'Rubik cube'), (['toycar', 'toy car'], 'toy car'), (['phone'], 'smartphone'), (['stapler'], 'stapler'), (['soap'], 'soap'), (['mouse'], 'computer mouse'), (['bread', 'loaf', 'bun', 'braided'], 'bread'), (['bell'], 'bell'), (['playingcards', 'playing cards', 'card deck', 'card holder', 'card case', 'cards holder'], 'card box'), (['black box'], 'black box'), (['block'], 'toy blocks'), (['box'], 'box')]:
        if any((word in text for word in words)):
            return prompt
    return None

def instruction_roles(instruction):
    text = instruction.strip()
    relation = re.search('\\b(left|right)(?:ward)?(?:\\s+side)?\\s+of\\b', text, re.I)
    if relation:
        first = text[:relation.start()]
        second = text[relation.end():]
        direction = relation.group(1).lower()
    else:
        relation = re.search("\\bon\\s+(.+?)(?:'s|’s)\\s+(left|right)\\s+side", text, re.I)
        if relation is None:
            return None
        first = text[:relation.start()]
        second = relation.group(1)
        direction = relation.group(2).lower()
    a = object_prompt(first)
    b = object_prompt(second)
    if a is None or b is None or a == b:
        return None
    arm = re.search('\\b(left|right)\\s+arm\\b', text, re.I)
    return {'A': a, 'B': b, 'direction': direction, 'specified_arm': None if arm is None else arm.group(1).lower()}

def candidates(vision, image, prompt, head=False):
    detected = vision.detect(image, prompt)
    h, w = image.shape[:2]
    result = []
    for score, box in zip(detected['scores'], detected['boxes']):
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        if score < 0.35 or not 50 < area < 0.45 * h * w:
            continue
        if x1 < 1 or y1 < 1 or x2 > w - 2 or (y2 > h - 2):
            continue
        if head and y2 < h * 0.34:
            continue
        result.append({'pixel': (box[:2] + box[2:]) / 2, 'box': box, 'confidence': float(score)})
    return sorted(result, key=lambda x: x['confidence'], reverse=True)[:4]

def stereo_object(views):
    head = views[0]
    options = []
    for camera in views[1:]:
        for a in head['candidates']:
            for b in camera['candidates']:
                X = triangulate([(head['P'], a['pixel']), (camera['P'], b['pixel'])])
                if not np.isfinite(X).all() or not (-0.6 < X[0] < 0.6 and -0.6 < X[1] < 0.6 and (0.45 < X[2] < 1.3)):
                    continue
                if any(((v['P'] @ np.r_[X, 1])[2] < 0.1 for v in [head, camera])):
                    continue
                errors = [np.linalg.norm(project(v['P'], X) - c['pixel']) for v, c in [(head, a), (camera, b)]]
                if max(errors) > 4:
                    continue
                other = [v for v in views[1:] if v is not camera]
                inconsistent = False
                for v in other:
                    uv = project(v['P'], X)
                    if v['candidates'] and 5 < uv[0] < 315 and (5 < uv[1] < 235):
                        if min((np.linalg.norm(uv - c['pixel']) for c in v['candidates'])) > 12:
                            inconsistent = True
                if inconsistent:
                    continue
                score = np.sqrt(a['confidence'] * b['confidence']) / (1 + max(errors))
                options.append({'point': X, 'score': float(score), 'reprojection_errors': list(map(float, errors)), 'head_pixel': a['pixel'], 'head_box': a['box'], 'camera': camera['name']})
    return max(options, key=lambda x: x['score']) if options else None

def perceive_roles(vision, head, left, right, state, kin):
    h, w = head.shape[:2]
    P, Q = projection_matrices(kin['right'], state[7:13], w, h)
    L = wrist_projection(kin['left'], state[:6], w, h)
    return [('head', head, P), ('left', left, L), ('right', right, Q)]

def estimate_roles(vision, images, state, kin, roles):
    cameras = perceive_roles(vision, *images, state, kin)
    result = {}
    for role in ['A', 'B']:
        views = [{'name': name, 'P': P, 'candidates': candidates(vision, img, roles[role], head=name == 'head')} for name, img, P in cameras]
        fitted = stereo_object(views)
        result[role] = fitted
        result[role + '_views'] = [{'camera': v['name'], 'candidates': v['candidates']} for v in views]
    return result

ALIASES = {'tea box': ['tea box', 'box'], 'coffee box': ['coffee box', 'box'], 'bread': ['bread', 'bun', 'bread loaf', 'loaf'], 'smartphone': ['smartphone', 'phone'], 'black box': ['black box', 'box'], 'card box': ['playing cards box', 'card box', 'box'], 'Rubik cube': ['Rubik cube', 'Rubik cube puzzle'], 'toy blocks': ['toy blocks', 'blocks']}

def detections(vision, image, prompt):
    old = vision.processor.confidence_threshold
    vision.processor.confidence_threshold = 0.2
    try:
        d = vision.detect(image, prompt)
    finally:
        vision.processor.confidence_threshold = old
    h, w = image.shape[:2]
    out = []
    for score, box, mask in zip(d['scores'], d['boxes'], d['masks']):
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        pixel = (box[:2] + box[2:]) / 2
        if score < 0.3 or not 100 < area < 0.15 * h * w or (not (18 < pixel[0] < w - 12 and pixel[1] > h * 0.39)):
            continue
        xy = np.argwhere(np.squeeze(mask) > 0.5)[:, [1, 0]]
        angle = None
        ratio = None
        if len(xy) > 20:
            values, vectors = np.linalg.eigh(np.cov(xy.T))
            axis = vectors[:, -1]
            angle = float(np.arctan2(axis[1], axis[0]))
            ratio = float(values[-1] / max(values[0], 1e-06))
        out.append({'pixel': pixel, 'box': box, 'confidence': float(score), 'prompt': prompt, 'mask_angle': angle, 'mask_axis_ratio': ratio})
    return sorted(out, key=lambda x: x['confidence'], reverse=True)

def detect_head_roles(vision, image, roles):
    found = {}
    cache = {}
    for role in ['A', 'B']:
        choices = []
        for prompt in ALIASES.get(roles[role], [roles[role]]):
            if prompt not in cache:
                cache[prompt] = detections(vision, image, prompt)
            choices.extend(cache[prompt])
            if choices and max((c['confidence'] for c in choices)) > 0.65:
                break
        found[role + '_candidates'] = choices
        found[role] = max(choices, key=lambda x: x['confidence']) if choices else None
    if any((found[role] is None for role in ['A', 'B'])):
        generic = detections(vision, image, 'object')
        for role, other in [('A', 'B'), ('B', 'A')]:
            if found[role] is None and found[other] is not None:
                choices = [c for c in generic if np.linalg.norm(c['pixel'] - found[other]['pixel']) > 25 and c['confidence'] > 0.35 and (c['box'][0] > 4) and (c['box'][2] < image.shape[1] - 4) and (c['box'][3] < image.shape[0] - 4)]
                choices.sort(key=lambda c: c['confidence'], reverse=True)
                if len(choices) == 1 or (len(choices) > 1 and choices[0]['confidence'] > 1.25 * choices[1]['confidence']):
                    found[role] = choices[0]
                    found[role + '_candidates'] = choices
    if found['A'] is not None and found['B'] is not None and (np.linalg.norm(found['A']['pixel'] - found['B']['pixel']) < 35):
        pairs = [(a, b) for a in found['A_candidates'] for b in found['B_candidates'] if np.linalg.norm(a['pixel'] - b['pixel']) > 35]
        if pairs:
            found['A'], found['B'] = max(pairs, key=lambda p: p[0]['confidence'] * p[1]['confidence'])
        else:
            found['A'] = None
            found['B'] = None
    return found

def point_at_height(pixel, height):
    ray = HEAD_RCW.T @ np.linalg.solve(camera_k(320, 240), np.r_[pixel, 1.0])
    return HEAD_C + ray * ((height - HEAD_C[2]) / ray[2])
TOOL = np.array([0.14, 0.0, 0.0])

def move_pose(kin, state, side, point, R=None, grip=1.0, hold=6):
    sl = slice(0, 6) if side == 'left' else slice(7, 13)
    gindex = 6 if side == 'left' else 13
    start = ee_world(kin, state[sl])
    R = start[:3, :3] if R is None else R
    angle = Rotation.from_matrix(R @ start[:3, :3].T).magnitude()
    segments = max(1, int(np.ceil(np.linalg.norm(point - start[:3, 3]) / 0.012)), int(np.ceil(angle / 0.15)))
    interpolator = Slerp([0, 1], Rotation.from_matrix(np.stack([start[:3, :3], R])))
    previous = state[sl].copy()
    rows = []
    for t in np.linspace(0, 1, segments + 1)[1:]:
        target = np.eye(4)
        target[:3, :3] = BASE_R.T @ interpolator([t]).as_matrix()[0]
        target[:3, 3] = BASE_R.T @ (start[:3, 3] + t * (point - start[:3, 3]) - BASE_P)
        q = kin.solve_pose(previous, target)
        count = max(2, int(np.ceil(np.max(np.abs(q - previous)) / 0.03)))
        if count > 65:
            raise ValueError('Discontinuous own-arm correction path')
        for alpha in np.linspace(0, 1, count + 1)[1:]:
            row = state.copy()
            row[sl] = previous + alpha * (q - previous)
            row[gindex] = grip
            rows.append(row)
        previous = q
    rows.extend((rows[-1].copy() for _ in range(hold)))
    return np.stack(rows)

def _plan_regrasp(kin, state, roles, found, priors, grasp_R, remaining, stereo=None, settings=None, transit_hold=6):
    if found['A'] is None or found['B'] is None:
        raise ValueError('Both roles not visible after clearing')
    heights = {r: priors['class_heights'].get(roles[r], {}).get('median', priors['default_height']) for r in ['A', 'B']}
    points = {r: point_at_height(found[r]['pixel'], heights[r]) for r in ['A', 'B']}
    used_stereo = []
    if stereo:
        for r in ['A', 'B']:
            candidate = stereo.get(r)
            if candidate is not None and np.linalg.norm(candidate['point'][:2] - points[r][:2]) < 0.1:
                points[r][:2] = candidate['point'][:2]
                used_stereo.append(r)
    gap = settings['placement_gap']
    sign = -1.0 if roles['direction'] == 'left' else 1.0
    target_A = points['A'].copy()
    target_A[:2] = points['B'][:2] + [sign * gap, 0.0]
    shift = target_A - points['A']
    distance = float(np.linalg.norm(shift[:2]))
    if distance < 0.025:
        raise ValueError('Estimated placement already close')
    if distance > 0.38:
        raise ValueError('Estimated correction exceeds workspace margin')
    sides = [roles['specified_arm']] if roles['specified_arm'] else ['left', 'right']
    feasible = []
    for side in sides:
        for R in [grasp_R, grasp_R @ rot([1, 0, 0], np.pi)]:
            current = state.copy()
            rows = []
            gindex = 6 if side == 'left' else 13
            pick = points['A'] - R @ TOOL
            place = target_A - R @ TOOL
            try:
                for point, grip, hold in [(pick + [0, 0, 0.07], 1.0, transit_hold), (pick, 1.0, 6)]:
                    segment = move_pose(kin[side], current, side, np.array(point), R, grip, hold=hold)
                    rows.append(segment)
                    current = segment[-1].copy()
                closed = []
                for value in [0.65, 0.3, 0.0]:
                    row = current.copy()
                    row[gindex] = value
                    closed.extend((row.copy() for _ in range(3)))
                rows.append(np.stack(closed))
                current = closed[-1].copy()
                for point, hold in [(pick + [0, 0, 0.07], transit_hold), (place + [0, 0, 0.07], transit_hold), (place, 6)]:
                    segment = move_pose(kin[side], current, side, np.array(point), R, 0.0, hold=hold)
                    rows.append(segment)
                    current = segment[-1].copy()
                opened = []
                for value in [0.3, 0.6, 1.0]:
                    row = current.copy()
                    row[gindex] = value
                    opened.extend((row.copy() for _ in range(3)))
                rows.append(np.stack(opened))
                current = opened[-1].copy()
                segment = move_pose(kin[side], current, side, np.array(place) + [0, 0, 0.06], R, 1.0, hold=transit_hold)
                rows.append(segment)
                actions = np.concatenate(rows)
            except ValueError:
                continue
            if len(actions) <= remaining - 10:
                details = {'arm': side, 'estimated_A': points['A'].tolist(), 'estimated_B': points['B'].tolist(), 'estimated_goal_A': target_A.tolist(), 'stereo_roles': used_stereo, 'gap': gap, 'counted_steps': len(actions), 'height_source': 'native RGB/model-action class statistics, no object GT'}
                if transit_hold == 0:
                    details.update(segment_lengths=[len(s) for s in rows], transit_dwell_steps=0,
                                   pre_grasp_dwell_steps=6, pre_release_dwell_steps=6)
                feasible.append((len(actions), actions, details))
    if not feasible:
        raise ValueError('No reachable regrasp within original action budget')
    _, actions, details = min(feasible, key=lambda x: x[0])
    return (actions, details)

def plan_regrasp(kin, state, roles, found, priors, grasp_R, remaining, stereo=None, settings=None):
    if (settings or {}).get('budget_contact_compaction'):
        from .contact_budget import plan_with_contact_budget
        return plan_with_contact_budget(_plan_regrasp, kin, state, roles, found, priors,
                                        grasp_R, remaining, stereo=stereo, settings=settings)
    return _plan_regrasp(kin, state, roles, found, priors, grasp_R, remaining,
                         stereo=stereo, settings=settings)


class PlacementRecovery(Actor):

    def _load_model(self):
        super()._load_model()
        if not hasattr(self, '_vision'):
            self._vision = self.vision(MugRGBPerception)
            urdf = os.environ['ROBOTWIN_ROOT'] + '/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf'
            self._kin = {s: RightArmKinematics(urdf, side=s) for s in ['left', 'right']}
            self._priors = self.settings['parameters']['grasp_heights']


    async def on_episode_start(self, config, ctx):
        if not hasattr(self, '_sessions'):
            self._sessions = {}
        self._sessions[ctx.session_id] = {'n': 0, 'phase': None, 'done': False, 'states': deque(maxlen=self.settings.get('stall_window', 33)), 'grasp_R': None, 'ever_closed': False}
        return await super().on_episode_start(config, ctx)

    def _normalize_result(self, result, ctx):
        self._session_chunk_sizes[ctx.session_id] = self.chunk_size
        native = super()._normalize_result(result, ctx)
        if self.settings.get('observe_before_close', False):
            from .preclose import placement_prefix
            return placement_prefix(self, native, ctx)
        return native


    async def on_observation(self, obs, ctx):
        sid = ctx.session_id
        m = self._sessions[sid]
        m['n'] += 1
        state = self._state_vector_from_obs(obs).copy()
        m['states'].append(state)
        if 'initial' not in m:
            m['initial'] = state.copy()
            m['roles'] = instruction_roles(obs.get('task_description', ''))
        side = 'left' if np.linalg.norm(state[:6] - m['initial'][:6]) > np.linalg.norm(state[7:13] - m['initial'][7:13]) else 'right'
        sl = slice(0, 6) if side == 'left' else slice(7, 13)
        grip = 6 if side == 'left' else 13
        if state[grip] < 0.2:
            m['ever_closed'] = True
            if m['grasp_R'] is None:
                m['grasp_R'] = ee_world(self._kin[side], state[sl])[:3, :3].copy()
        if m['phase'] == 'measure' and (sid not in self._chunk_buffers or self._chunk_buffers[sid].empty):
            images = [self._image_from_obs_dict(obs['images'], name, index) for name, index in [(self.image_name, 0), (self.wrist_image_name, 1), (self.right_wrist_image_name, 2)]]
            found = detect_head_roles(self._vision, images[0], m['roles'])
            m['phase'] = None
            m['done'] = True
            try:
                stereo = estimate_roles(self._vision, images, state, self._kin, m['roles'])
                R = m['grasp_R'] if m['grasp_R'] is not None else ee_world(self._kin[side], state[sl])[:3, :3]
                rows, details = plan_regrasp(self._kin, state, m['roles'], found, self._priors, R, self.settings['max_steps'] - m['n'], stereo, settings=self.settings)
            except (ValueError, np.linalg.LinAlgError) as exc:
                pass
            else:
                await self._send_script(obs, ctx, rows)
                return
        if not m['done'] and m['phase'] is None and (m['roles'] is not None) and m['ever_closed'] and (self.settings['trigger'][0] <= m['n'] <= self.settings['trigger'][1]) and (m['n'] % self.settings['trigger'][2] == 0) and (len(m['states']) == self.settings['stall_window']):
            spread = float(np.max(np.linalg.norm(np.stack(m['states'])[:, sl] - state[sl], axis=1)))
            if spread < self.settings['stall_threshold'] and (not self.settings['require_open'] or state[grip] > 0.8):
                start = ee_world(self._kin[side], state[sl])
                try:
                    rows = move_pose(self._kin[side], state, side, start[:3, 3] + [-self.settings['clear_outward'] if side == 'left' else self.settings['clear_outward'], 0, self.settings['clear_up']], grip=1.0, hold=self.settings['clear_hold'])
                except ValueError as exc:
                    pass
                else:
                    m['phase'] = 'measure'
                    await self._send_script(obs, ctx, rows)
                    return
        return await super().on_observation(obs, ctx)

    async def on_episode_end(self, result, ctx):
        try:
            return await super().on_episode_end(result, ctx)
        finally:
            m = self._sessions.pop(ctx.session_id, None)

def objects(vision, image, prompt, kind):
    threshold = vision.processor.confidence_threshold
    vision.processor.confidence_threshold = 0.25
    try:
        result = vision.detect(image, prompt)
    finally:
        vision.processor.confidence_threshold = threshold
    h, w = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(float)
    detections = []
    for score, box, raw_mask in zip(result['scores'], result['boxes'], result['masks']):
        x1, y1, x2, y2 = map(float, box)
        area = (x2 - x1) * (y2 - y1)
        pixel = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        if score < 0.32 or not (8 < pixel[0] < w - 8 and pixel[1] > h * 0.36):
            continue
        if kind == 'basket':
            if not 800 < area < h * w * 0.55:
                continue
        elif not 65 < area < h * w * 0.11:
            continue
        mask = (np.squeeze(raw_mask) > 0.5).astype(np.uint8)
        core = cv2.erode(mask, np.ones((3, 3), np.uint8)).astype(bool)
        if core.sum() < 20:
            core = mask.astype(bool)
        if core.sum() < 25:
            continue
        color = np.median(lab[core], axis=0)
        spread = np.median(np.abs(lab[core] - color), axis=0)
        if kind == 'cube' and (pixel[1] < h * 0.43 or not 0.42 < (x2 - x1) / max(y2 - y1, 1) < 1.9):
            continue
        point = point_at_height(pixel, 0.75)
        camera_depth = (HEAD_RCW @ (point - HEAD_C))[2]
        detections.append({'pixel': pixel, 'box': np.array(box), 'confidence': float(score), 'color': color, 'color_spread': spread, 'mask_area': int(mask.sum()), 'metric_area_proxy': float(mask.sum() * camera_depth ** 2)})
    return sorted(detections, key=lambda item: item['confidence'], reverse=True)

def cubes(vision, image):
    found = objects(vision, image, 'cube', 'cube')
    if len(found) != 3:
        for candidate in objects(vision, image, 'block', 'cube'):
            if all((np.linalg.norm(candidate['pixel'] - other['pixel']) > 18 for other in found)):
                found.append(candidate)
    if len(found) > 3:
        found = sorted(found, key=lambda item: item['pixel'][1], reverse=True)[:3]
    return found

def rank_initial(found):
    if len(found) != 3:
        raise ValueError('Three initial RGB cubes required')
    return sorted(found, key=lambda item: item['metric_area_proxy'], reverse=True)

def match_ranks(initial, found):
    if len(initial) != 3 or len(found) != 3:
        raise ValueError('Three current RGB cubes required')
    weights = np.array([0.25, 1.0, 1.0])
    costs = np.array([[np.linalg.norm((a['color'] - b['color']) * weights) for b in found] for a in initial])
    ranks, indices = linear_sum_assignment(costs)
    if np.max(costs[ranks, indices]) > 45:
        raise ValueError('Cube color association uncertain')
    return [found[int(indices[list(ranks).index(rank)])] for rank in range(3)]


def transfer(kin, state, source, destination, rotations, remaining):
    feasible = []
    for side in ['left', 'right']:
        grip_index = 6 if side == 'left' else 13
        orientation = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        for R in [orientation, orientation @ rot([1, 0, 0], np.pi)]:
            current, segments = (state.copy(), [])

            def move(point, grip, hold=5):
                nonlocal current
                actions = move_pose(kin[side], current, side, np.asarray(point), R, grip, hold=hold)
                segments.append(actions)
                current = actions[-1].copy()

            def set_grip(values):
                nonlocal current
                rows = []
                for value in values:
                    current = current.copy()
                    current[grip_index] = value
                    rows.extend((current.copy() for _ in range(3)))
                segments.append(np.stack(rows))
            pick = np.asarray(source) - R @ TOOL
            place = np.asarray(destination) - R @ TOOL
            try:
                move(pick + [0, 0, 0.08], 1.0)
                move(pick, 1.0, hold=8)
                set_grip([0.65, 0.3, 0.0])
                segments.append(np.stack([current.copy() for _ in range(12)]))
                move(pick + [0, 0, 0.08], 0.0)
                move(place + [0, 0, 0.08], 0.0)
                release = place.copy()
                release[2] += max(0.018, 0.773 - float(destination[2]))
                move(release, 0.0, hold=10)
                set_grip([0.3, 0.6, 1.0])
                move(place + [0, 0, 0.1], 1.0)
            except ValueError:
                continue
            actions = np.concatenate(segments)
            if len(actions) < remaining - 10:
                feasible.append((len(actions), actions, side))
    if not feasible:
        raise ValueError('No reachable cube transfer in original remaining budget')
    _, actions, side = min(feasible, key=lambda item: item[0])
    return (actions, side)

def plan_reorder(kin, state, points, goals, rotations, remaining):
    points = np.array(points).copy()
    current, rows, details = (state.copy(), [], [])
    for _ in range(7):
        wrong = [i for i in range(3) if np.linalg.norm(points[i, :2] - goals[i, :2]) > 0.024]
        if not wrong:
            break
        rank = wrong[0]
        occupied = [i for i in range(3) if i != rank and np.linalg.norm(points[i, :2] - goals[rank, :2]) < 0.055]
        if occupied:
            rank = occupied[0]
            alternatives = [np.array([x, -0.06, points[rank, 2]]) for x in [-0.19, 0.19]]
            destinations = [p for p in alternatives if all((i == rank or np.linalg.norm(p[:2] - points[i, :2]) > 0.07 for i in range(3)))]
        else:
            destinations = [goals[rank]]
        options = []
        for destination in destinations:
            try:
                actions, side = transfer(kin, current, points[rank], destination, rotations, remaining - sum((len(x) for x in rows)))
            except ValueError:
                continue
            options.append((len(actions), actions, side, np.asarray(destination)))
        if not options:
            raise ValueError('Complete reorder cannot fit reachable native action budget')
        _, actions, side, destination = min(options, key=lambda item: item[0])
        details.append({'rank': rank, 'arm': side, 'source': points[rank].tolist(), 'destination': destination.tolist(), 'counted_steps': len(actions), 'temporary': bool(occupied)})
        rows.append(actions)
        current = actions[-1].copy()
        points[rank] = destination
    if any((np.linalg.norm(points[i, :2] - goals[i, :2]) > 0.024 for i in range(3))):
        raise ValueError('Reorder did not produce a complete collision-free sequence')
    if not rows:
        raise ValueError('RGB ordering already close to command-derived target row')
    return (np.concatenate(rows), details)

class _BlockReordering(Actor):

    def _load_model(self):
        super()._load_model()
        if not hasattr(self, '_vision'):
            self._vision = self.vision(MugRGBPerception)
            urdf = os.environ['ROBOTWIN_ROOT'] + '/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf'
            self._kin = {side: RightArmKinematics(urdf, side=side) for side in ['left', 'right']}
            self._parameters = self.settings['parameters']['ranking_geometry']


    async def on_episode_start(self, config, ctx):
        if not hasattr(self, '_sessions'):
            self._sessions = {}
        self._sessions[ctx.session_id] = {'n': 0, 'done': False, 'phase': None, 'rotations': {}, 'states': deque(maxlen=self.settings.get('stall_window', 33)), 'previous_points': None}
        return await super().on_episode_start(config, ctx)

    def _normalize_result(self, result, ctx):
        self._session_chunk_sizes[ctx.session_id] = self.chunk_size
        return super()._normalize_result(result, ctx)


    def _points(self, initial, image):
        matched = match_ranks(initial, cubes(self._vision, image))
        return np.array([point_at_height(obj['pixel'], self._parameters['rank_heights'][str(rank)]['median']) for rank, obj in enumerate(matched)])

    async def on_observation(self, obs, ctx):
        sid = ctx.session_id
        memory = self._sessions[sid]
        memory['n'] += 1
        state = self._state_vector_from_obs(obs).copy()
        memory['states'].append(state)
        image = self._image_from_obs_dict(obs['images'], self.image_name, 0)
        if 'initial' not in memory:
            try:
                memory['initial'] = rank_initial(cubes(self._vision, image))
            except ValueError as exc:
                memory['initial'] = []
                memory['done'] = True
        for side, sl, grip in [('left', slice(0, 6), 6), ('right', slice(7, 13), 13)]:
            if state[grip] < 0.25 and side not in memory['rotations']:
                memory['rotations'][side] = ee_world(self._kin[side], state[sl])[:3, :3].copy()
        empty = sid not in self._chunk_buffers or self._chunk_buffers[sid].empty
        if memory['phase'] == 'measure' and empty:
            memory['phase'] = None
            memory['done'] = True
            try:
                points = self._points(memory['initial'], image)
                goals = self._relative_goals(points)
                goals[:, 2] = points[:, 2]
                try:
                    actions, moves = plan_reorder(self._kin, state, points, goals, memory['rotations'], 1200 - memory['n'])
                except ValueError as error:
                    if not self.settings.get('retain_visible_ranking_slots', False) or str(error) != 'Complete reorder cannot fit reachable native action budget':
                        raise
                    from .ranking_slots import retained_slot_plan
                    try:
                        actions, moves, proof = retained_slot_plan(self._kin, state, points, memory['rotations'], 1200 - memory['n'], transfer)
                    except ValueError:
                        raise error
                    goals = np.asarray(proof['goals'])
            except (ValueError, np.linalg.LinAlgError) as exc:
                pass
            else:
                await self._send_script(obs, ctx, actions)
                return
        if not memory['done'] and memory['phase'] is None and (self.settings['trigger'][0] <= memory['n'] <= self.settings['trigger'][1]) and (memory['n'] % self.settings['trigger'][2] == 0):
            try:
                points = self._points(memory['initial'], image)
                wrong = bool(points[0, 0] > points[1, 0] + 0.035 or points[1, 0] > points[2, 0] + 0.035 or np.ptp(points[:, 1]) > 0.055)
                near_row = bool(np.all(np.abs(points[:, 0]) < 0.22) and np.all((points[:, 1] > -0.24) & (points[:, 1] < -0.08)))
                stable = memory['previous_points'] is not None and np.max(np.linalg.norm(points[:, :2] - memory['previous_points'][:, :2], axis=1)) < 0.018
                memory['previous_points'] = points.copy()
                if wrong and stable and near_row:
                    current = state.copy()
                    rows = []
                    for side, sl in [('left', slice(0, 6)), ('right', slice(7, 13))]:
                        start = ee_world(self._kin[side], current[sl])
                        segment = move_pose(self._kin[side], current, side, start[:3, 3] + [0, 0, 0.08], grip=1.0, hold=10)
                        rows.append(segment)
                        current = segment[-1].copy()
                    memory['phase'] = 'measure'
                    actions = np.concatenate(rows)
                    await self._send_script(obs, ctx, actions)
                    return
            except (ValueError, np.linalg.LinAlgError) as exc:
                pass
        return await super().on_observation(obs, ctx)

    async def on_episode_end(self, result, ctx):
        try:
            return await super().on_episode_end(result, ctx)
        finally:
            memory = self._sessions.pop(ctx.session_id, None)

class _OcclusionClearing(_BlockReordering):

    async def on_observation(self, obs, ctx):
        sid = ctx.session_id
        m = self._sessions[sid]
        n = m['n'] + 1
        if not m['done'] and m['phase'] is None and (len(m.get('initial', [])) == 3) and (self.settings['trigger'][0] <= n <= self.settings['trigger'][1]) and (n % self.settings['trigger'][2] == 0):
            image = self._image_from_obs_dict(obs['images'], self.image_name, 0)
            try:
                self._points(m['initial'], image)
            except (ValueError, np.linalg.LinAlgError) as exc:
                m['occluded_checks'] = m.get('occluded_checks', 0) + 1
                state = self._state_vector_from_obs(obs).copy()
                history = list(m['states'])[-32:] + [state]
                joints = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
                spread = float(np.max(np.linalg.norm(np.stack(history)[:, joints] - state[joints], axis=1)))
                if m['occluded_checks'] >= 2 and len(history) == 33 and (spread < 0.2) and m['rotations']:
                    current = state.copy()
                    segments = []
                    try:
                        for side, sl in [('left', slice(0, 6)), ('right', slice(7, 13))]:
                            pose = ee_world(self._kin[side], current[sl])
                            rows = move_pose(self._kin[side], current, side, pose[:3, 3] + [-0.06 if side == 'left' else 0.06, 0, 0.06], grip=1.0, hold=10)
                            segments.append(rows)
                            current = rows[-1].copy()
                        actions = np.concatenate(segments)
                    except ValueError as clear_error:
                        pass
                    else:
                        m['n'] = n
                        m['states'].append(state)
                        m['phase'] = 'measure'
                        await self._send_script(obs, ctx, actions)
                        return
            else:
                m['occluded_checks'] = 0
        return await super().on_observation(obs, ctx)

def match_cube_colors(initial, image):
    if len(initial) != 3:
        raise ValueError('Three initial RGB color references required')
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    refs = np.asarray([x['color'] for x in initial], dtype=np.float32)
    chroma = np.linalg.norm(refs[:, 1:] - 128.0, axis=1)
    if np.min(chroma) < 12 or any((np.linalg.norm(refs[a, 1:] - refs[b, 1:]) < 12 for a in range(3) for b in range(a))):
        raise ValueError('Initial colors are too neutral or similar for a color-only fallback')
    weights = np.array([[0.6 if c < 12 else 0.18, 1.0, 1.0] for c in chroma])
    distances = np.linalg.norm((lab[:, :, None, :] - refs[None, None, :, :]) * weights[None, None, :, :], axis=3)
    winner = distances.argmin(axis=2)
    out = []
    for rank, reference in enumerate(initial):
        limit = 10.0 if chroma[rank] < 12 else 22.0
        selected = (winner == rank) & (distances[:, :, rank] < limit) & (lab[:, :, 0] > 35)
        selected[:96] = False
        selected[235:] = False
        selected[:, :18] = False
        selected[:, 303:] = False
        mask = cv2.morphologyEx(selected.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
        choices = []
        expected = float(reference.get('mask_area', 400))
        for k in range(1, count):
            x, y, w, h, area = stats[k]
            if not (max(28.0, 0.05 * expected) < area < min(3500.0, 3 * expected) and 6 <= w < 90 and (6 <= h < 90) and (0.3 < w / h < 3.2) and (area / (w * h) > 0.35)):
                continue
            pixels = labels == k
            error = float(np.median(distances[:, :, rank][pixels]))
            score = error + 0.8 * abs(np.log(max(area, 1) / expected))
            choices.append((score, {'pixel': np.array([x + w / 2, y + h / 2]), 'box': np.array([x, y, x + w, y + h]), 'confidence': float(np.exp(-error / 22.0)), 'color': np.median(lab[pixels], axis=0), 'mask_area': int(area), 'source': 'initial RGB color tracking', 'color_distance': error}))
        if not choices:
            raise ValueError('Initial RGB color is not sufficiently visible')
        out.append(min(choices, key=lambda x: x[0])[1])
    if any((np.linalg.norm(out[a]['pixel'] - out[b]['pixel']) < 14 for a in range(3) for b in range(a))):
        raise ValueError('RGB color centers overlap')
    return out

class RankingRecovery(_OcclusionClearing):

    async def on_observation(self, obs, ctx):
        return await super().on_observation(obs, ctx)

    def _points(self, initial, image):
        try:
            return super()._points(initial, image)
        except ValueError:
            found = match_cube_colors(initial, image)
            points = np.array([point_at_height(item['pixel'], self._parameters['rank_heights'][str(rank)]['median']) for rank, item in enumerate(found)])
            return points

    @staticmethod
    def _relative_goals(points):
        slots = np.sort(points[:, 0])
        if np.any(np.diff(slots) < 0.06) or np.any(np.diff(slots) > 0.115):
            slots = np.median(slots) + np.array([-0.085, 0, 0.085])
        goals = points.copy()
        goals[:, 0] = slots
        goals[:, 1] = np.median(points[:, 1])
        return goals

def bottles(vision, image):
    old = vision.processor.confidence_threshold
    vision.processor.confidence_threshold = 0.25
    try:
        result = vision.detect(image, 'bottle')
    finally:
        vision.processor.confidence_threshold = old
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(float)
    out = []
    for score, box, raw_mask in zip(result['scores'], result['boxes'], result['masks']):
        pixel = (box[:2] + box[2:]) / 2
        size = box[2:] - box[:2]
        if score < 0.35 or not (5 < pixel[0] < 315 and 10 < pixel[1] < 235) or (not 150 < np.prod(size) < 50000):
            continue
        if size[1] < size[0] * 1.05:
            continue
        mask = np.squeeze(raw_mask) > 0.5
        if mask.sum() < 40:
            continue
        item = {'pixel': pixel, 'box': box, 'confidence': float(score), 'color': np.median(lab[mask], axis=0), 'area': int(mask.sum())}
        if all((np.linalg.norm(pixel - x['pixel']) > 20 for x in out)):
            out.append(item)
    return out

class _BottleGraspCorrection(Actor):

    def _load_model(self):
        super()._load_model()
        self._bottle_vision = self.vision(MugRGBPerception)
        urdf = os.environ['ROBOTWIN_ROOT'] + '/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf'
        self._bottle_kin = {s: RightArmKinematics(urdf, side=s) for s in ['left', 'right']}


    async def on_episode_start(self, config, ctx):
        if not hasattr(self, '_bottle_sessions'):
            self._bottle_sessions = {}
        self._bottle_sessions[ctx.session_id] = {'n': 0, 'seen': {'left': False, 'right': False}}
        return await super().on_episode_start(config, ctx)

    async def on_observation(self, obs, ctx):
        m = self._bottle_sessions[ctx.session_id]
        m['n'] += 1
        m['head'] = self._image_from_obs_dict(obs['images'], self.image_name, 0)
        if 'initial' not in m:
            found = bottles(self._bottle_vision, m['head'])
            selected = sorted(found, key=lambda x: x['area'] * (1 + x['pixel'][1] / 120), reverse=True)[:2]
            m['initial'] = sorted(selected, key=lambda x: x['pixel'][0]) if len(selected) == 2 else []
        return await super().on_observation(obs, ctx)

    async def on_episode_end(self, result, ctx):
        try:
            return await super().on_episode_end(result, ctx)
        finally:
            self._bottle_sessions.pop(ctx.session_id, None)

class BottleRecovery(_BottleGraspCorrection):

    def _normalize_result(self, result, ctx):
        sid = ctx.session_id
        m = self._bottle_sessions[sid]
        self._session_chunk_sizes[sid] = self.chunk_size
        forecast = np.asarray(result['actions'], dtype=np.float32)
        native = Actor._normalize_result(self, result, ctx)
        actions = np.asarray(native['actions'], dtype=np.float32).copy()
        current = None
        for index, side, sl, g in [(0, 'left', slice(0, 6), 6), (1, 'right', slice(7, 13), 13)]:
            closed = np.flatnonzero(forecast[:, g] < 0.2)
            if m['seen'][side] or not len(closed):
                continue
            close_index = int(closed[0])
            if close_index < len(actions):
                m['seen'][side] = True
            if len(m['initial']) != 2:
                continue
            if current is None:
                current = bottles(self._bottle_vision, m['head'])
            initial = m['initial'][index]
            matches = []
            for item in current:
                distance = float(np.linalg.norm(item['pixel'] - initial['pixel']))
                color = float(np.linalg.norm((item['color'] - initial['color']) * [0.25, 1.0, 1.0]))
                if distance < 35 and color < 50:
                    matches.append((distance + 0.1 * color, item))
            if not matches:
                continue
            item = min(matches, key=lambda x: x[0])[1]
            pose = ee_world(self._bottle_kin[side], forecast[close_index, sl])
            point = pose[:3, 3] + pose[:3, :3] @ np.array(self.settings['parameters']['grasp_correction']['estimated_grasp_point'])
            visual = point_at_height(item['pixel'], point[2])
            delta = visual[:2] - point[:2]
            distance = float(np.linalg.norm(delta))
            if not 0.003 < distance < 0.08:
                continue
            outward = point[:2] - BASE_P[:2]
            outward /= max(float(np.linalg.norm(outward)), 1e-08)
            radial = float(delta @ outward)
            if radial >= -self.settings['parameters']['grasp_correction']['min_radial_pullback']:
                continue
            delta *= min(self.settings['parameters']['grasp_correction']['alpha'], self.settings['parameters']['grasp_correction']['max_xy_shift'] / distance)
            corrected = actions.copy()
            try:
                for j in range(len(actions)):
                    pose = ee_world(self._bottle_kin[side], actions[j, sl])
                    weight = min(1.0, (j + 1) / (close_index + 1))
                    pose[:2, 3] += weight * delta
                    target = np.eye(4)
                    target[:3, :3] = BASE_R.T @ pose[:3, :3]
                    target[:3, 3] = BASE_R.T @ (pose[:3, 3] - BASE_P)
                    q = self._bottle_kin[side].solve_pose(actions[j, sl], target)
                    if np.max(np.abs(q - actions[j, sl])) > 0.2:
                        raise ValueError('Correction would depart excessively from native joint path')
                    corrected[j, sl] = q
            except ValueError as exc:
                continue
            actions = corrected
        if self.settings.get('fully_close_native_grasp', False):
            if (forecast.shape != (50, 14) or actions.ndim != 2 or actions.shape[1] != 14
                    or not np.isfinite(forecast).all() or not np.isfinite(actions).all()):
                raise ValueError('Native bottle closure requires finite native50 and 14D targets')
            # Preserve RGB corrections; only complete an unchanged native grasp command.
            if 0 < len(actions) <= self.chunk_size and np.array_equal(actions, forecast[:len(actions)]):
                jaws = actions[:, [6, 13]].copy()
                closing = (jaws > 0) & (jaws < 0.2)
                if closing.any():
                    jaws[closing] = 0.
                    actions[:, [6, 13]] = jaws
                    self._chunk_buffers.pop(sid, None)
                    self._session_chunk_sizes[sid] = len(actions)
                    self._try_serve_from_buffer(ctx)
        return {**native, 'actions': actions, 'joints': actions}
