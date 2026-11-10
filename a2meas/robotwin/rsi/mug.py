"""Retained normal-observation RSI mug helpers."""
import os
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp
import cv2
from .actor import Actor
from .geometry import BASE_P, BASE_R, HEAD_C, HEAD_RCW, RightArmKinematics, camera_k, ee_world, fit_handle_plane, hanging_orientation, head_visible_pose, inspect_pose, project, projection_matrices, triangulate, wrist_projection
from .vision import MugTrackedRGBPerception, coherent_handles, mask_proposal, rod_proposals

def point_on_height_plane(P, pixel, height):
    u, v = pixel
    rows = np.stack([u * P[2] - P[0], v * P[2] - P[1]])
    A = np.vstack([rows[:, :3], [0.0, 0.0, 1.0]])
    b = np.r_[-rows[:, 3], height]
    return np.linalg.solve(A, b)

def monocular_hook(vision, image, prior, projection=None, extra_pass=False):
    h, w = image.shape[:2]
    proposals = []
    if not extra_pass:
        proposals = list(vision.proposals(image))
        simple = vision.rack_tip(image)
        if simple is not None:
            proposals.append({**simple, 'source': 'single_stand_mask', 'branch_mean': None})
    else:
        old = vision.processor.confidence_threshold
        vision.processor.confidence_threshold = 0.25
        try:
            for prompt in ('mug rack', 'cup rack', 'mug tree'):
                detected = vision.detect(image, prompt)
                for i, (mask, score) in enumerate(zip(detected['masks'], detected['scores'])):
                    if score < 0.3:
                        continue
                    proposal = mask_proposal(mask, score, f'{prompt}_{i}')
                    if proposal is not None:
                        proposals.append(proposal)
                    proposals.extend(rod_proposals(mask, score, f'{prompt}_{i}'))
        finally:
            vision.processor.confidence_threshold = old
    K = camera_k(w, h)
    P = K @ np.c_[HEAD_RCW, -HEAD_RCW @ HEAD_C] if projection is None else projection
    prior_axis = np.asarray(prior['branch_axis_median'], dtype=float)
    candidates = []
    for proposal in proposals:
        pixel = np.asarray(proposal['pixel'], dtype=float)
        if proposal['confidence'] < (0.3 if extra_pass else 0.45) or not (5 < pixel[0] < w - 5 and 5 < pixel[1] < h - 5):
            continue
        try:
            X = point_on_height_plane(P, pixel, float(prior['hook_height_median']))
        except np.linalg.LinAlgError:
            continue
        ylo, yhi = prior['hook_y_range']
        if (P @ np.r_[X, 1.0])[2] <= 0.1 or not (-0.5 < X[0] < 0.6 and ylo < X[1] < yhi):
            continue
        axis = prior_axis.copy()
        line = proposal.get('branch_line')
        if line is not None:
            plane = np.asarray(line) @ P
            normal = plane[:3]
            axis -= normal * (normal @ axis) / max(normal @ normal, 1e-12)
        if np.linalg.norm(axis) < 1e-06:
            continue
        axis /= np.linalg.norm(axis)
        outward = proposal.get('outward_pixel')
        if outward is not None and (project(P, X + 0.02 * axis) - pixel) @ outward < 0:
            axis = -axis
        if axis[2] < 0.2:
            continue
        score = float(proposal['confidence'])
        if line is not None:
            error = abs(np.asarray(line) @ np.r_[pixel, 1.0]) / max(np.linalg.norm(np.asarray(line)[:2]), 1e-12)
            if error > 3:
                continue
            score /= 1.0 + 0.2 * error
        candidates.append({'point': X, 'axis': axis, 'pixel': pixel, 'score': score, 'source': proposal['source'], 'height_is_RGB_estimate_not_GT': True})
    if candidates:
        return max(candidates, key=lambda x: x['score'])
    if not extra_pass:
        return monocular_hook(vision, image, prior, projection, extra_pass=True)
    return None

class HandlePixelTrack:

    def __init__(self, image, pixel, own_pixel, step, confidence):
        self.gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        self.origin_gray = self.gray.copy()
        self.center = np.asarray(pixel, dtype=np.float32).copy()
        self.origin = self.center.copy()
        self.own_origin = np.asarray(own_pixel, dtype=float).copy()
        self.step = step
        self.confidence = float(confidence)
        mask = np.zeros_like(self.gray)
        cv2.circle(mask, tuple(np.rint(pixel).astype(int)), 15, 255, -1)
        self.points = cv2.goodFeaturesToTrack(self.gray, maxCorners=40, qualityLevel=0.015, minDistance=2, mask=mask, blockSize=3)
        self.valid = self.points is not None and len(self.points) >= 4
        self.first_count = 0 if self.points is None else len(self.points)
        self.worst_fb = 0.0

    def update(self, image):
        if not self.valid:
            return
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        options = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        forward, status, _ = cv2.calcOpticalFlowPyrLK(self.gray, gray, self.points, None, **options)
        if forward is None:
            self.valid = False
            return
        backward, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.gray, forward, None, **options)
        if backward is None:
            self.valid = False
            return
        fb = np.linalg.norm(backward[:, 0] - self.points[:, 0], axis=1)
        keep = (status[:, 0] > 0) & (back_status[:, 0] > 0) & (fb <= 0.8)
        if keep.sum() < max(4, self.first_count * 0.5):
            self.valid = False
            return
        delta = forward[keep, 0] - self.points[keep, 0]
        shift = np.median(delta, axis=0)
        inliers = np.linalg.norm(delta - shift, axis=1) <= 0.9
        if inliers.sum() < max(4, self.first_count * 0.5) or np.linalg.norm(shift) > 8:
            self.valid = False
            return
        self.center += np.median(delta[inliers], axis=0)
        self.points = forward[keep][inliers].copy()
        self.worst_fb = max(self.worst_fb, float(fb[keep][inliers].max()))
        self.gray = gray
        h, w = gray.shape
        self.valid = bool(9 <= self.center[0] < w - 9 and 9 <= self.center[1] < h - 9)

    def measurement(self, own_pixel, step, parameters):
        if not self.valid:
            return (None, {'reason': 'insufficient consistent tracked RGB features'})
        motion = self.center - self.origin
        own_motion = np.asarray(own_pixel) - self.own_origin
        own_distance = float(np.linalg.norm(own_motion))
        distance = float(np.linalg.norm(motion))
        ratio = distance / max(own_distance, 1e-06)
        cosine = float(motion @ own_motion / max(distance * own_distance, 1e-06))
        old = cv2.getRectSubPix(self.origin_gray, (17, 17), tuple(self.origin)).astype(float)
        new = cv2.getRectSubPix(self.gray, (17, 17), tuple(self.center)).astype(float)
        old -= old.mean()
        new -= new.mean()
        contrast = float(np.std(old))
        correlation = float((old * new).sum() / max(np.linalg.norm(old) * np.linalg.norm(new), 1e-06))
        details = {'age': step - self.step, 'tracked_features': len(self.points), 'motion_pixels': motion.tolist(), 'own_command_motion_pixels': own_motion.tolist(), 'motion_ratio': ratio, 'motion_cosine': cosine, 'patch_correlation': correlation, 'patch_contrast': contrast, 'worst_forward_backward_error': self.worst_fb}
        accepted = step - self.step <= 80 and own_distance >= 3 and (distance >= 2) and (parameters['minimum_motion_ratio'] <= ratio <= parameters['maximum_motion_ratio']) and (cosine >= parameters['minimum_motion_cosine']) and (correlation >= parameters['minimum_patch_correlation']) and (contrast >= 8)
        return (self.center.copy() if accepted else None, details)

class MugRecovery(Actor):
    RECOVERABLE = {'Handle lost at lateral calibration view', 'Handle lost at final calibration view'}
    _minimum_prior_confidence = 0.65
    _native_prefix_release_min = 0.2

    async def on_episode_start(self, config, ctx):
        if not hasattr(self, '_sessions'):
            self._sessions = {}
        self._sessions[ctx.session_id] = {'n': 0, 'open_seen': False, 'held': 0, 'done': False, 'phase': None, 'tip': None, 'tip3d': None, 'views': [], 'forecast': None, 'ellipse_views': [], 'body_views': [], 'branch_axis': None, 'normal': None, 'body': None, 'visual_adjustments': 0, 'inspection_attempted': False, 'inspection_records': []}
        await Actor.on_episode_start(self, config, ctx)

    def _pose_trajectory(self, state, target, hold=10):
        count = max(12, int(np.ceil(np.max(np.abs(target - state[7:13])) / 0.025)))
        if count > 140:
            raise ValueError('Excessive joint excursion')
        rows = []
        for fraction in np.linspace(0, 1, count + 1)[1:]:
            row = state.copy()
            row[7:13] = state[7:13] + fraction * (target - state[7:13])
            row[13] = 0.0
            rows.append(row)
        rows.extend((rows[-1].copy() for _ in range(hold)))
        return np.stack(rows)

    def _move_handle(self, m, goal, Rworld=None):
        state = m['state']
        start = ee_world(self._kin, state[7:13])
        R = start[:3, :3] if Rworld is None else Rworld
        center = (start @ np.r_[m['relative'], 1])[:3]
        angle = Rotation.from_matrix(R @ start[:3, :3].T).magnitude()
        segments = max(1, int(np.ceil(np.linalg.norm(goal - center) / 0.012)), int(np.ceil(angle / 0.15)))
        interpolation = Slerp([0, 1], Rotation.from_matrix(np.stack([start[:3, :3], R])))
        previous = state[7:13].copy()
        rows = []
        for fraction in np.linspace(0, 1, segments + 1)[1:]:
            Ri = interpolation([fraction]).as_matrix()[0]
            point = center + fraction * (goal - center)
            target = np.eye(4)
            target[:3, :3] = BASE_R.T @ Ri
            target[:3, 3] = BASE_R.T @ (point - Ri @ m['relative'] - BASE_P)
            q = self._kin.solve_pose(previous, target)
            count = max(2, int(np.ceil(np.max(np.abs(q - previous)) / 0.025)))
            if count > 70:
                raise ValueError('Discontinuous Cartesian IK')
            for alpha in np.linspace(0, 1, count + 1)[1:]:
                row = state.copy()
                row[7:13] = previous + alpha * (q - previous)
                row[13] = 0.0
                rows.append(row)
            previous = q
        rows.extend((rows[-1].copy() for _ in range(10)))
        if len(rows) > 280:
            raise ValueError('Excessive Cartesian path length')
        return np.stack(rows)

    def _detect_handle_view(self, sid, m):
        handle = self._vision.handle_center(m['head'])
        if handle is None:
            return False
        P = m['headP'] @ ee_world(self._kin, m['state'][7:13])
        m['views'].append((P, handle['pixel']))
        if handle.get('ellipse') is not None:
            m['ellipse_views'].append((P, handle['ellipse']))
        body = self._vision.mug_center(m['head'], handle['pixel'])
        if body is not None:
            m['body_views'].append((P, body['pixel']))
        P = m['headP'] @ ee_world(self._kin, m['state'][7:13])
        m['last_visible_handle_grip_delta'] = np.asarray(m['views'][-1][1]) - project(P, np.array([0.14, 0.0, 0.0]))
        return True

    def _motion_step(self, sid, m):
        state = m['state']
        phase = m['phase']
        if phase == 'settle':
            m['phase'] = 'dither1'
            row = state.copy()
            row[13] = 0.0
            return np.stack([row.copy() for _ in range(16)])
        if phase == 'dither1':
            if not self._capture_view(sid, m):
                raise ValueError('Missing handle at calibration start')
            q = self._kin.translate_xyz(state[7:13], BASE_R.T @ np.array([-0.04, 0, 0]))
            m['phase'] = 'dither2'
            return self._pose_trajectory(state, q, hold=12)
        if phase == 'dither2':
            if not self._capture_view(sid, m):
                raise ValueError('Handle lost at lateral calibration view')
            q = self._kin.translate_xyz(state[7:13], BASE_R.T @ np.array([0, -0.035, 0]))
            m['phase'] = 'fit'
            return self._pose_trajectory(state, q, hold=12)
        if phase == 'fit':
            if not self._capture_view(sid, m):
                raise ValueError('Handle lost at final calibration view')
            views = m['views']
            initial = triangulate(views)

            def residual(X):
                return np.concatenate([project(P, X) - uv for P, uv in views])
            fit = least_squares(residual, initial, loss='soft_l1', f_scale=1.0, max_nfev=100)
            relative = fit.x
            error = np.linalg.norm(residual(relative).reshape(-1, 2), axis=1)
            if np.max(error) > 4 or np.linalg.norm(relative) > 0.35:
                raise ValueError('Uncertain RGB handle triangulation')
            if len(m['body_views']) >= 3:
                m['body'] = triangulate(m['body_views'])
            if len(m['ellipse_views']) >= 3:
                plane = fit_handle_plane(m['ellipse_views'], relative, m['body'])
                m['initial_plane_rmse'] = float(plane['pixel_rmse'])
                if plane['pixel_rmse'] < 3:
                    relative = plane['center']
                    m['normal'] = plane['normal']
            m['relative'] = relative
            T = ee_world(self._kin, state[7:13])
            handle = (T @ np.r_[relative, 1])[:3]
            m['clear_z'] = m['tip3d'][2] + 0.04
            goal = handle.copy()
            goal[2] = m['clear_z']
            m['phase'] = 'orient'
            return self._move_handle(m, goal)
        if phase == 'orient':
            T = ee_world(self._kin, state[7:13])
            handle = (T @ np.r_[m['relative'], 1])[:3]
            forecast = ee_world(self._kin, m['forecast'])[:3, :3] if m['forecast'] is not None else T[:3, :3]
            candidates = [forecast, T[:3, :3]]
            if m['normal'] is not None and m['body'] is not None and (m['branch_axis'] is not None):
                for desired in hanging_orientation(T[:3, :3], m['normal'], m['body'] - m['relative'], m['branch_axis']):
                    interpolation = Slerp([0, 1], Rotation.from_matrix(np.stack([forecast, desired])))
                    candidates.extend(interpolation([0.25, 0.5, 0.75, 1.0]).as_matrix())
            goal = m['tip3d'].copy()
            goal[2] = m['clear_z']
            feasible = []
            for R in candidates:
                score = 0.0
                if m['normal'] is not None and m['branch_axis'] is not None:
                    score += 2.0 * abs(R @ m['normal'] @ m['branch_axis'])
                if m['body'] is not None:
                    body = R @ (m['body'] - m['relative'])
                    score -= 0.5 * body[2] / max(np.linalg.norm(body), 1e-08)
                target = np.eye(4)
                target[:3, :3] = BASE_R.T @ R
                target[:3, 3] = BASE_R.T @ (m['tip3d'] - R @ m['relative'] - BASE_P)
                try:
                    self._kin.solve_pose(state[7:13], target)
                except ValueError:
                    continue
                feasible.append((score, R))
            for score, R in sorted(feasible, key=lambda item: item[0], reverse=True):
                try:
                    rows = self._move_handle(m, goal, R)
                except ValueError:
                    continue
                m['phase'] = 'lower'
                return rows
            raise ValueError('No reachable RGB-aligned placement pose')
        if phase == 'over':
            goal = m['tip3d'].copy()
            goal[2] = m['clear_z']
            m['phase'] = 'lower'
            return self._move_handle(m, goal)
        if phase == 'lower':
            goal = m['tip3d'] + np.array([0, 0, 0.004])
            m['phase'] = 'thread'
            return self._move_handle(m, goal)
        if phase == 'thread':
            detected = self._vision.handle_center(m['head'])
            goal_near = m['tip3d'] + np.array([0, 0, 0.004])
            if detected is not None and m['visual_adjustments'] < 2:
                pixel_error = float(np.linalg.norm(detected['pixel'] - project(m['headP'], goal_near)))
                if 4 < pixel_error < 35:
                    T = ee_world(self._kin, state[7:13])
                    predicted = (T @ np.r_[m['relative'], 1])[:3]
                    depth = float((HEAD_RCW @ (predicted - HEAD_C))[2])
                    K = camera_k(m['head'].shape[1], m['head'].shape[0])
                    ray = np.linalg.solve(K, np.r_[detected['pixel'], 1.0]) * depth
                    observed = HEAD_C + HEAD_RCW.T @ ray
                    m['relative'] = T[:3, :3].T @ (observed - T[:3, 3])
                    m['visual_adjustments'] += 1
                    return self._move_handle(m, goal_near)
            goal = m['tip3d'] - 0.018 * m['branch_axis'] if m['branch_axis'] is not None else m['tip3d'] + np.array([0.008, 0, -0.012])
            m['phase'] = 'release'
            return self._move_handle(m, goal)
        if phase == 'release':
            rows = []
            for value in (0.1, 0.3, 0.6, 1.0):
                row = state.copy()
                row[13] = value
                rows.extend((row.copy() for _ in range(3)))
            m['phase'] = 'retreat'
            return np.stack(rows)
        if phase == 'retreat':
            T = ee_world(self._kin, state[7:13])
            away = -(T[:3, :3] @ m['relative'])
            away = 0.05 * away / max(np.linalg.norm(away), 1e-08)
            q = self._kin.translate_xyz(state[7:13], BASE_R.T @ away)
            rows = self._pose_trajectory(state, q)
            rows[:, 13] = 1.0
            m['phase'] = None
            m['done'] = True
            return rows
        raise ValueError('Unknown active-vision stage')

    def _left_trajectory(self, state, target):
        count = max(16, int(np.ceil(np.max(np.abs(target - state[:6])) / 0.025)))
        if count > 150:
            raise ValueError('Excessive inspection camera motion')
        rows = []
        for t in np.linspace(0, 1, count + 1)[1:]:
            fraction = t * t * (3 - 2 * t)
            row = state.copy()
            row[:6] = state[:6] + fraction * (target - state[:6])
            row[6] = 1.0
            row[13] = 0.0
            rows.append(row)
        rows.extend((rows[-1].copy() for _ in range(12)))
        return np.stack(rows)

    def _inspection_record(self, sid, m, name, image, P):
        candidates = self._vision.handle_candidates(image)
        row = {'P': P, 'candidates': candidates, 'image': np.array(image).copy(), 'name': name}
        m['inspection_records'].append(row)

    def _stereo_inspection(self, sid, m):
        state = m['state']
        phase = m['phase']
        if phase == 'inspect_move':
            T = ee_world(self._kin, state[7:13])
            target = (T @ np.array([0.14, 0, 0, 1.0]))[:3]
            result = inspect_pose(self._leftkin, state[:6], target)
            if result is None:
                raise ValueError('No reachable left RGB inspection view')
            _, q, C, _ = result
            m['inspection_left_start'] = state[:6].copy()
            m['phase'] = 'inspect_measure1'
            return self._left_trajectory(state, q)
        if phase == 'inspect_measure1':
            Tr = ee_world(self._kin, state[7:13])
            leftP = wrist_projection(self._leftkin, state[:6], m['left_image'].shape[1], m['left_image'].shape[0])
            self._inspection_record(sid, m, 'head', m['head'], m['headP'] @ Tr)
            self._inspection_record(sid, m, 'left1', m['left_image'], leftP @ Tr)
            q = self._leftkin.translate_xyz(state[:6], BASE_R.T @ np.array([-0.025, 0, 0.035]))
            m['phase'] = 'inspect_measure2'
            return self._left_trajectory(state, q)
        if phase == 'inspect_measure2':
            Tr = ee_world(self._kin, state[7:13])
            leftP = wrist_projection(self._leftkin, state[:6], m['left_image'].shape[1], m['left_image'].shape[0])
            self._inspection_record(sid, m, 'left2', m['left_image'], leftP @ Tr)
            records = [r for r in m['inspection_records'] if r['candidates']]
            fitted = coherent_handles(records)
            if fitted is None:
                raise ValueError('No consistent RGB inspection match')
            relative = fitted['center']
            ellipses = []
            body_views = []
            for record, candidate in zip(records, fitted['selected']):
                if candidate['ellipse'] is not None:
                    ellipses.append((record['P'], candidate['ellipse']))
                body = self._vision.mug_center(record['image'], candidate['pixel'])
                if body is not None:
                    body_views.append((record['P'], body['pixel']))
            body = triangulate(body_views) if len(body_views) >= 2 else None
            if len(ellipses) >= 2:
                plane = fit_handle_plane(ellipses, relative, body)
                if plane['pixel_rmse'] < 3:
                    relative = plane['center']
                    m['normal'] = plane['normal']
            m['relative'] = relative
            m['body'] = body
            m['clear_z'] = m['tip3d'][2] + 0.04
            m['phase'] = 'inspect_restore'
            return self._left_trajectory(state, m['inspection_left_start'])
        if phase == 'inspect_restore':
            T = ee_world(self._kin, state[7:13])
            handle = (T @ np.r_[m['relative'], 1])[:3]
            goal = handle.copy()
            goal[2] = m['clear_z']
            m['phase'] = 'orient'
            return self._move_handle(m, goal)
        raise ValueError('Unknown inspection phase')

    def _select_recovery_actions(self, sid, m):
        if str(m['phase']).startswith('inspect_'):
            return self._inspection_actions(sid, m)
        try:
            return self._advance_motion(sid, m)
        except ValueError as exc:
            if str(exc) == 'Uncertain RGB handle triangulation' and (not m['inspection_attempted']):
                m['inspection_attempted'] = True
                m['phase'] = 'inspect_move'
                return self._inspection_actions(sid, m)
            raise

    async def _observe_recovery(self, obs, ctx):
        sid = ctx.session_id
        m = self._sessions[sid]
        m['n'] += 1
        m['state'] = self._state_vector_from_obs(obs).copy()
        m['head'] = self._image_from_obs_dict(obs['images'], self.image_name, 0)
        m['left_image'] = self._image_from_obs_dict(obs['images'], self.wrist_image_name, 1)
        if m['n'] == 1:
            tip = self._vision.rack_tip(m['head'])
            right = self._image_from_obs_dict(obs['images'], self.right_wrist_image_name, 2)
            tip_right = self._vision.rack_tip(right)
            m['headP'], rightP = projection_matrices(self._kin, m['state'][7:13], m['head'].shape[1], m['head'].shape[0])
            if tip is not None and tip_right is not None:
                views = [(m['headP'], tip['pixel']), (rightP, tip_right['pixel'])]
                X = triangulate(views)
                errors = [float(np.linalg.norm(project(P, X) - uv)) for P, uv in views]
                if max(errors) < 5 and np.isfinite(X).all():
                    m['tip3d'] = X
                if m['tip3d'] is not None and tip.get('branch_line') is not None and (tip_right.get('branch_line') is not None):
                    plane_h = tip['branch_line'] @ m['headP']
                    plane_r = tip_right['branch_line'] @ rightP
                    axis = np.cross(plane_h[:3], plane_r[:3])
                    if np.linalg.norm(axis) > 1e-06:
                        axis /= np.linalg.norm(axis)
                        if (project(m['headP'], X + 0.02 * axis) - project(m['headP'], X)) @ tip['outward_pixel'] < 0:
                            axis = -axis
                        m['branch_axis'] = axis
            if m['tip3d'] is None and self._mode == 'align':
                fallback = self._vision.fallback_hook(m['head'], right, m['headP'], rightP)
                if fallback is not None and fallback['score'] >= 0.45:
                    m['tip3d'] = fallback['point']
                    m['branch_axis'] = fallback['axis']
        g = m['state'][13]
        if g > 0.8:
            m['open_seen'] = True
        if m['open_seen'] and g < 0.2:
            m['held'] += 1
        if self._mode == 'align' and m['phase'] is None and (not m['done']) and (m['tip3d'] is not None) and (m['held'] >= 24) and (self.settings['trigger'][0] <= m['n'] <= self.settings['trigger'][1]) and (m['n'] % 8 == 0):
            handle = self._vision.handle_center(m['head'])
            if handle is not None:
                self._chunk_buffers.pop(sid, None)
                m['phase'] = 'settle'
        if m['phase'] is not None and (sid not in self._chunk_buffers or self._chunk_buffers[sid].empty):
            try:
                actions = self._next_actions(sid, m)
            except (ValueError, np.linalg.LinAlgError) as exc:
                m['done'] = True
                m['phase'] = None
                actions = None
            if actions is not None:
                self._try_serve_from_buffer(ctx)
                self._append_observation(obs, ctx)
                self._session_chunk_sizes[sid] = len(actions)
                await self._process_and_send({'actions': actions, 'joints': actions}, ctx)
                return
        await Actor.on_observation(self, obs, ctx)

    async def on_episode_end(self, result, ctx):
        try:
            await Actor.on_episode_end(self, result, ctx)
        finally:
            m = self._sessions.pop(ctx.session_id, None)

    def _thread_motion_step(self, sid, m):
        if m['phase'] == 'thread_probe':
            goal = m['thread_probe_goals'].pop(0)
            if not m['thread_probe_goals']:
                m['phase'] = 'release'
            return self._move_handle(m, goal)
        phase = m['phase']
        rows = self._motion_step(sid, m)
        if phase == 'thread' and m['phase'] == 'release' and (m['branch_axis'] is not None):
            goal = m['tip3d'] - 0.035 * m['branch_axis']
            goals = [m['tip3d'] - 0.05 * m['branch_axis'], goal] if self._thread_motion == 'backout' else [goal + np.array([0.0, 0.0, 0.008]), goal - np.array([0.0, 0.0, 0.008]), goal]
            m['thread_probe_goals'] = goals[1:]
            m['phase'] = 'thread_probe'
            return self._move_handle(m, goals[0])
        return rows

    def _inspection_with_head_view(self, sid, m):
        if m['phase'] == 'inspect_head_move':
            state = m['state']
            approx = (ee_world(self._kin, state[7:13]) @ np.array([0.14, 0.0, 0.0, 1.0]))[:3]
            detected = self._vision.handle_center(m['head'])
            if detected is not None:
                depth = float((HEAD_RCW @ (approx - HEAD_C))[2])
                ray = np.linalg.solve(camera_k(m['head'].shape[1], m['head'].shape[0]), np.r_[detected['pixel'], 1.0]) * depth
                approx = HEAD_C + HEAD_RCW.T @ ray
            result = head_visible_pose(self._leftkin, state[:6], approx)
            if result is None:
                raise ValueError('No reachable camera view near visible head direction')
            _, q, C = result
            m['inspection_records'] = []
            m['phase'] = 'inspect_measure1'
            return self._left_trajectory(state, q)
        try:
            return self._stereo_inspection(sid, m)
        except ValueError as exc:
            if str(exc) != 'No consistent RGB inspection match' or m.get('head_view_retried', False):
                raise
            m['head_view_retried'] = True
            m['phase'] = 'inspect_head_move'
            return self._inspection_actions(sid, m)

    def _next_actions(self, sid, m):
        try:
            return self._select_recovery_actions(sid, m)
        except ValueError as exc:
            if str(exc) not in self.RECOVERABLE or m['inspection_attempted']:
                raise
            delta = m.get('last_visible_handle_grip_delta')
            accepted = delta is not None and float(delta[0]) > -40.0 and (float(np.linalg.norm(delta)) < 80.0)
            if not accepted:
                raise
            m['inspection_attempted'] = True
            m['inspection_records'] = []
            m['phase'] = 'inspect_move'
            return self._inspection_actions(sid, m)

    def _mono_record(self, sid, m):
        detected = self._vision.handle_center(m['head'])
        if detected is None:
            raise ValueError('Handle missing during head parallax measurement')
        P = m['headP'] @ ee_world(self._kin, m['state'][7:13])
        body = self._vision.mug_center(m['head'], detected['pixel'])
        m['mono_views'].append((P, detected['pixel']))
        if detected['ellipse'] is not None:
            m['mono_ellipses'].append((P, detected['ellipse']))
        if body is not None:
            m['mono_bodies'].append((P, body['pixel']))

    def _parallax_inspection(self, sid, m):
        phase = m['phase']
        state = m['state']
        if phase == 'inspect_mono_restore_lower':
            rows = self._left_trajectory(state, m['inspection_left_start'])
            q = self._kin.translate_xyz(state[7:13], BASE_R.T @ np.array([0.0, 0.0, -0.035]))
            count = max(12, int(np.ceil(np.max(np.abs(q - state[7:13])) / 0.025)))
            for i, row in enumerate(rows):
                alpha = min(1.0, (i + 1) / count)
                row[7:13] = state[7:13] + alpha * (q - state[7:13])
            m['mono_views'] = []
            m['mono_ellipses'] = []
            m['mono_bodies'] = []
            m['phase'] = 'inspect_mono_side'
            return rows
        if phase == 'inspect_mono_side':
            self._mono_record(sid, m)
            sign = -1.0 if m['mono_views'][-1][1][0] > m['head'].shape[1] / 2 else 1.0
            q = self._kin.translate_xyz(state[7:13], BASE_R.T @ np.array([sign * 0.035, 0.0, 0.0]))
            m['phase'] = 'inspect_mono_forward'
            return self._pose_trajectory(state, q, hold=16)
        if phase == 'inspect_mono_forward':
            self._mono_record(sid, m)
            q = self._kin.translate_xyz(state[7:13], BASE_R.T @ np.array([0.0, -0.035, 0.0]))
            m['phase'] = 'inspect_mono_fit'
            return self._pose_trajectory(state, q, hold=16)
        if phase == 'inspect_mono_fit':
            self._mono_record(sid, m)
            views = m['mono_views']
            initial = triangulate(views)
            residual = lambda X: np.concatenate([project(P, X) - uv for P, uv in views])
            fitted = least_squares(residual, initial, loss='soft_l1', f_scale=1.0, max_nfev=120)
            relative = fitted.x
            error = np.linalg.norm(residual(relative).reshape(-1, 2), axis=1)
            if not np.isfinite(relative).all() or np.linalg.norm(relative) > 0.4 or max(error) > 4:
                raise ValueError('Inconsistent fresh head parallax geometry')
            body = triangulate(m['mono_bodies']) if len(m['mono_bodies']) >= 3 else None
            normal = None
            if len(m['mono_ellipses']) >= 3:
                plane = fit_handle_plane(m['mono_ellipses'], relative, body)
                if plane['pixel_rmse'] < 3 and np.linalg.norm(plane['center']) < 0.4:
                    relative = plane['center']
                    normal = plane['normal']
            m['relative'] = relative
            m['body'] = body
            m['normal'] = normal
            m['clear_z'] = m['tip3d'][2] + 0.04
            T = ee_world(self._kin, state[7:13])
            center = (T @ np.r_[relative, 1.0])[:3]
            goal = center.copy()
            goal[2] = m['clear_z']
            m['phase'] = 'orient'
            return self._move_handle(m, goal)
        try:
            return self._inspection_with_head_view(sid, m)
        except ValueError as exc:
            if str(exc) != 'No consistent RGB inspection match' or not m.get('head_view_retried') or m.get('mono_attempted') or (m['n'] > 550):
                raise
            m['mono_attempted'] = True
            m['phase'] = 'inspect_mono_restore_lower'
            return self._inspection_actions(sid, m)


    def _advance_motion(self, sid, m):
        if m['phase'] == 'orient' and (not m.get('mono_attempted')) and (not m['inspection_attempted']) and (not m.get('early_refresh_attempted')):
            m['early_refresh_attempted'] = True
            m['early_refresh_active'] = True
            m['early_original_state'] = m['state'].copy()
            m['early_original_geometry'] = {key: m[key].copy() if isinstance(m[key], np.ndarray) else m[key] for key in ('relative', 'body', 'normal', 'clear_z')}
            m['mono_attempted'] = True
            m.setdefault('inspection_left_start', m['state'][:6].copy())
            m['phase'] = 'inspect_mono_restore_lower'
            return self._inspection_actions(sid, m)
        return self._thread_motion_step(sid, m)

    def _return_to_original(self, sid, m, reason):
        for key, value in m['early_original_geometry'].items():
            m[key] = value.copy() if isinstance(value, np.ndarray) else value
        m['phase'] = 'inspect_early_continue'
        return self._pose_trajectory(m['state'], m['early_original_state'][7:13], hold=16)

    def _inspection_actions(self, sid, m):
        phase = m['phase']
        if phase == 'inspect_early_continue':
            m['early_refresh_active'] = False
            m['phase'] = 'orient'
            return self._advance_motion(sid, m)
        if not m.get('early_refresh_active'):
            return self._parallax_inspection(sid, m)
        try:
            actions = self._parallax_inspection(sid, m)
        except (ValueError, np.linalg.LinAlgError) as exc:
            return self._return_to_original(sid, m, str(exc))
        if phase == 'inspect_mono_fit':
            old_normal = m['early_original_geometry']['normal']
            new_normal = m['normal']
            angle = None
            if old_normal is not None and new_normal is not None:
                angle = float(np.rad2deg(np.arccos(np.clip(abs(old_normal @ new_normal), 0.0, 1.0))))
            if angle is not None and angle > 60.0 and (m.get('initial_plane_rmse', float('inf')) < 1.0):
                return self._return_to_original(sid, m, 'Conflicting RGB plane estimates: %.1f degrees' % angle)
            m['early_refresh_active'] = False
        return actions

    def _load_model(self):
        Actor._load_model(self)
        if not getattr(self, '_active_visual_checked', False):
            self._mode = self.settings.get('mode', 'align')
            if self._mode not in ('control', 'align'):
                raise ValueError(self._mode)
            self._kin = RightArmKinematics(os.environ['ROBOTWIN_ROOT'] + '/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf', rotation_log=False)
            self._leftkin = RightArmKinematics(os.environ['ROBOTWIN_ROOT'] + '/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf', side='left', rotation_log=False)
            self._vision = self.vision(MugTrackedRGBPerception)
            self._active_visual_checked = True
        self._thread_motion = self.settings.get('thread_motion', 'vertical')
        if self._thread_motion not in ('backout', 'vertical'):
            raise ValueError('Unknown threading motion')
        if not hasattr(self, '_rgb_hook_prior'):
            self._rgb_hook_prior = self.settings['parameters']['hook_prior']

    async def _observe_hook(self, obs, ctx):
        m = self._sessions[ctx.session_id]
        first = m['n'] == 0 and self._mode == 'align'
        if first:
            state = self._state_vector_from_obs(obs).copy()
            head = self._image_from_obs_dict(obs['images'], self.image_name, 0).copy()
            right = self._image_from_obs_dict(obs['images'], self.right_wrist_image_name, 2).copy()
            headP, rightP = projection_matrices(self._kin, state[7:13], head.shape[1], head.shape[0])
        await self._observe_recovery(obs, ctx)
        if first and m['tip3d'] is None:
            choices = []
            for name, image, P in (('head', head, headP), ('right_wrist', right, rightP)):
                fitted = monocular_hook(self._vision, image, self._rgb_hook_prior, P)
                if fitted is not None:
                    choices.append((name, fitted))
            if not choices:
                return
            name, fitted = max(choices, key=lambda x: x[1]['score'])
            if fitted['score'] < self._minimum_prior_confidence:
                return
            point = fitted['point']
            axis = fitted['axis']
            agreement = None
            if len(choices) == 2:
                agreement = float(np.linalg.norm(choices[0][1]['point'] - choices[1][1]['point']))
                if agreement < 0.025:
                    weights = np.array([x[1]['score'] for x in choices])
                    weights /= weights.sum()
                    point = sum((w * x[1]['point'] for w, x in zip(weights, choices)))
                    axis = sum((w * x[1]['axis'] for w, x in zip(weights, choices)))
                    axis /= np.linalg.norm(axis)
                    name = 'head_and_right_wrist_agree'
            m['tip3d'] = point
            m['branch_axis'] = axis

    def _normalize_result(self, result, ctx):
        m = self._sessions[ctx.session_id]
        if m['phase'] is None and (not m['done']):
            actions = np.asarray(result['actions'], dtype=np.float32)
            m['native_guard_plan'] = (m['n'], actions.copy())
        actions = np.asarray(result['actions'], dtype=np.float32)[:self.chunk_size].copy()
        self._sessions[ctx.session_id]['forecast'] = actions[-1, 7:13].copy()
        self._session_chunk_sizes[ctx.session_id] = self.chunk_size
        native = Actor._normalize_result(self, {**result, 'actions': actions, 'joints': actions}, ctx)
        if self.settings.get('observe_before_close', False):
            from .preclose import mug_prefix
            return mug_prefix(self, native, result, ctx)
        return native

    def _preserve_native_release(self, obs, ctx):
        m = self._sessions[ctx.session_id]
        n = m['n'] + 1
        if self._mode == 'align' and m['phase'] is None and (not m['done']) and (m['tip3d'] is not None) and (m['held'] >= 23) and (self.settings['trigger'][0] <= n <= self.settings['trigger'][1]) and (n % 8 == 0):
            plan = m.get('native_guard_plan')
            if plan is not None:
                start, actions = plan
                offset = max(0, n - start)
                future = actions[offset:, 13]
                opens = np.flatnonzero(future > 0.6)
                state = self._state_vector_from_obs(obs)
                if len(opens) and state[13] < 0.2:
                    head = self._image_from_obs_dict(obs['images'], self.image_name, 0)
                    detected = self._vision.handle_center(head)
                    if detected is not None:
                        gap = float(np.linalg.norm(detected['pixel'] - project(m['headP'], m['tip3d'])))
                        prefix = actions[offset:min(self.chunk_size, len(actions)), 13]
                        prefix_open = float(prefix.max()) if len(prefix) else 0.0
                        distances = []
                        for index in sorted(set([min(15, len(actions) - 1), min(31, len(actions) - 1), len(actions) - 1])):
                            T = ee_world(self._kin, actions[index, 7:13])
                            tool = (T @ np.array([0.14, 0.0, 0.0, 1.0]))[:3]
                            distances.append(float(np.linalg.norm(tool - m['tip3d'])))
                        accepted = min(distances) <= 0.105 and (gap <= 5.0 or (gap <= 12.0 and prefix_open > self._native_prefix_release_min))
                        if accepted:
                            m['done'] = True

    def _own_pixel(self, m):
        return project(m['headP'] @ ee_world(self._kin, m['state'][7:13]), np.array([0.14, 0.0, 0.0]))

    def _capture_view(self, sid, m):
        captured = self._detect_handle_view(sid, m)
        if captured:
            detection_pixel = m['views'][-1][1]
            m['flow_handle'] = HandlePixelTrack(m['head'], detection_pixel, self._own_pixel(m), m['n'], 0.55)
            return True
        delta = m.get('last_visible_handle_grip_delta')
        preserve_inspection = not m['inspection_attempted'] and delta is not None and (float(delta[0]) > -40.0) and (float(np.linalg.norm(delta)) < 80.0)
        height_error = abs(float(m['tip3d'][2]) - float(self._rgb_hook_prior['hook_height_median']))
        if preserve_inspection or height_error > self.settings['parameters']['tracking']['max_rgb_prior_height_delta']:
            return False
        tracker = m.get('flow_handle')
        if tracker is None:
            return False
        pixel, details = tracker.measurement(self._own_pixel(m), m['n'], self.settings['parameters']['tracking'])
        if pixel is not None and details.get('patch_contrast', 0.0) < self.settings['parameters']['tracking']['minimum_tracked_patch_contrast']:
            details['contrast_guard_declined'] = True
            pixel = None
        if pixel is None:
            return False
        P = m['headP'] @ ee_world(self._kin, m['state'][7:13])
        m['views'].append((P, pixel))
        body = self._vision.mug_center(m['head'], pixel)
        if body is not None:
            m['body_views'].append((P, body['pixel']))
        m['last_visible_handle_grip_delta'] = pixel - project(P, np.array([0.14, 0.0, 0.0]))
        return True

    def _update_handle_tracking(self, obs, ctx):
        m = self._sessions[ctx.session_id]
        head = self._image_from_obs_dict(obs['images'], self.image_name, 0)
        tracker = m.get('flow_handle')
        if tracker is not None and m['phase'] in ('dither2', 'fit'):
            tracker.update(head)

    def _existing_monocular_proposal(self, pending):
        state, head, right = (pending['state'], pending['head'], pending['right'])
        headP, rightP = projection_matrices(self._kin, state[7:13], head.shape[1], head.shape[0])
        choices = []
        for name, image, P in [('head', head, headP), ('right_wrist', right, rightP)]:
            fitted = monocular_hook(self._vision, image, self._rgb_hook_prior, P)
            if fitted is not None:
                choices.append((name, fitted))
        if not choices:
            return None
        name, fitted = max(choices, key=lambda item: item[1]['score'])
        if fitted['score'] < self._minimum_prior_confidence:
            return None
        point, axis = (fitted['point'].copy(), fitted['axis'].copy())
        agreement = None
        if len(choices) == 2:
            agreement = float(np.linalg.norm(choices[0][1]['point'] - choices[1][1]['point']))
            if agreement < 0.025:
                weights = np.array([item[1]['score'] for item in choices])
                weights /= weights.sum()
                point = sum((w * item[1]['point'] for w, item in zip(weights, choices)))
                axis = sum((w * item[1]['axis'] for w, item in zip(weights, choices)))
                axis /= np.linalg.norm(axis)
                name = 'head_and_right_wrist_agree'
        return dict(point=point, axis=axis, camera=name, confidence=float(fitted['score']), source=fitted['source'], two_camera_disagreement=agreement)

    async def on_observation(self, obs, ctx):
        m = self._sessions[ctx.session_id]
        first = m['n'] == 0
        old_phase = m['phase']
        self._update_handle_tracking(obs, ctx)
        self._preserve_native_release(obs, ctx)
        await self._observe_hook(obs, ctx)
        if not self.settings.get('consistent_hook_relocalization', False):
            return
        if first and m['tip3d'] is not None:
            difference = abs(float(m['tip3d'][2]) - self._rgb_hook_prior['hook_height_median'])
            if difference > self.settings['parameters']['tracking']['max_rgb_prior_height_delta']:
                m['_consistent_hook_pending'] = dict(old=m['tip3d'].copy(), state=self._state_vector_from_obs(obs).copy(), head=self._image_from_obs_dict(obs['images'], self.image_name, 0).copy(), right=self._image_from_obs_dict(obs['images'], self.right_wrist_image_name, 2).copy())
        pending = m.get('_consistent_hook_pending')
        if pending is None or old_phase is not None or m['phase'] != 'dither1':
            return
        m.pop('_consistent_hook_pending')
        if not np.array_equal(m['tip3d'], pending['old']):
            raise RuntimeError('Hook estimate changed during pending re-localization')
        proposed = self._existing_monocular_proposal(pending)
        if proposed is not None:
            m['tip3d'] = proposed['point']
            m['branch_axis'] = proposed['axis']
