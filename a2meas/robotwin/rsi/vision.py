import os
'Retained normal-observation RSI vision helpers.'
import cv2
import numpy as np
import torch
from PIL import Image
import itertools
from .geometry import project, triangulate

class MugRGBPerception:

    def __init__(self):
        from a2meas.grounding import load_grounder
        self.model, self.processor = load_grounder(os.environ['RSI_SAM3_WEIGHTS'], 'cuda', threshold=0.4)

    @torch.inference_mode()
    def detect(self, image, prompt):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            state = self.processor.set_image(Image.fromarray(np.asarray(image, dtype=np.uint8)))
            result = self.processor.set_text_prompt(prompt, state=state)
        return {k: result[k].detach().float().cpu().numpy() for k in ('masks', 'scores', 'boxes')}

    def rack_tip(self, image):
        h, w = image.shape[:2]
        old_threshold = self.processor.confidence_threshold
        self.processor.confidence_threshold = 0.25
        try:
            detections = self.detect(image, 'gray stand')
        finally:
            self.processor.confidence_threshold = old_threshold
        candidates = []
        for score, box, mask in zip(detections['scores'], detections['boxes'], detections['masks']):
            x1, y1, x2, y2 = box
            if score < 0.3 or y1 > h * 0.2 or y2 - y1 < 30 or (y2 - y1 < (x2 - x1) * 0.85):
                continue
            pixels = np.argwhere(np.squeeze(mask) > 0.5)
            pixels = pixels[pixels[:, 0] < max(0, y1) + 0.4 * (y2 - y1)]
            if len(pixels) < 5:
                continue
            left = pixels[pixels[:, 1] <= pixels[:, 1].min() + 3]
            point = np.array([np.mean(left[:, 1]), np.mean(left[:, 0])])
            top_pixels = np.argwhere(np.squeeze(mask) > 0.5)
            top = top_pixels[top_pixels[:, 0] < max(float(point[1]) - 5.0, float(y1) + 3.0)]
            branch_line = None
            outward = None
            if len(top) > 8:
                stem_left = float(np.quantile(top[:, 1], 0.15))
                branch = top_pixels[(top_pixels[:, 1] < stem_left - 2) & (top_pixels[:, 0] >= point[1] - 3) & (top_pixels[:, 0] < point[1] + (y2 - y1) * 0.45)]
                if len(branch) > 10:
                    xy = branch[:, [1, 0]].astype(np.float32)
                    vx, vy, x0, y0 = cv2.fitLine(xy, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
                    branch_line = np.array([-vy, vx, vy * x0 - vx * y0], dtype=float)
                    outward = point - xy.mean(axis=0)
            candidates.append((float(score) * (y2 - y1), float(score), point, branch_line, outward))
        if not candidates:
            return None
        _, score, point, line, outward = max(candidates, key=lambda item: item[0])
        return {'pixel': point, 'confidence': score, 'branch_line': line, 'outward_pixel': outward}

    def handle_center(self, image):
        d = self.detect(image, 'mug handle')
        h, w = image.shape[:2]
        candidates = []
        for score, box, mask in zip(d['scores'], d['boxes'], d['masks']):
            x1, y1, x2, y2 = box
            if score < 0.55 or (x2 - x1) * (y2 - y1) > h * w * 0.2:
                continue
            point = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            ellipse = None
            pixels = np.argwhere(np.squeeze(mask) > 0.5)[:, [1, 0]].astype(np.float32)
            if len(pixels) >= 12:
                center, axes, angle = cv2.fitEllipse(pixels.reshape(-1, 1, 2))
                if min(axes) > 2 and max(axes) < 2.5 * max(x2 - x1, y2 - y1) and (np.linalg.norm(np.asarray(center) - point) < 20):
                    ellipse = {'center': list(map(float, center)), 'axes': list(map(float, axes)), 'angle': float(angle)}
            candidates.append((float(score), point, ellipse))
        if not candidates:
            return None
        score, point, ellipse = max(candidates, key=lambda item: item[0])
        return {'pixel': point, 'confidence': score, 'ellipse': ellipse}

    def mug_center(self, image, near):
        d = self.detect(image, 'mug')
        candidates = []
        for score, box in zip(d['scores'], d['boxes']):
            if score < 0.45:
                continue
            point = (box[:2] + box[2:]) / 2
            candidates.append((float(np.linalg.norm(point - near)), point, float(score)))
        if not candidates:
            return None
        _, point, score = min(candidates, key=lambda item: item[0])
        return {'pixel': point, 'confidence': score}

def mask_proposal(mask, confidence, source):
    mask = np.squeeze(mask) > 0.5
    pixels = np.argwhere(mask)
    if len(pixels) < 30:
        return None
    y1, x1 = pixels.min(axis=0)
    y2, x2 = pixels.max(axis=0)
    h, w = mask.shape
    if y2 - y1 < 30 or y1 > h * 0.3 or y2 - y1 < (x2 - x1) * 0.75:
        return None
    rows = []
    for v in range(int(y1), int(y1 + 0.65 * (y2 - y1)) + 1):
        us = pixels[pixels[:, 0] == v, 1]
        if len(us) > 4:
            rows.append((v, float(us.min()), float(us.max())))
    if len(rows) < 6:
        return None
    rows = np.asarray(rows)
    width = float(np.quantile(rows[:, 2] - rows[:, 1], 0.25))
    slope, intercept = np.polyfit(rows[:, 0], rows[:, 2], 1)
    left_edge = slope * pixels[:, 0] + intercept - width
    branch = pixels[(pixels[:, 1] < left_edge - 1) & (pixels[:, 0] < y1 + 0.6 * (y2 - y1))]
    if len(branch) < 10:
        return None
    coarse = branch[branch[:, 1] <= branch[:, 1].min() + 3].mean(axis=0)
    branch = branch[branch[:, 0] > coarse[0] - 6]
    if len(branch) < 10:
        return None
    xy = branch[:, [1, 0]].astype(np.float32)
    covariance = np.cov(xy.T)
    eigen = np.linalg.eigvalsh(covariance)
    if eigen[-1] < max(4.0, eigen[0] * 2.0):
        return None
    vx, vy, x0, y0 = cv2.fitLine(xy, cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(-1)
    if abs(vx) < 0.1 or not 0.25 < abs(vy / vx) < 7.0:
        return None
    left = xy[xy[:, 0] <= xy[:, 0].min() + 3]
    tip = left.mean(axis=0)
    line = np.array([-vy, vx, vy * x0 - vx * y0], dtype=float)
    return {'pixel': tip.astype(float), 'branch_line': line, 'outward_pixel': tip - xy.mean(axis=0), 'branch_mean': xy.mean(axis=0), 'confidence': float(confidence), 'visible_tip': bool(branch[:, 1].min() > 2 and branch[:, 1].max() < w - 3 and (branch[:, 0].min() > 2) and (branch[:, 0].max() < h - 3)), 'source': source}

def rod_proposals(mask, confidence, source):
    mask = np.squeeze(mask) > 0.5
    h, w = mask.shape
    proposals = []
    lines = cv2.HoughLinesP(mask.astype(np.uint8) * 255, 1, np.pi / 180, threshold=8, minLineLength=9, maxLineGap=3)
    if lines is None:
        return []
    for x1, y1, x2, y2 in lines[:, 0][:120]:
        delta = np.array([x2 - x1, y2 - y1], float)
        if abs(delta[0]) < 2 or not 0.3 < abs(delta[1] / delta[0]) < 6:
            continue
        if delta[0] * delta[1] < 0:
            continue
        delta /= np.linalg.norm(delta)
        if delta[0] < 0:
            delta = -delta
        normal = np.array([-delta[1], delta[0]])
        mid = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        widths = []
        shifts = []
        for fraction in (0.25, 0.5, 0.75):
            point = np.array([x1, y1]) + fraction * np.array([x2 - x1, y2 - y1])
            samples = point[None, :] + np.arange(-20, 21)[:, None] * normal
            u = np.rint(samples[:, 0]).astype(int)
            v = np.rint(samples[:, 1]).astype(int)
            valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            hits = np.zeros(41, bool)
            hits[valid] = mask[v[valid], u[valid]]
            if not hits[20]:
                continue
            lo = hi = 20
            while lo > 0 and hits[lo - 1]:
                lo -= 1
            while hi < 40 and hits[hi + 1]:
                hi += 1
            widths.append(hi - lo + 1)
            shifts.append((lo + hi) / 2 - 20)
        if len(widths) < 2 or np.median(widths) > 12:
            continue
        mid += normal * np.median(shifts)
        tip = mid.copy()
        clipped = False
        for step in range(1, 260):
            point = mid - delta * step
            u, v = np.rint(point).astype(int)
            if not (0 <= u < w and 0 <= v < h):
                clipped = True
                break
            if not mask[v, u]:
                break
            tip = point
        if np.linalg.norm(tip - mid) < 5:
            continue
        visible = not clipped and 3 < tip[0] < w - 4 and (3 < tip[1] < h - 4)
        line = np.r_[normal, -normal @ mid]
        proposals.append({'pixel': tip, 'branch_line': line, 'outward_pixel': -delta, 'branch_mean': mid, 'confidence': float(confidence), 'visible_tip': bool(visible), 'source': source + '_rod', 'length': float(np.linalg.norm(tip - mid))})
    return sorted(proposals, key=lambda p: p['length'], reverse=True)[:5]

def point_line_triangulate(P, point, Q, line):
    u, v = point
    A = np.stack([u * P[2] - P[0], v * P[2] - P[1], line @ Q])
    _, _, vh = np.linalg.svd(A)
    X = vh[-1]
    return X[:3] / X[3]

def line_axis(P, left, Q, right, X):
    first = left['branch_line'] @ P
    second = right['branch_line'] @ Q
    axis = np.cross(first[:3], second[:3])
    if np.linalg.norm(axis) < 1e-06:
        return None
    axis /= np.linalg.norm(axis)
    if (project(P, X + 0.02 * axis) - project(P, X)) @ left['outward_pixel'] < 0:
        axis = -axis
    return axis

class MugRobustRGBPerception(MugRGBPerception):

    def proposals(self, image):
        old = self.processor.confidence_threshold
        self.processor.confidence_threshold = 0.2
        proposals = []
        try:
            gray = self.detect(image, 'gray stand')
            for i, (mask, score) in enumerate(zip(gray['masks'], gray['scores'])):
                p = mask_proposal(mask, score, f'gray_stand_{i}')
                if p is not None:
                    proposals.append(p)
                proposals.extend(rod_proposals(mask, score, f'gray_stand_{i}'))
            poles = self.detect(image, 'pole')
            masks = np.squeeze(poles['masks'], axis=1) > 0.5 if len(poles['masks']) else []
            for i, (mask, score, box) in enumerate(zip(masks, poles['scores'], poles['boxes'])):
                proposals.extend(rod_proposals(mask, score, f'pole_{i}'))
                x1, y1, x2, y2 = box
                if score < 0.3 or y2 - y1 < 45 or y2 - y1 < (x2 - x1) * 1.8:
                    continue
                merged = mask.copy()
                for _ in range(2):
                    adjacent = cv2.dilate(merged.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
                    for other, other_score in zip(masks, poles['scores']):
                        if other_score >= 0.2 and np.count_nonzero(adjacent & other) > 5:
                            merged |= other
                p = mask_proposal(merged, score, f'pole_union_{i}')
                if p is not None:
                    proposals.append(p)
        finally:
            self.processor.confidence_threshold = old
        return sorted(proposals, key=lambda p: p['confidence'], reverse=True)[:12]

    def fallback_hook(self, head, right, P, Q):
        heads = self.proposals(head)
        wrists = self.proposals(right)
        candidates = []
        for a in heads:
            for b in wrists:
                possibilities = []
                if a['visible_tip'] and b['visible_tip']:
                    X = triangulate([(P, a['pixel']), (Q, b['pixel'])])
                    error = max(np.linalg.norm(project(P, X) - a['pixel']), np.linalg.norm(project(Q, X) - b['pixel']))
                    if error < 4:
                        possibilities.append((X, 'two_visible_tips', float(error)))
                if a['visible_tip']:
                    possibilities.append((point_line_triangulate(P, a['pixel'], Q, b['branch_line']), 'head_tip_wrist_line', 0.0))
                if b['visible_tip']:
                    possibilities.append((point_line_triangulate(Q, b['pixel'], P, a['branch_line']), 'wrist_tip_head_line', 0.0))
                for X, method, error in possibilities:
                    if not np.isfinite(X).all() or not (-0.5 < X[0] < 0.6 and -0.4 < X[1] < 0.6 and (0.5 < X[2] < 1.4)):
                        continue
                    if (P @ np.r_[X, 1])[2] < 0.1 or (Q @ np.r_[X, 1])[2] < 0.1:
                        continue
                    if any(((project(M, X) - proposal['branch_mean']) @ proposal['outward_pixel'] < 0 for M, proposal in ((P, a), (Q, b)))):
                        continue
                    line_errors = [abs(proposal['branch_line'] @ np.r_[project(M, X), 1.0]) / np.linalg.norm(proposal['branch_line'][:2]) for M, proposal in ((P, a), (Q, b))]
                    if max(line_errors) > 4:
                        continue
                    score = np.sqrt(a['confidence'] * b['confidence']) / (1.0 + error + max(line_errors) * 0.25)
                    if method == 'two_visible_tips':
                        score *= 1.15
                    candidates.append({'point': X, 'axis': line_axis(P, a, Q, b, X), 'score': float(score), 'method': method, 'head': a, 'wrist': b, 'line_errors': line_errors})
        return max(candidates, key=lambda candidate: candidate['score']) if candidates else None

def coherent_handles(records):
    if len(records) < 2 or any((not record['candidates'] for record in records)):
        return None
    best = None
    for choice in itertools.product(*(record['candidates'][:6] for record in records)):
        views = [(record['P'], candidate['pixel']) for record, candidate in zip(records, choice)]
        center = triangulate(views)
        if not np.isfinite(center).all() or np.linalg.norm(center) > 0.35:
            continue
        error = np.array([np.linalg.norm(project(P, center) - pixel) for P, pixel in views])
        if max(error) > 2.5:
            continue
        confidence = np.array([candidate['confidence'] for candidate in choice])
        score = float(error.mean() + 0.5 * error.max() - 0.5 * np.log(confidence.clip(0.01, 1)).mean())
        if best is None or score < best['score']:
            best = {'center': center, 'errors': error, 'selected': choice, 'score': score}
    return best

class MugTrackedRGBPerception(MugRobustRGBPerception):

    def handle_candidates(self, image):
        old = self.processor.confidence_threshold
        self.processor.confidence_threshold = 0.25
        try:
            d = self.detect(image, 'mug handle')
        finally:
            self.processor.confidence_threshold = old
        h, w = image.shape[:2]
        candidates = []
        for score, box, mask in zip(d['scores'], d['boxes'], d['masks']):
            x1, y1, x2, y2 = box
            if score < 0.25 or (x2 - x1) * (y2 - y1) > h * w * 0.2:
                continue
            point = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            ellipse = None
            pixels = np.argwhere(np.squeeze(mask) > 0.5)[:, [1, 0]].astype(np.float32)
            if len(pixels) >= 12:
                center, axes, angle = cv2.fitEllipse(pixels.reshape(-1, 1, 2))
                if min(axes) > 2 and max(axes) < 2.5 * max(x2 - x1, y2 - y1) and (np.linalg.norm(np.asarray(center) - point) < 20):
                    ellipse = {'center': list(map(float, center)), 'axes': list(map(float, axes)), 'angle': float(angle)}
            candidates.append({'pixel': point, 'confidence': float(score), 'ellipse': ellipse, 'point_type': 'bbox'})
            if ellipse is not None:
                candidates.append({'pixel': np.array(ellipse['center']), 'confidence': float(score), 'ellipse': ellipse, 'point_type': 'ellipse'})
        return sorted(candidates, key=lambda candidate: candidate['confidence'], reverse=True)[:6]
