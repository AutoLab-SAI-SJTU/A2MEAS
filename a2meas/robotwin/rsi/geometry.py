"""Retained normal-observation RSI geometry helpers."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

def rot(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

def rpy_matrix(rpy):
    r, p, y = rpy
    return rot([0, 0, 1], y) @ rot([0, 1, 0], p) @ rot([1, 0, 0], r)

class RightArmKinematics:

    def __init__(self, urdf, side='right', rotation_log=True):
        self.rotation_log = rotation_log
        robot = ET.parse(Path(urdf)).getroot()
        parents = {j.find('child').get('link'): j for j in robot.findall('joint')}
        prefix = 'fr' if side == 'right' else 'fl'
        self.names = [f'{prefix}_joint{i}' for i in range(1, 7)]
        chain = []
        link = f'{prefix}_link6'
        while link in parents:
            j = parents[link]
            chain.append(j)
            link = j.find('parent').get('link')
        self.chain = []
        self.limits = {}
        for j in reversed(chain):
            origin = j.find('origin')
            axis = j.find('axis')
            limit = j.find('limit')
            xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            a = np.fromstring(axis.get('xyz'), sep=' ') if axis is not None else np.array([1.0, 0, 0])
            self.chain.append((j.get('name'), xyz, rpy_matrix(rpy), a, j.get('type')))
            if j.get('name') in self.names and limit is not None:
                self.limits[self.names.index(j.get('name'))] = (float(limit.get('lower', '-6.28')), float(limit.get('upper', '6.28')))
        if not all(any(j[0] == name for j in self.chain) for name in self.names):
            raise ValueError(f"Robot model is missing required {side} arm joints")

    def forward(self, q):
        T = np.eye(4)
        origins = {}
        axes = {}
        for name, p, R, a, kind in self.chain:
            U = np.eye(4)
            U[:3, :3] = R
            U[:3, 3] = p
            T = T @ U
            if name in self.names:
                i = self.names.index(name)
                origins[i] = T[:3, 3].copy()
                axes[i] = T[:3, :3] @ a
                V = np.eye(4)
                V[:3, :3] = rot(a, float(q[i]))
                T = T @ V
            elif kind not in ('fixed',):
                pass
        J = np.empty((6, 6))
        for i in range(6):
            J[:3, i] = np.cross(axes[i], T[:3, 3] - origins[i])
            J[3:, i] = axes[i]
        return (T, J)

    def translate_xyz(self, q, delta):
        q = np.asarray(q, dtype=float).copy()
        start, _ = self.forward(q)
        target = start.copy()
        target[:3, 3] += np.asarray(delta, dtype=float)
        return self.solve_pose(q, target)

    def solve_pose(self, q, target):
        q = np.asarray(q, dtype=float).copy()
        for _ in range(80):
            now, J = self.forward(q)
            angular = 0.5 * sum((np.cross(now[:3, i], target[:3, i]) for i in range(3)), np.zeros(3))
            relative_rotation = target[:3, :3] @ now[:3, :3].T
            if self.rotation_log and np.trace(relative_rotation) < 0.0:
                angular = Rotation.from_matrix(relative_rotation).as_rotvec()
            error = np.r_[target[:3, 3] - now[:3, 3], angular]
            if np.linalg.norm(error[:3]) < 0.0003 and np.linalg.norm(error[3:]) < 0.003:
                return q
            dq = J.T @ np.linalg.solve(J @ J.T + np.eye(6) * 1e-05, error)
            q += np.clip(dq, -0.12, 0.12)
            for i, (lo, hi) in self.limits.items():
                q[i] = np.clip(q[i], lo + 1e-05, hi - 1e-05)
        raise ValueError('Robot-only local IK did not converge')
BASE_R = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
BASE_P = np.array([0.0, -0.65, 0.0])
HEAD_RCW = np.array([[1.0, 0, 0], [0, -0.8, -0.6], [0, 0.6, -0.8]])
HEAD_C = np.array([-0.032, -0.45, 1.35])

def camera_k(width=320, height=240):
    f = height / 2 / np.tan(np.deg2rad(18.5))
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])

def ee_world(kin, q):
    T, _ = kin.forward(q)
    world = np.eye(4)
    world[:3, :3] = BASE_R @ T[:3, :3]
    world[:3, 3] = BASE_R @ T[:3, 3] + BASE_P
    return world

def projection_matrices(kin, q, width=320, height=240):
    K = camera_k(width, height)
    head = K @ np.c_[HEAD_RCW, -HEAD_RCW @ HEAD_C]
    T = ee_world(kin, q)
    Rcam = T[:3, :3] @ rpy_matrix([0, 0.4, 0])
    C = T[:3, 3] + T[:3, :3] @ np.array([0.07, 0.032, 0.065])
    Rcw = np.stack([-Rcam[:, 1], -Rcam[:, 2], Rcam[:, 0]])
    wrist = K @ np.c_[Rcw, -Rcw @ C]
    return (head, wrist)

def triangulate(views):
    A = []
    for P, uv in views:
        u, v = uv
        A.extend([u * P[2] - P[0], v * P[2] - P[1]])
    _, _, vh = np.linalg.svd(np.stack(A))
    X = vh[-1]
    return X[:3] / X[3]

def project(P, X):
    homogeneous = P @ np.r_[X, 1.0]
    return homogeneous[:2] / homogeneous[2]

def wrist_projection(kin, q, width=320, height=240):
    T = ee_world(kin, q)
    Rcam = T[:3, :3] @ rpy_matrix([0, 0.4, 0])
    center = T[:3, 3] + T[:3, :3] @ np.array([0.07, 0.032, 0.065])
    Rcw = np.stack([-Rcam[:, 1], -Rcam[:, 2], Rcam[:, 0]])
    return camera_k(width, height) @ np.c_[Rcw, -Rcw @ center]

def inspect_pose(kin, current_q, target_world):
    candidates = []
    for offset in ([-0.25, -0.18, 0.1], [-0.25, -0.12, 0.15], [-0.3, -0.1, 0.08], [-0.2, -0.22, 0.08], [-0.25, -0.22, 0.18], [-0.3, -0.15, 0.18]):
        C = np.asarray(target_world) + offset
        forward = target_world - C
        forward /= np.linalg.norm(forward)
        left = np.cross([0, 0, 1.0], forward)
        left /= np.linalg.norm(left)
        up = np.cross(forward, left)
        Rcam = np.stack([forward, left, up], axis=1)
        for roll in (0.0, np.pi / 2, -np.pi / 2, np.pi):
            Rcam_roll = rot(forward, roll) @ Rcam
            Rworld = Rcam_roll @ rpy_matrix([0, 0.4, 0]).T
            position = C - Rworld @ np.array([0.07, 0.032, 0.065])
            target = np.eye(4)
            target[:3, :3] = BASE_R.T @ Rworld
            target[:3, 3] = BASE_R.T @ (position - BASE_P)
            for seed in (current_q, np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0]), np.array([0.0, 1.5, 0.5, 0.0, -0.5, 0.0])):
                try:
                    q = kin.solve_pose(seed, target)
                except ValueError:
                    continue
                score = float(np.linalg.norm(q - current_q) + 0.25 * abs(roll))
                candidates.append((score, q, C, target_world))
                break
    return min(candidates, key=lambda item: item[0]) if candidates else None

def fit_handle_plane(views, center_prior, body_center=None):
    center_prior = np.asarray(center_prior, dtype=float)
    theta = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    cos_t, sin_t = (np.cos(theta), np.sin(theta))

    def residual(params):
        center = params[:3]
        R = Rotation.from_rotvec(params[3:6]).as_matrix()
        r1, r2 = np.exp(params[6:8])
        xyz = center[:, None] + r1 * R[:, 0, None] * cos_t + r2 * R[:, 1, None] * sin_t
        X = np.vstack([xyz, np.ones((1, len(theta)))])
        res = []
        for P, ellipse in views:
            projected = P @ X
            uv = (projected[:2] / projected[2]).T
            c = np.array(ellipse['center'])
            axes = np.array(ellipse['axes']) / 2
            angle = np.deg2rad(ellipse['angle'])
            U = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            local = (uv - c) @ U
            distance = np.sqrt(np.sum((local / axes) ** 2, axis=1)) - 1
            res.extend(distance * min(axes))
            G = np.zeros((4, 3))
            G[:3, 0] = r1 * R[:, 0]
            G[:3, 1] = r2 * R[:, 1]
            G[:3, 2] = center
            G[3, 2] = 1.0
            H = P @ G
            try:
                inverse = np.linalg.inv(H)
                conic = inverse.T @ np.diag([1.0, 1.0, -1.0]) @ inverse
                predicted_center = -np.linalg.solve(conic[:2, :2], conic[:2, 2])
                res.extend((predicted_center - c) * 2.0)
            except np.linalg.LinAlgError:
                res.extend([1000.0, 1000.0])
        res.extend((center - center_prior) / 0.03)
        if body_center is not None:
            res.append(float(R[:, 2] @ (np.asarray(body_center) - center)) * 100)
        return np.asarray(res)
    rng = np.random.default_rng(7)
    best = None
    bounds_low = np.r_[center_prior - 0.04, [-6.4] * 3, np.log([0.004, 0.004])]
    bounds_high = np.r_[center_prior + 0.04, [6.4] * 3, np.log([0.07, 0.07])]
    initials = [np.zeros(3), np.array([np.pi / 2, 0, 0]), np.array([0, np.pi / 2, 0])] + [Rotation.random(random_state=rng).as_rotvec() for _ in range(7)]
    for r in initials:
        initial = np.r_[center_prior, r, np.log([0.025, 0.018])]
        result = least_squares(residual, initial, bounds=(bounds_low, bounds_high), loss='soft_l1', f_scale=1.0, max_nfev=180)
        score = float(np.mean(np.square(residual(result.x))))
        if best is None or score < best[0]:
            best = (score, result.x)
    score, p = best
    R = Rotation.from_rotvec(p[3:6]).as_matrix()
    return {'center': p[:3], 'normal': R[:, 2], 'radii': np.exp(p[6:8]), 'pixel_rmse': np.sqrt(score)}

def align_directions(source, target):
    source = np.asarray(source) / np.linalg.norm(source)
    target = np.asarray(target) / np.linalg.norm(target)
    axis = np.cross(source, target)
    length = np.linalg.norm(axis)
    dot = np.clip(source @ target, -1.0, 1.0)
    if length < 1e-08:
        if dot > 0:
            return np.eye(3)
        helper = np.eye(3)[np.argmin(np.abs(source))]
        axis = np.cross(source, helper)
        return rot(axis, np.pi)
    return rot(axis / length, np.arctan2(length, dot))

def hanging_orientation(current_R, normal_local, body_relative, branch_axis):
    candidates = []
    for sign in (1.0, -1.0):
        desired_n = np.asarray(branch_axis) * sign
        normal = current_R @ normal_local
        A = align_directions(normal, desired_n)
        b = A @ current_R @ body_relative
        b = b - desired_n * (desired_n @ b)
        down = np.array([0.0, 0.0, -1.0])
        down -= desired_n * (desired_n @ down)
        if np.linalg.norm(b) > 1e-06 and np.linalg.norm(down) > 1e-06:
            b /= np.linalg.norm(b)
            down /= np.linalg.norm(down)
            angle = np.arctan2(desired_n @ np.cross(b, down), np.clip(b @ down, -1, 1))
            A = rot(desired_n, angle) @ A
        candidates.append(A @ current_R)
    return candidates

def head_visible_pose(kin, current_q, target_world):
    direction = HEAD_C - target_world
    direction /= np.linalg.norm(direction)
    side = np.cross(direction, [0.0, 0.0, 1.0])
    side /= np.linalg.norm(side)
    candidates = []
    for distance in (0.32, 0.36, 0.4, 0.28):
        for lateral in (-0.025, 0.025, -0.05):
            C = target_world + direction * distance + side * lateral
            forward = target_world - C
            forward /= np.linalg.norm(forward)
            left = np.cross([0.0, 0.0, 1.0], forward)
            left /= np.linalg.norm(left)
            up = np.cross(forward, left)
            Rcam = np.stack([forward, left, up], axis=1)
            for roll in (0.0, np.pi / 2, -np.pi / 2, np.pi):
                Rworld = rot(forward, roll) @ Rcam @ rpy_matrix([0.0, 0.4, 0.0]).T
                target = np.eye(4)
                target[:3, :3] = BASE_R.T @ Rworld
                target[:3, 3] = BASE_R.T @ (C - Rworld @ np.array([0.07, 0.032, 0.065]) - BASE_P)
                for seed in (current_q, np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0]), np.array([0.0, 1.5, 0.5, 0.0, -0.5, 0.0])):
                    try:
                        q = kin.solve_pose(seed, target)
                    except ValueError:
                        continue
                    score = float(np.linalg.norm(q - current_q) + 0.25 * abs(roll) + 4 * abs(lateral))
                    candidates.append((score, q, C))
                    break
    return min(candidates, key=lambda x: x[0]) if candidates else None
