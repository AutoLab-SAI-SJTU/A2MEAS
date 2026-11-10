"""Local geometry bounds and actuator scope for sixteen-step corrections."""

import numpy as np


def action_summary(base, proposed, kinematics):
    b = np.asarray(base, dtype=np.float32)[:16]
    c = np.asarray(proposed, dtype=np.float32)[:16]

    out = {
        "frame": "R1Pro base_link; FK of robot joint commands only, not contact or object prediction",
        "prefix_steps": 16,
        "gripper_controller": "official eval/r1pro.yaml: MultiFingerGripperController, mode=smooth; -1 closed, +1 open; interpolated position, not force or binary sign",
    }
    for side, sl, gi in [("left", slice(7, 14), 14), ("right", slice(15, 22), 22)]:
        pb = np.stack([kinematics.fk(a, side) for a in b])
        pc = np.stack([kinematics.fk(a, side) for a in c])
        rot = pc[:, :3, :3] @ np.swapaxes(pb[:, :3, :3], -1, -2)
        angles = np.arccos(np.clip((np.trace(rot, axis1=1, axis2=2) - 1) / 2, -1, 1))
        out[side] = {
            "joint_delta_max_rad": float(np.abs(c[:, sl] - b[:, sl]).max()),
            "ee_delta_max_mm": float(
                np.linalg.norm(pc[:, :3, 3] - pb[:, :3, 3], axis=-1).max() * 1000
            ),
            "rotation_delta_max_rad": float(angles.max()),
            "residual_step_jump_rad": float(
                np.abs(np.diff(c[:, sl] - b[:, sl], axis=0)).max()
            ),
            "base_ee_first_last_m": np.round(pb[[0, 15], :3, 3], 4).tolist(),
            "corrected_ee_first_last_m": np.round(pc[[0, 15], :3, 3], 4).tolist(),
            "base_gripper_commands": np.round(b[:, gi], 3).tolist(),
            "corrected_gripper_commands": np.round(c[:, gi], 3).tolist(),
        }
    return out


def guarded_actions(base, proposed, trained_mask, fit, summary):
    """Restrict existing bounded residuals; never replace base commands by hold targets."""
    b = np.asarray(base, dtype=np.float32)
    p = np.asarray(proposed, dtype=np.float32)
    if b.shape != p.shape or b.shape[1:] != (23,) or not np.isfinite(p).all():
        raise ValueError("Invalid correction")
    rejected = lambda reason: (
        b.copy(),
        np.zeros(23, np.float32),
        {"accepted": False, "reason": reason},
    )

    if fit["verdict"] != "compatible" or fit["confidence"] < 0.65:
        return rejected("action_fit_rejected")
    mask = np.zeros(23, np.float32)
    fixed = b.copy()
    for side, sl, gi in [("left", slice(7, 14), 14), ("right", slice(15, 22), 22)]:
        if fit["arms"] not in (side, "both"):
            continue
        x = summary[side]
        if (
            x["ee_delta_max_mm"] > 30
            or x["rotation_delta_max_rad"] > 0.20
            or x["residual_step_jump_rad"] > 0.05
        ):
            return rejected("action_geometry_outside_local_scope")
        mask[sl] = trained_mask[sl]
        fixed[:, sl] = b[:, sl] + (p[:, sl] - b[:, sl]) * mask[sl]
        mode = fit[side + "_gripper"]
        if mode != "keep_base" and trained_mask[gi]:
            delta = p[:, gi] - b[:, gi]
            delta = (
                np.minimum(delta, 0) if mode == "closing_only" else np.maximum(delta, 0)
            )
            fixed[:, gi] = b[:, gi] + delta
            mask[gi] = 1
    if np.abs(fixed[:16] - b[:16]).max() < 1e-5:
        return rejected("action_correction_negligible")
    assert np.array_equal(fixed[:, mask == 0], b[:, mask == 0])
    return (
        fixed,
        mask,
        {
            "accepted": True,
            "reason": "observable_fit_and_local_bounds",
        },
    )
