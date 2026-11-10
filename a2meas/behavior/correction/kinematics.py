"""R1Pro forward kinematics in the robot base frame."""

from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
import yaml
from ..observations import _project_behavior_b1k_state

ARM_SLICES = {"left": slice(7, 14), "right": slice(15, 22)}


def finite_array(value, shape, name):
    result = np.asarray(value, dtype=np.float32)
    if result.shape != tuple(shape) or not np.isfinite(result).all():
        raise ValueError(f"{name}: expected finite {tuple(shape)}, got {result.shape}")
    return result


def pose(position, quaternion):
    xyz = finite_array(position, (3,), "position")
    quat = finite_array(quaternion, (4,), "quaternion_xyzw")
    if abs(float(np.linalg.norm(quat)) - 1) > 0.01:
        raise ValueError("Expected a unit xyzw quaternion")
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(quat).as_matrix()
    result[:3, 3] = xyz
    return result


def pose_error(target, current):
    return np.r_[
        target[:3, 3] - current[:3, 3],
        Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T).as_rotvec(),
    ]


class R1ProKinematics:
    def __init__(self, robot_dir):
        root = Path(robot_dir)
        urdf = root / "urdf/r1pro.urdf"
        offsets = root / "urdf/r1pro_source_cfg.yaml"
        definition = yaml.safe_load((root / "r1pro.yaml").read_text())
        if definition["base_footprint_link_name"] != "base_link":
            raise ValueError("Unsupported R1Pro base frame")
        self.joints = {}
        for element in ET.parse(urdf).getroot().findall("joint"):
            origin = element.find("origin")
            transform = np.eye(4)
            if origin is not None:
                transform[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                transform[:3, :3] = Rotation.from_euler(
                    "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
                ).as_matrix()
            axis = element.find("axis")
            self.joints[element.find("child").get("link")] = {
                "name": element.get("name"),
                "type": element.get("type"),
                "parent": element.find("parent").get("link"),
                "origin": transform,
                "axis": np.fromstring(axis.get("xyz"), sep=" ")
                if axis is not None
                else np.zeros(3),
            }
        self.eef_offsets = {}
        for item in yaml.safe_load(offsets.read_text())["eef_vis_links"]:
            self.eef_offsets[item["link"]] = (
                item["parent_link"],
                pose(
                    **{
                        "position": item["offset"]["position"],
                        "quaternion": item["offset"]["orientation"],
                    }
                ),
            )
        self.chains = {}
        self.index = {f"torso_joint{i + 1}": i + 3 for i in range(4)}
        for arm, indices in ARM_SLICES.items():
            self.index.update(
                {f"{arm}_arm_joint{i + 1}": indices.start + i for i in range(7)}
            )
            eef_name = definition["manipulation"]["eef_link_names"][arm]
            parent, offset = self.eef_offsets[eef_name]
            chain = []
            while parent != "base_link":
                joint = self.joints[parent]
                if joint["type"] not in {"fixed", "revolute"}:
                    raise ValueError(
                        "Unexpected movable joint in R1Pro arm/trunk chain"
                    )
                if joint["type"] != "fixed" and joint["name"] not in self.index:
                    raise ValueError("Robot chain requires unavailable proprioception")
                chain.append(joint)
                parent = joint["parent"]
            self.chains[arm] = (list(reversed(chain)), offset)

    def fk(self, state, arm):
        matrix = np.eye(4)
        chain, offset = self.chains[arm]
        for joint in chain:
            matrix = matrix @ joint["origin"]
            if joint["type"] == "revolute":
                rotation = np.eye(4)
                rotation[:3, :3] = Rotation.from_rotvec(
                    joint["axis"] * state[self.index[joint["name"]]]
                ).as_matrix()
                matrix = matrix @ rotation
        matrix = matrix @ offset
        return matrix

    def verify_observation(self, raw):
        """Cross-check static FK against the allowed, base-relative proprio EEF."""
        raw = finite_array(raw, (61,), "official R1Pro proprio")
        state = _project_behavior_b1k_state(raw)
        errors = {}
        for arm, start in (("left", 17), ("right", 42)):
            measured = pose(raw[start : start + 3], raw[start + 3 : start + 7])
            error = pose_error(measured, self.fk(state, arm))
            errors[arm] = {
                "position_m": float(np.linalg.norm(error[:3])),
                "rotation_rad": float(np.linalg.norm(error[3:])),
            }
        if any(
            (
                e["position_m"] > 0.015 or e["rotation_rad"] > 0.05
                for e in errors.values()
            )
        ):
            raise ValueError(f"R1Pro FK/proprio frame mismatch: {errors}")
        return errors
