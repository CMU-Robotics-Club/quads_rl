"""Hardware-dependent numbers in one place. Pick a preset with the LEG_PRESET environment variable."""
import os
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class LegParams:
    name: str
    hip_torque: float = 15.0  # N*m at the joint
    knee_torque: float = 15.0
    kp_hip: float = 30.0  # N*m/rad
    kp_knee: float = 30.0
    kd_hip: float = 2.0  # N*m/(rad/s)
    kd_knee: float = 2.0
    delay_min: int = 0  # physics steps (1/240 s)
    delay_max: int = 8
    default_hip: float = 0.4  # rad
    default_knee: float = 1.1  # rad
    spawn_slider: float = -0.4  # m
    phantom_mass: float | None = None  # kg for base/foot_link; None keeps PhysX's 1 kg default
    # stand reset (None = old reset: default pose +-0.1 rad, which drops the leg from 10 cm)
    stand_hip: float | None = None  # rad
    stand_knee: float | None = None  # rad
    stand_slider: float = 0.0  # m, foot just on the floor at the stand pose (URDF model, L-08)
    depth_per_hip: float = 0.254  # m of slider per rad of hip near the stand, so noise cannot push the foot into the floor
    depth_per_knee: float = 0.080
    reset_pose_noise: float = 0.03  # rad, hip and knee
    reset_lift: float = 0.003  # m, always spawn at least this far above the floor
    reset_lift_noise: float = 0.01  # m, extra upward-only random lift


# (low, high) we sweep or randomize over; not used by the env yet
RANGES = {
    "hip_torque": (3.0, 40.0),
    "knee_torque": (3.0, 40.0),
    "kp_hip": (10.0, 100.0),
    "kp_knee": (10.0, 100.0),
    "kd_hip": (0.5, 5.0),
    "kd_knee": (0.5, 5.0),
    "delay_max": (0, 8),
}

PRESETS = {
    "LEGACY": LegParams("LEGACY"),  # exactly what the 2026-10-11 run trained on
    "A": LegParams("A", phantom_mass=1e-3),  # 15 N*m, phantom masses fixed
    "B": LegParams("B", hip_torque=10.0, knee_torque=6.0, phantom_mass=1e-3),  # real clamp read as joint-side
}
# same limits, new stand: default (action zero) = stand pose + hold torque / kp, spawn on the floor at the stand pose (L-08, L-09)
_STAND = dict(default_hip=0.48, default_knee=1.58, spawn_slider=-0.43, stand_hip=0.3, stand_knee=1.5, stand_slider=-0.4435)
PRESETS["A_STAND"] = replace(PRESETS["A"], name="A_STAND", **_STAND)  # fallback: foot 25 cm ahead of the hip
PRESETS["B_STAND"] = replace(PRESETS["B"], name="B_STAND", **_STAND)
# main stand: foot under the hip, 0.50 m of 0.75 m reach, like A1 and ANYmal (L-14, L-15)
_UNDER = dict(default_hip=0.500, default_knee=1.405, spawn_slider=-0.474, stand_hip=0.55, stand_knee=1.13, stand_slider=-0.4843,
              depth_per_hip=0.0035, depth_per_knee=0.295)
PRESETS["A_UNDER"] = replace(PRESETS["A"], name="A_UNDER", **_UNDER)
PRESETS["B_UNDER"] = replace(PRESETS["B"], name="B_UNDER", **_UNDER)


def get_params() -> LegParams:
    name = os.environ.get("LEG_PRESET", "LEGACY")
    if name not in PRESETS:
        raise ValueError(f"LEG_PRESET={name!r}, expected one of {list(PRESETS)}")
    return PRESETS[name]


PARAMS = get_params()
print(f"[leg_params] preset {PARAMS.name}")
