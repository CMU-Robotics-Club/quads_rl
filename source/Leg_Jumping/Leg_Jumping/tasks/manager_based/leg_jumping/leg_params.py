"""Hardware-dependent numbers in one place. Pick a preset with the LEG_PRESET environment variable."""
import os
from dataclasses import dataclass


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


def get_params() -> LegParams:
    name = os.environ.get("LEG_PRESET", "LEGACY")
    if name not in PRESETS:
        raise ValueError(f"LEG_PRESET={name!r}, expected one of {list(PRESETS)}")
    return PRESETS[name]


PARAMS = get_params()
print(f"[leg_params] preset {PARAMS.name}")
