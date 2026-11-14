# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp

##
# Pre-defined configs
##

from .leg_asset import SLIDING_LEG_CFG
from .leg_params import PARAMS

# stand numbers for rewards and the fall line (LEGACY has no stand pose: use the old measured height)
_STAND_HIP = PARAMS.stand_hip if PARAMS.stand_hip is not None else PARAMS.default_hip
_STAND_KNEE = PARAMS.stand_knee if PARAMS.stand_knee is not None else PARAMS.default_knee
_STAND_SLIDER = PARAMS.stand_slider if PARAMS.stand_hip is not None else -0.55
_FALL_SLIDER = _STAND_SLIDER - 0.40 if PARAMS.stand_hip is not None else -0.75


##
# Scene definition
##


@configclass
class LegJumpingSceneCfg(InteractiveSceneCfg):

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot
    # ENV_REGEX_NS is just a shortcut to the path to the current environment in the scene
    robot: ArticulationCfg = SLIDING_LEG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    foot_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*_link", history_length=4, update_period=0.0)


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=100.0)
    leg_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["hip_joint", "knee_joint"],
        scale=0.6,
        use_default_offset=True # Makes it so policy outputs are relative to intial position
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel_knee_backlash,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["hip_joint", "knee_joint"])},
            noise=Unoise(n_min=-0.01, n_max=0.01), # rad noise
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["hip_joint", "knee_joint"])},
            noise=Unoise(n_min=-0.5, n_max=0.5), # rad/s noise
        )

        last_action = ObsTerm(func=mdp.last_action)
        jump_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "jump"})

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True
            # Real leg has no contact/height; infer stance from the last 10 control steps.
            self.history_length = 10
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged extras for the critic only (not on the real robot)."""

        slider_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_base"])},
        )
        slider_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_base"])},
        )
        airborne = ObsTerm(func=mdp.airborne_flag)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class CommandsCfg:
    jump = mdp.JumpCommandCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    randomize_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            # Hip, thigh, shin only. Not the 100 kg slider rail, not phantom base/foot.
            "asset_cfg": SceneEntityCfg("robot", body_names=["hip_link", "upper_link", "lower_link"]),
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
        },
    )

    # After mass rand so a future ".*" scale cannot wipe these. Tiny masses PhysX would otherwise set to 1 kg.
    set_phantom_mass = EventTerm(
        func=mdp.set_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base", "foot_link"]),
            "mass": PARAMS.phantom_mass,
            "preset": PARAMS.name,
        },
    )

    randomize_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",  # Use startup to prevent CPU overhead / PhysX crashes
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.2, 1.2),
            "dynamic_friction_range": (0.2, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,  # Creates 64 different random materials to sample from
        },
    )

    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup", # done once per env
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["hip_joint", "knee_joint"]),
            "stiffness_distribution_params": (0.8, 1.2),  # ±20% variation in Kp
            "damping_distribution_params": (0.8, 1.2),    # ±20% variation in Kd
            "operation": "scale",
            "distribution": "uniform",                    # "uniform" or "log_uniform"
        },
    )
    # Knee encoder slack 0–3° (includes perfect). Not the old-leg 7°. Physics knee is still ideal.
    randomize_knee_backlash = EventTerm(
        func=mdp.sample_knee_backlash,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot"), "min_rad": 0.0, "max_rad": 0.05},
    )
    # Constant knee zero error ±2° (includes perfect). Not backlash; not hip.
    randomize_knee_encoder_offset = EventTerm(
        func=mdp.sample_knee_encoder_offset,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot"), "min_rad": -0.035, "max_rad": 0.035},
    )

    # reset
    reset_leg_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["hip_joint", "knee_joint", "slider_to_base"]),
            # TODO: update pos/vel perturbation vals
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.1, 0.1),
        },
    )

    reset_knee_encoder = EventTerm(
        func=mdp.reset_knee_backlash_state,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    def __post_init__(self):
        # presets with a stand pose start on the floor there instead of the old drop
        if PARAMS.stand_hip is not None:
            self.reset_leg_position = EventTerm(
                func=mdp.reset_to_stand,
                mode="reset",
                params={"asset_cfg": SceneEntityCfg("robot")},
            )

@configclass
class RewardsCfg:
    """Level 2 rewards (spec in rl_notes rewards_l2.md)."""

    alive = RewTerm(func=mdp.is_alive, weight=0.2)
    stand_pose = RewTerm(
        func=mdp.stand_pose,
        weight=1.0,
        params={"stand_hip": _STAND_HIP, "stand_knee": _STAND_KNEE, "asset_cfg": SceneEntityCfg("robot")},
    )
    stand_height = RewTerm(
        func=mdp.stand_height,
        weight=1.0,
        params={"stand_slider": _STAND_SLIDER, "asset_cfg": SceneEntityCfg("robot")},
    )
    false_air = RewTerm(func=mdp.false_air, weight=-2.0)
    idle_motion = RewTerm(
        func=mdp.idle_motion,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    rise_air = RewTerm(
        func=mdp.rise_air,
        weight=40.0,
        params={"stand_slider": _STAND_SLIDER, "cap_m": 0.25, "y_scale": 0.08, "asset_cfg": SceneEntityCfg("robot", body_names=["foot_link"])},
    )
    # Left in from an earlier hop; weight 0 so it does not train.
    foot_air = RewTerm(
        func=mdp.foot_air,
        weight=0.0,
        params={"rest_z": 0.012, "cap_m": 0.13, "y_scale": 0.08, "max_s": 0.20, "asset_cfg": SceneEntityCfg("robot", body_names=["foot_link"])},
    )
    push_up = RewTerm(
        func=mdp.push_up,
        weight=8.0,
        params={"cap_mps": 2.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    takeoff = RewTerm(func=mdp.takeoff, weight=8.0)
    extra_takeoff = RewTerm(func=mdp.extra_takeoff, weight=-4.0)
    land_soft = RewTerm(
        func=mdp.land_soft,
        weight=-5.0,
        params={"v_ok": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    land_still = RewTerm(
        func=mdp.land_still,
        weight=-2.0,
        params={"after_s": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    return_stand_pose = RewTerm(
        func=mdp.return_stand_pose,
        weight=1.0,
        params={"stand_hip": _STAND_HIP, "stand_knee": _STAND_KNEE, "after_s": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    return_stand_height = RewTerm(
        func=mdp.return_stand_height,
        weight=1.0,
        params={"stand_slider": _STAND_SLIDER, "after_s": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    foot_slide = RewTerm(
        func=mdp.foot_slide,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["foot_link"])},
    )
    delta_action_penalty = RewTerm(func=mdp.action_l2, weight=-0.04)
    action_rate_penalty = RewTerm(func=mdp.action_rate_l2, weight=-0.12)
    torque_penalty = RewTerm(
        func=mdp.joint_torque_penalty,
        weight=-0.004,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["hip_joint", "knee_joint"])},
    )
    # Left in from an earlier hop; weight 0 (it wrecked the sit).
    torque_spike = RewTerm(
        func=mdp.torque_spike,
        weight=0.0,
        params={"frac": 0.7, "asset_cfg": SceneEntityCfg("robot", joint_names=["hip_joint", "knee_joint"])},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # TODO: Add termination criteria for minimum height
    height_term = DoneTerm(
        func=mdp.height_termination,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_base"]), "target": _FALL_SLIDER},
    )


##
# Environment configuration
##


@configclass
class LegJumpingEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: LegJumpingSceneCfg = LegJumpingSceneCfg(num_envs=8192, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        # decimation is number of sim dt per rl action
        self.decimation = 4 # 1/(decimation*sim.dt) = frequency
        self.episode_length_s = 12
        # viewer settings
        self.viewer.eye = (3.0, 3.0, 3.0)
        # simulation settings
        self.sim.dt = 1 / 240
        self.sim.render_interval = self.decimation
