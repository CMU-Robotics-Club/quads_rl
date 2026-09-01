# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
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

# from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # isort:skip
from .leg_asset import SLIDING_LEG_CFG


##
# Scene definition
##


@configclass
class LegJumpingSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

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
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["hip_joint", "knee_joint"])},
            noise=Unoise(n_min=-0.01, n_max=0.01), # rad noise
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["hip_joint", "knee_joint"])},
            noise=Unoise(n_min=-0.5, n_max=0.5), # rad/s noise
        )

        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True # Randomized noise TODO enable
            self.concatenate_terms = True

    # observation groups
    # Just one group for everything the policy uses (both actor and critic)
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    randomize_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
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

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Constant baseline alive reward
    # alive = RewTerm(func=mdp.is_alive, weight=1.0)
    alive = RewTerm(func=mdp.is_alive, weight=0.2)

    # Reward for jumping (reduce distance between top of pole and current slider position)
    jump_height = RewTerm(
        # func=mdp.joint_pos_target_l2,
        # func=mdp.joint_pos_target_exp,
        func=mdp.target_above_threshold,
        # weight=-1.0, # Negate since closer target = less l2 distance = higher reward
        # weight=2.0,
        # weight=100.0,
        # weight = 60.0,
        weight = 40.0,
        # weight=15.0,
        # Target 0.2
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_base"]), "target": -0.4},
    )

    takeoff_velocity = RewTerm(
        func=mdp.upward_velocity,
        # weight=1.5,
        # weight=2.0,
        # weight=5.0,
        # weight=15.0,
        weight=20.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_base"])},
    )

    foot_centered_penalty = RewTerm(
        func=mdp.body_pos_target_y_l2,
        # weight=-5.0,
        # weight=-17.0,
        # weight=-40.0,
        # weight=-70.0,
        # weight=-200.0,
        weight=-220.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["foot_link", "slider"])}
    )

    # delta penalty
    delta_action_penalty = RewTerm(
        func=mdp.action_l2, # Literally penalizes actions far from 0 (i.e. far from nominal pose since actions are deltas)
        weight=-0.04,
        # weight=-0.1,
    )

    # Penalize large action difference
    action_rate_penalty = RewTerm(
        func=mdp.action_rate_l2,
        # weight=-0.03,
        weight=-0.1,
    )

    # # Penalize joint velocities
    # joint_velocity_penalty = RewTerm(
    #     func=mdp.joint_vel_l2,
    #     weight=-0.01,  # Adjust the weight to scale the penalty severity
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names=["lower_link", "upper_link"])}, # Targets the whole robot
    # )

    # Torque spike penalty
    torque_penalty = RewTerm(
        func=mdp.joint_torque_penalty,
        # weight=-0.001,
        weight=-0.004,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["hip_joint", "knee_joint"])},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # TODO: Add termination criteria for minimum height
    height_term = DoneTerm(
        func=mdp.height_termination,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_base"]), "target": -0.75},
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
        self.episode_length_s = 3 # Time out length for reset
        # viewer settings
        self.viewer.eye = (3.0, 3.0, 3.0)
        # simulation settings
        self.sim.dt = 1 / 240
        self.sim.render_interval = self.decimation
