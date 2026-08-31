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
from isaaclab.sensors import ContactSensorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp

##
# Pre-defined configs
##

# from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # isort:skip
from .dog_asset import DOG_LOC_CFG


##
# Scene definition
##


@configclass
class DogLocSceneCfg(InteractiveSceneCfg):

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot
    # ENV_REGEX_NS is just a shortcut to the path to the current environment in the scene
    robot: ArticulationCfg = DOG_LOC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    dome_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True, # for feet_air_time
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
        joint_names=["(fl|fr|br|bl)_(lat|hip|knee)_joint"],
        # scale=0.4,
        scale=0.25,
        use_default_offset=True # Makes it so policy outputs are relative to intial position
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        base_orientation = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")}
        )
        last_action = ObsTerm(func=mdp.last_action, history_length=5)

        def __post_init__(self) -> None:
            self.enable_corruption = False # Randomized noise TODO enable
            self.concatenate_terms = True

    # observation groups
    # Just one group for everything the policy uses (both actor and critic)
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    
    # randomize_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "mass_distribution_params": (0.95, 1.05),
    #         "operation": "scale",
    #     },
    # )
    #
    # randomize_friction = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",  # Use startup to prevent CPU overhead / PhysX crashes
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "static_friction_range": (0.2, 1.2),
    #         "dynamic_friction_range": (0.2, 1.0),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 64,  # Creates 64 different random materials to sample from
    #     },
    # )

    # reset
    reset_leg_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["(fl|fr|br|bl)_(lat|hip|knee)_joint"]),
            # TODO: update pos/vel perturbation vals
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.01, 0.01),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                # "z": (0.35, 0.4),
                # "z": (0.3, 0.35),
                "z": (0.25, 0.30),
                "yaw": (-math.pi, math.pi),
                # "yaw": (math.pi, math.pi),
                # "yaw": (-math.pi/2, -math.pi/2),
                # tried pi/4, 3pi/4, -3pi/4, 5pi/4 (goes backwards), 0 (goes sideways left)
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Constant baseline alive reward
    alive = RewTerm(func=mdp.is_alive, weight=5.0)
    # alive = RewTerm(func=mdp.is_alive, weight=10.0)

    dog_velocity = RewTerm(
        func=mdp.velocity_rew,
        # weight=16.0,
        weight=40.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["base"])},
    )

    # Penalized walking in circles (rotational velocity around z axis)
    dog_rotation_penalty = RewTerm(
        func=mdp.rotation_penalty,
        # weight=-100.0,
        weight=-200.0,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )

    # dog_roll_balanced = RewTerm(
    #     func=mdp.balanced_roll_rew,
    #     weight=-5.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names=["base"])}
    # )

    dog_pitch_balanced = RewTerm(
            func=mdp.balanced_pitch_rew,
        weight=-320.0,
        # weight=-350.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["base"])}
    )

    dog_roll_balanced = RewTerm(
        func=mdp.balanced_roll_rew,
        weight=-320.0,
        # weight=-350.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["base"])}
    )

    dog_vert_vel_penalty = RewTerm(
        func=mdp.vertical_velocity_penalty,
        weight=-15.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["base"])}
    )

    # delta penalty
    delta_action_penalty = RewTerm(
        func=mdp.action_l2, # Literally penalizes actions far from 0 (i.e. far from nominal pose since actions are deltas)
        weight=-0.07, # changed from 0.05
    )

    # # Penalize large action difference
    action_rate_penalty = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.35,
    )

    # # Penalize joint velocities
    joint_velocity_penalty = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.05,  # Adjust the weight to scale the penalty severity
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["(fl|fr|br|bl)_(lat|hip|knee)_joint"])},
    )

    # Torque spike penalty
    torque_penalty = RewTerm(
        func=mdp.joint_torque_penalty,
        # weight=-0.001,
        weight=-0.0005,
        # weight=-0.003,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["(fl|fr|br|bl)_(lat|hip|knee)_joint"])},
    )

    #######################################################

    # The built in mdp.feet_air_time required command manager, so we use the custom one
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=300.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_lower_link"),
            "threshold": 0.05, # Minimum time for a foot to be in the air
            "max_air": 0.45, # Max time for a foot to be in the air
        },
    )
    
    # Penalize foot velocity when on ground (sliding)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-6.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_lower_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_lower_link"),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # base_roll = DoneTerm(
    #     func=mdp.bad_orientation,
    #     params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": math.pi / 6},
    # )

    base_roll = DoneTerm(
        func=mdp.bad_roll,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    base_pitch = DoneTerm(
        func=mdp.bad_pitch,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )

    base_height = DoneTerm(
        func=mdp.bad_height,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )

    # base_yaw = DoneTerm(
    #     func=mdp.bad_yaw,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names=["base"])}
    #     )

##
# Environment configuration
##


@configclass
class DogLocEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    # scene: DogLocSceneCfg = DogLocSceneCfg(num_envs=4096, env_spacing=100.0)
    scene: DogLocSceneCfg = DogLocSceneCfg(num_envs=8192, env_spacing=10.0)
    # scene: DogLocSceneCfg = DogLocSceneCfg(num_envs=16384, env_spacing=0.0)
    # scene: DogLocSceneCfg = DogLocSceneCfg(num_envs=16384, env_spacing=10.0)
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
        self.episode_length_s = 5 # Time out length for reset
        # viewer settings
        self.viewer.eye = (3.0, 3.0, 3.0)
        # simulation settings
        self.sim.dt = 1 / 240
        self.sim.render_interval = self.decimation

        # To prevent gpu buffer overflow errors, this increases preallocation
        self.sim.physx.gpu_max_rigid_patch_count = 1024 * 1024 * 4
        self.sim.physx.gpu_max_rigid_contact_count = 1024 * 1024 * 8
        self.sim.physx.gpu_found_lost_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
