# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi, euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    # compute the reward
    return torch.sum(torch.pow(joint_pos - target, 2), dim=1)

def joint_pos_target_exp(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """penalize joint position deviation from target and reward highly for close vals"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    print(joint_pos)
    # Standing up gives a joint position of -0.19
    # compute the reward
    return torch.sum(torch.exp(-torch.square(joint_pos-target) / 1.0), dim=1)

def target_above_threshold(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """reward only if jump height is above a threshold"""
    asset: Articulation = env.scene[asset_cfg.name]
    # Slider to base height
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids[0]]
    # print(joint_pos)
    return torch.clamp(joint_pos - target, min=0.0, max=0.2)

def velocity_rew(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Rewards velocity"""
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = -asset.data.root_lin_vel_b[:, 1] # 0 is X, 1 is Y, and 2 is Z. 
    return torch.clamp(body_vel, max=1.5)

def rotation_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """penalized rotation"""
    asset: Articulation = env.scene[asset_cfg.name]
    yaw_vel = asset.data.root_ang_vel_b[:, 2] # 2 is Z
    return torch.square(yaw_vel)

def vertical_velocity_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """penalized vertical velocity"""
    asset: Articulation = env.scene[asset_cfg.name]
    vert_vel = asset.data.root_lin_vel_b[:, 2] # 2 is Z
    return torch.square(vert_vel)

# def balanced_pitch_rew(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
#     """rewards an upright pose"""
#     asset: Articulation = env.scene[asset_cfg.name]
#     base_idx = asset_cfg.body_ids[0]
#     body_quat = asset.data.body_state_w[:, base_idx, 3:7] # 3:7 gives quaternion
#     roll, pitch, yaw = euler_xyz_from_quat(body_quat)
#     # print(f"pitch: {pitch}")
#     # print(f"roll: {roll}")
#     return torch.square(pitch)

def balanced_pitch_rew(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalizes pitch using projected gravity"""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.projected_gravity_b[:, 1]) # Y component

def balanced_roll_rew(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalizes pitch using projected gravity"""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.projected_gravity_b[:, 0]) # X component

def balanced_yaw_rew(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """rewards an upright pose"""
    asset: Articulation = env.scene[asset_cfg.name]
    base_idx = asset_cfg.body_ids[0]
    body_quat = asset.data.body_state_w[:, base_idx, 3:7] # 3:7 gives quaternion
    roll, pitch, yaw = euler_xyz_from_quat(body_quat)
    # print(pitch)
    return torch.square(yaw)

def bad_height(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """rewards an upright pose"""
    asset: Articulation = env.scene[asset_cfg.name]
    # Was 0.2 -> 0.18 (worked well) -> 0.16 (worked well)
    return asset.data.root_pos_w[:, 2]<0.16

# def bad_yaw(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
#     """rewards an upright pose"""
#     asset: Articulation = env.scene[asset_cfg.name]
#     base_idx = asset_cfg.body_ids[0]
#     body_quat = asset.data.body_state_w[:, base_idx, 3:7] # 3:7 gives quaternion
#     roll, pitch, yaw = euler_xyz_from_quat(body_quat)
#     # yaw_cos = torch.cos(yaw-torch.pi/4)
#     yaw_cos = torch.cos(yaw+torch.pi/2)
#     # print(yaw_sin)
#     return yaw_cos<0.95
#
def bad_pitch(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """rewards an upright pose"""
    asset: Articulation = env.scene[asset_cfg.name]
    # base_idx = asset_cfg.body_ids[0]
    # body_quat = asset.data.body_state_w[:, base_idx, 3:7] # 3:7 gives quaternion
    # roll, pitch, yaw = euler_xyz_from_quat(body_quat)
    # pitch_cos = torch.cos(pitch)
    # # print(pitch_cos)
    # return pitch_cos<0.95
    proj_grav = asset.data.projected_gravity_b # Shape (num_envs, 3). gravity vector is a unit vector down.
    pitch_sin = proj_grav[:, 1]
    # Using cos(x) = sqrt(1 - sin(x)^2)
    pitch_cos = torch.sqrt(torch.clamp(1.0 - torch.square(pitch_sin), min=0.0))
    return pitch_cos < 0.85

def bad_roll(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """rewards an upright pose"""
    asset: Articulation = env.scene[asset_cfg.name]
    # base_idx = asset_cfg.body_ids[0]
    # body_quat = asset.data.body_state_w[:, base_idx, 3:7] # 3:7 gives quaternion
    # roll, pitch, yaw = euler_xyz_from_quat(body_quat)
    # roll_cos = torch.cos(roll)
    # # print(roll_cos)
    proj_grav = asset.data.projected_gravity_b # Shape (num_envs, 3). gravity vector is a unit vector down.
    roll_sin = proj_grav[:, 0]
    # Using cos(x) = sqrt(1 - sin(x)^2)
    roll_cos = torch.sqrt(torch.clamp(1.0 - torch.square(roll_sin), min=0.0))
    return roll_cos < 0.85

def body_pos_target_y_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    y_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, 1]
    # print(y_pos[:, 0:5])
    return torch.square(y_pos[:, 0] - y_pos[:, 1])

def joint_torque_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name] # Get Leg from the env scene
    joint_ids = asset_cfg.joint_ids # Getting specified joint ids for the robot
    torques = robot.data.applied_torque[:, joint_ids]
    # Square the torques to discourage large spikes in torque
    return torch.sum(torch.square(torques), dim=1)

def feet_air_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 0.25, max_air: float = 0.45) -> torch.Tensor:
    """Reward feet for remaining in the air longer than threshold upon touchdown."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # Get touchdown mask (1 on touchdown step, 0 otherwise)
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    # Air time measured before touchdown
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    air_time_excess = torch.clamp(last_air_time - threshold, min=0.0, max=max_air)
    
    sqr_per_foot = torch.square(air_time_excess)
    # Reward positive air time on touchdown
    return torch.sum(sqr_per_foot * first_contact.float(), dim=1)
