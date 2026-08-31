# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

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
    # return torch.clamp(joint_pos - target, min=0.0, max=0.2)
    ret = torch.clamp(joint_pos - target, min=0.0)
    return torch.where(ret > 0.18, -1.0, ret) # make too high negative reward
    # return (joint_pos > target).float()

def height_termination(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """terminate if height is below a certain value"""
    asset: Articulation = env.scene[asset_cfg.name]
    # Slider to base height
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids[0]]
    return joint_pos < target

def upward_velocity(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Rewards positive upward velocity to encourage the explosive push-off phase."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids[0]]
    return torch.clamp(joint_vel, min=0.0, max=4.0)

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
