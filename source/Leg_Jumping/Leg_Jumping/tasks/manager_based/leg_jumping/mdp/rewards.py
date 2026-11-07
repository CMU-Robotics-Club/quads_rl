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


def _jump(env: ManagerBasedRLEnv):
    return env.command_manager.get_term("jump")


def _airborne(env: ManagerBasedRLEnv, force_n: float = 1.0) -> torch.Tensor:
    f = env.scene["foot_contact"].data.net_forces_w_history.norm(dim=-1).amax(dim=(1, 2))
    return f < force_n


def airborne_flag(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Privileged: 1 if the foot is off the floor. Critic only."""
    return _airborne(env).float().unsqueeze(-1)


def _joints(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    robot: Articulation = env.scene[asset_cfg.name]
    names = list(robot.data.joint_names)
    return robot, names.index("slider_to_base"), names.index("hip_joint"), names.index("knee_joint")


def stand_pose(env: ManagerBasedRLEnv, stand_hip: float, stand_knee: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot, _, hi, ki = _joints(env, asset_cfg)
    q = robot.data.joint_pos
    err = (q[:, hi] - stand_hip) ** 2 + (q[:, ki] - stand_knee) ** 2
    idle = ~_jump(env).cmd.latch
    return idle.float() * torch.exp(-err / 0.05)


def stand_height(env: ManagerBasedRLEnv, stand_slider: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot, si, _, _ = _joints(env, asset_cfg)
    err = ((robot.data.joint_pos[:, si] - stand_slider) / 0.05) ** 2
    idle = ~_jump(env).cmd.latch
    return idle.float() * torch.exp(-err)


def false_air(env: ManagerBasedRLEnv) -> torch.Tensor:
    return ((~_jump(env).cmd.latch) & _airborne(env)).float()


def rise_air(env: ManagerBasedRLEnv, stand_slider: float, cap_m: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Body rise on first flight, only while the slider is still going up."""
    robot, si, _, _ = _joints(env, asset_cfg)
    rise = torch.clamp(robot.data.joint_pos[:, si] - stand_slider, min=0.0, max=cap_m)
    going_up = robot.data.joint_vel[:, si] > 0.0
    return _jump(env).in_first_flight.float() * going_up.float() * rise


def foot_air(env: ManagerBasedRLEnv, rest_z: float, cap_m: float, asset_cfg: SceneEntityCfg, y_scale: float = 0.08, max_s: float = 0.20) -> torch.Tensor:
    """Foot height on the first ~200 ms of first flight, scaled down if the foot swings off the rail."""
    robot: Articulation = env.scene[asset_cfg.name]
    p = robot.data.body_pos_w[:, asset_cfg.body_ids[0], :] - env.scene.env_origins
    clear = torch.clamp(p[:, 2] - rest_z, min=0.0, max=cap_m)
    align = torch.exp(-(p[:, 1] / y_scale) ** 2)
    term = _jump(env)
    early = term.flight_s <= max_s
    return term.in_first_flight.float() * early.float() * clear * align


def return_stand_pose(env: ManagerBasedRLEnv, stand_hip: float, stand_knee: float, after_s: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot, _, hi, ki = _joints(env, asset_cfg)
    term = _jump(env)
    late = term.cmd.latch & term.hopped & (term.cmd.time_since_request > after_s)
    err = (robot.data.joint_pos[:, hi] - stand_hip) ** 2 + (robot.data.joint_pos[:, ki] - stand_knee) ** 2
    return late.float() * torch.exp(-err / 0.05)


def return_stand_height(env: ManagerBasedRLEnv, stand_slider: float, after_s: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot, si, _, _ = _joints(env, asset_cfg)
    term = _jump(env)
    late = term.cmd.latch & term.hopped & (term.cmd.time_since_request > after_s)
    err = ((robot.data.joint_pos[:, si] - stand_slider) / 0.05) ** 2
    return late.float() * torch.exp(-err)


def takeoff(env: ManagerBasedRLEnv) -> torch.Tensor:
    """1 on the first leave-floor of this jump request."""
    return _jump(env).takeoff.float()


def extra_takeoff(env: ManagerBasedRLEnv) -> torch.Tensor:
    """1 on later leave-floor events in the same 2 s lockout (bounce)."""
    return _jump(env).extra_takeoff.float()


def land_soft(env: ManagerBasedRLEnv, v_ok: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Downward slider speed above v_ok on the touchdown step, while a jump is latched."""
    robot, si, _, _ = _joints(env, asset_cfg)
    down = torch.clamp(-robot.data.joint_vel[:, si] - v_ok, min=0.0)
    return _jump(env).touch.float() * down


def foot_slide(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    v = robot.data.body_lin_vel_w[:, asset_cfg.body_ids[0], :2]
    contact = ~_airborne(env)
    return contact.float() * torch.sum(torch.square(v), dim=1)
