from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

from ..leg_params import PARAMS


def set_body_mass(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    mass: float | None,
    preset: str = "",
    inertia: float = 1e-8,
):
    """Give bodies the URDF left massless a tiny mass (PhysX would load them as 1 kg). None = do nothing."""
    if mass is None:
        return
    asset = env.scene[asset_cfg.name]
    ids = torch.arange(env.scene.num_envs, device="cpu") if env_ids is None else env_ids.cpu()
    body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")
    view = asset.root_physx_view

    masses = view.get_masses()
    masses[ids[:, None], body_ids] = mass
    view.set_masses(masses, ids)

    inertias = view.get_inertias()
    inertias[ids[:, None], body_ids] = 0.0
    for k in (0, 4, 8):
        inertias[ids[:, None], body_ids, k] = inertia
    view.set_inertias(inertias, ids)


def reset_to_stand(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    velocity_range: tuple[float, float] = (-0.02, 0.02),
):
    """Put the leg at its stand pose with small noise, never inside the floor (slider moves up with the pose noise)."""
    # Read PARAMS here. Hydra cannot write into the frozen sheet if it lives in the env cfg.
    p = PARAMS
    asset = env.scene[asset_cfg.name]
    n = len(env_ids)
    dev = asset.data.joint_pos.device
    dh = (torch.rand(n, device=dev) * 2 - 1) * p.reset_pose_noise
    dk = (torch.rand(n, device=dev) * 2 - 1) * p.reset_pose_noise
    lift = p.reset_lift + torch.rand(n, device=dev) * p.reset_lift_noise
    slider = p.stand_slider + p.depth_per_hip * dh + p.depth_per_knee * dk + lift
    jn = list(asset.joint_names)
    pos = asset.data.default_joint_pos[env_ids].clone()
    pos[:, jn.index("slider_to_base")] = slider
    pos[:, jn.index("hip_joint")] = p.stand_hip + dh
    pos[:, jn.index("knee_joint")] = p.stand_knee + dk
    vel = torch.rand_like(pos) * (velocity_range[1] - velocity_range[0]) + velocity_range[0]
    asset.write_joint_state_to_sim(pos, vel, env_ids=env_ids)
