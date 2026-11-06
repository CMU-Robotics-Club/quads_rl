from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg


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
