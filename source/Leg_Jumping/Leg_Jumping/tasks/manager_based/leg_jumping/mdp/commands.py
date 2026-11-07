"""Isaac Lab command term wrapping the jump timer (item 2.4)."""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from ..leg_params import PARAMS
from .jump_command import HeldRequests, JumpCommand, ScriptedRequests, TrainingRequests, T_LOCK


class JumpCommandTerm(CommandTerm):
    def __init__(self, cfg: JumpCommandCfg, env):
        super().__init__(cfg, env)
        dt = env.step_dt
        n, dev = env.num_envs, env.device
        self.cmd = JumpCommand(n, dev, dt=dt, t_lock=cfg.t_lock)
        if cfg.mode == "train":
            self.src = TrainingRequests(n, dev, dt=dt, idle_s=cfg.idle_s, none_frac=cfg.none_frac, t_lock=cfg.t_lock, seed=env.cfg.seed or 0)
        elif cfg.mode == "eval":
            self.src = ScriptedRequests(cfg.eval_times, n, dev, dt=dt)
        elif cfg.mode == "hold":
            self.src = HeldRequests(n, dev, dt=dt, t_hop=cfg.t_lock)
        else:
            raise ValueError(f"JumpCommandCfg.mode={cfg.mode!r}")
        self._command = torch.zeros(n, 2, device=dev)
        self.prev_air = torch.zeros(n, dtype=torch.bool, device=dev)
        self.hopped = torch.zeros(n, dtype=torch.bool, device=dev)
        self.takeoff = torch.zeros(n, dtype=torch.bool, device=dev)
        self.extra_takeoff = torch.zeros(n, dtype=torch.bool, device=dev)
        self.in_first_flight = torch.zeros(n, dtype=torch.bool, device=dev)
        self.touch = torch.zeros(n, dtype=torch.bool, device=dev)
        self.cleared = torch.zeros(n, dtype=torch.bool, device=dev)
        self.had_cleared = torch.zeros(n, dtype=torch.bool, device=dev)
        names = list(env.scene["robot"].data.joint_names)
        self._si = names.index("slider_to_base")
        self._stand_slider = float(PARAMS.stand_slider) if PARAMS.stand_slider is not None else None

    @property
    def command(self):
        return self._command

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        self.cmd.reset(env_ids)
        self.src.reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        for name in ("prev_air", "hopped", "takeoff", "extra_takeoff", "in_first_flight", "touch", "cleared", "had_cleared", "_command"):
            t = getattr(self, name)
            if t.is_inference():
                setattr(self, name, t.clone())
        self.prev_air[ids] = False
        self.hopped[ids] = False
        self.takeoff[ids] = False
        self.extra_takeoff[ids] = False
        self.in_first_flight[ids] = False
        self.touch[ids] = False
        self.cleared[ids] = False
        self.had_cleared[ids] = False
        self._command[ids] = 0.0
        return {}

    def compute(self, dt: float):
        req = self.src.poll()
        acc = self.cmd.step(req)
        if hasattr(self.src, "note_accepted"):
            self.src.note_accepted(acc)
        self._command[:, 0] = self.cmd.latch.float()
        self._command[:, 1] = self.cmd.time_since_request
        f = self._env.scene["foot_contact"].data.net_forces_w_history.norm(dim=-1).amax(dim=(1, 2))
        air = f < 1.0
        latch = self.cmd.latch
        for name in ("hopped", "in_first_flight", "had_cleared"):
            t = getattr(self, name)
            if t.is_inference():
                setattr(self, name, t.clone())
        self.hopped = torch.where(acc, torch.zeros_like(self.hopped), self.hopped)
        self.in_first_flight = torch.where(acc, torch.zeros_like(self.in_first_flight), self.in_first_flight)
        self.had_cleared = torch.where(acc, torch.zeros_like(self.had_cleared), self.had_cleared)
        leaving = (~self.prev_air) & air
        leaving_lock = leaving & latch
        landing = self.prev_air & ~air & latch
        self.takeoff = leaving_lock & ~self.hopped
        self.extra_takeoff = (leaving_lock & self.hopped) | (leaving & ~latch)
        self.touch = landing
        self.in_first_flight = (self.in_first_flight | self.takeoff) & ~landing & latch
        self.hopped = (self.hopped | leaving_lock) & latch
        self.cleared = torch.zeros_like(air)
        if self._stand_slider is not None:
            rise = self._env.scene["robot"].data.joint_pos[:, self._si] - self._stand_slider
            self.cleared = self.in_first_flight & (rise >= self.cfg.clear_m) & ~self.had_cleared
            self.had_cleared = (self.had_cleared | self.cleared) & latch
        self.prev_air = air

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids):
        pass

    def _update_command(self):
        pass


@configclass
class JumpCommandCfg(CommandTermCfg):
    class_type: type = JumpCommandTerm
    resampling_time_range: tuple[float, float] = (math.inf, math.inf)
    mode: str = "train"  # train | eval | hold
    t_lock: float = T_LOCK
    idle_s: tuple[float, float] = (1.0, 8.0)
    none_frac: float = 0.35
    eval_times: tuple[float, ...] = (2.0, 3.0, 8.0)
    clear_m: float = 0.22
