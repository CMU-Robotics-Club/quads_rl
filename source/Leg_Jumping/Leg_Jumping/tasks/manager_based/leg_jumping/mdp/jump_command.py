"""Jump request timer. One timer for training, tests and watching; only the request source changes."""
import torch

T_LOCK = 2.0
NEVER = 10**9


def _ids(env_ids, n, device):
    if env_ids is None or env_ids == slice(None):
        return torch.arange(n, device=device)
    if not torch.is_tensor(env_ids):
        return torch.as_tensor(env_ids, device=device, dtype=torch.long)
    return env_ids.to(device=device, dtype=torch.long)


def _writable(t):
    """Buffers written under torch.inference_mode cannot be updated in place afterwards."""
    return t.clone() if t.is_inference() else t


class JumpCommand:
    def __init__(self, num_envs, device, dt=1 / 60, t_lock=T_LOCK):
        self.n = num_envs
        self.n_lock = int(round(t_lock / dt))
        self.dt = dt
        self.since = torch.full((num_envs,), self.n_lock + 1, dtype=torch.long, device=device)

    def step(self, request):
        self.since = (self.since + 1).clamp(max=10 * self.n_lock)
        accepted = request & ~self.latch
        self.since = torch.where(accepted, torch.zeros_like(self.since), self.since)
        return accepted

    def reset(self, env_ids=None):
        self.since = _writable(self.since)
        self.since[_ids(env_ids, self.n, self.since.device)] = self.n_lock + 1

    @property
    def latch(self):
        return self.since < self.n_lock

    @property
    def time_since_request(self):
        return (self.since.float() * self.dt).clamp(max=self.n_lock * self.dt)


class ScriptedRequests:
    """Per-env step counter so a fallen leg can restart the 2/3/8 s script."""

    def __init__(self, times, num_envs, device, dt=1 / 60):
        self.times = torch.tensor([int(round(t / dt)) for t in times], device=device, dtype=torch.long)
        self.n, self.dev = num_envs, device
        self.k = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset(self, env_ids=None):
        self.k = _writable(self.k)
        self.k[_ids(env_ids, self.n, self.dev)] = 0

    def poll(self):
        out = (self.k.unsqueeze(1) == self.times).any(dim=1)
        self.k = self.k + 1
        return out


class TrainingRequests:
    def __init__(self, num_envs, device, dt=1 / 60, idle_s=(0.5, 4.0), none_frac=0.1, t_lock=T_LOCK, seed=0):
        self.n, self.dev, self.dt = num_envs, device, dt
        self.n_lock = int(round(t_lock / dt))
        self.lo = int(round(idle_s[0] / dt))
        self.hi = int(round(idle_s[1] / dt))
        self.none_frac = none_frac
        self.g = torch.Generator(device=device).manual_seed(seed)
        self.k = 0
        self.next = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.reset()

    def _gap(self, n, allow_never):
        gap = torch.randint(self.lo, self.hi + 1, (n,), device=self.dev, generator=self.g)
        if allow_never:
            never = torch.rand(n, device=self.dev, generator=self.g) < self.none_frac
            gap = torch.where(never, torch.full((n,), NEVER, device=self.dev, dtype=torch.long), gap)
        return gap

    def reset(self, env_ids=None):
        env_ids = _ids(env_ids, self.n, self.dev)
        self.next = _writable(self.next)
        self.next[env_ids] = self.k + self._gap(int(env_ids.numel()), allow_never=True)

    def poll(self):
        out = self.next == self.k
        self.k += 1
        return out

    def note_accepted(self, accepted):
        idx = accepted.nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            return
        self.next[idx] = self.k + (self.n_lock - 1) + self._gap(int(idx.numel()), allow_never=False)


class WatchRequests:
    """One hop per pulse. Keyboard/terminal set a Python flag; poll() runs on the sim thread."""

    def __init__(self, num_envs, device):
        self.n, self.dev = num_envs, device
        self._hit = False

    def pulse(self):
        self._hit = True

    def poll(self):
        hit, self._hit = self._hit, False
        return torch.full((self.n,), hit, dtype=torch.bool, device=self.dev)

    def reset(self, env_ids=None):
        self._hit = False


class HeldRequests:
    def __init__(self, num_envs, device, dt=1 / 60, t_hop=T_LOCK):
        self.n, self.dev = num_envs, device
        self.n_hop = int(round(t_hop / dt))
        self.k = 0
        self.held = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.last = torch.full((num_envs,), -self.n_hop, dtype=torch.long, device=device)

    def set_held(self, held):
        if not torch.is_tensor(held):
            held = torch.full((self.n,), bool(held), dtype=torch.bool, device=self.dev)
        rise = held & ~self.held
        self.last = torch.where(rise, torch.full_like(self.last, self.k - self.n_hop), self.last)
        self.held = held

    def poll(self):
        issue = self.held & ((self.k - self.last) >= self.n_hop)
        self.last = torch.where(issue, torch.full_like(self.last, self.k), self.last)
        self.k += 1
        return issue

    def reset(self, env_ids=None):
        ids = _ids(env_ids, self.n, self.dev)
        self.held = _writable(self.held)
        self.last = _writable(self.last)
        self.held[ids] = False
        self.last[ids] = -self.n_hop
