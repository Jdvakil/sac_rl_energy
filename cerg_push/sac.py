"""Small SAC implementation: twin Q, tanh Gaussian actor, learned temperature.

References: Haarnoja et al., arXiv:1801.01290 and arXiv:1812.05905.
No dependency on Stable-Baselines3. Checkpoints include optimizer/RNG/replay state.
"""
import random
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


def mlp(inputs, outputs):
    return nn.Sequential(nn.Linear(inputs, 128), nn.ReLU(), nn.Linear(128, 128),
                         nn.ReLU(), nn.Linear(128, outputs))


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = mlp(obs_dim, 2 * act_dim)

    def forward(self, obs, deterministic=False):
        mean, log_std = self.net(obs).chunk(2, dim=-1)
        dist = Normal(mean, log_std.clamp(-5, 2).exp())
        z = mean if deterministic else dist.rsample()
        action = z.tanh()
        # Stable tanh Jacobian correction, including near-saturation actions.
        correction = 2 * (np.log(2) - z - torch.nn.functional.softplus(-2 * z))
        log_prob = (dist.log_prob(z) - correction).sum(-1, keepdim=True)
        return action, log_prob


class SAC:
    def __init__(self, obs_dim=23, act_dim=4, device='cpu', capacity=100000, seed=0):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        self.device = torch.device(device)
        self.actor = Actor(obs_dim, act_dim).to(device)
        self.q = nn.ModuleList([mlp(obs_dim+act_dim, 1) for _ in range(2)]).to(device)
        self.target = nn.ModuleList([mlp(obs_dim+act_dim, 1) for _ in range(2)]).to(device)
        self.target.load_state_dict(self.q.state_dict())
        self.target.requires_grad_(False)
        self.log_alpha = torch.tensor(np.log(.1), dtype=torch.float32, device=device, requires_grad=True)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=3e-4)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=3e-4)
        self.target_entropy = -act_dim
        self.capacity, self.pos, self.size = capacity, 0, 0
        self.buffer = {k: np.zeros((capacity, d), np.float32) for k, d in
                       [('obs', obs_dim), ('act', act_dim), ('rew', 1), ('next', obs_dim), ('done', 1)]}
        self.steps = 0

    @torch.no_grad()
    def predict(self, obs):
        action, _ = self.actor(torch.as_tensor(obs, dtype=torch.float32, device=self.device), True)
        return action.cpu().numpy()

    @torch.no_grad()
    def sample(self, obs):
        action, _ = self.actor(torch.as_tensor(obs, dtype=torch.float32, device=self.device))
        return action.cpu().numpy()

    def add(self, obs, act, rew, next_obs, done):
        for k, value in zip(self.buffer, [obs, act, rew, next_obs, done]):
            self.buffer[k][self.pos] = value
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.steps += 1

    def update(self, batch_size=256):
        ids = np.random.randint(self.size, size=batch_size)
        o, a, r, no, done = [torch.as_tensor(v[ids], device=self.device) for v in self.buffer.values()]
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            na, nlp = self.actor(no)
            nq = torch.minimum(*[q(torch.cat([no, na], -1)) for q in self.target])
            y = r + .99 * (1 - done) * (nq - alpha * nlp)
        q_loss = sum((q(torch.cat([o, a], -1)) - y).square().mean() for q in self.q)
        self.q_opt.zero_grad(); q_loss.backward(); self.q_opt.step()
        self.q.requires_grad_(False)
        pa, lp = self.actor(o)
        pq = torch.minimum(*[q(torch.cat([o, pa], -1)) for q in self.q])
        actor_loss = (alpha * lp - pq).mean()
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()
        self.q.requires_grad_(True)
        alpha_loss = -(self.log_alpha * (lp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()
        with torch.no_grad():
            for p, t in zip(self.q.parameters(), self.target.parameters()):
                t.lerp_(p, .005)
        return dict(q_loss=q_loss.item(), actor_loss=actor_loss.item(), alpha=alpha.item())

    def save(self, path, replay=True):
        state = dict(actor=self.actor.state_dict(), q=self.q.state_dict(), target=self.target.state_dict(),
                     log_alpha=self.log_alpha.detach(), actor_opt=self.actor_opt.state_dict(),
                     q_opt=self.q_opt.state_dict(), alpha_opt=self.alpha_opt.state_dict(), steps=self.steps,
                     numpy_rng=np.random.get_state(), python_rng=random.getstate(), torch_rng=torch.get_rng_state())
        if replay:
            state.update(buffer={k: v[:self.size] for k, v in self.buffer.items()}, pos=self.pos, size=self.size)
        torch.save(state, path)

    def load(self, path, resume=False):
        # Only load checkpoints you trust: PyTorch checkpoints use pickle.
        s = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(s['actor']); self.q.load_state_dict(s['q'])
        self.target.load_state_dict(s['target'])
        self.log_alpha.data.copy_(s['log_alpha']); self.steps = s['steps']
        if resume:
            for opt, key in [(self.actor_opt, 'actor_opt'), (self.q_opt, 'q_opt'), (self.alpha_opt, 'alpha_opt')]:
                opt.load_state_dict(s[key])
            if 'buffer' not in s or s['size'] > self.capacity:
                raise ValueError('Resume requires replay and sufficient buffer capacity')
            self.pos, self.size = s['pos'], s['size']
            for k in self.buffer:
                self.buffer[k][:self.size] = s['buffer'][k]
            np.random.set_state(s['numpy_rng']); random.setstate(s['python_rng'])
            torch.set_rng_state(s['torch_rng'].cpu())
        return self
