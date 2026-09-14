"""
rl_experiments/nets.py — Q-networks, n-step replay buffer, Double-DQN core.

One DDQNCore = one learner (Q-net + frozen target net + replay + optimiser).
The four experiment framings each hold 1 or N of these cores:

    B / D : 1 core        (central / universal)
    A     : 1 per truck   (independent learners)
    C     : 1 per mine    (auction bidders)

Everything is plain PyTorch — no RL libraries.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------- nets
class QNet(nn.Module):
    def __init__(self, in_dim, n_actions, hidden=256, dueling=True):
        super().__init__()
        self.dueling = dueling
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        if dueling:
            self.value = nn.Linear(hidden, 1)
            self.adv = nn.Linear(hidden, n_actions)
        else:
            self.head = nn.Linear(hidden, n_actions)

    def forward(self, x):
        h = self.body(x)
        if self.dueling:
            v = self.value(h)
            a = self.adv(h)
            return v + a - a.mean(dim=1, keepdim=True)
        return self.head(h)


# ---------------------------------------------------------------- replay
class NStepBuffer:
    """Replay with n-step returns: R = r0 + γ r1 + ... + γ^{n-1} r_{n-1}."""

    def __init__(self, capacity=50_000, n_step=3, gamma=0.97):
        self.buf = deque(maxlen=capacity)
        self.tmp = deque()
        self.n = n_step
        self.gamma = gamma

    def __len__(self):
        return len(self.buf)

    def _pack(self, items):
        R = sum((self.gamma ** i) * tr[2] for i, tr in enumerate(items))
        s, a = items[0][0], items[0][1]
        s2, done, m2 = items[-1][3], items[-1][4], items[-1][5]
        return (s, a, R, s2, done, m2)

    def push(self, s, a, r, s2, done, mask2):
        self.tmp.append((s, a, r, s2, done, mask2))
        if len(self.tmp) == self.n:
            self.buf.append(self._pack(list(self.tmp)))
            self.tmp.popleft()
        if done:                                    # flush tail of episode
            while self.tmp:
                self.buf.append(self._pack(list(self.tmp)))
                self.tmp.popleft()
            self.tmp.clear()

    def sample(self, batch, rng):
        idx = rng.choice(len(self.buf), size=min(batch, len(self.buf)), replace=False)
        rows = [self.buf[i] for i in idx]
        s = torch.as_tensor(np.stack([r[0] for r in rows]))
        a = torch.as_tensor(np.array([r[1] for r in rows]), dtype=torch.long)
        r = torch.as_tensor(np.array([r[2] for r in rows]), dtype=torch.float32)
        s2 = torch.as_tensor(np.stack([r[3] for r in rows]))
        d = torch.as_tensor(np.array([r[4] for r in rows]), dtype=torch.float32)
        m2 = torch.as_tensor(np.stack([r[5] for r in rows]))
        return s, a, r, s2, d, m2


# ---------------------------------------------------------------- DDQN core
class DDQNCore:
    """Double DQN with target net, action masking and n-step targets."""

    def __init__(self, in_dim, n_actions, device="cpu", lr=1e-3, gamma=0.97,
                 hidden=256, dueling=True, n_step=3, buffer_size=50_000,
                 batch=64, target_sync=500, min_buffer=1000, seed=0):
        self.n_actions = n_actions
        self.device = torch.device(device)
        self.gamma = gamma
        self.n_step = n_step
        self.batch = batch
        self.target_sync = target_sync
        self.min_buffer = min_buffer
        self.learn_steps = 0
        self.rng = np.random.default_rng(seed)

        torch.manual_seed(seed)
        self.q = QNet(in_dim, n_actions, hidden, dueling).to(self.device)
        self.tq = QNet(in_dim, n_actions, hidden, dueling).to(self.device)
        self.tq.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.buf = NStepBuffer(buffer_size, n_step, gamma)

    # ---- behaviour ---------------------------------------------------------
    def act(self, obs, mask, eps):
        if self.rng.random() < eps:
            valid = np.flatnonzero(np.asarray(mask) > 0)
            return int(self.rng.choice(valid)) if len(valid) else 0
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.q(x).squeeze(0).cpu().numpy()
        q[np.asarray(mask) <= 0] = -1e9
        return int(np.argmax(q))

    # ---- memory -------------------------------------------------------------
    def store(self, s, a, r, s2, done, mask2):
        self.buf.push(np.asarray(s, dtype=np.float32), int(a), float(r),
                      np.asarray(s2, dtype=np.float32), bool(done),
                      np.asarray(mask2, dtype=np.float32))

    # ---- learning -----------------------------------------------------------
    def learn(self):
        if len(self.buf) < self.min_buffer:
            return None
        s, a, r, s2, d, m2 = self.buf.sample(self.batch, self.rng)
        s, a, r, s2, d, m2 = (t.to(self.device) for t in (s, a, r, s2, d, m2))

        q = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_online_next = self.q(s2)
            q_online_next[m2 <= 0] = -1e9                      # mask invalid
            a_star = q_online_next.argmax(dim=1, keepdim=True)   # online picks
            q_target_next = self.tq(s2).gather(1, a_star).squeeze(1)  # target prices
            y = r + (self.gamma ** self.n_step) * (1.0 - d) * q_target_next

        loss = self.loss_fn(q, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_sync == 0:
            self.tq.load_state_dict(self.q.state_dict())
        return float(loss.item())
