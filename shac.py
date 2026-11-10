"""
SHAC pieces: the critic network, the actor loss, and the critic update.

Reference: Xu et al., "Accelerated Policy Learning with Parallel
Differentiable Simulation" (SHAC), ICLR 2022.

  - actor loss = -(sum_t gamma^t r_t + gamma^H V(o_H)) / H
    (gradients flow through the simulator into the rewards and into V's input)
  - the critic regresses TD(lambda) targets computed with a slow target critic
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    """V(observation): expected discounted return over the next few seconds.

    Inputs are the policy's own: image input x, state vector, and its GRU
    hidden state (detached; gives the critic memory of what the drone saw earlier).
    """

    def __init__(self, in_ch, dim_state, dim_hidden, width=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 2, 1), nn.LeakyReLU(0.05),
            nn.Conv2d(32, 64, 3, 2, 1), nn.LeakyReLU(0.05),
            nn.Conv2d(64, 64, 3, 1, 1), nn.LeakyReLU(0.05),
            nn.AdaptiveAvgPool2d((3, 4)), nn.Flatten())
        self.mlp = nn.Sequential(
            nn.Linear(64 * 12 + dim_state + dim_hidden, width), nn.ELU(),
            nn.Linear(width, width), nn.ELU(),
            nn.Linear(width, 1))

    def forward(self, x, state, h):
        return self.mlp(torch.cat([self.conv(x), state, h], -1))


def actor_loss(rewards, last_obs, critic, gamma):
    """SHAC actor loss for one window: rewards (H, B) live, last_obs = (x, state, h) live."""
    H = rewards.shape[0]
    disc = gamma ** torch.arange(H, device=rewards.device, dtype=rewards.dtype)
    critic.requires_grad_(False)          # gradients go into the policy, not the critic
    v_end = critic(*last_obs).squeeze(-1)
    critic.requires_grad_(True)
    ret = (disc[:, None] * rewards).sum(0) + gamma ** H * v_end
    return -ret.mean() / H, v_end.detach().mean()


def train_critic(args, obs, rewards, critic, target, opt):
    """Fit the critic to TD(lambda) targets for one window, then soft-update the target.

    obs: list over steps of (x, state, h), detached; rewards (H, B).
    Uses args.gamma, lam, critic_epochs, critic_minibatch, critic_huber_delta,
    grad_clip, target_alpha.
    """
    parts = [torch.stack(t) for t in zip(*obs)]          # x, state, h, each (H, B, ...)
    r = rewards.detach()
    H = r.shape[0]
    flat = lambda t: t.flatten(0, 1)

    # TD(lambda) targets for obs[0..H-2]; reward r[i+1] follows obs[i].
    with torch.no_grad():
        v_t = target(*[flat(t) for t in parts]).view(H, -1)
        G = [None] * (H - 1)
        nxt = v_t[H - 1]
        for i in range(H - 2, -1, -1):
            nxt = r[i + 1] + args.gamma * ((1 - args.lam) * v_t[i + 1] + args.lam * nxt)
            G[i] = nxt
        Y = flat(torch.stack(G))

    inputs = [flat(t[:-1]) for t in parts]
    n = Y.shape[0]
    loss = torch.zeros((), device=Y.device)
    for _ in range(args.critic_epochs):
        perm = torch.randperm(n, device=Y.device)
        for j in range(0, n, args.critic_minibatch):
            idx = perm[j:j + args.critic_minibatch]
            pred = critic(*[t[idx] for t in inputs]).squeeze(-1)
            if args.critic_huber_delta > 0:
                loss = F.huber_loss(pred, Y[idx], delta=args.critic_huber_delta)
            else:
                loss = F.mse_loss(pred, Y[idx])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), args.grad_clip)
            opt.step()

    with torch.no_grad():
        for pt, pc in zip(target.parameters(), critic.parameters()):
            pt.mul_(args.target_alpha).add_(pc, alpha=1 - args.target_alpha)
    return loss.detach(), Y.mean()


def detach_graph(obj):
    """Cut the autograd graph on every non-leaf tensor attribute of obj.

    Needed between SHAC windows so each window backprops only through its own
    steps. If you see 'Trying to backward through the graph a second time',
    some tensor kept across windows lives somewhere this does not reach
    (a list or dict attribute); detach it here as well.
    """
    for k, v in list(vars(obj).items()):
        if torch.is_tensor(v) and v.grad_fn is not None:
            setattr(obj, k, v.detach())
