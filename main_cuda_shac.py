"""
SHAC fine-tuning for the DiffPhysDrone reactive policy, in the original
main_cuda.py training scenario (same scenes, 150-step episodes, same DiffPhys
loss terms and defaults). Three things differ from the original:

  1. TTC input (--use_ttc): the policy sees time-to-contact, depth / forward speed,
     instead of depth.
  2. Velocity-aligned heading (--heading velocity, default): the camera follows the
     flight direction instead of facing the goal, so the drone sees where it is going
     while flying around a building. --heading goal restores main_cuda.py.
     isaac/isaac_eval.py --heading velocity evaluates it the same way.
  3. SHAC (shac.py): each episode is rolled out in windows of H steps (default 30,
     5 per episode); actor loss = SHAC loss + DiffPhys losses, and a critic that sees
     only the policy's inputs is trained on TD(lambda) targets. The sim state is
     detached between windows; the episode itself continues.

The reward is the negative of the original per-step costs, so the critic
estimates the cost-to-go of the same objective main_cuda.py optimises:
  r_t = -(r_track * |v - target_v| / v_max
          + r_obj * relu(1 - clearance)^2
          + r_collide * softplus(-32 * clearance)) * reward_scale

Run from the repo root so configs/ and checkpoints/ resolve:

  python main_cuda_shac.py $(cat configs/single_agent.args) --use_ttc \\
      --resume checkpoints/base/cmd_checkpoint0000.pth
"""
import argparse
import copy
import math
import os
from collections import defaultdict, deque
from contextlib import nullcontext
from random import normalvariate

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from env_cuda import Env
from model import Model
from shac import Critic, actor_loss, detach_graph, train_critic


# --------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--resume', default=None,
                   help='policy checkpoint to start from; omit to train from scratch')

    # Env / model flags, same names and defaults as main_cuda.py so configs/*.args work.
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--grad_decay', type=float, default=0.4)
    p.add_argument('--speed_mtp', type=float, default=1.0)
    p.add_argument('--fov_x_half_tan', type=float, default=0.53)
    p.add_argument('--cam_angle', type=int, default=10)
    for flag in ['single', 'gate', 'ground_voxels', 'scaffold', 'random_rotation',
                 'gimbal_stabilized', 'no_odom', 'yaw_drift', 'use_ttc']:
        p.add_argument(f'--{flag}', default=False, action='store_true')

    # DiffPhys losses, same defaults as main_cuda.py
    p.add_argument('--coef_v', type=float, default=1.0)
    p.add_argument('--v_avg_window', type=int, default=30, help='hardcoded 30 in main_cuda.py')
    p.add_argument('--coef_v_pred', type=float, default=2.0)
    p.add_argument('--coef_collide', type=float, default=2.0)
    p.add_argument('--coef_obj_avoidance', type=float, default=1.5)
    p.add_argument('--coef_d_acc', type=float, default=0.01)
    p.add_argument('--coef_d_jerk', type=float, default=0.001)

    # SHAC
    p.add_argument('--num_episodes', type=int, default=10000)
    p.add_argument('--episode_steps', type=int, default=150, help='main_cuda.py --timesteps')
    p.add_argument('--horizon', type=int, default=30, help='actor window H (2 s); 5 windows per episode')
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--lam', type=float, default=0.95, help='TD(lambda) for critic targets')
    p.add_argument('--coef_shac', type=float, default=1.0)
    p.add_argument('--actor_lr', type=float, default=1e-4)
    p.add_argument('--actor_lr_cosine', default=False, action='store_true',
                   help='cosine-decay the actor lr to 1%% over training, as main_cuda.py '
                        '(use with --actor_lr 1e-3 when training from scratch)')
    p.add_argument('--critic_lr', type=float, default=5e-4)
    p.add_argument('--critic_epochs', type=int, default=4)
    p.add_argument('--critic_minibatch', type=int, default=1024)
    p.add_argument('--critic_huber_delta', type=float, default=0.5,
                   help='Huber loss for the critic; 0 = plain MSE')
    p.add_argument('--target_alpha', type=float, default=0.2,
                   help='target <- alpha*target + (1-alpha)*critic after each window')
    p.add_argument('--critic_warmup_windows', type=int, default=300,
                   help='train only the critic for this many windows before touching the policy')
    p.add_argument('--grad_clip', type=float, default=1.0)

    # Reward: negative of the original per-step costs
    p.add_argument('--r_track', type=float, default=1.0, help='|v - target_v| / max speed')
    p.add_argument('--r_obj', type=float, default=1.0, help='relu(1 - clearance)^2, as loss_obj_avoidance')
    p.add_argument('--r_collide', type=float, default=1.0, help='softplus(-32 * clearance), as loss_collide')
    p.add_argument('--collide_clip', type=float, default=5.0,
                   help='cap on the per-step collision penalty (before reward_scale)')
    p.add_argument('--reward_scale', type=float, default=0.1)

    # Heading
    p.add_argument('--heading', choices=['velocity', 'goal'], default='velocity',
                   help="'velocity': camera looks where the drone flies; 'goal': main_cuda.py behaviour")
    p.add_argument('--heading_min_speed', type=float, default=1.0,
                   help='start following the flight direction above this speed (m/s)')
    p.add_argument('--heading_release_speed', type=float, default=0.3,
                   help='stop following below this speed; in between, keep the last flight direction')

    p.add_argument('--save_dir', default='checkpoints/shac/')
    p.add_argument('--save_every', type=int, default=1000)
    p.add_argument('--print_every', type=int, default=10,
                   help='print all episode-mean losses every N episodes (0 = off)')
    p.add_argument('--ckpt_tag', default='orig_',
                   help='checkpoints are saved as shac_checkpoint_<tag>NNNNN.pth / critic_<tag>NNNNN.pth')

    args, unknown = p.parse_known_args()
    if unknown:
        print('[shac] ignoring unrecognised flags:', unknown)
    assert args.critic_warmup_windows >= 1
    return args


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def level_frame(env):
    """Gravity-level frame (fwd, left, up), exactly as in main_cuda.py."""
    fwd = env.R[:, :, 0].clone()
    fwd[:, 2] = 0
    fwd = F.normalize(fwd, 2, -1)
    up = torch.zeros_like(fwd)
    up[:, 2] = 1
    return torch.stack([fwd, torch.cross(up, fwd, dim=-1), up], -1)


def heading_command(args, env, goal_vec, carry):
    """Yaw command passed to env.run (only its direction matters).

    'goal' is main_cuda.py. 'velocity' follows the horizontal flight direction with
    hysteresis: start following above heading_min_speed, keep following down to
    heading_release_speed, and below that hold the last flight direction (snapping
    back to the goal would turn the camera to the wall being avoided).
    """
    if args.heading == 'goal':
        return goal_vec
    v = env.v.detach().clone()
    v[:, 2] = 0
    spd = v.norm(dim=-1, keepdim=True)
    following = carry.get('head_on', torch.zeros_like(spd, dtype=torch.bool))
    following = (spd > args.heading_min_speed) | (following & (spd > args.heading_release_speed))
    held = torch.where(following, v, carry.get('head_dir', goal_vec))
    carry['head_on'], carry['head_dir'] = following, held
    return held


def image_input(args, depth, local_v):
    """Policy image input, as main_cuda.build_model_input: inverse depth, or with
    --use_ttc inverse time-to-contact (depth / forward speed), max-pooled 4x."""
    if args.use_ttc:
        img = (depth / local_v[:, 0].clamp_min(1e-3)[:, None, None]).clamp(0.3, 10.0)
    else:
        img = depth.clamp(0.3, 48)
    x = 3 / img - 0.6 + torch.randn_like(img) * 0.02
    return F.max_pool2d(x[:, None], 4, 4)


# --------------------------------------------------------------------------
# Rollout of one SHAC window
# --------------------------------------------------------------------------
def rollout_window(args, env, model, carry, B, n_steps):
    """Run n_steps steps, mirroring the step logic in main_cuda.py."""
    rec = defaultdict(list)
    pending, h = carry['pending'], carry['h']

    for _ in range(n_steps):
        ctl_dt = normalvariate(1 / 15, 0.1 / 15)
        depth, _ = env.render(ctl_dt)
        rec['vec_to_pt'].append(env.find_vec_to_nearest_pt())

        if args.yaw_drift:
            carry['tv_raw'] = torch.squeeze(carry['tv_raw'][:, None] @ carry['R_drift'], 1)
            target_v_raw = carry['tv_raw']
        else:
            target_v_raw = env.p_target - env.p.detach()
        env.run(pending.popleft(), ctl_dt, heading_command(args, env, target_v_raw, carry))

        R = level_frame(env)
        target_v_norm = target_v_raw.norm(dim=-1, keepdim=True)
        target_v = target_v_raw / target_v_norm * torch.minimum(target_v_norm, env.max_speed)
        local_v = torch.squeeze(env.v[:, None] @ R, 1)

        x = image_input(args, depth, local_v)
        state = [torch.squeeze(target_v[:, None] @ R, 1), env.R[:, 2], env.margin[:, None]]
        if not args.no_odom:
            state.insert(0, local_v)
        state = torch.cat(state, -1)

        act, _, h = model(x, state, h)
        a_pred, v_pred = (R @ act.reshape(B, 3, -1)).unbind(-1)[:2]
        act = (a_pred - v_pred - env.g_std) * env.thr_est_error[:, None] + env.g_std
        pending.append(act)

        # Reward for this step, from the post-step state (differentiable).
        clearance = env.find_vec_to_nearest_pt()[0].norm(dim=-1) - env.margin
        track = (env.v - target_v).norm(dim=-1) / env.max_speed.reshape(-1)
        r = -(args.r_track * track
              + args.r_obj * (1 - clearance).relu().pow(2)
              + args.r_collide * F.softplus(-32 * clearance).clamp_max(args.collide_clip)
              ) * args.reward_scale

        obs = (x, state, h.detach())
        rec['r'].append(r)
        rec['obs'].append(tuple(t.detach() for t in obs))
        rec['act'].append(act)
        rec['v'].append(env.v)
        rec['v_pred'].append(v_pred)
        rec['target_v'].append(target_v)
        last_obs = obs  # live graph for the actor's bootstrap

    carry['h'] = h
    return rec, last_obs


# --------------------------------------------------------------------------
# DiffPhys losses
# --------------------------------------------------------------------------
def diffphys_losses(args, rec, carry, margin):
    """The original main_cuda.py losses, computed over one window.

    loss_v compares the average of v over v_avg_window steps with the target at
    the start of that span, exactly as main_cuda.py; spans that start in an
    earlier window use that window's (detached) velocities, so over an episode
    loss_v has the same terms as the original.
    """
    acts = torch.stack([carry['last_act']] + rec['act'])
    loss_d_acc = acts.pow(2).sum(-1).mean()
    loss_d_jerk = acts.diff(1, 0).mul(15).pow(2).sum(-1).mean()

    w = args.v_avg_window
    v_win, tv_win = torch.stack(rec['v']), torch.stack(rec['target_v'])
    v_all = torch.cat([carry['v_tail'], v_win])
    tv_all = torch.cat([carry['tv_tail'], tv_win])
    # main_cuda.py pairs mean(v[e-w+1..e]) with target_v[e-w+1] for every step e >= w.
    # This window owns the spans that end inside it; indices below are into v_all.
    start = carry['t']
    offset = start - carry['v_tail'].shape[0]
    ends = torch.arange(max(start, w), start + len(rec['v']), device=v_all.device) - offset
    if len(ends):
        cum = F.pad(v_all.cumsum(0), (0, 0, 0, 0, 1, 0))
        v_avg = (cum[ends + 1] - cum[ends + 1 - w]) / w
        delta_v = torch.norm(v_avg - tv_all[ends - w + 1], 2, -1)
        loss_v = F.smooth_l1_loss(delta_v, torch.zeros_like(delta_v))
    else:
        loss_v = torch.zeros((), device=v_all.device)
    carry['v_tail'] = v_all[-(w - 1):].detach()
    carry['tv_tail'] = tv_all[-(w - 1):].detach()

    loss_v_pred = F.mse_loss(torch.stack(rec['v_pred']), v_win.detach())

    # Same clearance losses as main_cuda.py (same tensor layout and diff axis).
    distance = torch.norm(torch.stack(rec['vec_to_pt']), 2, -1) - margin
    with torch.no_grad():
        v_to_pt = (-torch.diff(distance, 1, 1) * 135).clamp_min(1)
    loss_obj = (v_to_pt * (1 - distance[:, 1:]).relu().pow(2)).mean()
    loss_collide = F.softplus(distance[:, 1:].mul(-32)).mul(v_to_pt).mean()

    total = (args.coef_v * loss_v + args.coef_v_pred * loss_v_pred
             + args.coef_obj_avoidance * loss_obj + args.coef_collide * loss_collide
             + args.coef_d_acc * loss_d_acc + args.coef_d_jerk * loss_d_jerk)
    logs = dict(loss_v=loss_v, loss_v_pred=loss_v_pred, loss_obj_avoidance=loss_obj,
                loss_collide=loss_collide, loss_d_acc=loss_d_acc, loss_d_jerk=loss_d_jerk)
    return total, logs


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
def main():
    args = parse_args()
    print(args)
    device = torch.device('cuda')
    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(comment='_shac_orig')
    writer.add_text('args', '  \n'.join(f'{k}: {v}' for k, v in sorted(vars(args).items())))

    env = Env(args.batch_size, 64, 48, args.grad_decay, device,
              fov_x_half_tan=args.fov_x_half_tan, single=args.single,
              gate=args.gate, ground_voxels=args.ground_voxels,
              scaffold=args.scaffold, speed_mtp=args.speed_mtp,
              random_rotation=args.random_rotation, cam_angle=args.cam_angle,
              gimbal_stabilized=args.gimbal_stabilized)
    B = args.batch_size

    dim_obs = 7 if args.no_odom else 10
    model = Model(dim_obs, 6).to(device)
    if args.resume:
        missing, unexpected = model.load_state_dict(torch.load(args.resume, map_location=device), False)
        if missing or unexpected:
            print('missing:', missing, 'unexpected:', unexpected)
    else:
        print('[shac] no --resume: training the policy from scratch')
    actor_opt = AdamW(model.parameters(), args.actor_lr)

    critic = Critic(1, dim_obs, model.gru.hidden_size).to(device)
    target = copy.deepcopy(critic).requires_grad_(False)
    critic_opt = AdamW(critic.parameters(), args.critic_lr)

    # Windows of `horizon` steps; the last one is shorter if episode_steps is not a multiple.
    windows = [min(args.horizon, args.episode_steps - s) for s in range(0, args.episode_steps, args.horizon)]
    actor_sched = None
    if args.actor_lr_cosine:
        n_updates = max(1, args.num_episodes * len(windows) - args.critic_warmup_windows)
        actor_sched = CosineAnnealingLR(actor_opt, n_updates, args.actor_lr * 0.01)
    global_window = 0

    pbar = tqdm(range(args.num_episodes), ncols=140)
    for ep in pbar:
        env.reset()
        model.reset()
        a0 = env.act.detach()
        empty = torch.zeros((0, B, 3), device=device)
        carry = dict(pending=deque([a0, a0]), h=None, last_act=a0, t=0, v_tail=empty, tv_tail=empty)
        if args.yaw_drift:
            # Same drift model as main_cuda.py: a constant turn rate per drone (~5 deg/s std)
            drift_av = torch.randn(B, device=device) * (5 * math.pi / 180 / 15)
            zeros, ones = torch.zeros_like(drift_av), torch.ones_like(drift_av)
            carry['R_drift'] = torch.stack([torch.cos(drift_av), -torch.sin(drift_av), zeros,
                                            torch.sin(drift_av), torch.cos(drift_av), zeros,
                                            zeros, zeros, ones], -1).reshape(B, 3, 3)
            carry['tv_raw'] = (env.p_target - env.p).detach()

        logs = defaultdict(list)
        min_dist = torch.full((B,), float('inf'), device=device)
        speed_sum = torch.zeros(B, device=device)
        speed_max = torch.zeros(B, device=device)
        n_steps = 0

        for n_win in windows:
            warm = global_window < args.critic_warmup_windows
            with torch.no_grad() if warm else nullcontext():
                rec, last_obs = rollout_window(args, env, model, carry, B, n_win)
                l_dp, dp_logs = diffphys_losses(args, rec, carry, env.margin)
            rewards = torch.stack(rec['r'])

            if not warm:
                l_shac, v_end = actor_loss(rewards, last_obs, critic, args.gamma)
                loss = args.coef_shac * l_shac + l_dp
                if torch.isnan(loss):
                    raise RuntimeError('NaN actor loss')
                actor_opt.zero_grad()
                loss.backward()
                gnorm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                actor_opt.step()
                if actor_sched is not None:
                    actor_sched.step()
                logs['loss_actor'].append(float(loss))
                logs['loss_shac'].append(float(l_shac))
                logs['v_end'].append(float(v_end))
                logs['actor_grad_norm'].append(float(gnorm))
            for k, v in dp_logs.items():
                logs[k].append(float(v))

            if n_win > 1:
                c_loss, c_target = train_critic(args, rec['obs'], rewards, critic, target, critic_opt)
                logs['critic_loss'].append(float(c_loss))
                logs['critic_target'].append(float(c_target))
            logs['reward'].append(float(rewards.mean()))
            with torch.no_grad():
                dist = torch.norm(torch.stack(rec['vec_to_pt']), 2, -1) - env.margin   # (T, K, B)
                min_dist = torch.minimum(min_dist, dist.flatten(0, 1).min(0).values)
                speed = torch.stack(rec['v']).norm(dim=-1)
                speed_sum += speed.sum(0)
                speed_max = torch.maximum(speed_max, speed.max(0).values)
                n_steps += speed.shape[0]

            # Cut the graph before the next window; the episode keeps going.
            detach_graph(env)
            detach_graph(model)
            carry['h'] = carry['h'].detach()
            carry['pending'] = deque(a.detach() for a in carry['pending'])
            carry['last_act'] = carry['pending'][-1]
            carry['t'] += n_win
            global_window += 1

        # Episode metrics, same names as main_cuda.py
        success = (min_dist > 0).float()
        avg_speed = speed_sum / max(n_steps, 1)
        writer.add_scalar('success', success.mean(), ep)
        writer.add_scalar('avg_speed', avg_speed.mean(), ep)
        writer.add_scalar('max_speed', speed_max.mean(), ep)
        writer.add_scalar('ar', (success * avg_speed).mean(), ep)
        for k, v in logs.items():
            writer.add_scalar(f'shac/{k}', sum(v) / len(v), ep)
        mean = {k: sum(v) / len(v) for k, v in logs.items()}
        pbar.set_description_str(
            f'success {success.mean():.2f} speed {avg_speed.mean():.1f}'
            + (f' loss {mean["loss_actor"]:.3f}' if 'loss_actor' in mean else '')
            + (f' critic {mean["critic_loss"]:.4f}' if 'critic_loss' in mean else '')
            + (' [critic warmup]' if global_window <= args.critic_warmup_windows else ''))
        if args.print_every and (ep + 1) % args.print_every == 0:
            tqdm.write(f'[ep {ep + 1}] ' + ' '.join(f'{k} {v:.4g}' for k, v in sorted(mean.items())))

        if (ep + 1) % args.save_every == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'shac_checkpoint_{args.ckpt_tag}{ep + 1:05d}.pth'))
            torch.save(critic.state_dict(), os.path.join(args.save_dir, f'critic_{args.ckpt_tag}{ep + 1:05d}.pth'))


if __name__ == '__main__':
    main()
