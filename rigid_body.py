import math
import torch


def _skew(w: torch.Tensor) -> torch.Tensor:
    """(...,3) angular-velocity vector -> (...,3,3) skew-symmetric matrix (skew(w) @ x == w cross x)."""
    zeros = torch.zeros_like(w[..., 0])
    wx, wy, wz = w.unbind(-1)
    row0 = torch.stack([zeros, -wz, wy], -1)
    row1 = torch.stack([wz, zeros, -wx], -1)
    row2 = torch.stack([-wy, wx, zeros], -1)
    return torch.stack([row0, row1, row2], -2)


def expm_so3(w: torch.Tensor) -> torch.Tensor:
    """Closed-form matrix exponential of a so(3) element (Rodrigues' formula).

    addresses point-mass limitation: this is a globally valid rotation integrator (no small-angle
    assumption, no singularity), unlike the point-mass model's algebraic per-step heading solve --
    so a sustained body rate can integrate through a full aerobatic loop/flip.

    w: (...,3) rotation vector (axis * angle for this step). Stable as w -> 0 via torch.sinc.
    """
    theta = w.norm(dim=-1, keepdim=True)
    K = _skew(w)
    sinc = torch.sinc(theta / math.pi)                        # sin(theta)/theta, stable at theta=0
    cosc = torch.sinc(theta / (2 * math.pi)).pow(2) * 0.5      # (1-cos(theta))/theta^2, stable at theta=0
    eye = torch.eye(3, device=w.device, dtype=w.dtype).expand(*w.shape[:-1], 3, 3)
    return eye + sinc[..., None] * K + cosc[..., None] * (K @ K)
    # end point-mass limitation


def rigid_body_step(R, omega, v, p, thrust_cmd, omega_cmd, rate_ctl_delay,
                     drag_2, z_drag_coef, v_wind, g_std, ctl_dt):
    """One differentiable step of thrust + body-rate quadrotor dynamics.

    addresses point-mass limitation: replaces the point-mass model's algebraic
    "attitude = desired thrust direction" proxy with a real integrated rigid-body state
    (angular velocity + orientation), enabling maneuvers (flips, loops) the point-mass
    model's small-angle heading solve cannot represent.

    R: (B,3,3) world-from-body rotation, columns = (forward, left, up)
    omega: (B,3) body-frame angular velocity (rad/s)
    v, p: (B,3) world-frame velocity, position
    thrust_cmd: (B,) collective specific thrust (m/s^2), applied along body 'up' (R[...,2])
    omega_cmd: (B,3) desired body rate (rad/s)
    rate_ctl_delay: (B,3) inner-loop rate-controller time constant per axis (1/s) -- plays the
        same role pitch_ctl_delay/yaw_ctl_delay play for the point-mass model's attitude lag
    drag_2: (B,2) [quadratic_coef, linear_coef], same convention as the point-mass CUDA kernel
    z_drag_coef, v_wind, g_std, ctl_dt: same convention as Env/quadsim_kernel.cu

    Returns: R_next, omega_next, v_next, p_next
    """
    # inner-loop rate controller: first-order lag toward the commanded body rate, mirroring the
    # act_pred -> act lag the point-mass model uses for its attitude response
    alpha = torch.exp(-rate_ctl_delay * ctl_dt)
    omega_next = omega_cmd * (1 - alpha) + omega * alpha

    R_next = R @ expm_so3(omega_next * ctl_dt)

    fwd, left, up = R[..., 0], R[..., 1], R[..., 2]
    v_rel = v - v_wind
    v_fwd = (v_rel * fwd).sum(-1, keepdim=True)
    v_left = (v_rel * left).sum(-1, keepdim=True)
    v_up = (v_rel * up).sum(-1, keepdim=True)
    a_drag_1 = v_fwd * fwd + v_left * left + v_up * z_drag_coef * up
    a_drag_2 = v_fwd.abs() * v_fwd * fwd + v_left.abs() * v_left * left + v_up.abs() * v_up * z_drag_coef * up
    drag = -drag_2[:, :1] * a_drag_2 - drag_2[:, 1:] * a_drag_1

    a = up * thrust_cmd[:, None] + g_std + drag
    v_next = v + a * ctl_dt
    p_next = p + v_next * ctl_dt
    return R_next, omega_next, v_next, p_next
    # end point-mass limitation


def rigid_body_step_ctbr_paper(R, omega, thrust, v, p, thrust_cmd, omega_cmd, rate_ctl_delay, thrust_ctl_delay,
                                lin_drag_coef, v_wind, g_std, ctl_dt):
    """CTBR step matching Eq. 6-11 of "Vision-Based End-to-End Learning for UAV Traversal of
    Irregular Gaps via Differentiable Simulation" (arXiv:2604.02779) -- only used when
    --ctbr_paper_match is passed to main_cuda_rigid.py; rigid_body_step() above is the default.

    addresses point-mass limitation: differs from rigid_body_step() in two ways the paper's
    ablation (their Table I) found to matter for large-tilt maneuvers: (1) thrust gets the same
    first-order lag as body rate, not applied instantly; (2) drag is a single linear coefficient
    in world frame (-k_v * v), not the fwd/left/up-decomposed quadratic+linear drag reused from
    the point-mass CUDA kernel.

    thrust, thrust_cmd: (B,1) collective specific thrust (state, command)
    thrust_ctl_delay: (B,1) inner-loop thrust-controller time constant (1/s), matches the paper's tau_c
    lin_drag_coef: (B,1) linear drag coefficient k_v, s.t. a_drag = -k_v * (v - v_wind)
    (other args match rigid_body_step())

    Returns: R_next, omega_next, thrust_next, v_next, p_next
    """
    alpha_omega = torch.exp(-rate_ctl_delay * ctl_dt)
    omega_next = omega_cmd * (1 - alpha_omega) + omega * alpha_omega
    alpha_thrust = torch.exp(-thrust_ctl_delay * ctl_dt)
    thrust_next = thrust_cmd * (1 - alpha_thrust) + thrust * alpha_thrust

    R_next = R @ expm_so3(omega_next * ctl_dt)

    up = R[..., 2]
    drag = -lin_drag_coef * (v - v_wind)

    a = up * thrust_next + g_std + drag
    v_next = v + a * ctl_dt
    p_next = p + v_next * ctl_dt
    return R_next, omega_next, thrust_next, v_next, p_next
    # end point-mass limitation
