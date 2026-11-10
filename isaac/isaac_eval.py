"""
Fly a TTC policy (main_cuda_shac.py, or the DiffPhysDrone baseline) through a USD city in
Isaac Sim 5.1 + Pegasus.

What is reproduced exactly from training:
  - 64x48 depth camera, fov_x_half_tan, cam_angle tilt, rigidly mounted on the body
  - TTC = planar depth / forward speed in the gravity-level frame
  - model input 3/clamp(ttc, 0.3, 10) - 0.6 + noise, max-pooled by 4
  - state [local_v, target_v (level frame), body-z row of R, margin]
  - action decoding: act = a_pred - v_pred (world)
  - 15 Hz control, one step of action lag
  - camera heading: --heading velocity follows the horizontal flight direction with
    the same hysteresis and hold as main_cuda_shac.heading_command; --heading goal
    faces the goal, as main_cuda.py. The yaw command is turned by the same rule as the
    training sim (env_cuda.Env.run -> update_state_vec).
What is different on purpose (the point of the test):
  - rendering (RTX depth), rigid-body physics (PhysX), Pegasus Iris rotors + linear drag,
  - a geometric attitude controller turning the commanded acceleration into rotor speeds.

Run with Isaac's python, from the repo root:

    isaac_run isaac/isaac_eval.py --headless \\
        --resume checkpoints/shac/shac_checkpoint_orig_03000.pth \\
        --start 71.8 368.7 --goal 91.8 204.7 --max_speed 8 --out results/r3/shac/v8

    # baseline (main_cuda.py) checkpoint: camera on the goal, as it was trained
    ... --resume checkpoints/base/cmd_checkpoint0000.pth --heading goal

Start/goal are x y in metres (z = --alt above the ground below; the script prints the
world bounds). The city is NVIDIA's AECO City Demo Pack (World_CityDemopack.usd); pass
its path with --env or the CITY_USD environment variable. --record writes the first episode to <out>/<checkpoint folder>_ep000.mp4.
Output: <out>/summary.json (args, summary, per-episode results) and <out>/epNNN.npz
(trajectory).
"""

import argparse
import json
import math
import os
import sys
import time
from collections import deque

# ----------------------------------------------------------------------------- args
ENVIRONMENT = os.environ.get("CITY_USD")

p = argparse.ArgumentParser()
p.add_argument("--resume", required=True, help="policy checkpoint")
p.add_argument("--env", default=ENVIRONMENT, required=ENVIRONMENT is None,
               help="USD world to fly in (default: $CITY_USD)")
p.add_argument("--start", type=float, nargs="+", required=True,
               help="start 'x y' or 'x y z' in metres (z omitted = --alt above the ground below)")
p.add_argument("--goal", type=float, nargs="+", required=True, help="goal, same format as --start")
p.add_argument("--start_jitter", type=float, default=3.0, help="random xy offset of the start per episode (m)")
p.add_argument("--episodes", type=int, default=10)
p.add_argument("--timesteps", type=int, default=2100, help="control steps per episode (140 s at 15 Hz)")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--max_speed", type=float, default=8.0, help="commanded speed (m/s)")
p.add_argument("--margin", type=float, default=0.2, help="drone radius for collisions (m)")
p.add_argument("--alt", type=float, default=6.0, help="flight altitude above the ground (m)")
p.add_argument("--goal_radius", type=float, default=5.0,
               help="goal reached within this horizontal distance (m), checked every physics step")
p.add_argument("--goal_dz", type=float, default=5.0, help="... and within this vertical distance (m)")
p.add_argument("--max_tilt", type=float, default=35.0,
               help="cap the commanded tilt (deg) by clipping the horizontal part of the acceleration "
                    "command, like a flight controller's tilt limit (PX4 MPC_TILTMAX_AIR); 0 = off")
p.add_argument("--speed_ramp", type=float, default=3.0,
               help="seconds over which the commanded speed rises from 0 to --max_speed at the start "
                    "(soft start); the target never drops below 1 m/s; 0 = off")
p.add_argument("--headless", action="store_true")
p.add_argument("--out", default="isaac_results")
p.add_argument("--record", action="store_true",
               help="record the first episode through a chase camera (works headless), speed overlaid")
p.add_argument("--record_res", type=int, nargs=2, default=[1280, 720], metavar=("W", "H"))
p.add_argument("--record_fps", type=int, default=30)
# camera heading
p.add_argument("--heading", choices=["velocity", "goal"], default="velocity",
               help="velocity: camera follows the flight direction, as in SHAC training (default). "
                    "goal: camera on the goal, as in main_cuda.py")
p.add_argument("--heading_min_speed", type=float, default=1.0,
               help="start following the flight direction above this speed (m/s)")
p.add_argument("--heading_release_speed", type=float, default=0.3,
               help="stop following below this speed; in between, keep the last flight direction")
p.add_argument("--yaw_ctl_delay", type=float, default=6.0, help="yaw response rate (training samples 6 +- 0.6)")
p.add_argument("--yaw_inertia", type=float, default=5.0,
               help="weight of the current heading against the heading vector (training: 5)")
# timing / vehicle
p.add_argument("--physics_hz", type=int, default=240)
p.add_argument("--ctl_hz", type=int, default=15)
p.add_argument("--act_lag", type=int, default=1, help="control steps of action delay (training: 1)")
p.add_argument("--mass", type=float, default=1.5, help="Pegasus Iris mass (kg)")
p.add_argument("--kr", type=float, default=3.5, help="attitude gain")
p.add_argument("--kw", type=float, default=0.5, help="angular rate gain")
# must match the checkpoint (defaults = configs/single_agent.args)
p.add_argument("--fov_x_half_tan", type=float, default=0.82)
p.add_argument("--cam_angle", type=float, default=20.0, help="camera pitch-up (deg)")
args, _unknown = p.parse_known_args()
if _unknown:
    print("[eval] ignoring unrecognised flags:", _unknown)

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": args.headless, "extra_args": ["--/app/extensions/fsWatcherEnabled=0"]})

# ----------------------------------------------------------------------------- imports needing Isaac
import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from omni.isaac.core.world import World  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from pegasus.simulator.logic.backends import Backend  # noqa: E402
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface  # noqa: E402
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig  # noqa: E402
from pegasus.simulator.params import ROBOTS  # noqa: E402

try:
    from isaacsim.sensors.camera import Camera
except ImportError:  # older module name
    from omni.isaac.sensor import Camera

# model.py lives in the repo root, one folder up from this file.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import Model  # noqa: E402

G = 9.80665
IMG_W, IMG_H = 64, 48
VEHICLE = "/World/quadrotor"
ENV_ROOT = "/World/Env"
CAM_PATH = f"{VEHICLE}/body/ttc_camera"
VIEW_CAM = "/OmniverseKit_Persp"   # Isaac's default viewport camera


# ============================================================================= controller backend
class AccelYawBackend(Backend):
    """Tracks a world-frame net acceleration (what the policy commands, gravity included)
    and a yaw angle, with a geometric attitude controller (same structure as Pegasus'
    NonlinearController, minus the position loop)."""

    def __init__(self, mass, kr, kw):
        super().__init__(None)
        self.m = mass
        self.Kr = np.diag([kr, kr, kr])
        self.Kw = np.diag([kw, kw, kw])
        self.a_cmd = np.zeros(3)
        self.yaw_cmd = 0.0
        self.input_ref = [0.0] * 4
        self.R = np.eye(3)
        self.w = np.zeros(3)
        self.have_state = False

    def set_command(self, a_cmd, yaw):
        self.a_cmd = np.asarray(a_cmd, dtype=np.float64)
        self.yaw_cmd = float(yaw)

    def update_state(self, state):
        self.R = Rotation.from_quat(state.attitude).as_matrix()
        self.w = np.asarray(state.angular_velocity)
        self.have_state = True

    def input_reference(self):
        return self.input_ref

    def update(self, dt):
        if not self.have_state or self.vehicle is None:
            return
        f = self.m * (self.a_cmd + np.array([0.0, 0.0, G]))
        f[2] = max(f[2], 0.1 * self.m * G)            # never command upside-down thrust
        z_b = self.R[:, 2]
        u1 = max(float(f @ z_b), 0.0)
        z_des = f / np.linalg.norm(f)
        x_c = np.array([math.cos(self.yaw_cmd), math.sin(self.yaw_cmd), 0.0])
        y_des = np.cross(z_des, x_c)
        y_des /= max(np.linalg.norm(y_des), 1e-6)
        x_des = np.cross(y_des, z_des)
        R_des = np.c_[x_des, y_des, z_des]
        S = R_des.T @ self.R - self.R.T @ R_des
        e_R = 0.5 * np.array([-S[1, 2], S[0, 2], -S[0, 1]])
        tau = -(self.Kr @ e_R) - (self.Kw @ self.w)
        self.input_ref = self.vehicle.force_and_torques_to_velocities(u1, tau)

    def update_sensor(self, sensor_type, data):
        pass

    def update_graphical_sensor(self, sensor_type, data):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def reset(self):
        self.a_cmd = np.zeros(3)


# ============================================================================= USD world
def load_env(stage, path):
    """Reference the world under /World/Env, converted to metres and Z-up."""
    if not os.path.exists(path):
        raise SystemExit(f"environment not found: {path}")
    probe = Usd.Stage.Open(path, Usd.Stage.LoadNone)          # metadata only, no payloads
    mpu = UsdGeom.GetStageMetersPerUnit(probe) or 0.01
    up = UsdGeom.GetStageUpAxis(probe)
    del probe
    prim = stage.DefinePrim(ENV_ROOT, "Xform")
    prim.GetReferences().AddReference(path)
    api = UsdGeom.XformCommonAPI(prim)
    api.SetScale(Gf.Vec3f(mpu, mpu, mpu))
    if up == UsdGeom.Tokens.y:
        api.SetRotate(Gf.Vec3f(90.0, 0.0, 0.0))
    print(f"[env] {os.path.basename(path)}: metersPerUnit={mpu}, upAxis={up}")


def add_env_colliders(stage):
    """Static triangle-mesh colliders on every mesh, so PhysX can tell when the drone hits the city.
    Instanced prims are de-instanced first (instance proxies can't take new APIs)."""
    root = stage.GetPrimAtPath(ENV_ROOT)
    while True:
        inst = [p for p in Usd.PrimRange(root) if p.IsInstance()]
        if not inst:
            break
        for p in inst:
            p.SetInstanceable(False)
    n = 0
    for p in Usd.PrimRange(root):
        if p.IsA(UsdGeom.Mesh) and not p.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(p)
            UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr("none")
            n += 1
    print(f"[env] added colliders to {n} meshes (first physics step cooks them; can take a minute)")


def env_bounds(stage):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    r = cache.ComputeWorldBound(stage.GetPrimAtPath(ENV_ROOT)).ComputeAlignedRange()
    return np.array(r.GetMin()), np.array(r.GetMax())


def _sq():
    from omni.physx import get_physx_scene_query_interface
    return get_physx_scene_query_interface()


def env_overlap(pos, radius):
    """True if any world collider (not the drone) is within radius of pos."""
    hit = []

    def cb(h):
        if str(h.collision).startswith(VEHICLE):
            return True                      # ignore the drone's own parts, keep looking
        hit.append(str(h.collision))
        return False
    _sq().overlap_sphere(float(radius), carb.Float3(*map(float, pos)), cb, False)
    return bool(hit)


def env_clearance(pos, margin, ladder=(0.1, 0.25, 0.5, 1.0, 2.0)):
    """Coarse clearance (m beyond margin): the largest ladder step that is still free."""
    if env_overlap(pos, margin):
        return -0.01
    free = 0.0
    for r in ladder:
        if env_overlap(pos, margin + r):
            return free
        free = r
    return float(ladder[-1])


def ground_z(x, y, z_top):
    """Height of the first world surface below (x, y), ignoring the drone."""
    best = [None]

    def cb(h):
        if not str(h.collision).startswith(VEHICLE):
            if best[0] is None or h.distance < best[0]:
                best[0] = h.distance
        return True
    _sq().raycast_all(carb.Float3(float(x), float(y), float(z_top)), carb.Float3(0.0, 0.0, -1.0),
                      float(z_top + 1000.0), cb)
    return None if best[0] is None else z_top - best[0]


def make_scenes(rng, hi):
    """Start/goal per episode: --start/--goal at --alt above the ground, start jittered."""
    def place(v, name):
        if len(v) >= 3:
            return np.array(v[:3], dtype=np.float64)
        gz = ground_z(v[0], v[1], hi[2] + 5.0)
        if gz is None:
            print(f"[env] WARNING: no ground found under --{name} {v}; using z = --alt")
            gz = 0.0
        return np.array([v[0], v[1], gz + args.alt])

    start, goal = place(args.start, "start"), place(args.goal, "goal")
    print(f"[env] start {np.round(start, 2).tolist()}  goal {np.round(goal, 2).tolist()}  "
          f"({np.linalg.norm(goal - start):.1f} m apart)")
    for name, pt in (("start", start), ("goal", goal)):
        if env_overlap(pt, args.margin + 0.3):
            print(f"[env] WARNING: {name} is inside or touching geometry; move it")
    scenes = []
    for _ in range(args.episodes):
        p0 = start.copy()
        if args.start_jitter > 0:
            p0[:2] += rng.uniform(-args.start_jitter, args.start_jitter, 2)
        scenes.append(dict(p0=p0, goal=goal.copy()))
    return scenes


# ============================================================================= cameras
def setup_camera(stage):
    cam = Camera(prim_path=CAM_PATH, resolution=(IMG_W, IMG_H))
    prim = stage.GetPrimAtPath(CAM_PATH)
    h_ap = 20.955
    prim.GetAttribute("horizontalAperture").Set(h_ap)
    prim.GetAttribute("verticalAperture").Set(h_ap * IMG_H / IMG_W)   # square pixels
    prim.GetAttribute("focalLength").Set(h_ap / (2.0 * args.fov_x_half_tan))
    prim.GetAttribute("clippingRange").Set(Gf.Vec2f(0.05, 1000.0))
    return cam


def mount_camera(cam):
    # world camera axes: +x forward, +z up. Rotation about body y by -angle tilts the view up.
    q = Rotation.from_euler("y", -args.cam_angle, degrees=True).as_quat()      # xyzw
    cam.set_local_pose(translation=np.array([0.1, 0.0, 0.0]),
                       orientation=np.array([q[3], q[0], q[1], q[2]]), camera_axes="world")


def read_depth(cam):
    d = None
    try:
        d = cam.get_depth()
    except Exception:
        pass
    if d is None or np.size(d) == 0:
        d = cam.get_current_frame().get("distance_to_image_plane")
    if d is None or np.size(d) == 0:
        return None
    d = np.asarray(d, dtype=np.float32).reshape(IMG_H, IMG_W)
    return np.where(np.isfinite(d) & (d > 0), d, 1000.0)


def set_view(eye, target):
    try:
        from isaacsim.core.utils.viewports import set_camera_view
    except ImportError:
        from omni.isaac.core.utils.viewports import set_camera_view
    set_camera_view(eye=eye, target=target, camera_prim_path=VIEW_CAM)


def chase_view(p, fwd):
    """Chase cam: 4 m behind, 1.5 m above, looking 3 m ahead along fwd."""
    set_view(p - 4.0 * fwd + np.array([0.0, 0.0, 1.5]), p + 3.0 * fwd)


def set_camera_lighting():
    """Viewport 'Camera Light' (visual only: the policy's depth input does not depend on lighting)."""
    try:
        import omni.kit.actions.core
        omni.kit.actions.core.execute_action("omni.kit.viewport.menubar.lighting", "set_lighting_mode_camera")
    except Exception:                        # menu extension missing (e.g. headless): set the flag directly
        import carb.settings
        carb.settings.get_settings().set("/rtx/useViewLightingMode", True)


# ============================================================================= recording (--record)
class Recorder:
    """Third-person video of an episode through the viewport camera, chase framing, heading
    smoothed over 0.4 s. Encoded with ffmpeg/x264 when available, else OpenCV mp4v.
    Speed / time / goal distance are drawn on every frame."""

    SMOOTH = 0.4

    def __init__(self, stage, path, fps):
        import cv2
        existed = stage.GetPrimAtPath(VIEW_CAM).IsValid()
        self.cam = Camera(prim_path=VIEW_CAM, resolution=tuple(args.record_res))
        if not existed:            # headless without a viewport: give it Kit's default perspective lens
            prim = stage.GetPrimAtPath(VIEW_CAM)
            prim.GetAttribute("focalLength").Set(18.147562)
            prim.GetAttribute("horizontalAperture").Set(20.955)
            prim.GetAttribute("verticalAperture").Set(15.2908)
            prim.GetAttribute("clippingRange").Set(Gf.Vec2f(0.01, 1.0e7))
        self.cv2, self.path, self.fps = cv2, path, fps
        self.proc = self.writer = None
        self.frames = 0
        self.fwd = None              # smoothed horizontal heading of the camera

    def place(self, p, Rb):
        fwd = level_frame(Rb)[:, 0]
        if self.fwd is None:
            self.fwd = fwd
        else:
            a = 1.0 - math.exp(-1.0 / (self.fps * self.SMOOTH))
            self.fwd = self.fwd + a * (fwd - self.fwd)
            self.fwd[2] = 0.0
            self.fwd /= max(np.linalg.norm(self.fwd), 1e-6)
        chase_view(p, self.fwd)

    def _open(self, w, h):
        import shutil
        import subprocess
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        if shutil.which("ffmpeg"):
            self.proc = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                 "-s", f"{w}x{h}", "-r", str(self.fps), "-i", "-",
                 "-c:v", "libx264", "-preset", "slow", "-crf", "18",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", self.path],
                stdin=subprocess.PIPE)
        else:
            print("    (no ffmpeg on PATH; recording with OpenCV mp4v, lower quality)")
            self.writer = self.cv2.VideoWriter(self.path, self.cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))

    def grab(self, t, speed, goal_dist, max_speed, pos):
        cv2 = self.cv2
        img = self.cam.get_rgba()
        if img is None or np.size(img) == 0:
            return
        img = np.ascontiguousarray(np.asarray(img)[..., :3][..., ::-1], dtype=np.uint8)   # RGB -> BGR
        h, w = img.shape[:2]
        if self.proc is None and self.writer is None:
            self._open(w, h)
        s = h / 720.0
        lines = [f"speed {speed:5.2f} m/s", f"t {t:5.1f} s", f"goal {goal_dist:5.1f} m",
                 f"x {pos[0]:7.1f}  y {pos[1]:7.1f}  z {pos[2]:5.1f}"]
        box = img[int(10 * s):int(190 * s), int(10 * s):int(420 * s)]
        box[:] = (box * 0.45).astype(np.uint8)                 # translucent backing
        for i, txt in enumerate(lines):
            cv2.putText(img, txt, (int(22 * s), int((48 + 40 * i) * s)), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0 * s, (255, 255, 255), max(1, int(2 * s)), cv2.LINE_AA)
        # speed bar, full = the commanded max speed
        frac = min(speed / max(max_speed, 1e-6), 1.0)
        x0, y0, bw, bh = int(22 * s), int(172 * s), int(296 * s), int(8 * s)
        cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), (80, 80, 80), -1)
        cv2.rectangle(img, (x0, y0), (x0 + int(bw * frac), y0 + bh), (0, 200, 255), -1)
        if self.proc is not None:
            self.proc.stdin.write(img.tobytes())
        else:
            self.writer.write(img)
        self.frames += 1

    def close(self):
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait()
        elif self.writer is not None:
            self.writer.release()
        else:
            print("    !! recording produced no frames")
            return
        print(f"    recorded {self.frames} frames ({self.frames / self.fps:.1f} s) -> {self.path}")


# ============================================================================= policy
def level_frame(Rb):
    fwd = Rb[:, 0].copy()
    fwd[2] = 0.0
    fwd /= max(np.linalg.norm(fwd), 1e-6)
    up = np.array([0.0, 0.0, 1.0])
    return np.stack([fwd, np.cross(up, fwd), up], 1)        # columns: fwd, left, up


class Policy:
    def __init__(self, device):
        self.device = device
        self.model = Model(10, 6, 1).to(device)
        missing, unexpected = self.model.load_state_dict(torch.load(args.resume, map_location=device), False)
        if missing or unexpected:
            print("!! checkpoint mismatch  missing:", missing, " unexpected:", unexpected)
        self.model.eval()
        self.h = None

    def reset(self):
        self.h = None

    @torch.no_grad()
    def act(self, depth, v, Rb, target_v, margin):
        R = level_frame(Rb)
        dev = self.device
        local_v = R.T @ v
        ttc = torch.from_numpy(depth).to(dev)[None] / max(local_v[0], 1e-3)
        x = 3 / ttc.clamp(0.3, 10.0) - 0.6 + torch.randn_like(ttc) * 0.02
        x = F.max_pool2d(x[:, None], 4, 4)
        state = np.concatenate([local_v, R.T @ target_v, Rb[2, :], [margin]])
        state = torch.tensor(state, dtype=torch.float32, device=dev)[None]
        act, _, self.h = self.model(x.float(), state, self.h)
        Rt = torch.tensor(R, dtype=torch.float32, device=dev)[None]
        a_pred, v_pred = (Rt @ act.reshape(1, 3, -1)).unbind(-1)[:2]
        return (a_pred - v_pred)[0].cpu().numpy().astype(np.float64)


def turn_yaw(yaw_cmd, head):
    """One control step of the training sim's yaw rule (env_cuda.Env.run -> update_state_vec):
      target = normalize(yaw_inertia * forward + heading)   (heading NOT normalized)
      forward <- alpha * forward + (1 - alpha) * target,  alpha = exp(-yaw_ctl_delay * dt)
    Applied to the yaw command; Pegasus' attitude loop then tracks it."""
    fwd = np.array([math.cos(yaw_cmd), math.sin(yaw_cmd)])
    tgt = args.yaw_inertia * fwd + np.asarray(head[:2], dtype=np.float64)
    if np.linalg.norm(tgt) < 1e-6:
        return yaw_cmd, 0.0
    alpha = math.exp(-args.yaw_ctl_delay / args.ctl_hz)
    new = alpha * fwd + (1 - alpha) * tgt / np.linalg.norm(tgt)
    step = math.atan2(math.sin(math.atan2(new[1], new[0]) - yaw_cmd),
                      math.cos(math.atan2(new[1], new[0]) - yaw_cmd))
    return math.atan2(math.sin(yaw_cmd + step), math.cos(yaw_cmd + step)), abs(step)


def clip_tilt(a_cmd):
    """|a_h| <= (a_z + g) * tan(max_tilt): the thrust vector stays within max_tilt of vertical."""
    if args.max_tilt <= 0:
        return a_cmd
    a_h_max = max(a_cmd[2] + G, 0.1 * G) * math.tan(math.radians(args.max_tilt))
    a_h = np.linalg.norm(a_cmd[:2])
    if a_h <= a_h_max:
        return a_cmd
    a = a_cmd.copy()
    a[:2] *= a_h_max / a_h
    return a


# ============================================================================= episode
def run_episode(world, vehicle, backend, cam, policy, scene, idx, rec=None):
    p0, goal = scene["p0"], scene["goal"]
    yaw0 = math.atan2(goal[1] - p0[1], goal[0] - p0[0])
    q = Rotation.from_euler("z", yaw0).as_quat()
    q_wxyz = np.array([q[3], q[0], q[1], q[2]])

    # Hold the drone at the start while the controller and renderer settle
    backend.set_command(np.zeros(3), yaw0)
    for _ in range(int(0.5 * args.physics_hz)):
        vehicle.set_world_pose(position=p0, orientation=q_wxyz)
        vehicle.set_linear_velocity(np.zeros(3))
        vehicle.set_angular_velocity(np.zeros(3))
        world.step(render=False)
    for _ in range(3):
        world.step(render=True)

    policy.reset()
    sub = args.physics_hz // args.ctl_hz
    rec_every = max(1, round(args.physics_hz / rec.fps)) if rec is not None else sub   # substeps per video frame
    act_q = deque([np.zeros(3)] * (args.act_lag + 1))
    depth = np.full((IMG_H, IMG_W), 1000.0, np.float32)
    log = {"p": [], "v": [], "speed": [], "a_cmd": [], "t": [], "yaw_cmd": []}
    crashed = reached = False
    min_clear = np.inf
    t_reach = None
    s = vehicle.state
    yaw_cmd, yaw_turned = yaw0, 0.0
    head_on, head_dir = False, None
    for k in range(args.timesteps):
        d = read_depth(cam)                  # rendered before this step's physics
        if d is not None:
            depth = d

        # Camera heading, as main_cuda_shac.heading_command: follow the horizontal flight
        # direction above heading_min_speed, keep following down to heading_release_speed,
        # below that hold the last flight direction. --heading goal: face the goal.
        head = goal - s.position
        if args.heading == "velocity":
            v_h = np.array([s.linear_velocity[0], s.linear_velocity[1], 0.0])
            spd_h = np.linalg.norm(v_h)
            head_on = spd_h > args.heading_min_speed or (head_on and spd_h > args.heading_release_speed)
            if head_on:
                head_dir = v_h
            if head_dir is not None:
                head = head_dir
        yaw_cmd, dyaw = turn_yaw(yaw_cmd, head)
        yaw_turned += dyaw

        a_cmd = act_q.popleft()
        backend.set_command(clip_tilt(a_cmd), yaw_cmd)
        for j in range(sub):
            snap = rec is not None and (k * sub + j + 1) % rec_every == 0
            if snap:
                rec.place(vehicle.state.position, Rotation.from_quat(vehicle.state.attitude).as_matrix())
            world.step(render=(j == sub - 1) or snap)
            vs = vehicle.state
            if snap:
                rec.grab(k / args.ctl_hz + (j + 1) / args.physics_hz, float(np.linalg.norm(vs.linear_velocity)),
                         float(np.linalg.norm(goal - vs.position)), args.max_speed, vs.position)
            dg = goal - vs.position
            if np.linalg.norm(dg[:2]) <= args.goal_radius and abs(dg[2]) <= args.goal_dz:
                reached, t_reach = True, k / args.ctl_hz + (j + 1) / args.physics_hz
                break
            if env_overlap(vs.position, args.margin):
                crashed, min_clear = True, min(min_clear, -0.01)
                break
            if j == sub - 1:
                min_clear = min(min_clear, env_clearance(vs.position, args.margin))
        s = vehicle.state
        p, v = s.position.copy(), s.linear_velocity.copy()
        log["p"].append(p); log["v"].append(v); log["a_cmd"].append(a_cmd); log["t"].append(k / args.ctl_hz)
        log["speed"].append(float(np.linalg.norm(v))); log["yaw_cmd"].append(yaw_cmd)
        if crashed or reached:
            break

        target_raw = goal - p
        n = np.linalg.norm(target_raw)
        v_cap = args.max_speed
        if args.speed_ramp > 0:
            v_cap = min(v_cap, max(1.0, args.max_speed * (k + 1) / (args.speed_ramp * args.ctl_hz)))
        target_v = target_raw / max(n, 1e-6) * min(n, v_cap)
        Rb = Rotation.from_quat(s.attitude).as_matrix()
        if rec is None and not args.headless:   # while recording, the recorder drives the view camera
            chase_view(p, level_frame(Rb)[:, 0])
        act_q.append(policy.act(depth, v, Rb, target_v, args.margin))

    P = np.array(log["p"])
    speed = np.array(log["speed"]) if log["speed"] else np.zeros(1)
    res = {
        "episode": idx,
        "crashed": crashed, "reached": reached,
        "success": not crashed,
        "time_to_goal": t_reach,
        "avg_speed": float(speed.mean()), "max_speed": float(speed.max()),
        "max_speed_cmd": args.max_speed,
        "final_goal_dist": float(np.linalg.norm(goal - P[-1])) if len(P) else None,
        "closest_goal_dist": float(np.min(np.linalg.norm(P - goal, axis=1))) if len(P) else None,
        "min_clearance": float(min_clear),
        "duration": len(P) / args.ctl_hz,
        "yaw_turns": yaw_turned / (2 * math.pi),   # full turns of the yaw command
    }
    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, f"ep{idx:03d}.npz"), goal=goal, p0=p0,
             **{k: np.array(v) for k, v in log.items()})
    return res


# ============================================================================= main
def main():
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    timeline = omni.timeline.get_timeline_interface()
    pg = PegasusInterface()
    settings = dict(pg._world_settings)
    settings.update(physics_dt=1.0 / args.physics_hz, rendering_dt=1.0 / args.physics_hz,
                    stage_units_in_meters=1.0)
    pg._world = World(**settings)
    world = pg.world
    load_env(world.stage, args.env)
    for _ in range(30):
        simulation_app.update()           # let the references and textures load
    set_camera_lighting()
    add_env_colliders(world.stage)
    lo, hi = env_bounds(world.stage)
    print(f"[env] world bounds (m): min {np.round(lo, 1).tolist()}  max {np.round(hi, 1).tolist()}")

    backend = AccelYawBackend(args.mass, args.kr, args.kw)
    cfg = MultirotorConfig()
    cfg.backends = [backend]
    vehicle = Multirotor(VEHICLE, ROBOTS["Iris"], 0, list(map(float, hi + 20.0)),   # parked outside the city
                         [0.0, 0.0, 0.0, 1.0], config=cfg)
    cam = setup_camera(world.stage)
    fps = args.physics_hz / max(1, round(args.physics_hz / args.record_fps))
    ckpt_name = os.path.basename(os.path.dirname(os.path.abspath(args.resume)))
    rec = Recorder(world.stage, os.path.join(args.out, f"{ckpt_name}_ep000.mp4"), fps) if args.record else None
    world.reset()
    cam.initialize()
    cam.add_distance_to_image_plane_to_frame()
    if rec is not None:
        rec.cam.initialize()
    mount_camera(cam)
    timeline.play()
    for _ in range(10):
        world.step(render=True)
    scenes = make_scenes(rng, hi)

    policy = Policy(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[eval] checkpoint {args.resume}, heading {args.heading}")
    results = []
    for idx, scene in enumerate(scenes):
        if not simulation_app.is_running():
            break
        t0 = time.time()
        ep_rec = rec if idx == 0 else None
        try:
            r = run_episode(world, vehicle, backend, cam, policy, scene, idx, ep_rec)
        finally:
            if ep_rec is not None:
                ep_rec.close()
        results.append(r)
        tag = "CRASH" if r["crashed"] else ("GOAL" if r["reached"] else "timeout")
        print(f"[ep {idx:3d}] {tag:7s} avg {r['avg_speed']:.2f} m/s  max {r['max_speed']:.2f}  "
              f"closest-to-goal {r['closest_goal_dist']:.2f} m  min-clear {r['min_clearance']:.2f} m  "
              f"yaw-turns {r['yaw_turns']:.1f}  ({time.time() - t0:.0f}s wall)")

    if results:
        def frac(k):
            return float(np.mean([bool(r.get(k)) for r in results]))
        summary = {"episodes": len(results),
                   "success (no crash)": frac("success"), "reached": frac("reached"),
                   "crashed": frac("crashed"),
                   "avg_speed": float(np.mean([r["avg_speed"] for r in results])),
                   "heading": args.heading,
                   "yaw_turns_per_episode": float(np.mean([r["yaw_turns"] for r in results])),
                   "checkpoint": args.resume}
        print("\n" + json.dumps(summary, indent=2))
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "summary.json"), "w") as f:
            json.dump({"args": vars(args), "summary": summary, "episodes": results}, f, indent=2)

    carb.log_warn("Isaac policy eval finished.")
    timeline.stop()
    simulation_app.close()


if __name__ == "__main__":
    main()
