"""
run_drowsy.py
--------------
Normal driving  → pretrained RL model controls the car.
Drowsy detected → A self-contained PID (replicating LaneChangePolicy's
                  internal logic) steers the car to the rightmost lane,
                  then brakes to a stop.
Eyes reopen     → RL immediately resumes.

Why not use LaneChangePolicy directly?
  LaneChangePolicy asserts discrete_action=True at construction time.
  HumanInTheLoopEnv uses continuous actions, so the assert always fires.
  Solution: replicate the same PID logic here without the assert.

The PID works like LaneChangePolicy does internally:
  - Identify the target lane (rightmost = highest lane index)
  - Compute lateral offset of vehicle from target lane centre
  - Compute heading error relative to road direction
  - PID on (lateral_error + heading_error) → steering
  - Speed control → throttle/brake
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import time
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict
from enum import Enum, auto

# ── Project paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in [
    PROJECT_ROOT,
    os.path.join(PROJECT_ROOT, "env_gym"),
    os.path.join(PROJECT_ROOT, "utils"),
    os.path.join(PROJECT_ROOT, "networks"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)

from gym_metadrivepvp_data import HumanInTheLoopEnv
from mlp import StochaPolicy

try:
    import pygame
    _PYGAME = True
except ImportError:
    _PYGAME = False

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

PRETRAINED_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "results",
    "DSAC_V2_PVP_RL_gym_metadrivepvp",
    "260128-222729", "apprfunc", "apprfunc_73000.pkl"
)
LANDMARK_PATH = os.path.join(PROJECT_ROOT, "env_gym",
                             "shape_predictor_68_face_landmarks.dat")

NUM_EPISODES           = 6
MAX_STEPS              = 2000
DROWSY_CONFIRM_SECONDS = 2.0

# PID gains (same order of magnitude as MetaDrive's LaneChangePolicy)
PID_KP_LAT  = 0.5   # proportional gain on lateral error
PID_KP_HDG  = 0.3   # proportional gain on heading error
STEER_MAX   = 0.5   # clip steering to ±this

LANE_CHANGE_SPEED_KMH = 12.0   # target speed while changing lanes
BRAKE_SPEED_KMH       = 3.0    # target speed while braking in rightmost lane
STOP_SPEED_KMH        = 0.5    # considered stopped below this

BEEP_INTERVAL = 2.0


# ─────────────────────────────────────────────────────────────────────────────
#  LaneChange-style PID (mirrors LaneChangePolicy internals)
# ─────────────────────────────────────────────────────────────────────────────

def _get_road_info(env):
    """
    Returns (road_network, road_id, section_id, lane_id, lanes)
    or None on failure.
    """
    try:
        vehicle      = env.agent
        road_network = env.engine.current_map.road_network
        road_id, section_id, lane_id = vehicle.lane_index
        lanes        = road_network.graph[road_id][section_id]
        return road_network, road_id, section_id, lane_id, lanes
    except Exception:
        return None


def _in_rightmost_lane(env):
    info = _get_road_info(env)
    if info is None:
        return False
    _, _, _, lane_id, lanes = info
    return lane_id == len(lanes) - 1


def _pid_steer_to_lane(env, target_lane_obj):
    """
    Compute steering to follow target_lane_obj.
    Mirrors what LaneChangePolicy's PID does:
      error = lateral_offset + k * heading_error
      steer = kp * error
    """
    vehicle = env.agent
    try:
        long, lat = target_lane_obj.local_coordinates(vehicle.position)
        road_hdg  = target_lane_obj.heading_theta_at(long)
        hdg_err   = road_hdg - vehicle.heading_theta
        # wrap to [-π, π]
        hdg_err   = (hdg_err + np.pi) % (2 * np.pi) - np.pi
        # combined error: lateral offset + scaled heading correction
        error = PID_KP_LAT * lat + PID_KP_HDG * hdg_err
        steer = float(np.clip(error, -STEER_MAX, STEER_MAX))
    except Exception:
        steer = 0.0
    return steer


def _speed_accel(env, target_kmh):
    """Simple P-controller for speed."""
    speed = getattr(env.agent, "speed_km_h", 0.0)
    diff  = target_kmh - speed
    # map diff to [-1, 1]: positive = accelerate, negative = brake
    accel = float(np.clip(diff / max(target_kmh, 1.0), -1.0, 1.0))
    return accel


def _lane_change_action(env, ramp=1.0):
    """
    Action to move toward the rightmost lane.
    Ramp smoothly increases steering from 0 → full over time.
    """
    info = _get_road_info(env)
    if info is None:
        return np.array([0.0, 0.1], dtype=np.float32)

    _, _, _, _, lanes = info
    target_lane = lanes[len(lanes) - 1]   # rightmost

    steer = _pid_steer_to_lane(env, target_lane) * ramp
    accel = _speed_accel(env, LANE_CHANGE_SPEED_KMH)
    return np.array([steer, accel], dtype=np.float32)


def _keep_lane_brake_action(env):
    """
    Keep the current (rightmost) lane and slow to a stop.
    """
    info = _get_road_info(env)
    if info is None:
        return np.array([0.0, -0.3], dtype=np.float32)

    _, _, _, lane_id, lanes = info
    current_lane = lanes[lane_id]

    steer = _pid_steer_to_lane(env, current_lane)
    accel = _speed_accel(env, BRAKE_SPEED_KMH)
    return np.array([steer, accel], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  State machine
# ─────────────────────────────────────────────────────────────────────────────

class DrowsyState(Enum):
    IDLE          = auto()
    CONFIRMING    = auto()
    LANE_CHANGING = auto()
    LANE_KEEPING  = auto()   # in rightmost lane, braking to stop
    STOPPED       = auto()


class DrowsinessController:

    def __init__(self):
        self.state             = DrowsyState.IDLE
        self._confirm_start    = 0.0
        self._lc_start         = 0.0
        self._last_beep        = 0.0
        self._beep_sound       = None
        self._detector         = None
        self.drowsy_events     = 0
        self.lane_change_times = []
        self._setup_sound()
        self._setup_detector()

    def _setup_detector(self):
        try:
            from drowsiness_detector import DrowsinessDetector
            self._detector = DrowsinessDetector(
                ear_threshold=0.25,
                consec_frames=20,
                camera_index=0,
                landmark_path=LANDMARK_PATH,
                show_window=True,
            )
            self._detector.start()
            print("[Drowsy] Detector started.")
        except Exception as e:
            print(f"[Drowsy] Detector unavailable: {e}")
            self._detector = None

    def _setup_sound(self):
        if not _PYGAME:
            return
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
            n    = int(44100 * 0.7)
            t    = np.linspace(0, 0.7, n, endpoint=False)
            half = n // 2
            wave = np.zeros(n, dtype=np.int16)
            wave[:half] = (np.sin(2 * np.pi * 880       * t[:half]) * 28000).astype(np.int16)
            wave[half:] = (np.sin(2 * np.pi * 880 * 1.3 * t[half:]) * 28000).astype(np.int16)
            self._beep_sound = pygame.sndarray.make_sound(np.column_stack([wave, wave]))
        except Exception as e:
            print(f"[Drowsy] Sound init failed: {e}")

    def reset(self):
        self.state             = DrowsyState.IDLE
        self._confirm_start    = 0.0
        self._lc_start         = 0.0
        self.drowsy_events     = 0
        self.lane_change_times = []
        if self._detector:
            self._detector.reset()
        self._stop_sound()

    def stop(self):
        if self._detector:
            self._detector.stop()

    # ── Main ──────────────────────────────────────────────────────────────────

    def get_action(self, rl_action, env):
        """Returns (action, override, state_name, ear)."""
        is_drowsy = self._detector.is_drowsy if self._detector else False
        ear       = self._detector.current_ear if self._detector else 1.0
        now       = time.time()

        # Eyes open → immediately back to RL
        if self.state != DrowsyState.IDLE and not is_drowsy:
            self._reset_to_idle()
            return rl_action, False, self.state.name, ear

        # IDLE
        if self.state == DrowsyState.IDLE:
            if is_drowsy:
                self.state          = DrowsyState.CONFIRMING
                self._confirm_start = now
                print(f"[Drowsy] Detected — confirming ({DROWSY_CONFIRM_SECONDS}s) …")
            return rl_action, False, self.state.name, ear

        # CONFIRMING — RL still drives
        if self.state == DrowsyState.CONFIRMING:
            if now - self._confirm_start >= DROWSY_CONFIRM_SECONDS:
                self.state         = DrowsyState.LANE_CHANGING
                self._lc_start     = now
                self.drowsy_events += 1
                print("[Drowsy] CONFIRMED — PID steering to rightmost lane.")
                self._play_beep(force=True)
            return rl_action, False, self.state.name, ear

        # LANE_CHANGING — PID steers toward rightmost lane
        if self.state == DrowsyState.LANE_CHANGING:
            self._play_beep()
            elapsed = now - self._lc_start
            ramp    = float(np.clip(elapsed / 2.0, 0.0, 1.0))  # smooth ramp over 2s

            if _in_rightmost_lane(env):
                self.state = DrowsyState.LANE_KEEPING
                print("[Drowsy] Rightmost lane reached — braking to stop.")
                action = _keep_lane_brake_action(env)
            else:
                action = _lane_change_action(env, ramp)

            return action, True, self.state.name, ear

        # LANE_KEEPING — hold lane, decelerate
        if self.state == DrowsyState.LANE_KEEPING:
            self._play_beep()
            speed = getattr(env.agent, "speed_km_h", 0.0)
            if speed <= STOP_SPEED_KMH:
                elapsed = now - self._lc_start
                self.lane_change_times.append(elapsed)
                self.state = DrowsyState.STOPPED
                print(f"[Drowsy] STOPPED. Override time: {elapsed:.1f}s.")
            action = _keep_lane_brake_action(env)
            return action, True, self.state.name, ear

        # STOPPED — hold position
        if self.state == DrowsyState.STOPPED:
            self._play_beep()
            # Full brake, zero steer
            action = np.array([0.0, -1.0], dtype=np.float32)
            return action, True, self.state.name, ear

        return rl_action, False, self.state.name, ear

    def _play_beep(self, force=False):
        if not self._beep_sound:
            return
        now = time.time()
        if force or now - self._last_beep >= BEEP_INTERVAL:
            try:
                self._beep_sound.play()
            except Exception:
                pass
            self._last_beep = now

    def _stop_sound(self):
        if self._beep_sound:
            try:
                self._beep_sound.stop()
            except Exception:
                pass

    def _reset_to_idle(self):
        prev = self.state
        self._stop_sound()
        self.state = DrowsyState.IDLE
        if prev != DrowsyState.IDLE:
            print(f"[Drowsy] AWAKE — RL resumes. (was: {prev.name})")


# ─────────────────────────────────────────────────────────────────────────────
#  Pretrained RL policy
# ─────────────────────────────────────────────────────────────────────────────

def load_policy(model_path, obs_dim, act_dim, act_high_lim, act_low_lim):
    print(f"[RunDrowsy] Loading: {model_path}")
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["state_dict"]

    prefix    = "policy_rl."
    policy_sd = OrderedDict(
        {k[len(prefix):]: v
         for k, v in state_dict.items()
         if k.startswith(prefix) and not k.startswith("policy_rl_target.")}
    )
    print(f"[RunDrowsy] policy_rl keys (sample): {list(policy_sd.keys())[:4]}")

    policy = StochaPolicy(**{
        "obs_dim"                : obs_dim,
        "act_dim"                : act_dim,
        "hidden_sizes"           : [256, 256, 256],
        "hidden_activation"      : "gelu",
        "output_activation"      : "linear",
        "std_type"               : "mlp_shared",
        "min_log_std"            : -5,
        "max_log_std"            : 2,
        "act_high_lim"           : act_high_lim,
        "act_low_lim"            : act_low_lim,
        "action_distribution_cls": "TanhGaussDistribution",
    })
    policy.load_state_dict(policy_sd, strict=True)
    policy.to(device)
    policy.eval()
    print("[RunDrowsy] Policy loaded.\n")
    return policy, checkpoint


def get_rl_action(policy, obs):
    with torch.no_grad():
        device  = next(policy.parameters()).device
        obs_t   = torch.FloatTensor(obs).unsqueeze(0).to(device)
        output  = policy(obs_t)
        act_dim = output.shape[-1] // 2
        action  = output[:, :act_dim].squeeze(0).cpu().numpy()
    return np.clip(action, -1.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Drowsy Driver — RL + LaneChangePolicy-style PID")
    print("  AWAKE  : pretrained RL model drives")
    print("  DROWSY : PID (mirrors LaneChangePolicy internals)")
    print("           steers to rightmost lane then stops")
    print("  AWAKE  : RL immediately resumes")
    print("=" * 60 + "\n")

    env = HumanInTheLoopEnv()
    env.use_drowsy_reward = True 
    obs = env.reset()

    obs_dim      = obs.shape[0]
    act_dim      = env.action_space.shape[0]
    act_high_lim = env.action_space.high.astype(np.float32)
    act_low_lim  = env.action_space.low.astype(np.float32)
    print(f"[RunDrowsy] obs_dim={obs_dim}  act_dim={act_dim}\n")

    policy, checkpoint = load_policy(
        PRETRAINED_MODEL_PATH, obs_dim, act_dim, act_high_lim, act_low_lim
    )
    env.activate_rl = checkpoint.get("activate_rl", True)

    drowsy = DrowsinessController()

    episode_rewards       = []
    episode_lengths       = []
    successes = crashes = out_of_roads = 0
    total_drowsy_events   = 0
    all_lane_change_times = []

    try:
        for episode in range(NUM_EPISODES):
            if episode > 0:
                obs = env.reset()
            drowsy.reset()
            ep_reward = 0.0
            env.external_reward = 0
            print(f"\n─── Episode {episode + 1} / {NUM_EPISODES} ───")

            for step in range(MAX_STEPS):
                rl_action = get_rl_action(policy, obs)
                action, override, state_name, ear = drowsy.get_action(rl_action, env)
                obs, reward, done, info = env.step(action)
                env.external_reward += reward
                ep_reward += reward

                if step % 100 == 0:
                    speed = getattr(env.agent, "speed_km_h", 0.0)
                    lane  = getattr(env.agent, "lane_index", ("?", "?", "?"))[2]
                    print(f"  step={step:4d} | state={state_name:14s} | "
                          f"EAR={ear:.3f} | lane={lane} | "
                          f"speed={speed:.1f} km/h | reward={reward:+.3f}")
                if done:
                    print(f"  Done at step {step}. Reward={ep_reward:.2f}")
                    break
            else:
                print(f"  Max steps. Reward={ep_reward:.2f}")

            episode_rewards.append(ep_reward)
            episode_lengths.append(step + 1)
            crash_keys = {'crash_vehicle', 'crash_object', 'crash_human'}
            is_crash   = done and any(info.get(k, False) for k in crash_keys)
            is_oor     = done and info.get('out_of_road', False)
            if is_crash:              crashes += 1
            if is_oor:                out_of_roads += 1
            if not is_crash and not is_oor: successes += 1
            total_drowsy_events   += drowsy.drowsy_events
            all_lane_change_times += drowsy.lane_change_times

        print("\n" + "=" * 60)
        print("  SUMMARY")
        print("=" * 60)
        print(f"  Episodes         : {NUM_EPISODES}")
        print(f"  Success rate     : {successes/NUM_EPISODES*100:.1f}%")
        print(f"  Crash rate       : {crashes/NUM_EPISODES*100:.1f}%")
        print(f"  Out-of-road      : {out_of_roads/NUM_EPISODES*100:.1f}%")
        print(f"  Avg reward       : {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"  Drowsy events    : {total_drowsy_events}")
        if all_lane_change_times:
            print(f"  Avg override time: {np.mean(all_lane_change_times):.2f}s")

        plt.figure(figsize=(10, 4))
        plt.plot(episode_rewards, marker='o')
        plt.xlabel("Episode"); plt.ylabel("Reward")
        plt.title("Episode Rewards"); plt.grid(True); plt.tight_layout()
        plt.savefig("episode_rewards.png")
        print("\nPlot → episode_rewards.png")

    finally:
        drowsy.stop()
        env.close()
        print("[RunDrowsy] Done.")


if __name__ == "__main__":
    main()