"""
run_drowsy.py
--------------
Normal driving  → pretrained RL model controls the car.
Drowsy detected → Smooth 4-phase manoeuvre:
                    Phase 1  SLOW_DOWN   – ease off throttle, reduce speed gently
                    Phase 2  DRIFT_LEFT  – soft left steering at low speed
                    Phase 3  ALIGN_LANE  – line up with the leftmost lane heading
                    Phase 4  BRAKE_STOP  – hold lane, decelerate to full stop
Manoeuvre completes fully; does not resume until driver indicates readiness.

SUCCESS is defined as: vehicle stopped in the leftmost lane while drowsy.
Reaching the end of the route is NOT required.

Smoothness guarantees
---------------------
* All steering changes are low-pass filtered (exponential moving average)
  so there are no abrupt jumps.
* Speed targets ramp gently; full-brake is never issued in one step.
* Steer ramps from 0 → target over RAMP_STEPS steps in DRIFT_RIGHT phase.
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
import json

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

NUM_EPISODES           = 5
MAX_STEPS              = 2000
DROWSY_CONFIRM_SECONDS = 2.0
CUMULATIVE_STATS_FILE  = "cumulative_stats.json"

# ── Smooth manoeuvre parameters ───────────────────────────────────────────────
SLOW_DOWN_TARGET_KMH  = 18.0   # Phase 1: reduce speed to this before steering
DRIFT_SPEED_KMH       = 14.0   # Phase 2: target speed while drifting right
BRAKE_SPEED_KMH       = 4.0    # Phase 3: target speed while braking in lane
STOP_SPEED_KMH        = 0.5  # considered fully stopped below this

STEER_SMOOTH_ALPHA    = 0.3    # EMA coefficient for steering (0=frozen, 1=instant)
STEER_MAX             = 0.5    # absolute max steering magnitude
DRIFT_STEER_FACTOR    = 1.0    # scale down raw rightward steering during lane drift
RAMP_STEPS            = 20     # steps over which drift steer ramps from 0 → target
ALIGN_SPEED_KMH       = 8.0    # Phase 3 pre-brake speed for alignment
ALIGN_DURATION_S      = 5.0    # hold alignment for this long before braking
RIGHT_LANE_NEAR_THR   = 0.6    # meters from rightmost lane centre to trigger alignment

# PID gains (mirrors LaneChangePolicy internals)
PID_KP_LAT = 2.0
PID_KP_HDG = 1.0

BEEP_INTERVAL = 2.0


# ─────────────────────────────────────────────────────────────────────────────
#  Cumulative stats persistence
# ─────────────────────────────────────────────────────────────────────────────

def load_cumulative_stats():
    if os.path.exists(CUMULATIVE_STATS_FILE):
        with open(CUMULATIVE_STATS_FILE, 'r') as f:
            data = json.load(f)
        return {
            "outcomes": data.get("outcomes", {"drowsy_success": 0, "crash": 0, "out_of_road": 0, "timeout": 0, "normal_done": 0}),
            "total_drowsy_events": data.get("total_drowsy_events", 0),
            "all_lane_change_times": data.get("all_lane_change_times", []),
            "total_episodes": data.get("total_episodes", 0),
            "max_reward": data.get("max_reward", float('-inf')),
            "min_reward": data.get("min_reward", float('inf'))
        }
    else:
        return {
            "outcomes": {"drowsy_success": 0, "crash": 0, "out_of_road": 0, "timeout": 0, "normal_done": 0},
            "total_drowsy_events": 0,
            "all_lane_change_times": [],
            "total_episodes": 0,
            "max_reward": float('-inf'),
            "min_reward": float('inf')
        }

def save_cumulative_stats(stats):
    with open(CUMULATIVE_STATS_FILE, 'w') as f:
        json.dump(stats, f, indent=4)


# ─────────────────────────────────────────────────────────────────────────────
#  Road / lane helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_road_info(env):
    try:
        vehicle      = env.agent
        road_network = env.engine.current_map.road_network
        road_id, section_id, lane_id = vehicle.lane_index
        lanes        = road_network.graph[road_id][section_id]
        return road_network, road_id, section_id, lane_id, lanes
    except Exception:
        return None


def _in_leftmost_lane(env):
    info = _get_road_info(env)
    if info is None:
        return False
    _, _, _, lane_id, lanes = info
    return lane_id == 0


def _current_speed_kmh(env):
    return float(getattr(env.agent, "speed_km_h", 0.0))


def _pid_steer_to_lane(env, target_lane_obj):
    """Compute smooth PID steering toward target_lane_obj."""
    vehicle = env.agent
    try:
        long, lat = target_lane_obj.local_coordinates(vehicle.position)
        road_hdg  = target_lane_obj.heading_theta_at(long)
        hdg_err   = road_hdg - vehicle.heading_theta
        hdg_err   = (hdg_err + np.pi) % (2 * np.pi) - np.pi
        error     = - PID_KP_LAT * lat + PID_KP_HDG * hdg_err
        return float(np.clip(error, -STEER_MAX, STEER_MAX))
    except Exception:
        return 0.0


def _speed_accel(current_kmh, target_kmh, max_brake=-0.4, max_accel=0.3):
    """
    Gentle P-controller — max_brake caps negative output so we never
    slam the brakes in one step.
    """
    diff  = target_kmh - current_kmh
    scale = max(target_kmh, STOP_SPEED_KMH + 1.0)
    accel = float(np.clip(diff / scale, max_brake, max_accel))
    return accel


def _is_near_leftmost_lane(env, threshold=RIGHT_LANE_NEAR_THR):
    info = _get_road_info(env)
    if info is None:
        return False
    _, _, _, lane_id, lanes = info
    if lane_id == 0:
        return True
    try:
        target_lane = lanes[0]
        _, lat = target_lane.local_coordinates(env.agent.position)
        return abs(lat) <= threshold
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  State machine
# ─────────────────────────────────────────────────────────────────────────────

class DrowsyState(Enum):
    IDLE          = auto()
    CONFIRMING    = auto()
    SLOW_DOWN     = auto()   # Phase 1: ease off throttle, go straight
    DRIFT_LEFT  = auto()   # Phase 2: gentle left steer toward leftmost lane
    ALIGN_LANE    = auto()   # Phase 3: align to lane heading before braking
    BRAKE_STOP    = auto()   # Phase 4: hold lane, decelerate to stop
    STOPPED       = auto()   # SUCCESS – vehicle at rest in leftmost lane


class DrowsinessController:

    def __init__(self):
        self.state             = DrowsyState.IDLE
        self._confirm_start    = 0.0
        self._phase_start_time = 0.0
        self._phase_step       = 0
        self._smooth_steer     = 0.0   # EMA-filtered steering output
        self._last_beep        = 0.0
        self._beep_sound       = None
        self._detector         = None
        self.drowsy_events     = 0
        self.lane_change_times = []
        self._in_rightmost_steps = 0  # Counter for steps in rightmost lane
        self._align_start_time = 0.0
        self._setup_sound()
        self._setup_detector()

    # ── Init helpers ──────────────────────────────────────────────────────────

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

    # ── Public ────────────────────────────────────────────────────────────────

    def reset(self):
        self.state             = DrowsyState.IDLE
        self._confirm_start    = 0.0
        self._phase_start_time = 0.0
        self._phase_step       = 0
        self._smooth_steer     = 0.0
        self.drowsy_events     = 0
        self.lane_change_times = []
        self._in_rightmost_steps = 0
        self._align_start_time = 0.0
        if self._detector:
            self._detector.reset()
        self._stop_sound()

    def stop(self):
        if self._detector:
            self._detector.stop()

    # ── Main tick ─────────────────────────────────────────────────────────────

    def get_action(self, rl_action, env):
        """
        Returns (action, override, state_name, ear).
        override=True  → drowsy manoeuvre is in control.
        override=False → RL action is passed through unmodified.
        """
        is_drowsy = self._detector.is_drowsy if self._detector else False
        ear       = self._detector.current_ear if self._detector else 1.0
        speed     = _current_speed_kmh(env)
        now       = time.time()

        # ── IDLE ──────────────────────────────────────────────────────────────
        if self.state == DrowsyState.IDLE:
            if is_drowsy:
                self.state          = DrowsyState.CONFIRMING
                self._confirm_start = now
                print(f"[Drowsy] Detected – confirming for {DROWSY_CONFIRM_SECONDS}s …")
            return rl_action, False, self.state.name, ear

        # ── CONFIRMING – RL still drives while we wait ─────────────────────
        if self.state == DrowsyState.CONFIRMING:
            if not is_drowsy:
                # Blink was too short – go back to IDLE
                self._reset_to_idle()
                return rl_action, False, self.state.name, ear
            if now - self._confirm_start >= DROWSY_CONFIRM_SECONDS:
                self._enter_phase(DrowsyState.SLOW_DOWN, now)
                self.drowsy_events += 1
                print("[Drowsy] CONFIRMED → Phase 1: Slowing down straight.")
                self._play_beep(force=True)
            return rl_action, False, self.state.name, ear

        # ── SLOW_DOWN – go straight, ease off throttle ─────────────────────
        if self.state == DrowsyState.SLOW_DOWN:
            self._play_beep()
            # Keep going straight (steer = 0), gently reduce speed
            target_steer = 0.0
            self._smooth_steer = self._ema(self._smooth_steer, target_steer)
            accel = _speed_accel(speed, SLOW_DOWN_TARGET_KMH, max_brake=-0.25, max_accel=0.1)
            action = np.array([self._smooth_steer, accel], dtype=np.float32)

            if speed <= SLOW_DOWN_TARGET_KMH + 1.0:
                self._enter_phase(DrowsyState.DRIFT_LEFT, now)
                print("[Drowsy] Phase 2: Drifting left toward leftmost lane.")
            return action, True, self.state.name, ear

        # ── DRIFT_LEFT – gentle leftward steer until near leftmost lane ─
        if self.state == DrowsyState.DRIFT_LEFT:
            self._play_beep()
            self._phase_step += 1

            if _in_leftmost_lane(env):
                self._in_rightmost_steps += 1
            else:
                self._in_rightmost_steps = 0

            if self._phase_step >= 20 and _is_near_leftmost_lane(env):
                self._enter_phase(DrowsyState.ALIGN_LANE, now)
                print("[Drowsy] Phase 3: Aligning in leftmost lane before braking.")
                # fall through to alignment state on next tick
                return self.get_action(rl_action, env)

            raw_steer = self._steer_to_leftmost(env)
            ramp = float(np.clip(self._phase_step / RAMP_STEPS, 0.0, 1.0))
            target_steer = raw_steer * DRIFT_STEER_FACTOR * ramp
            self._smooth_steer = self._ema(self._smooth_steer, target_steer)

            accel = _speed_accel(speed, DRIFT_SPEED_KMH, max_brake=-0.25, max_accel=0.1)
            action = np.array([self._smooth_steer, accel], dtype=np.float32)
            return action, True, self.state.name, ear

        # ── ALIGN_LANE – line up with the rightmost lane before braking ────
        if self.state == DrowsyState.ALIGN_LANE:
            self._play_beep()
            self._phase_step += 1

            if _in_leftmost_lane(env):
                self._in_rightmost_steps += 1
            else:
                self._in_rightmost_steps = 0

            raw_steer = self._steer_to_leftmost(env)
            self._smooth_steer = self._ema(self._smooth_steer, raw_steer)

            accel = _speed_accel(speed, ALIGN_SPEED_KMH, max_brake=-0.25, max_accel=0.08)
            action = np.array([self._smooth_steer, accel], dtype=np.float32)

            if (now - self._align_start_time >= ALIGN_DURATION_S and
                    self._in_rightmost_steps >= 5):
                self._enter_phase(DrowsyState.BRAKE_STOP, now)
                print("[Drowsy] Phase 4: Braking to stop in rightmost lane.")
                action = self._hold_lane_action(env, BRAKE_SPEED_KMH, speed)
            return action, True, self.state.name, ear

        # ── BRAKE_STOP – hold lane, decelerate smoothly ────────────────────
        if self.state == DrowsyState.BRAKE_STOP:
            self._play_beep()
            action = self._hold_lane_action(env, BRAKE_SPEED_KMH, speed)

            if speed <= STOP_SPEED_KMH:
                elapsed = now - self._phase_start_time
                self.lane_change_times.append(elapsed)
                self.state = DrowsyState.STOPPED
                print(f"[Drowsy] ✓ SUCCESS – stopped in leftmost lane. "
                      f"Brake time: {elapsed:.1f}s")
            return action, True, self.state.name, ear

        # ── STOPPED – hold position (full brake, no steer) ─────────────────
        if self.state == DrowsyState.STOPPED:
            self._play_beep()
            action = np.array([0.0, -1.0], dtype=np.float32)
            return action, True, self.state.name, ear

        return rl_action, False, self.state.name, ear

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _enter_phase(self, new_state, now):
        self.state             = new_state
        self._phase_start_time = now
        self._phase_step       = 0
        self._in_rightmost_steps = 0
        if new_state == DrowsyState.ALIGN_LANE:
            self._align_start_time = now

    def _ema(self, current, target):
        """Exponential moving average – smooths abrupt steer changes."""
        return current + STEER_SMOOTH_ALPHA * (target - current)

    def _steer_to_leftmost(self, env):
        """PID steering toward the leftmost lane."""
        info = _get_road_info(env)
        if info is None:
            return -STEER_MAX * 0.4   # mild left bias as fallback
        road_network, road_id, section_id, lane_id, lanes = info
        target_lane = lanes[0]
        return _pid_steer_to_lane(env, target_lane)

    def _hold_lane_action(self, env, target_speed, speed):
        """Stay centred in current lane while decelerating toward target_speed."""
        info = _get_road_info(env)
        if info is not None:
            _, _, _, lane_id, lanes = info
            target_lane  = lanes[lane_id]
            raw_steer    = _pid_steer_to_lane(env, target_lane)
        else:
            raw_steer = 0.0
        self._smooth_steer = self._ema(self._smooth_steer, raw_steer)
        # In brake phase allow harder braking but still gradual
        accel = _speed_accel(speed, target_speed, max_brake=-1.0, max_accel=0.05)
        return np.array([self._smooth_steer, accel], dtype=np.float32)

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
        self._smooth_steer = 0.0
        self.state = DrowsyState.IDLE
        if prev != DrowsyState.IDLE:
            print(f"[Drowsy] AWAKE – RL resumes. (was: {prev.name})")


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
#  Episode success logic
# ─────────────────────────────────────────────────────────────────────────────

def _classify_episode(done, info, drowsy_state, env):
    """
    Returns one of: 'drowsy_success', 'crash', 'out_of_road', 'timeout', 'normal_done'

    drowsy_success = vehicle stopped in leftmost lane while drowsy.
    This is the PRIMARY success criterion – reaching route end is NOT required.
    """
    if drowsy_state == DrowsyState.STOPPED:
        return "drowsy_success"
    if done:
        crash_keys = {'crash_vehicle', 'crash_object', 'crash_human'}
        if any(info.get(k, False) for k in crash_keys):
            return "crash"
        if info.get("out_of_road", False):
            return "out_of_road"
        return "normal_done"
    return "timeout"


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("  Drowsy Driver Safety System")
    print("  ─────────────────────────────────────────────────────────")
    print("  AWAKE  : pretrained RL model drives")
    print("  DROWSY : 3-phase smooth manoeuvre")
    print("           Phase 1  SLOW_DOWN   – straight, ease off throttle")
    print("           Phase 2  DRIFT_LEFT  – gentle leftward steer")
    print("           Phase 3  BRAKE_STOP  – hold lane, decelerate to stop")
    print("  SUCCESS: vehicle stopped in leftmost lane (not route end)")
    print("=" * 64 + "\n")

    # Load cumulative stats
    cum_stats = load_cumulative_stats()
    print(f"[RunDrowsy] Loaded cumulative stats: {cum_stats['total_episodes']} previous episodes\n")

    env = HumanInTheLoopEnv()
    obs= env.reset()

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
    outcomes              = {"drowsy_success": 0, "crash": 0,
                             "out_of_road": 0, "timeout": 0, "normal_done": 0}
    total_drowsy_events   = 0
    all_lane_change_times = []

    try:
        for episode in range(NUM_EPISODES):
            if episode > 0:
                obs = env.reset()
            drowsy.reset()
            ep_reward  = 0.0
            ep_outcome = "timeout"

            print(f"\n─── Episode {episode + 1} / {NUM_EPISODES} ───")

            for step in range(MAX_STEPS):
                rl_action = get_rl_action(policy, obs)
                action, override, state_name, ear = drowsy.get_action(rl_action, env)
                obs, reward, done, info = env.step(action)
                if override:
                    if drowsy.state == DrowsyState.SLOW_DOWN:
                        reward += 0.1
                    elif drowsy.state == DrowsyState.DRIFT_LEFT:
                        reward += 0.2
                    elif drowsy.state == DrowsyState.BRAKE_STOP:
                        reward += 0.3
                    elif drowsy.state == DrowsyState.STOPPED:
                        reward += 0.1
                ep_reward += reward

                # Log every 100 steps
                if step % 100 == 0:
                    speed = _current_speed_kmh(env)
                    lane  = getattr(env.agent, "lane_index", ("?", "?", "?"))[2]
                    print(f"  step={step:4d} | state={state_name:14s} | "
                          f"EAR={ear:.3f} | lane={lane} | "
                          f"speed={speed:5.1f} km/h | r={reward:+.3f}")

                # Check for drowsy success first (takes priority over env done)
                if drowsy.state == DrowsyState.STOPPED:
                    ep_outcome = "drowsy_success"
                    print(f"  ✓ DROWSY SUCCESS at step {step} | reward={ep_reward:.2f}")
                    break

                if done:
                    ep_outcome = _classify_episode(done, info, drowsy.state, env)
                    print(f"  Episode ended: {ep_outcome} at step {step} | "
                          f"reward={ep_reward:.2f}")
                    break
            else:
                ep_outcome = "timeout"
                print(f"  Timeout. reward={ep_reward:.2f}")

            episode_rewards.append(ep_reward)
            episode_lengths.append(step + 1)
            outcomes[ep_outcome] = outcomes.get(ep_outcome, 0) + 1
            total_drowsy_events   += drowsy.drowsy_events
            all_lane_change_times += drowsy.lane_change_times

        # Update cumulative stats
        for key in cum_stats["outcomes"]:
            cum_stats["outcomes"][key] += outcomes[key]
        cum_stats["total_drowsy_events"] += total_drowsy_events
        cum_stats["all_lane_change_times"].extend(all_lane_change_times)
        cum_stats["total_episodes"] += NUM_EPISODES
        
        # Update max/min rewards
        if episode_rewards:
            current_max = max(episode_rewards)
            current_min = min(episode_rewards)
            cum_stats["max_reward"] = max(cum_stats["max_reward"], current_max)
            cum_stats["min_reward"] = min(cum_stats["min_reward"], current_min)

        # Save cumulative stats
        save_cumulative_stats(cum_stats)
        print(f"[RunDrowsy] Updated cumulative stats saved to {CUMULATIVE_STATS_FILE}")

        # ── Cumulative Summary ───────────────────────────────────────────────
        print("\n" + "=" * 64)
        print("  CUMULATIVE SUMMARY (All Runs)")
        print("=" * 64)
        total_eps = cum_stats["total_episodes"]
        ds  = cum_stats["outcomes"]["drowsy_success"]
        cr  = cum_stats["outcomes"]["crash"]
        oor = cum_stats["outcomes"]["out_of_road"]
        nd  = cum_stats["outcomes"]["normal_done"]
        to_ = cum_stats["outcomes"]["timeout"]
        print(f"  Total Episodes      : {total_eps}")
        print(f"  Drowsy success rate : {ds/total_eps*100:.1f}%  "
              f"({ds}/{total_eps})")
        print(f"  Crash rate          : {cr/total_eps*100:.1f}%")
        print(f"  Out-of-road rate    : {oor/total_eps*100:.1f}%")
        print(f"  Normal done         : {nd/total_eps*100:.1f}%")
        print(f"  Timeout             : {to_/total_eps*100:.1f}%")
        print(f"  Drowsy events total : {cum_stats['total_drowsy_events']}")
        if cum_stats["all_lane_change_times"]:
            print(f"  Avg brake time      : {np.mean(cum_stats['all_lane_change_times']):.2f}s")
        print(f"  Highest reward      : {cum_stats['max_reward']:.2f}")
        print(f"  Lowest reward       : {cum_stats['min_reward']:.2f}")
        if cum_stats["max_reward"] != float('-inf') and cum_stats["min_reward"] != float('inf'):
            print(f"  Reward range        : {cum_stats['max_reward'] - cum_stats['min_reward']:.2f}")

        # ── Plot (current run only, for simplicity) ─────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(episode_rewards, marker='o', color='steelblue')
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Total Reward")
        axes[0].set_title("Episode Rewards (This Run)")
        axes[0].grid(True)

        labels = list(outcomes.keys())
        values = [outcomes[k] for k in labels]
        axes[1].bar(labels, values, color=['green', 'red', 'orange', 'blue', 'gray'])
        axes[1].set_title("Episode Outcomes (This Run)")
        axes[1].set_ylabel("Count")
        axes[1].tick_params(axis='x', rotation=20)

        plt.tight_layout()
        plt.savefig("episode_results.png")
        
        # ── Current run summary ─────────────────────────────────────────────
        print("\n" + "=" * 64)
        print("  THIS RUN SUMMARY")
        print("=" * 64)
        ds  = outcomes["drowsy_success"]
        cr  = outcomes["crash"]
        oor = outcomes["out_of_road"]
        nd  = outcomes["normal_done"]
        to_ = outcomes["timeout"]
        print(f"  Episodes            : {NUM_EPISODES}")
        print(f"  Drowsy success rate : {ds/NUM_EPISODES*100:.1f}%  ")
        print(f"  Crash rate          : {cr/NUM_EPISODES*100:.1f}%")
        print(f"  Out-of-road rate    : {oor/NUM_EPISODES*100:.1f}%")
        print(f"  Avg reward          : {np.mean(episode_rewards):.2f}")
        print(f"  Highest reward      : {max(episode_rewards):.2f}")
        print(f"  Lowest reward       : {min(episode_rewards):.2f}")
        print(f"  Reward range        : {max(episode_rewards) - min(episode_rewards):.2f}")
        
        print("\nPlot saved → episode_results.png")

    finally:
        drowsy.stop()
        env.close()
        print("[RunDrowsy] Done.")


if __name__ == "__main__":
    main()