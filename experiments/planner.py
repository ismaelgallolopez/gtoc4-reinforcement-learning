# WP12/WP13: greedy tour planner + per-leg piloting.
#
# Planning: at the spacecraft's current state/epoch, estimate the delta-v to every unvisited pool
# asteroid over the next leg window (generalises curriculum._delta_v_needed, which only handles
# departure from Earth, to an arbitrary current heliocentric state via its osculating semi-major
# axis) and greedily picks the cheapest one that clears margin_min x the thrust-limited budget --
# same reachability filter curriculum.build_reachable_pool already uses for Earth departure.
#
# Piloting: each leg gets its own Gtoc4ControlEnv (flyby mode: no velocity match) and its own
# policy, warm-started from flyby_seed0_warmstart and fine-tuned for a modest step budget --
# WP5's randomised-target run already showed a single policy doesn't generalise across different
# targets (0/20, indistinguishable from coasting), so a shared multi-target pilot isn't an option
# here; a fresh warm-started fine-tune per leg is the affordable alternative. The final leg is
# flown in rendezvous mode (velocity match required, warm-started from stage1_seed0) since GTOC4
# requires the mission to end with one rendezvous, not a flyby.
#
# Scoring: WP11 found the true 1000km GTOC4 flyby tolerance is ~1000x tighter than what this
# direct-control PPO setup reaches in a comparable training budget. Rather than loosen the
# tolerance to manufacture successes, every leg here is scored against the true tolerance
# (accuracy_position) regardless of what tolerance training used -- an honest score, even if it
# comes out at 0 flybys.
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import catalog
import curriculum
from gtoc4_env import Gtoc4ControlEnv
from constants import (sun_gravitational_parameter as mu, spacecraft_wet_mass, thrust_max,
                        accuracy_position, accuracy_velocity, time_mission_max)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
CATALOG_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'gtoc4_problem_data.txt')
LEG_TIME_RANGE = curriculum.ASTEROID_TIME_LIMIT_RANGE  # days
LEG_CONTROL_INTERVAL = curriculum.ASTEROID_CONTROL_INTERVAL
INFO_KEYWORDS = ('success', 'delta_r_norm', 'delta_v_norm', 'mass', 'total_impulse')

def estimate_leg_delta_v(r_from, v_from, epoch_from, target, leg_time):
    """Generalises curriculum._delta_v_needed (Earth-departure only) to an arbitrary current
    spacecraft state: energy term from the vis-viva semi-major axis of the current osculating
    orbit vs. the target's, plus a phasing term from the current-to-target position angle *right
    now* (epoch_from), matching curriculum._delta_v_needed's own convention (it uses Earth's and
    the target's position at START_EPOCH, not at estimated arrival) -- using arrival-epoch
    position instead was tried first and gave a 4x-different estimate for the same Earth-departure
    target/leg_time, which would make the pool-reachability margin (computed with the start-epoch
    convention) inconsistent with what this function selects on."""
    r_norm = np.linalg.norm(r_from)
    v_norm = np.linalg.norm(v_from)
    a_from = 1.0 / (2.0 / r_norm - v_norm**2 / mu)
    v_circ = np.sqrt(mu / abs(a_from))

    delta_a = target['a'] - a_from
    dv_energy = abs(delta_a) * np.sqrt(mu) / (2 * abs(a_from)**1.5)

    target_now = catalog.target_states([target], epoch_from, mu)[0]
    cos_angle = np.dot(r_from, target_now[:3]) / (r_norm * np.linalg.norm(target_now[:3]))
    delta_theta = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return dv_energy + v_circ * delta_theta

def select_next_target(r_from, v_from, epoch_from, remaining_time, visited_names, pool, margin_min=1.5):
    """Greedy: among unvisited `pool` asteroids reachable within min(remaining_time, the pool's
    longest leg window) at margin_min x the thrust-limited delta-v budget, returns the cheapest
    one as (target, leg_time, dv_estimate) -- or (None, None, None) if nothing clears the bar."""
    a_max = thrust_max / spacecraft_wet_mass
    leg_time = min(max(LEG_TIME_RANGE) * 86400.0, remaining_time)
    dv_budget = a_max * leg_time

    best = None
    for ast in pool:
        if ast['name'] in visited_names:
            continue
        dv = estimate_leg_delta_v(r_from, v_from, epoch_from, ast, leg_time)
        if dv < dv_budget / margin_min and (best is None or dv < best[2]):
            best = (ast, leg_time, dv)
    return best if best else (None, None, None)

def train_leg(initial_state, target, start_epoch, leg_time, require_velocity_match,
              warm_start_run, run_dir, timesteps, seed=0):
    os.makedirs(run_dir, exist_ok=True)
    env_fn = lambda: Gtoc4ControlEnv(
        initial_state, target, start_epoch, leg_time, control_interval=LEG_CONTROL_INTERVAL,
        position_tolerance=accuracy_position, velocity_tolerance=accuracy_velocity,
        require_velocity_match=require_velocity_match,
        shaping_weight_velocity=(1.0 if require_velocity_match else 0.0))
    env = DummyVecEnv([lambda: Monitor(env_fn(), filename=os.path.join(run_dir, 'monitor.csv'),
                                        info_keywords=INFO_KEYWORDS)])
    warm_dir = os.path.join(RESULTS_DIR, warm_start_run)
    env = VecNormalize.load(os.path.join(warm_dir, 'vecnormalize.pkl'), env)
    model = PPO.load(os.path.join(warm_dir, 'ppo_model'), env=env, seed=seed)
    model.learn(total_timesteps=timesteps)
    model.save(os.path.join(run_dir, 'ppo_model'))
    env.save(os.path.join(run_dir, 'vecnormalize.pkl'))
    return model, env

def fly_leg(model, env):
    """Runs one deterministic episode to completion and returns (final_info, min_delta_r_m) --
    min_delta_r_m is the closest approach over the *whole* leg (env already tracks this
    internally for flyby-mode termination; we need it for the rendezvous-mode final leg too, and
    for scoring against the true tolerance independent of what the leg was trained/terminated
    against), so it's tracked here step by step rather than read off the terminal step alone."""
    obs = env.reset()
    done = False
    info = {}
    min_dr = np.inf
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = env.step(action)
        done = bool(dones[0])
        info = infos[0]
        min_dr = min(min_dr, info['delta_r_norm'])
    return info, min_dr

def run_tour(max_legs=3, pilot_timesteps=300_000, pool_size=50, seed=0, tour_name='tour'):
    pool = curriculum.build_reachable_pool(CATALOG_PATH, pool_size=pool_size,
                                            time_limit=max(LEG_TIME_RANGE) * 86400.0)
    print(f"reachable pool: {len(pool)} asteroids")

    state = curriculum.initial_state()
    epoch_now = curriculum.START_EPOCH
    remaining = time_mission_max
    visited = []
    legs = []
    true_flybys = 0

    for leg_idx in range(max_legs):
        target, leg_time, dv = select_next_target(state[:3], state[3:6], epoch_now, remaining, visited, pool)
        if target is None:
            print(f"leg {leg_idx}: no reachable unvisited target within remaining {remaining/86400:.0f} days -- stopping")
            break

        run_dir = os.path.join(RESULTS_DIR, f"{tour_name}_leg{leg_idx}_{target['name']}")
        print(f"leg {leg_idx}: targeting {target['name']} (dv_estimate={dv:.1f} m/s, "
              f"leg_time={leg_time/86400:.0f} days) -- training pilot ({pilot_timesteps} steps)...")
        model, env = train_leg(state, target, epoch_now, leg_time, require_velocity_match=False,
                                warm_start_run='flyby_seed0_warmstart', run_dir=run_dir,
                                timesteps=pilot_timesteps, seed=seed)
        info, min_dr = fly_leg(model, env)

        is_true_flyby = bool(min_dr < accuracy_position)
        true_flybys += int(is_true_flyby)
        state = np.concatenate([info['position'], info['velocity'], [info['mass']]])
        epoch_now += leg_time
        remaining -= leg_time
        visited.append(target['name'])
        legs.append(dict(name=target['name'], leg_time_days=leg_time / 86400, dv_estimate_ms=dv,
                          achieved_min_delta_r_km=min_dr / 1e3, true_flyby=is_true_flyby,
                          mass_after_kg=info['mass']))
        print(f"  -> min delta_r over leg: {min_dr/1e3:.0f} km (true flyby: {is_true_flyby}), "
              f"mass: {info['mass']:.1f} kg, remaining budget: {remaining/86400:.0f} days")
        if remaining <= 0:
            print("mission time budget exhausted")
            break

    # final rendezvous attempt: one more leg, velocity match required, warm-started from the
    # stage-1 rendezvous policy rather than the flyby one
    final_target, final_leg_time, final_dv = select_next_target(
        state[:3], state[3:6], epoch_now, remaining, visited, pool)
    final_success = False
    if final_target is not None:
        run_dir = os.path.join(RESULTS_DIR, f"{tour_name}_final_{final_target['name']}")
        print(f"final leg: rendezvous attempt on {final_target['name']} "
              f"(dv_estimate={final_dv:.1f} m/s, leg_time={final_leg_time/86400:.0f} days)...")
        model, env = train_leg(state, final_target, epoch_now, final_leg_time, require_velocity_match=True,
                                warm_start_run='stage1_seed0', run_dir=run_dir,
                                timesteps=pilot_timesteps, seed=seed)
        info, min_dr = fly_leg(model, env)
        final_success = bool(info['success'])
        state = np.concatenate([info['position'], info['velocity'], [info['mass']]])
        remaining -= final_leg_time
        print(f"  -> final rendezvous success: {final_success}, min delta_r: {min_dr/1e3:.0f} km, "
              f"final delta_v: {info['delta_v_norm']:.1f} m/s, mass: {info['mass']:.1f} kg")
    else:
        print("no reachable target left for a final rendezvous attempt")

    print("\n=== GTOC4-style score ===")
    print(f"true flybys (delta_r < {accuracy_position/1e3:.0f} km at closest approach): {true_flybys}")
    print(f"final rendezvous achieved: {final_success}")
    print(f"final spacecraft mass: {state[6]:.1f} kg")
    print(f"legs attempted: {len(legs)} + {'1' if final_target is not None else '0'} final rendezvous attempt")

    return dict(legs=legs, true_flybys=true_flybys, final_success=final_success,
                final_mass=float(state[6]), final_target=final_target['name'] if final_target else None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-legs', type=int, default=3)
    parser.add_argument('--timesteps', type=int, default=300_000)
    parser.add_argument('--pool-size', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tour-name', type=str, default='tour')
    args = parser.parse_args()
    run_tour(max_legs=args.max_legs, pilot_timesteps=args.timesteps, pool_size=args.pool_size,
              seed=args.seed, tour_name=args.tour_name)
