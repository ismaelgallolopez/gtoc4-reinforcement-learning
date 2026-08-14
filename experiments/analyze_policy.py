# WP7: analysis of the learned stage-1 policy (the only curriculum stage that actually
# converged) -- thrust profile over a rollout, bang-bang structure, trajectory vs target,
# probing the policy's response to synthetic position offsets, a failure-mode sweep as the
# target is pushed towards stage 2's difficulty, and the mass-optimality gap against a
# closed-form impulsive delta-v estimate.
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import curriculum
from constants import spacecraft_wet_mass, Isp_engine
from dynamics import g0

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
RUN_NAME = 'stage1_seed0'

def load_policy(run_name=RUN_NAME, env_fn=None):
    run_dir = os.path.join(RESULTS_DIR, run_name)
    env_fn = env_fn or (lambda: curriculum.make_env(1))
    env = DummyVecEnv([lambda: Monitor(env_fn())])
    env = VecNormalize.load(os.path.join(run_dir, 'vecnormalize.pkl'), env)
    env.training = False
    env.norm_reward = False
    model = PPO.load(os.path.join(run_dir, 'ppo_model'))
    return model, env

def rollout(model, env):
    """One deterministic episode, recording per-step thrust/state for analysis. Everything is
    read from `info`, not from the wrapped env's attributes after step(): SB3's VecEnv
    auto-resets the underlying env internally as soon as a step returns done=True, so reading
    e.g. raw_env.elapsed_time *after* that step silently returns the next episode's value."""
    obs = env.reset()
    raw_env = env.venv.envs[0].unwrapped
    control_interval = raw_env.control_interval
    target, start_epoch = raw_env.target, raw_env.start_epoch

    log = {'t': [], 'throttle': [], 'rtn': [], 'delta_r': [], 'delta_v': [], 'mass': []}
    done = False
    info = {}
    t = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = env.step(action)
        done = bool(dones[0])
        info = infos[0]
        t += control_interval
        log['t'].append(t)
        log['throttle'].append(info['throttle'])
        log['rtn'].append(info['rtn_action'].copy())
        log['delta_r'].append(info['delta_r'].copy())
        log['delta_v'].append(info['delta_v'].copy())
        log['mass'].append(info['mass'])
    for k in log:
        log[k] = np.array(log[k])
    log['success'] = bool(info['success'])
    log['target'] = target
    log['start_epoch'] = start_epoch
    return log

def plot_thrust_profile(log, out_path):
    days = log['t'] / 86400.0
    thrust_rtn = log['rtn'] * log['throttle'][:, None]

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(days, log['throttle'], color='k')
    axes[0].set_ylabel('throttle [0-1]')
    axes[0].set_title(f"learned thrust profile ({RUN_NAME}, success={log['success']})")
    axes[0].grid(alpha=0.3)

    for i, label in enumerate(['radial', 'transverse', 'normal']):
        axes[1].plot(days, thrust_rtn[:, i], label=label)
    axes[1].set_xlabel('mission day')
    axes[1].set_ylabel('thrust fraction x RTN direction')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {out_path}")

def plot_bang_bang_histogram(log, out_path):
    """Checks whether the learned throttle clusters near 0/1 (bang-bang, as expected from
    Pontryagin's minimum principle for a mass/time-optimal low-thrust transfer) or sits in
    the middle of the range."""
    throttle = log['throttle']
    near_off = np.mean(throttle < 0.1)
    near_full = np.mean(throttle > 0.9)
    mid = 1.0 - near_off - near_full

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(throttle, bins=20, range=(0, 1), color='steelblue', edgecolor='k')
    ax.set_xlabel('throttle'); ax.set_ylabel('steps')
    ax.set_title(f'throttle distribution: {near_off:.0%} off, {mid:.0%} mid, {near_full:.0%} full')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {out_path}")
    return near_off, mid, near_full

def plot_trajectory(log, out_path):
    import catalog
    from constants import sun_gravitational_parameter as mu
    target_r = np.array([catalog.target_states([log['target']], log['start_epoch'] + t, mu)[0][:3]
                          for t in log['t']])
    spacecraft_r = target_r + log['delta_r']  # delta_r = spacecraft_r - target_r, by definition
    AU = 1.495978707e11

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(spacecraft_r[:, 0] / AU, spacecraft_r[:, 1] / AU, label='spacecraft', color='C0')
    ax.plot(target_r[:, 0] / AU, target_r[:, 1] / AU, label='target', color='C1', linestyle='--')
    ax.scatter(*(spacecraft_r[0, :2] / AU), color='C0', marker='o', label='start')
    ax.scatter(*(spacecraft_r[-1, :2] / AU), color='C0', marker='x', label='end (spacecraft)')
    ax.scatter(0, 0, color='gold', marker='*', s=150, label='Sun')
    ax.set_xlabel('x [AU]'); ax.set_ylabel('y [AU]')
    ax.set_title('trajectory, ecliptic-plane projection')
    ax.legend(); ax.set_aspect('equal'); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {out_path}")

def probe_policy(model, env, n_directions=8, delta_r_mag=5e8, out_path=None):
    """Feeds the policy synthetic observations with delta_r rotated around a full circle in
    the R-T plane (everything else at a 'mid-mission, on-track' baseline) and records the
    commanded thrust direction, to check whether the policy has learned a sensible geometric
    rule (thrust roughly anti-parallel to delta_r, i.e. 'push towards the target')."""
    AU = 1.495978707e11
    angles = np.linspace(0, 2 * np.pi, n_directions, endpoint=False)
    commanded = []
    for theta in angles:
        delta_r = delta_r_mag * np.array([np.cos(theta), np.sin(theta), 0.0])
        obs_raw = np.concatenate([
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],   # r/AU, v/V_REF: on a unit circular orbit, moving +y
            delta_r / AU, [0.0, 0.0, 0.0],       # delta_r/AU, delta_v/V_REF
            [0.9], [0.5],                        # mass fraction, time remaining fraction
        ]).astype(np.float32)
        obs_norm = env.normalize_obs(obs_raw[None, :])
        action, _ = model.predict(obs_norm, deterministic=True)
        rtn_raw = action[0][:3]
        norm = np.linalg.norm(rtn_raw)
        rtn_unit = rtn_raw / norm if norm > 1e-6 else np.zeros(3)
        throttle = (action[0][3] + 1.0) / 2.0
        commanded.append((theta, rtn_unit, throttle))

    if out_path:
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
        for theta, rtn_unit, throttle in commanded:
            thrust_theta = np.arctan2(rtn_unit[1], rtn_unit[0])
            ax.annotate('', xy=(thrust_theta, throttle), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color='C0'))
            ax.plot([theta], [1.0], 'rx')
        ax.set_title('probe: delta_r direction (x) vs commanded thrust direction/magnitude (arrow)')
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"saved {out_path}")
    return commanded

def failure_mode_sweep(model_env_fn, delta_a_values, n_episodes=10):
    """Stress-tests the stage-1 policy (no retraining) as delta_a is pushed from stage 1's
    trained value (0.02 AU) towards stage 2's (0.1 AU), to find where success collapses."""
    results = []
    for delta_a_au in delta_a_values:
        target = curriculum._earth_like_target(delta_a=delta_a_au * 1.495978707e11, delta_i=0.0,
                                                 delta_M=np.deg2rad(1.0))
        env_fn = lambda: __import__('gtoc4_env').Gtoc4ControlEnv(
            curriculum.initial_state(), target, curriculum.START_EPOCH, curriculum.STAGES[1]['time_limit'],
            position_tolerance=curriculum.STAGES[1]['position_tolerance'],
            velocity_tolerance=curriculum.STAGES[1]['velocity_tolerance'])
        model, env = model_env_fn(env_fn)
        successes = 0
        for _ in range(n_episodes):
            obs = env.reset()
            done = False
            info = {}
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, dones, infos = env.step(action)
                done = bool(dones[0])
                info = infos[0]
            successes += int(info['success'])
        results.append((delta_a_au, successes / n_episodes))
        print(f"delta_a={delta_a_au:.3f} AU: success rate {successes}/{n_episodes}")
    return results

def plot_failure_sweep(results, out_path):
    delta_as, rates = zip(*results)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(delta_as, rates, marker='o')
    ax.axvline(0.02, color='gray', linestyle=':', label='trained on (stage 1)')
    ax.axvline(0.1, color='red', linestyle=':', label='stage 2 (never solved)')
    ax.set_xlabel('delta_a [AU]'); ax.set_ylabel('success rate')
    ax.set_ylim(-0.05, 1.05); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title('stage-1 policy generalisation as the target offset grows')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {out_path}")

def mass_optimality_gap(log):
    dv_min = curriculum._delta_v_needed(log['target'], curriculum.STAGES[1]['time_limit'])
    m0 = spacecraft_wet_mass
    mf_actual = log['mass'][-1]
    dv_actual = Isp_engine * g0 * np.log(m0 / mf_actual)
    gap = (dv_actual - dv_min) / dv_min
    print(f"impulsive delta-v estimate (lower bound): {dv_min:.1f} m/s")
    print(f"learned policy's equivalent delta-v used:  {dv_actual:.1f} m/s")
    print(f"optimality gap: {gap:+.1%}")
    return dv_min, dv_actual, gap

if __name__ == '__main__':
    os.makedirs(FIGURES_DIR, exist_ok=True)
    model, env = load_policy()
    log = rollout(model, env)
    print(f"rollout: success={log['success']}, episode length={log['t'][-1]/86400:.0f} days, "
          f"final mass={log['mass'][-1]:.1f} kg")

    plot_thrust_profile(log, os.path.join(FIGURES_DIR, 'wp7_thrust_profile.png'))
    plot_bang_bang_histogram(log, os.path.join(FIGURES_DIR, 'wp7_bang_bang_histogram.png'))
    plot_trajectory(log, os.path.join(FIGURES_DIR, 'wp7_trajectory.png'))
    probe_policy(model, env, out_path=os.path.join(FIGURES_DIR, 'wp7_policy_probe.png'))
    mass_optimality_gap(log)

    sweep = failure_mode_sweep(lambda env_fn: load_policy(env_fn=env_fn),
                                delta_a_values=[0.02, 0.04, 0.06, 0.08, 0.1])
    plot_failure_sweep(sweep, os.path.join(FIGURES_DIR, 'wp7_failure_sweep.png'))
