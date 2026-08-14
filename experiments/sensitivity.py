# WP6: one-at-a-time sensitivity analysis around the stage-1 baseline (the only curriculum
# stage that reliably converges -- stage 2 and the randomised-target config both failed to beat
# coasting, so there's no successful config there to perturb). Each of 5 env/reward knobs is
# swept across 3 values, 3 seeds each, at 500k timesteps per run (half the WP4/WP5 budget: stage
# 1's return is still climbing at 1M steps, but the ranking between configs is already visible
# well before that, and 45 full-budget runs would take too long). The middle value of every
# sweep is the same stage-1 baseline, so it's trained once (3 seeds) and reused across all 5
# parameters instead of being retrained 5 times.
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
torch.set_num_threads(1)  # each run is one of many parallel processes, not a lone job -- letting
                           # every process grab torch's default thread pool oversubscribes the
                           # CPU and collapsed throughput ~18x in the WP5 multi-seed run
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import curriculum

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'sensitivity')
SEEDS = [0, 1, 2]
TIMESTEPS = 500_000
INFO_KEYWORDS = ('success', 'delta_r_norm', 'delta_v_norm', 'mass', 'total_impulse')

BASELINE = dict(control_interval=86400.0, position_tolerance=1e6 * 1e3, velocity_tolerance=500.0,
                 propellant_penalty=0.0, rendezvous_bonus=10.0)

# non-baseline values only -- the baseline value itself is shared across every parameter (see
# module docstring) and lives under run name "baseline", not under any of these parameter names
PARAMETERS = {
    'control_interval':   [43200.0, 172800.0],
    'position_tolerance': [1e5 * 1e3, 1e7 * 1e3],
    'velocity_tolerance': [100.0, 2000.0],
    'propellant_penalty': [0.01, 0.05],
    'rendezvous_bonus':   [2.0, 30.0],
}

def run_name(param, value, seed):
    return f"{param}_{value:g}_seed{seed}"

def make_env(param, value):
    kwargs = dict(BASELINE)
    if param != 'baseline':
        kwargs[param] = value
    return curriculum.make_stage1_variant(**kwargs)

def train_one(param, value, seed):
    run_dir = os.path.join(RESULTS_DIR, run_name(param, value, seed))
    os.makedirs(run_dir, exist_ok=True)

    env = DummyVecEnv([lambda: Monitor(make_env(param, value), filename=os.path.join(run_dir, 'monitor.csv'),
                                        info_keywords=INFO_KEYWORDS)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    model = PPO('MlpPolicy', env, policy_kwargs=dict(net_arch=[64, 64], activation_fn=torch.nn.Tanh),
                learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99,
                gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, seed=seed, verbose=0)
    model.learn(total_timesteps=TIMESTEPS)
    model.save(os.path.join(run_dir, 'ppo_model'))
    env.save(os.path.join(run_dir, 'vecnormalize.pkl'))
    print(f"done: {run_name(param, value, seed)}")

def evaluate_one(param, value, seed, n_episodes=20):
    run_dir = os.path.join(RESULTS_DIR, run_name(param, value, seed))
    env = DummyVecEnv([lambda: Monitor(make_env(param, value))])
    env = VecNormalize.load(os.path.join(run_dir, 'vecnormalize.pkl'), env)
    env.training = False
    env.norm_reward = False
    model = PPO.load(os.path.join(run_dir, 'ppo_model'))

    successes, returns = 0, []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        ep_return = 0.0
        info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            done = bool(dones[0])
            ep_return += reward[0]
            info = infos[0]
        successes += int(info['success'])
        returns.append(ep_return)
    return successes / n_episodes, float(np.mean(returns))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--param', required=True, help="one of PARAMETERS, or 'baseline'")
    parser.add_argument('--value', type=float, default=0.0)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--mode', choices=['train', 'eval'], default='train')
    args = parser.parse_args()
    if args.mode == 'train':
        train_one(args.param, args.value, args.seed)
    else:
        rate, ret = evaluate_one(args.param, args.value, args.seed)
        print(f"{args.param}={args.value} seed{args.seed}: success={rate:.2f} return={ret:.3f}")
