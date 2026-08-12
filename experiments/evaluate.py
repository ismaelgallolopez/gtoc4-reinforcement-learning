# deterministic evaluation of a trained policy -- separate from training-time reward, which is
# noisy (stochastic actions) and not what we report as "performance".
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import curriculum

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
CATALOG_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'gtoc4_problem_data.txt')

def evaluate(run_name, stage=1, randomize=False, n_episodes=20, seed=1234):
    """seed is used for the randomised-target sampler (should differ from training seeds, to test
    generalisation to unseen targets) and, for stage 3, to pick the fixed asteroid -- pass the
    same seed the run was trained with there, since stage 3 fixes one target for the whole run."""
    run_dir = os.path.join(RESULTS_DIR, run_name)
    if randomize:
        env_fn = lambda: curriculum.make_randomized_env(CATALOG_PATH, rng=np.random.default_rng(seed))
    elif stage == 3:
        env_fn = lambda: curriculum.make_env(stage, catalog_path=CATALOG_PATH, rng=np.random.default_rng(seed))
    else:
        env_fn = lambda: curriculum.make_env(stage)
    env = DummyVecEnv([lambda: Monitor(env_fn())])
    env = VecNormalize.load(os.path.join(run_dir, 'vecnormalize.pkl'), env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(os.path.join(run_dir, 'ppo_model'))

    successes = 0
    final_drs, final_dvs, final_masses, returns = [], [], [], []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        episode_return = 0.0
        info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            done = bool(dones[0])
            episode_return += reward[0]
            info = infos[0]
        successes += int(info['success'])
        final_drs.append(np.linalg.norm(info['delta_r']))
        final_dvs.append(np.linalg.norm(info['delta_v']))
        final_masses.append(info['mass'])
        returns.append(episode_return)

    label = "randomised targets" if randomize else f"stage {stage}"
    print(f"{run_name} on {label}, {n_episodes} deterministic episodes:")
    print(f"  success rate: {successes}/{n_episodes}")
    print(f"  final |delta_r|: {np.mean(final_drs) / 1e3:.1f} +- {np.std(final_drs) / 1e3:.1f} km")
    print(f"  final |delta_v|: {np.mean(final_dvs):.2f} +- {np.std(final_dvs):.2f} m/s")
    print(f"  final mass:      {np.mean(final_masses):.1f} kg")
    print(f"  mean return:     {np.mean(returns):.3f}")
    return dict(success_rate=successes / n_episodes, mean_delta_r=np.mean(final_drs),
                mean_delta_v=np.mean(final_dvs), mean_mass=np.mean(final_masses), mean_return=np.mean(returns))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', type=str, required=True)
    parser.add_argument('--stage', type=int, default=1)
    parser.add_argument('--randomize', action='store_true')
    parser.add_argument('--episodes', type=int, default=20)
    args = parser.parse_args()
    evaluate(args.run_name, stage=args.stage, randomize=args.randomize, n_episodes=args.episodes)
