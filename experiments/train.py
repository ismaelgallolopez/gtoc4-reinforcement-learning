# trains PPO on a curriculum stage. Takes a seed + total timesteps; everything else uses the
# plan's starting hyperparameters (2x64 tanh MLP, SB3 defaults for the rest). Supports warm-
# starting from a previous run's weights (WP5's stage progression) and the randomised-target
# generalisation mode.
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import curriculum

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
CATALOG_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'gtoc4_problem_data.txt')
INFO_KEYWORDS = ('success', 'delta_r_norm', 'delta_v_norm', 'mass', 'total_impulse')

def train(stage=1, randomize=False, seed=0, total_timesteps=1_000_000, run_name=None, warm_start=None):
    run_name = run_name or f"stage{stage}_seed{seed}"
    run_dir = os.path.join(RESULTS_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)

    if randomize:
        env_fn = lambda: curriculum.make_randomized_env(CATALOG_PATH, rng=np.random.default_rng(seed))
    elif stage == 3:
        # fixed single asteroid for the whole run: same seed -> same sampled target every reset,
        # since make_env(stage=3) draws once from a fresh rng and Gtoc4ControlEnv keeps it fixed
        env_fn = lambda: curriculum.make_env(stage, catalog_path=CATALOG_PATH, rng=np.random.default_rng(seed))
    else:
        env_fn = lambda: curriculum.make_env(stage)

    env = DummyVecEnv([lambda: Monitor(env_fn(), filename=os.path.join(run_dir, 'monitor.csv'),
                                        info_keywords=INFO_KEYWORDS)])

    if warm_start:
        warm_dir = os.path.join(RESULTS_DIR, warm_start)
        env = VecNormalize.load(os.path.join(warm_dir, 'vecnormalize.pkl'), env)
        model = PPO.load(os.path.join(warm_dir, 'ppo_model'), env=env, seed=seed)
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=False)  # don't normalise reward while debugging shaping
        model = PPO(
            'MlpPolicy', env,
            policy_kwargs=dict(net_arch=[64, 64], activation_fn=torch.nn.Tanh),
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            seed=seed,
            verbose=1,
        )

    model.learn(total_timesteps=total_timesteps)

    model.save(os.path.join(run_dir, 'ppo_model'))
    env.save(os.path.join(run_dir, 'vecnormalize.pkl'))
    print(f"saved model and VecNormalize stats to {run_dir}")
    return model, env

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int, default=1)
    parser.add_argument('--randomize', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--timesteps', type=int, default=1_000_000)
    parser.add_argument('--run-name', type=str, default=None)
    parser.add_argument('--warm-start', type=str, default=None, help='run-name to load weights from')
    args = parser.parse_args()
    train(stage=args.stage, randomize=args.randomize, seed=args.seed, total_timesteps=args.timesteps,
          run_name=args.run_name, warm_start=args.warm_start)
