# trains PPO on a curriculum stage. Takes a seed + total timesteps; everything else uses the
# plan's starting hyperparameters (2x64 tanh MLP, SB3 defaults for the rest).
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import curriculum

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

def train(stage=1, seed=0, total_timesteps=1_000_000, run_name=None):
    run_name = run_name or f"stage{stage}_seed{seed}"
    run_dir = os.path.join(RESULTS_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)

    env = DummyVecEnv([lambda: Monitor(curriculum.make_env(stage), filename=os.path.join(run_dir, 'monitor.csv'))])
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
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--timesteps', type=int, default=1_000_000)
    parser.add_argument('--run-name', type=str, default=None)
    args = parser.parse_args()
    train(stage=args.stage, seed=args.seed, total_timesteps=args.timesteps, run_name=args.run_name)
