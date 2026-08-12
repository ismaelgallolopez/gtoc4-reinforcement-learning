# plots the PPO learning curve (from SB3 Monitor's CSV) against the coast/tangential baselines
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import curriculum
import baselines

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')

def plot(run_name, stage=1, window=20):
    monitor = pd.read_csv(os.path.join(RESULTS_DIR, run_name, 'monitor.csv'), skiprows=1)
    timesteps = monitor['l'].cumsum()
    smoothed_return = monitor['r'].rolling(window, min_periods=1).mean()

    env = curriculum.make_env(stage)
    coast_return = baselines.coast(env)['episode_return']
    tangential_return = baselines.tangential_thrust(env)['episode_return']

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(timesteps, smoothed_return, label=f'PPO ({window}-episode rolling mean)')
    ax.axhline(coast_return, color='gray', linestyle='--', label='coast baseline')
    ax.axhline(tangential_return, color='orange', linestyle='--', label='tangential-thrust baseline')
    ax.set_xlabel('training timesteps')
    ax.set_ylabel('episode return')
    ax.set_title(f'PPO learning curve, stage {stage} ({run_name})')
    ax.legend()
    ax.grid(alpha=0.3)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, f'learning_curve_{run_name}.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"saved {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', type=str, required=True)
    parser.add_argument('--stage', type=int, default=1)
    args = parser.parse_args()
    plot(args.run_name, stage=args.stage)
