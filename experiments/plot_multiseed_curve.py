# canonical multi-seed learning curve: mean +- std across seeds, baselines as reference lines.
import argparse
import glob
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
CATALOG_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'gtoc4_problem_data.txt')

def load_curve(run_dir, grid, window=20):
    monitor = pd.read_csv(os.path.join(run_dir, 'monitor.csv'), skiprows=1)
    timesteps = monitor['l'].cumsum().values
    smoothed = monitor['r'].rolling(window, min_periods=1).mean().values
    return np.interp(grid, timesteps, smoothed, left=smoothed[0], right=smoothed[-1])

def baseline_mean_return(policy, n_targets=10, seed=999):
    rng = np.random.default_rng(seed)
    returns = []
    for _ in range(n_targets):
        env = curriculum.make_randomized_env(CATALOG_PATH, rng=rng)
        returns.append(policy(env)['episode_return'])
    return np.mean(returns)

def plot(pattern, n_grid_points=200):
    run_dirs = sorted(d for d in glob.glob(os.path.join(RESULTS_DIR, pattern)) if os.path.isdir(d))
    assert run_dirs, f"no runs matched {pattern}"
    print(f"found {len(run_dirs)} seeds: {[os.path.basename(d) for d in run_dirs]}")

    max_t = min(pd.read_csv(os.path.join(d, 'monitor.csv'), skiprows=1)['l'].cumsum().values[-1]
                for d in run_dirs)
    grid = np.linspace(0, max_t, n_grid_points)
    curves = np.stack([load_curve(d, grid) for d in run_dirs])
    mean, std = curves.mean(axis=0), curves.std(axis=0)

    coast_return = baseline_mean_return(baselines.coast)
    tangential_return = baseline_mean_return(baselines.tangential_thrust)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grid, mean, label=f'PPO (mean of {len(run_dirs)} seeds)')
    ax.fill_between(grid, mean - std, mean + std, alpha=0.2, label='+-1 std across seeds')
    ax.axhline(coast_return, color='gray', linestyle='--', label='coast baseline (mean over targets)')
    ax.axhline(tangential_return, color='orange', linestyle='--', label='tangential-thrust baseline (mean over targets)')
    ax.set_xlabel('training timesteps')
    ax.set_ylabel('episode return')
    ax.set_title('PPO learning curve, randomised targets, multi-seed')
    ax.legend()
    ax.grid(alpha=0.3)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, 'learning_curve_randomized_multiseed.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"saved {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pattern', type=str, default='randomized_seed*')
    args = parser.parse_args()
    plot(args.pattern)
