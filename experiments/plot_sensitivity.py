# WP6: aggregates the sensitivity sweep (experiments/sensitivity.py) into one plot per
# parameter -- success rate and mean return vs value, mean +- std over 3 seeds.
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt

import sensitivity

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')

def collect(param, value):
    rates, returns = [], []
    for seed in sensitivity.SEEDS:
        rate, ret = sensitivity.evaluate_one(param, value, seed)
        rates.append(rate)
        returns.append(ret)
    return np.mean(rates), np.std(rates), np.mean(returns), np.std(returns)

def plot_all():
    baseline_value = sensitivity.BASELINE
    os.makedirs(FIGURES_DIR, exist_ok=True)
    summary = []

    for param, extra_values in sensitivity.PARAMETERS.items():
        values = sorted([baseline_value[param]] + extra_values)
        rate_means, rate_stds, ret_means, ret_stds = [], [], [], []
        for v in values:
            key = 'baseline' if v == baseline_value[param] else param
            m_rate, s_rate, m_ret, s_ret = collect(key, v)
            rate_means.append(m_rate); rate_stds.append(s_rate)
            ret_means.append(m_ret); ret_stds.append(s_ret)
            summary.append((param, v, m_rate, s_rate, m_ret, s_ret))
            print(f"{param}={v:g}: success={m_rate:.2f}+-{s_rate:.2f} return={m_ret:.2f}+-{s_ret:.2f}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.errorbar(values, rate_means, yerr=rate_stds, marker='o', capsize=4)
        ax1.axvline(baseline_value[param], color='gray', linestyle=':', label='stage-1 baseline')
        ax1.set_xlabel(param); ax1.set_ylabel('success rate (20 eval episodes)')
        ax1.set_ylim(-0.05, 1.05); ax1.grid(alpha=0.3); ax1.legend()

        ax2.errorbar(values, ret_means, yerr=ret_stds, marker='o', capsize=4, color='C1')
        ax2.axvline(baseline_value[param], color='gray', linestyle=':')
        ax2.set_xlabel(param); ax2.set_ylabel('mean deterministic return')
        ax2.grid(alpha=0.3)

        fig.suptitle(f'sensitivity to {param} (stage-1 config, 3 seeds, 500k steps)')
        fig.tight_layout()
        out_path = os.path.join(FIGURES_DIR, f'sensitivity_{param}.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"saved {out_path}")

    with open(os.path.join(FIGURES_DIR, '..', 'results', 'sensitivity_summary.csv'), 'w') as f:
        f.write('param,value,success_rate_mean,success_rate_std,return_mean,return_std\n')
        for row in summary:
            f.write(','.join(str(x) for x in row) + '\n')

if __name__ == '__main__':
    plot_all()
