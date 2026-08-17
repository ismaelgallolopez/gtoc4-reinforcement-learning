# Phase 2 of the legality track: how many catalogue asteroids are reachable from Earth departure,
# as a function of the three quantities the curriculum pinned at their worst legal value --
# launch epoch (was launch_interval[0], one day out of a 4018-day legal window), launch hyperbolic
# excess |v_inf| (was 0, the rules grant 4 km/s in any direction) and mission window (was 400-600 d
# or 5-7 yr, the rules allow 10 years).
#
# Reachability is curriculum._delta_v_needed < curriculum.delta_v_budget(window) / margin, the same
# filter build_reachable_pool already uses. It is a coarse closed-form screen, not a trajectory --
# its job is to say whether the candidate set is empty, not to certify any individual transfer.
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt

import catalog
import curriculum
from constants import launch_interval

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
CATALOG_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'gtoc4_problem_data.txt')

YEAR = 365.25 * 86400.0
V_INFINITY_VALUES = [0.0, 1e3, 2e3, 3e3, 4e3]        # m/s, 4 km/s is the legal maximum
WINDOWS = [('600 d', 600 * 86400.0), ('5 yr', 5 * YEAR), ('10 yr', 10 * YEAR)]

def scan(n_epochs=41, margin_min=1.5):
    """Returns (epochs_mjd, dv_needed) where dv_needed[i, j] is the zero-v_inf delta-v requirement
    of asteroid j for a launch at epoch i. v_inf and the mission window only enter as thresholds
    afterwards, so the expensive part is computed once."""
    asteroids = catalog.parse_asteroids(CATALOG_PATH)
    epochs = np.linspace(launch_interval[0], launch_interval[1], n_epochs)
    dv_needed = np.empty((n_epochs, len(asteroids)))
    for i, t in enumerate(epochs):
        for j, ast in enumerate(asteroids):
            dv_needed[i, j] = curriculum._delta_v_needed(ast, None, start_epoch=t)
        print(f"  epoch {i+1}/{n_epochs} (MJD {t/86400 + catalog.MJD_J2000:.0f}): "
              f"cheapest target {dv_needed[i].min()/1e3:.2f} km/s")
    return epochs, dv_needed, asteroids

def counts(dv_needed, v_infinity, window_seconds, margin_min):
    """Reachable count per launch epoch, at one (v_inf, window, margin) setting."""
    threshold = curriculum.delta_v_budget(window_seconds) / margin_min
    return np.sum(np.maximum(0.0, dv_needed - v_infinity) < threshold, axis=1)

def acceptance_test(dv_needed, epochs):
    """With v_inf = 0 and the epoch pinned to launch_interval[0], the scan must reproduce the
    previously recorded count of 2 asteroids reachable at all (margin 1.0) in a 600-day window."""
    print("--- acceptance: v_inf = 0, epoch = launch_interval[0], 600-day window ---")
    assert abs(epochs[0] - launch_interval[0]) < 1.0, "epoch grid does not start at launch_interval[0]"
    n_1x = counts(dv_needed, 0.0, 600 * 86400.0, 1.0)[0]
    n_15x = counts(dv_needed, 0.0, 600 * 86400.0, 1.5)[0]
    print(f"budget                 : {curriculum.delta_v_budget(600*86400.0)/1e3:.3f} km/s")
    print(f"cheapest target        : {dv_needed[0].min()/1e3:.3f} km/s")
    print(f"reachable, margin 1.0x : {n_1x}   (recorded value: 2)")
    print(f"reachable, margin 1.5x : {n_15x}   (recorded value: 0)")
    assert n_1x == 2, f"expected 2 reachable asteroids at margin 1.0, got {n_1x}"
    assert n_15x == 0, f"expected 0 reachable asteroids at margin 1.5, got {n_15x}"
    print("PASS: recorded baseline reproduced\n")

def report(epochs, dv_needed, margin_min):
    epochs_mjd = epochs / 86400.0 + catalog.MJD_J2000
    print(f"--- reachable-asteroid count at a {margin_min}x margin (1436-asteroid catalogue) ---")
    print(f"{'window':<8}{'|v_inf|':>9}{'min':>7}{'median':>8}{'max':>7}{'best epoch (MJD)':>19}")
    rows = []
    for label, window in WINDOWS:
        for v_infinity in V_INFINITY_VALUES:
            c = counts(dv_needed, v_infinity, window, margin_min)
            best = int(np.argmax(c))
            print(f"{label:<8}{v_infinity/1e3:>7.0f} k{c.min():>7}{int(np.median(c)):>8}{c.max():>7}"
                  f"{epochs_mjd[best]:>19.0f}")
            rows.append((label, v_infinity, int(c.min()), float(np.median(c)), int(c.max()),
                         float(epochs_mjd[best])))
    print()
    return rows

def plot(epochs, dv_needed, margin_min):
    epochs_mjd = epochs / 86400.0 + catalog.MJD_J2000
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    window_label, window = WINDOWS[-1]
    for v_infinity in V_INFINITY_VALUES:
        ax1.plot(epochs_mjd, counts(dv_needed, v_infinity, window, margin_min),
                 marker='.', label=f'|v_inf| = {v_infinity/1e3:.0f} km/s')
    ax1.axvline(epochs_mjd[0], color='gray', linestyle=':', label='pinned START_EPOCH')
    ax1.set_xlabel('launch epoch (MJD)')
    ax1.set_ylabel(f'reachable asteroids ({window_label} window)')
    ax1.grid(alpha=0.3); ax1.legend(fontsize=8)

    for label, window in WINDOWS:
        best = [counts(dv_needed, v, window, margin_min).max() for v in V_INFINITY_VALUES]
        median = [np.median(counts(dv_needed, v, window, margin_min)) for v in V_INFINITY_VALUES]
        line, = ax2.plot(np.array(V_INFINITY_VALUES) / 1e3, best, marker='o', label=f'{label}, best epoch')
        ax2.plot(np.array(V_INFINITY_VALUES) / 1e3, median, marker='.', linestyle='--',
                 color=line.get_color(), label=f'{label}, median epoch')
    ax2.set_xlabel('launch |v_inf| (km/s)')
    ax2.set_ylabel('reachable asteroids')
    ax2.grid(alpha=0.3); ax2.legend(fontsize=8)

    fig.suptitle(f'reachable-asteroid count vs launch epoch and v_inf '
                 f'({margin_min}x delta-v margin, 1436-asteroid catalogue)')
    fig.tight_layout()
    out_path = os.path.join(FIGURES_DIR, 'legality_reachability_scan.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {out_path}")
    return out_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-epochs', type=int, default=41)
    parser.add_argument('--margin', type=float, default=1.5)
    args = parser.parse_args()

    print(f"scanning {args.n_epochs} launch epochs over MJD "
          f"{launch_interval[0]/86400 + catalog.MJD_J2000:.0f}-"
          f"{launch_interval[1]/86400 + catalog.MJD_J2000:.0f} "
          f"({(launch_interval[1]-launch_interval[0])/86400:.0f} days)")
    epochs, dv_needed, asteroids = scan(n_epochs=args.n_epochs)
    print()
    acceptance_test(dv_needed, epochs)
    rows = report(epochs, dv_needed, args.margin)
    plot(epochs, dv_needed, args.margin)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, 'legality_reachability_scan.csv')
    with open(csv_path, 'w') as f:
        f.write('window,v_infinity_ms,count_min,count_median,count_max,best_epoch_mjd\n')
        for row in rows:
            f.write(','.join(str(x) for x in row) + '\n')
    print(f"saved {csv_path}")
