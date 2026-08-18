# Guidance diagnostic track, Phase 3: locate the ToF where the failure rate transitions.
#
# Phase 6 measured near-0% failure at ToF>=300 d and 54.7% at ToF<=120 d. Phase 2 showed the
# short-ToF number is dominated by candidates that were never oracle-feasible to begin with (the
# top-3-ranked pool at short ToF is almost entirely "trivial and free" or "non-trivial and hopeless
# -- essentially no middle ground), and found, informally, that a genuinely testable population
# (non-trivial AND within the oracle's own 0.6x-duty-cycle margin) only starts appearing from
# ~180 d onward, and does not close reliably even then. This phase repeats that check formally, at
# the specified ToF sweep, with both of Phase 2's numbers: the raw Phase-6-style failure rate (for
# direct comparability to "54.7%" and "near-0%") and the failure rate restricted to non-trivial,
# oracle-feasible candidates (the more meaningful number once Phase 2's confound is accounted for).
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

import catalog
import curriculum
import sequencer
import guidance_trace1 as gt1   # this track's own module

CATALOG_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'gtoc4_problem_data.txt')
DAY = 86400.0
LAUNCH_MJD = 58128.0
TOF_SWEEP_DAYS = (150, 180, 210, 250, 300)
SAMPLES_PER_TOF = 15
TRIVIAL_DV1_MS = 30.0
DUTY_CYCLE = 0.6
SEED = 4


def sample_tof(rng, pool, pool_index, tof_days, epoch_range, n):
    rows = []
    attempts = 0
    while len(rows) < n and attempts < 10 * n:
        attempts += 1
        kind = 'earth' if rng.random() < 0.5 else 'mid'
        state0, epoch, exclude_name, credit = gt1._departure(rng, pool, kind, epoch_range)
        visited = {exclude_name} if exclude_name else set()
        candidates = sequencer.leg_candidates(state0, epoch, pool, pool_index, tof_days, visited,
                                              credit)
        if not candidates:
            continue
        rank = min(rng.choice(len(gt1.RANK_WEIGHTS), p=gt1.RANK_WEIGHTS), len(candidates) - 1)
        candidate = candidates[rank]
        state_start = state0.copy()
        if kind == 'earth':
            correction = candidate['v_departure'] - state0[3:6]
            norm = np.linalg.norm(correction)
            if norm > 0:
                state_start[3:6] = state_start[3:6] + correction / norm * min(4000.0, norm)
        target = pool[candidate['pool_slot']]
        flown = sequencer.fly_flyby_leg(state_start, epoch, target, candidate['arrival'])
        budget = curriculum.delta_v_budget(tof_days * DAY, state_start[6])
        feasible = candidate['dv1'] < DUTY_CYCLE * budget
        trivial = candidate['dv1'] < TRIVIAL_DV1_MS
        if flown is None:
            status = 'fly_fail'
        else:
            _, _, miss = flown
            status = 'ok' if miss < 1000e3 else 'missed'
        rows.append(dict(feasible=feasible, trivial=trivial, status=status, dv1=candidate['dv1']))
    return rows, attempts


def main():
    rng = np.random.default_rng(SEED)
    pool = catalog.parse_asteroids(CATALOG_PATH)
    pool_index = list(range(len(pool)))
    launch_epoch = (LAUNCH_MJD - catalog.MJD_J2000) * DAY
    epoch_range = (launch_epoch, launch_epoch + 3000 * DAY)

    print(f"--- ToF sweep: {TOF_SWEEP_DAYS} d, {SAMPLES_PER_TOF} legs each, MJD {LAUNCH_MJD} ---")
    results = {}
    for tof_days in TOF_SWEEP_DAYS:
        rows, attempts = sample_tof(rng, pool, pool_index, tof_days, epoch_range, SAMPLES_PER_TOF)
        results[tof_days] = rows
        n = len(rows)
        n_fail = sum(1 for r in rows if r['status'] != 'ok')
        raw_rate = 100 * n_fail / n if n else float('nan')

        nontrivial_feasible = [r for r in rows if r['feasible'] and not r['trivial']]
        nf_fail = sum(1 for r in nontrivial_feasible if r['status'] != 'ok')
        nf_rate = (100 * nf_fail / len(nontrivial_feasible)
                   if nontrivial_feasible else float('nan'))

        n_trivial = sum(1 for r in rows if r['trivial'])
        n_infeasible_nontrivial = sum(1 for r in rows if not r['feasible'] and not r['trivial'])

        print(f"  ToF {tof_days:4d} d (n={n}, {attempts} attempts): "
              f"raw failure rate = {raw_rate:5.1f}%  |  "
              f"trivial={n_trivial:2d}  nontrivial+feasible={len(nontrivial_feasible):2d} "
              f"(failure {nf_rate:5.1f}%)  nontrivial+infeasible={n_infeasible_nontrivial:2d}")

    print(f"\n--- raw (Phase-6-style) failure rate by ToF ---")
    raw_rates = {}
    for tof_days, rows in results.items():
        n = len(rows)
        n_fail = sum(1 for r in rows if r['status'] != 'ok')
        raw_rates[tof_days] = 100 * n_fail / n if n else float('nan')
        print(f"  ToF {tof_days:4d} d: {raw_rates[tof_days]:5.1f}%")

    crossing_50 = next((t for t in TOF_SWEEP_DAYS if raw_rates[t] < 50), None)
    below_10 = next((t for t in TOF_SWEEP_DAYS if raw_rates[t] < 10), None)
    print(f"\nraw failure rate first drops below 50% at ToF={crossing_50}"
          if crossing_50 else "\nraw failure rate does not drop below 50% within the sampled range")
    print(f"raw failure rate first drops below 10% at ToF={below_10}"
          if below_10 else "raw failure rate does not drop below 10% within the sampled range")

    print(f"\n--- non-trivial, oracle-feasible-only failure rate by ToF ---")
    nf_rates = {}
    for tof_days, rows in results.items():
        subset = [r for r in rows if r['feasible'] and not r['trivial']]
        if not subset:
            print(f"  ToF {tof_days:4d} d: no non-trivial+feasible candidates sampled")
            continue
        n_fail = sum(1 for r in subset if r['status'] != 'ok')
        nf_rates[tof_days] = 100 * n_fail / len(subset)
        print(f"  ToF {tof_days:4d} d (n={len(subset)}): {nf_rates[tof_days]:5.1f}%")

    return results


if __name__ == '__main__':
    main()
    print("\nPhase 3 sweep complete")
