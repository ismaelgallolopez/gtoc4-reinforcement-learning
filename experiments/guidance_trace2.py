# Guidance diagnostic track, Phase 2: classify the short-ToF failure mode across a sample.
#
# Phase 1 traced one fly_fail and one missed case at ToF=90 d in full: both saturated at
# thrust_max from the first control step onward, and both were candidates that already exceeded
# the oracle's own 0.6x-duty-cycle feasibility margin (leg_feasible). That raised the obvious next
# question -- does the guidance also fail on candidates the oracle itself calls affordable? -- which
# this phase answers two ways:
#
#   1. A full reproduction of Phase 6's exact ToF=60/90/120 d samples (same seed, n=120 each, the
#      very data that produced the 54.7% failure figure), split four ways: feasible-and-trivial
#      (dv1_est < 30 m/s, essentially free), feasible-and-non-trivial, infeasible-and-trivial, and
#      infeasible-and-non-trivial. This is necessary because a first pass conflated "oracle-
#      feasible" with "closes reliably" -- most of what looked like clean 100% success on feasible
#      candidates turned out to be candidates so cheap (v_inf-credit-covered) that success was
#      never actually being tested.
#   2. A fresh 30-attempt, two-epoch, fully-traced batch (this phase's literal acceptance test),
#      used to classify each *failure's* proximate cause (saturation / oscillation / exhaustion /
#      other) -- a question the feasibility cross-tab alone cannot answer, since it only says
#      whether a leg closed, not how it failed when it didn't.
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

import catalog
import curriculum
import sequencer
from constants import thrust_max
import guidance_trace1 as gt1   # this track's own Phase 1 module -- not the halted Phase 6 file

CATALOG_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'gtoc4_problem_data.txt')
DAY = 86400.0
TRIVIAL_DV1_MS = 30.0             # Phase 6's own "uninformative" threshold, reused for consistency
DUTY_CYCLE = 0.6                  # the oracle's own leg_feasible margin

# -- part 1: reproduce Phase 6's exact short-ToF samples, feasibility x triviality cross-tab -----

PHASE6_LAUNCH_MJD = 58128.0
PHASE6_SEED = 2
PHASE6_TOF_BUCKETS = (60, 90, 120)
PHASE6_MAX_ATTEMPTS = 120


def reproduce_feasibility_crosstab(tof_buckets=PHASE6_TOF_BUCKETS):
    """Replays Phase 6's exact RNG stream (seed=2, MJD 58128) through each short-ToF bucket in
    turn and records, per attempt: whether the candidate was oracle-feasible (dv1 < 0.6x budget),
    whether it was trivial (dv1 < 30 m/s), and whether the leg actually closed."""
    rng = np.random.default_rng(PHASE6_SEED)
    pool = catalog.parse_asteroids(CATALOG_PATH)
    pool_index = list(range(len(pool)))
    launch_epoch = (PHASE6_LAUNCH_MJD - catalog.MJD_J2000) * DAY
    epoch_range = (launch_epoch, launch_epoch + 3000 * DAY)

    all_rows = {}
    for tof_days in tof_buckets:
        rows = []
        for _ in range(PHASE6_MAX_ATTEMPTS):
            kind = 'earth' if rng.random() < 0.5 else 'mid'
            state0, epoch, exclude_name, credit = gt1._departure(rng, pool, kind, epoch_range)
            visited = {exclude_name} if exclude_name else set()
            candidates = sequencer.leg_candidates(state0, epoch, pool, pool_index, tof_days,
                                                  visited, credit)
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
            rows.append(dict(feasible=feasible, trivial=trivial, status=status,
                             dv1=candidate['dv1']))
        all_rows[tof_days] = rows
    return all_rows


def report_crosstab(all_rows):
    print("--- part 1: Phase 6's exact seed=2 short-ToF samples, feasibility x triviality ---")
    print(f"{'ToF':>6} {'n':>5} {'trivial+feasible':>20} {'nontrivial+feasible':>22} "
          f"{'trivial+infeasible':>21} {'nontrivial+infeasible':>24}")
    for tof_days, rows in all_rows.items():
        def cell(feasible, trivial):
            subset = [r for r in rows if r['feasible'] == feasible and r['trivial'] == trivial]
            ok = sum(1 for r in subset if r['status'] == 'ok')
            return f"{ok}/{len(subset)}" if subset else "0/0"
        print(f"{tof_days:>6} {len(rows):>5} {cell(True, True):>20} {cell(True, False):>22} "
              f"{cell(False, True):>21} {cell(False, False):>24}")


# -- part 2: fresh, fully-traced two-epoch batch, classify failure modes -------------------------

LAUNCH_EPOCHS_MJD = (58128.0, 58430.0)
TOF_BUCKETS_DAYS = (60, 90, 120)
SAMPLES_PER_CELL = 5   # 3 buckets x 2 epochs x 5 = 30
BATCH_SEED = 3


def classify(trace):
    """Primary proximate cause of a failed leg, plus the numbers behind the call.

    saturation: thrust >= 99% of thrust_max for more than 80% of the leg's control steps.
    oscillation: position error reaches a minimum well before the leg ends and then grows back by
                 more than 5% of that minimum -- a reversal, not simple (if slow) convergence.
    exhaustion: the leg's terminal event is propellant exhaustion (mass hit the dry-mass floor).
    other: none of the above -- reported, not forced into a bucket.

    Checked in that order (exhaustion first, as the most concrete signature; oscillation next, a
    specific pattern; saturation last, as the default explanation for a leg pinned at max thrust
    with neither of the other two). A case can show more than one signature -- the non-primary ones
    also present are kept as secondary."""
    steps = [s for s in trace if 'thrust_magnitude' in s]
    terminal = trace[-1] if trace and 'event' in trace[-1] else {}
    n = len(steps)
    if n == 0:
        return 'other', dict(reason='no control step reached (Lambert failed on the first solve)',
                             n_steps=0, frac_saturated=None, reversal=False, exhausted=False,
                             secondary=[], position_error_start_km=None,
                             position_error_min_km=None, position_error_end_km=None,
                             terminal_event=terminal.get('event', '?'))

    frac_saturated = sum(1 for s in steps if s['thrust_magnitude'] >= 0.99 * thrust_max) / n
    position_error = np.array([s['position_error_km'] for s in steps])
    min_index = int(np.argmin(position_error))
    reversal = (min_index < n - 2 and
                position_error[-1] > position_error[min_index] * 1.05)
    exhausted = terminal.get('event') == 'propellant_exhausted'
    saturated = frac_saturated > 0.8

    if exhausted:
        primary = 'exhaustion'
    elif reversal:
        primary = 'oscillation'
    elif saturated:
        primary = 'saturation'
    else:
        primary = 'other'
    secondary = [c for c, present in
                 (('exhaustion', exhausted), ('oscillation', reversal), ('saturation', saturated))
                 if present and c != primary]

    return primary, dict(n_steps=n, frac_saturated=frac_saturated, reversal=reversal,
                         exhausted=exhausted, secondary=secondary,
                         position_error_start_km=float(position_error[0]),
                         position_error_min_km=float(position_error[min_index]),
                         position_error_end_km=float(position_error[-1]),
                         terminal_event=terminal.get('event', 'ok'))


def sample_batch(rng, pool, pool_index, launch_epochs, tof_buckets, per_cell):
    rows = []
    for launch_mjd in launch_epochs:
        launch_epoch = (launch_mjd - catalog.MJD_J2000) * DAY
        epoch_range = (launch_epoch, launch_epoch + 3000 * DAY)
        for tof_days in tof_buckets:
            collected = 0
            attempts = 0
            while collected < per_cell and attempts < 200:
                attempts += 1
                trace = []
                result = gt1._fly_one(rng, pool, pool_index, tof_days, epoch_range, trace=trace)
                if result is None:
                    continue
                collected += 1
                mass_start = trace[0]['mass_kg'] if trace and 'mass_kg' in trace[0] else None
                budget = (curriculum.delta_v_budget(tof_days * DAY, mass_start)
                          if mass_start is not None else None)
                oracle_feasible = (budget is not None and result['dv1_est'] < DUTY_CYCLE * budget)
                trivial = result['dv1_est'] < TRIVIAL_DV1_MS
                primary, detail = (('ok', dict(n_steps=len(trace), frac_saturated=None,
                                               reversal=False, exhausted=False, secondary=[],
                                               terminal_event='ok',
                                               position_error_start_km=None,
                                               position_error_min_km=None,
                                               position_error_end_km=None))
                                   if result['status'] == 'ok' else classify(trace))
                rows.append(dict(launch_mjd=launch_mjd, tof_days=tof_days, status=result['status'],
                                 dv1_est=result['dv1_est'], oracle_feasible=oracle_feasible,
                                 trivial=trivial, primary=primary, **detail))
            print(f"  MJD {launch_mjd} ToF {tof_days:3d} d: {collected} attempts collected "
                  f"in {attempts}")
    return rows


def report_batch(rows):
    from collections import Counter
    print(f"\n--- part 2: fresh two-epoch batch, n={len(rows)} ---")
    print(f"outcome counts: {dict(Counter(r['status'] for r in rows))}")

    print(f"\nfeasibility x triviality (this batch, for comparison with part 1):")
    for feasible in (True, False):
        for trivial in (True, False):
            subset = [r for r in rows if r['oracle_feasible'] == feasible and r['trivial'] == trivial]
            ok = sum(1 for r in subset if r['status'] == 'ok')
            label = f"{'feasible' if feasible else 'infeasible'}+{'trivial' if trivial else 'nontrivial'}"
            print(f"  {label:24s}: {ok}/{len(subset)}" if subset else f"  {label:24s}: 0/0")

    print(f"\n--- primary failure-mode classification, by ToF bucket ---")
    failures = [r for r in rows if r['status'] != 'ok']
    for tof_days in TOF_BUCKETS_DAYS:
        subset = [r for r in failures if r['tof_days'] == tof_days]
        if not subset:
            print(f"  ToF {tof_days:3d} d: no failures sampled")
            continue
        counts = Counter(r['primary'] for r in subset)
        total = len(subset)
        parts = ", ".join(f"{k}={v} ({100*v/total:.0f}%)" for k, v in counts.most_common())
        print(f"  ToF {tof_days:3d} d (n={total}): {parts}")

    print(f"\n--- primary failure-mode classification, by outcome type ---")
    for status in ('fly_fail', 'missed'):
        subset = [r for r in failures if r['status'] == status]
        if not subset:
            print(f"  {status}: none sampled")
            continue
        counts = Counter(r['primary'] for r in subset)
        total = len(subset)
        parts = ", ".join(f"{k}={v} ({100*v/total:.0f}%)" for k, v in counts.most_common())
        print(f"  {status} (n={total}): {parts}")

    print(f"\n--- overall primary classification (all {len(failures)} failures) ---")
    counts = Counter(r['primary'] for r in failures)
    for k, v in counts.most_common():
        print(f"  {k:12s}: {v:3d} ({100*v/len(failures):.1f}%)")

    print(f"\n--- representative traces per category actually observed ---")
    for category in ('saturation', 'oscillation', 'exhaustion', 'other'):
        matching = [r for r in failures if r['primary'] == category]
        if not matching:
            print(f"  {category}: none observed")
            continue
        print(f"  {category} ({len(matching)} total, showing up to 3):")
        for r in matching[:3]:
            print(f"    MJD {r['launch_mjd']} ToF {r['tof_days']}d {r['status']:9s} "
                  f"dv1_est={r['dv1_est']:9.1f} feasible={r['oracle_feasible']!s:5s} "
                  f"trivial={r['trivial']!s:5s} n_steps={r['n_steps']:3d} "
                  f"frac_sat={r['frac_saturated']} "
                  f"pos_err(start->min->end km)={r['position_error_start_km']:.3g}->"
                  f"{r['position_error_min_km']:.3g}->{r['position_error_end_km']:.3g} "
                  f"terminal={r['terminal_event']} secondary={r['secondary']}")
    return failures


def main():
    all_rows = reproduce_feasibility_crosstab()
    report_crosstab(all_rows)

    print(f"\n--- sampling {len(TOF_BUCKETS_DAYS)} ToF buckets x {len(LAUNCH_EPOCHS_MJD)} epochs x "
          f"{SAMPLES_PER_CELL} attempts "
          f"= {len(TOF_BUCKETS_DAYS)*len(LAUNCH_EPOCHS_MJD)*SAMPLES_PER_CELL} ---")
    rng = np.random.default_rng(BATCH_SEED)
    pool = catalog.parse_asteroids(CATALOG_PATH)
    pool_index = list(range(len(pool)))
    rows = sample_batch(rng, pool, pool_index, LAUNCH_EPOCHS_MJD, TOF_BUCKETS_DAYS, SAMPLES_PER_CELL)
    report_batch(rows)
    return dict(crosstab=all_rows, batch=rows)


if __name__ == '__main__':
    main()
    print("\nPhase 2 classification complete")
