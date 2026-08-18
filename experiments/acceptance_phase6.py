# Multi-flyby track, Phase 6: empirical correction for the Lambert leg-cost oracle.
#
# The legality track measured the oracle underestimating one long flyby leg's flown cost by ~2.5x
# (8267 m/s flown vs 3270 m/s estimated at ToF=1000 d) and hit a case with no zero-revolution
# Lambert solution at all, 73 days from arrival. This phase calibrates the gap empirically: fly
# real candidates (the shrinking-aim guidance the sequencer actually uses) across a range of ToF,
# fit flown/estimated as a function of ToF, and apply the fit as a correction factor in the leg
# feasibility test.
#
# Sampling is confined to *plausible* candidates -- the top few ranked by the oracle's own dv1 from
# a real leg_candidates() call at a real departure state, not uniform-random (target, ToF) pairs.
# A uniform draw is essentially always unreachable (curriculum.py's own finding: ~50 km/s for a
# naive draw against a ~3-30 km/s budget), so it would not measure the oracle's error in the regime
# either search actually operates in -- it would just measure "how bad is the estimate for
# transfers nobody would ever attempt." A candidate also has to be *verified* successful (closes
# within the 1000 km flyby tolerance) to enter the fit: a shrinking-aim leg can run to completion
# without exhausting propellant and still land millions of km away if the candidate wasn't
# realistically flyable at all, and that "flown" delta-v is not a measurement of transfer cost.
import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

import catalog
import dynamics
import sequencer
from constants import sun_gravitational_parameter as mu, Isp_engine, scape_velocity_max, \
    accuracy_position

CATALOG_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'gtoc4_problem_data.txt')
DAY = 86400.0
LAUNCH_MJD = 58128.0
TOF_BUCKETS_DAYS = (60, 90, 120, 180, 300, 450, 600, 900, 1200)
SAMPLES_PER_BUCKET = 6
RANK_WEIGHTS = (0.5, 0.3, 0.2)   # bias toward the cheapest-ranked candidate, as a real search would


def _departure(rng, pool, kind, epoch_range):
    """A plausible leg-start state: Earth departure (v_inf credit available) or a state already on
    a real asteroid's own orbit at a random epoch (proxy for 'just flew by X')."""
    if kind == 'earth':
        epoch = rng.uniform(*epoch_range)
        return sequencer.launch_state(epoch), epoch, None, scape_velocity_max
    ast = pool[rng.integers(len(pool))]
    epoch = rng.uniform(*epoch_range)
    s = catalog.target_states([ast], epoch, mu)[0]
    return np.concatenate([s, [rng.uniform(700.0, 1500.0)]]), epoch, ast['name'], 0.0


def _fly_one(rng, pool, pool_index, tof_days, epoch_range):
    """One attempt: draw a departure state and a ranked candidate at tof_days, fly it, return a
    result dict or None if the attempt itself could not be set up (no candidates at all)."""
    kind = 'earth' if rng.random() < 0.5 else 'mid'
    state0, epoch, exclude_name, credit = _departure(rng, pool, kind, epoch_range)
    visited = {exclude_name} if exclude_name else set()

    candidates = sequencer.leg_candidates(state0, epoch, pool, pool_index, tof_days, visited, credit)
    if not candidates:
        return None
    rank = min(rng.choice(len(RANK_WEIGHTS), p=RANK_WEIGHTS), len(candidates) - 1)
    candidate = candidates[rank]

    state_start = state0.copy()
    if kind == 'earth':
        correction = candidate['v_departure'] - state0[3:6]
        norm = np.linalg.norm(correction)
        if norm > 0:
            state_start[3:6] = state_start[3:6] + correction / norm * min(scape_velocity_max, norm)

    target = pool[candidate['pool_slot']]
    flown = sequencer.fly_flyby_leg(state_start, epoch, target, candidate['arrival'])
    if flown is None:
        return dict(kind=kind, tof_days=tof_days, rank=int(rank), status='fly_fail',
                     dv1_est=candidate['dv1'])
    samples, final_state, miss = flown
    dv_flown = Isp_engine * dynamics.g0 * np.log(state_start[6] / final_state[6])
    closed = miss < accuracy_position
    return dict(kind=kind, tof_days=tof_days, rank=int(rank),
                status='ok' if closed else 'missed', dv1_est=candidate['dv1'],
                dv_flown=dv_flown, miss_km=miss / 1e3)


def collect(rng, pool, tof_buckets, samples_per_bucket, max_attempts_per_bucket=120,
            min_dv1_est=30.0, verbose=True):
    """For each ToF bucket, keeps attempting plausible legs until `samples_per_bucket` genuine
    (closed within tolerance, non-trivial cost) successes are collected or the attempt cap is hit.
    Returns (samples, per_bucket_stats) -- stats records every outcome, samples only the fit-worthy
    ones (closed, dv1_est > min_dv1_est)."""
    pool_index = list(range(len(pool)))
    launch_epoch = (LAUNCH_MJD - catalog.MJD_J2000) * DAY
    epoch_range = (launch_epoch, launch_epoch + 3000 * DAY)

    samples, stats = [], {}
    for tof_days in tof_buckets:
        counts = dict(attempts=0, ok=0, missed=0, fly_fail=0, no_candidates=0, trivial=0)
        collected = 0
        while collected < samples_per_bucket and counts['attempts'] < max_attempts_per_bucket:
            counts['attempts'] += 1
            result = _fly_one(rng, pool, pool_index, tof_days, epoch_range)
            if result is None:
                counts['no_candidates'] += 1
                continue
            if result['status'] == 'fly_fail':
                counts['fly_fail'] += 1
                continue
            if result['status'] == 'missed':
                counts['missed'] += 1
                continue
            if result['dv1_est'] < min_dv1_est:
                counts['trivial'] += 1
                continue
            counts['ok'] += 1
            samples.append(result)
            collected += 1
        stats[tof_days] = counts
        if verbose:
            print(f"  ToF {tof_days:5d} d: {counts['ok']}/{counts['attempts']} usable "
                  f"(missed={counts['missed']}, fly_fail={counts['fly_fail']}, "
                  f"no_candidates={counts['no_candidates']}, trivial={counts['trivial']})")
    return samples, stats


def fit_binned(samples):
    """Attempt 1: piecewise-linear correction factor(tof_days) = mean(flown/estimated) at each
    bucket that has samples, interpolated between bucket midpoints and held flat beyond the ends --
    the simplest curve that tracks the binned data, no functional form assumed."""
    by_bucket = {}
    for s in samples:
        by_bucket.setdefault(s['tof_days'], []).append(s['dv_flown'] / s['dv1_est'])
    tofs = sorted(by_bucket)
    factors = [float(np.mean(by_bucket[t])) for t in tofs]
    return dict(kind='binned', tofs=tofs, factors=factors)


def fit_binned_by_kind(samples):
    """Attempt 2: same as fit_binned, but a separate curve for 'earth' (departure, v_inf-credited)
    and 'mid' (already on some body's orbit) legs -- the two populations start from different kinds
    of state and might not share one correction."""
    curves = {}
    for kind in ('earth', 'mid'):
        subset = [s for s in samples if s['kind'] == kind]
        by_bucket = {}
        for s in subset:
            by_bucket.setdefault(s['tof_days'], []).append(s['dv_flown'] / s['dv1_est'])
        tofs = sorted(by_bucket)
        factors = [float(np.mean(by_bucket[t])) for t in tofs]
        curves[kind] = (tofs, factors)
    return dict(kind='by_kind', curves=curves)


def fit_loglinear(samples):
    """Attempt 3: log(ratio) = a + b*log(tof_days), least squares over all samples -- a single
    smooth monotone curve, less sensitive to any one bucket's small-n noise than the binned fits."""
    tof = np.array([s['tof_days'] for s in samples], dtype=np.float64)
    ratio = np.array([s['dv_flown'] / s['dv1_est'] for s in samples], dtype=np.float64)
    b, a = np.polyfit(np.log(tof), np.log(ratio), 1)
    return dict(kind='loglinear', a=float(a), b=float(b))


def apply_correction(fit, tof_days, kind=None):
    if fit['kind'] == 'binned':
        return float(np.interp(tof_days, fit['tofs'], fit['factors']))
    if fit['kind'] == 'by_kind':
        tofs, factors = fit['curves'][kind or 'earth']
        if not tofs:
            tofs, factors = fit['curves']['earth' if kind == 'mid' else 'mid']
        return float(np.interp(tof_days, tofs, factors))
    if fit['kind'] == 'loglinear':
        return float(np.exp(fit['a']) * tof_days ** fit['b'])
    raise ValueError(fit['kind'])


def held_out_legs(rng, pool, tof_range, n=5):
    """n fresh legs, drawn independently of the fit sample, spanning tof_range."""
    pool_index = list(range(len(pool)))
    launch_epoch = (LAUNCH_MJD - catalog.MJD_J2000) * DAY
    epoch_range = (launch_epoch, launch_epoch + 3000 * DAY)
    tof_choices = list(range(tof_range[0], tof_range[1] + 1, 10))

    legs = []
    attempts = 0
    while len(legs) < n and attempts < 300:
        attempts += 1
        tof_days = int(rng.choice(tof_choices))
        result = _fly_one(rng, pool, pool_index, tof_days, epoch_range)
        if result is None or result['status'] != 'ok' or result['dv1_est'] < 30.0:
            continue
        legs.append(result)
    return legs


def evaluate_fit(label, fit, legs, tolerance=0.20):
    print(f"\n--- held-out test: {label} ---")
    results = []
    for leg in legs:
        corrected = leg['dv1_est'] * apply_correction(fit, leg['tof_days'], leg.get('kind'))
        error = abs(corrected - leg['dv_flown']) / leg['dv_flown']
        results.append(dict(**leg, corrected=corrected, error=error))
        print(f"  ToF {leg['tof_days']:5d} d  kind={leg['kind']:5s}  est={leg['dv1_est']:8.1f}  "
              f"corrected={corrected:8.1f}  flown={leg['dv_flown']:8.1f}  "
              f"error={error*100:5.1f} %  {'PASS' if error < tolerance else 'FAIL'}")
    n_pass = sum(1 for r in results if r['error'] < tolerance)
    print(f"  {n_pass}/{len(results)} within {tolerance*100:.0f} %")
    return results, n_pass


def main():
    rng = np.random.default_rng(2)
    pool = catalog.parse_asteroids(CATALOG_PATH)

    print(f"--- sampling: {len(TOF_BUCKETS_DAYS)} ToF buckets x up to {SAMPLES_PER_BUCKET} "
          f"usable legs each, launch MJD {LAUNCH_MJD} ---")
    t0 = time.time()
    samples, stats = collect(rng, pool, TOF_BUCKETS_DAYS, SAMPLES_PER_BUCKET)
    print(f"collected {len(samples)} usable samples in {time.time()-t0:.1f} s")

    print("\n--- per-bucket outcome counts ---")
    total_attempts = sum(s['attempts'] for s in stats.values())
    total_fly_fail = sum(s['fly_fail'] for s in stats.values())
    total_missed = sum(s['missed'] for s in stats.values())
    print(f"total attempts: {total_attempts}, fly_fail: {total_fly_fail} "
          f"({100*total_fly_fail/total_attempts:.1f} %), missed: {total_missed} "
          f"({100*total_missed/total_attempts:.1f} %)")
    short_tof = [t for t in TOF_BUCKETS_DAYS if t <= 120]
    short_fail = sum(stats[t]['fly_fail'] + stats[t]['missed'] for t in short_tof)
    short_attempts = sum(stats[t]['attempts'] for t in short_tof)
    print(f"short ToF (<=120 d) failure rate: {short_fail}/{short_attempts} "
          f"({100*short_fail/short_attempts:.1f} %)" if short_attempts else "no short-ToF attempts")

    fit_a = fit_binned(samples)
    print("\n--- attempt 1: binned mean, combined ---")
    for t, f in zip(fit_a['tofs'], fit_a['factors']):
        n_bucket = sum(1 for s in samples if s['tof_days'] == t)
        print(f"  ToF {t:5d} d: correction = {f:6.3f}  (n={n_bucket})")

    fit_b = fit_binned_by_kind(samples)
    print("\n--- attempt 2: binned mean, split by kind ---")
    for kind, (tofs_k, factors_k) in fit_b['curves'].items():
        print(f"  {kind}: " + ", ".join(f"{t}d={f:.3f}" for t, f in zip(tofs_k, factors_k)))

    fit_c = fit_loglinear(samples)
    print("\n--- attempt 3: log-linear regression ---")
    print(f"  ratio = exp({fit_c['a']:.4f}) * tof_days^{fit_c['b']:.4f}")

    tof_range = (min(TOF_BUCKETS_DAYS), max(TOF_BUCKETS_DAYS))
    held_out = held_out_legs(rng, pool, tof_range, n=5)

    results_a, pass_a = evaluate_fit("attempt 1 (binned, combined)", fit_a, held_out)
    results_b, pass_b = evaluate_fit("attempt 2 (binned, by kind)", fit_b, held_out)
    results_c, pass_c = evaluate_fit("attempt 3 (log-linear)", fit_c, held_out)

    print(f"\n--- summary: {len(held_out)} held-out legs ---")
    print(f"  attempt 1 (binned, combined) : {pass_a}/{len(held_out)} within 20 %")
    print(f"  attempt 2 (binned, by kind)  : {pass_b}/{len(held_out)} within 20 %")
    print(f"  attempt 3 (log-linear)       : {pass_c}/{len(held_out)} within 20 %")

    best_label, best_fit, best_pass = max(
        [("binned_combined", fit_a, pass_a), ("binned_by_kind", fit_b, pass_b),
         ("loglinear", fit_c, pass_c)], key=lambda x: x[2])
    print(f"\n  best of the three: {best_label} ({best_pass}/{len(held_out)})")

    import json
    out_path = os.path.join(os.path.dirname(__file__), '..', 'results',
                             'multiflyby_phase6_correction.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(dict(fit_a=fit_a, fit_b={k: list(v) for k, v in fit_b['curves'].items()},
                        fit_c=fit_c, samples=samples, held_out=held_out,
                        pass_counts=dict(binned_combined=pass_a, binned_by_kind=pass_b,
                                          loglinear=pass_c),
                        best=best_label),
                   f, indent=2)
    print(f"\nsaved correction curves and raw samples to {out_path}")
    return dict(samples=samples, stats=stats, fit_a=fit_a, fit_b=fit_b, fit_c=fit_c,
                held_out=held_out, best=best_label)


if __name__ == '__main__':
    main()
    print("\nPhase 6 acceptance data collected")
