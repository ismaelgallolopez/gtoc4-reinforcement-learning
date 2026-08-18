# Guidance diagnostic track, Phase 1: trace one failing short-ToF leg in full.
#
# Reproduces Phase 6's sampling exactly (same RNG seed, same candidate-selection method: a real
# leg_candidates() call at a real departure state, rank drawn 50/30/20% toward rank 0/1/2) so the
# fly_fail and missed cases traced here are the same kind of attempt Phase 6 already measured
# failing 54.7% of the time at ToF <= 120 d -- not a fresh, possibly unrepresentative resample.
#
# The sampling logic (_departure, _fly_one) is duplicated from experiments/acceptance_phase6.py
# rather than imported, per this track's "do not touch anything from the halted Phase 6
# calibration" rule -- reading its method to reproduce it is fine, importing and driving its module
# state is not. Kept byte-identical to the original so the RNG stream matches call for call.
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt

import catalog
import sequencer
from constants import sun_gravitational_parameter as mu, scape_velocity_max

CATALOG_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'gtoc4_problem_data.txt')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
DAY = 86400.0
LAUNCH_MJD = 58128.0
TOF_BUCKETS_DAYS = (60, 90, 120, 180, 300, 450, 600, 900, 1200)   # Phase 6's bucket order
SAMPLES_PER_BUCKET = 6                                            # Phase 6's target per bucket
MAX_ATTEMPTS_PER_BUCKET = 120                                     # Phase 6's attempt cap
RANK_WEIGHTS = (0.5, 0.3, 0.2)
TARGET_TOF_DAYS = 90


def _departure(rng, pool, kind, epoch_range):
    if kind == 'earth':
        epoch = rng.uniform(*epoch_range)
        return sequencer.launch_state(epoch), epoch, None, scape_velocity_max
    ast = pool[rng.integers(len(pool))]
    epoch = rng.uniform(*epoch_range)
    s = catalog.target_states([ast], epoch, mu)[0]
    return np.concatenate([s, [rng.uniform(700.0, 1500.0)]]), epoch, ast['name'], 0.0


def _fly_one(rng, pool, pool_index, tof_days, epoch_range, trace=None):
    """Identical to acceptance_phase6.py's _fly_one, plus an optional trace passthrough to
    sequencer.fly_flyby_leg. Consumes exactly the same random draws in exactly the same order."""
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
    flown = sequencer.fly_flyby_leg(state_start, epoch, target, candidate['arrival'], trace=trace)
    if flown is None:
        return dict(kind=kind, tof_days=tof_days, rank=int(rank), status='fly_fail',
                     dv1_est=candidate['dv1'], target_name=target['name'])
    samples, final_state, miss = flown
    from constants import accuracy_position
    closed = miss < accuracy_position
    return dict(kind=kind, tof_days=tof_days, rank=int(rank),
                status='ok' if closed else 'missed', dv1_est=candidate['dv1'],
                miss_km=miss / 1e3, target_name=target['name'])


def replay_to_bucket(rng, pool, pool_index, launch_epoch, up_to_tof_days):
    """Burns through the same RNG draws Phase 6's collect() consumed for every bucket strictly
    before `up_to_tof_days`, so the following draws land on the same attempts Phase 6 made."""
    epoch_range = (launch_epoch, launch_epoch + 3000 * DAY)
    for tof_days in TOF_BUCKETS_DAYS:
        if tof_days == up_to_tof_days:
            return
        collected = 0
        attempts = 0
        while collected < SAMPLES_PER_BUCKET and attempts < MAX_ATTEMPTS_PER_BUCKET:
            attempts += 1
            result = _fly_one(rng, pool, pool_index, tof_days, epoch_range)
            if result is not None and result['status'] == 'ok' and result['dv1_est'] >= 30.0:
                collected += 1
    raise ValueError(f"{up_to_tof_days} not in TOF_BUCKETS_DAYS")


def find_case(rng, pool, pool_index, tof_days, epoch_range, want_status, max_attempts=120):
    """Scans forward through the reproduced RNG stream for the first attempt matching
    `want_status`, tracing it. Returns (trace, result) or (None, None) if not found."""
    for _ in range(max_attempts):
        trace = []
        result = _fly_one(rng, pool, pool_index, tof_days, epoch_range, trace=trace)
        if result is not None and result['status'] == want_status:
            return trace, result
    return None, None


def plot_trace(trace, result, label, out_path):
    steps = [s for s in trace if 'thrust_magnitude' in s]
    if not steps:
        print(f"  ({label}: no per-step data to plot -- failed before the first control step)")
        return
    elapsed = [s['elapsed_days'] for s in steps]
    thrust = [s['thrust_magnitude'] for s in steps]
    position_error = [s['position_error_km'] for s in steps]
    time_to_go = [s['time_remaining_days'] for s in steps]

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(elapsed, thrust, marker='.', color='C0')
    axes[0].axhline(0.135, color='gray', linestyle=':', label='thrust_max')
    axes[0].set_ylabel('thrust magnitude (N)')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(elapsed, position_error, marker='.', color='C1')
    axes[1].set_ylabel('position error to aim point (km)')
    axes[1].set_yscale('log')
    axes[1].grid(alpha=0.3)

    axes[2].plot(elapsed, time_to_go, marker='.', color='C2')
    axes[2].set_ylabel('time-to-go (days)')
    axes[2].set_xlabel('elapsed leg time (days)')
    axes[2].grid(alpha=0.3)

    terminal = trace[-1]
    fig.suptitle(f"{label}: ToF={result['tof_days']} d, target={result['target_name']}, "
                f"outcome={terminal.get('event', '?')}")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {out_path}")


def print_tail(trace, label, n=10):
    print(f"\n--- {label}: final {min(n, len(trace))} trace entries ---")
    for entry in trace[-n:]:
        print(f"  {entry}")


def main():
    rng = np.random.default_rng(2)   # Phase 6's seed
    pool = catalog.parse_asteroids(CATALOG_PATH)
    pool_index = list(range(len(pool)))
    launch_epoch = (LAUNCH_MJD - catalog.MJD_J2000) * DAY
    epoch_range = (launch_epoch, launch_epoch + 3000 * DAY)

    print(f"--- replaying Phase 6's RNG stream up to the ToF={TARGET_TOF_DAYS} d bucket ---")
    replay_to_bucket(rng, pool, pool_index, launch_epoch, TARGET_TOF_DAYS)
    print("replay complete; RNG stream now matches Phase 6's state at the start of this bucket")

    print(f"\n--- scanning for a fly_fail case at ToF={TARGET_TOF_DAYS} d ---")
    trace_fail, result_fail = find_case(rng, pool, pool_index, TARGET_TOF_DAYS, epoch_range,
                                        'fly_fail')
    if trace_fail is None:
        print("no fly_fail case found in the remaining stream")
    else:
        print(f"found: {result_fail}")
        print_tail(trace_fail, "fly_fail case")
        plot_trace(trace_fail, result_fail, "fly_fail",
                   os.path.join(FIGURES_DIR, 'guidance_trace_flyfail.png'))

    print(f"\n--- scanning for a missed case at ToF={TARGET_TOF_DAYS} d ---")
    trace_missed, result_missed = find_case(rng, pool, pool_index, TARGET_TOF_DAYS, epoch_range,
                                            'missed')
    if trace_missed is None:
        print("no missed case found in the remaining stream")
    else:
        print(f"found: {result_missed}")
        print_tail(trace_missed, "missed case")
        plot_trace(trace_missed, result_missed, "missed",
                   os.path.join(FIGURES_DIR, 'guidance_trace_missed.png'))

    return dict(trace_fail=trace_fail, result_fail=result_fail,
                trace_missed=trace_missed, result_missed=result_missed)


if __name__ == '__main__':
    main()
    print("\nPhase 1 trace collection complete")
