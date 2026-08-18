# Multi-flyby track, Phase 5: audit the leg cost model for flybys.
#
# Hypothesis to check: does the Phase 4 oracle (sequencer.leg_feasible) or Mission charge
# dv1 + dv2 (arrival velocity matching) on a flyby leg, when a flyby only ever needs to match
# position? If so, every nearby flyby candidate is priced as if it needed a full rendezvous, which
# would filter out cheap near-misses and could force the greedy toward longer legs, where the
# thrust-limited budget is larger and can absorb the inflated cost.
#
# Static audit: grep sequencer.py and train_sequencer.py for every leg_feasible call site and
# confirm what require_velocity_match is passed.
#
# Live audit: since the static read finds the flyby-only-charges-dv1 fix already in place (see
# sequencer.leg_feasible's docstring, written during the legality track's Phase 4 -- this was
# already identified and fixed before this track started), there is no code change to make here.
# To still produce the before/after comparison the acceptance test asks for, this script
# reconstructs the counterfactual "buggy" oracle (dv1 + dv2 charged on every leg, flyby included)
# entirely inside this file via monkey-patching sequencer.leg_feasible for the "before" run, then
# restores it and reruns for "after". No production code is touched.
import contextlib
import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

import catalog
import curriculum
import sequencer
from constants import sun_gravitational_parameter as mu

CATALOG_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'gtoc4_problem_data.txt')
DAY = 86400.0
LAUNCH_MJD = 58128.0   # the legality track's best-K launch epoch; same one used throughout


def static_audit():
    """Every leg_feasible call site and the require_velocity_match value it passes."""
    print("--- static audit: every leg_feasible call site ---")
    import inspect
    sites = []
    for module_name, module in (('sequencer', sequencer),):
        source = inspect.getsource(module)
        for lineno, line in enumerate(source.splitlines(), 1):
            if 'leg_feasible(' in line and 'def leg_feasible' not in line:
                sites.append((module_name, lineno, line.strip()))
    import train_sequencer  # noqa: E402  (path inserted above)
    source = inspect.getsource(train_sequencer)
    for lineno, line in enumerate(source.splitlines(), 1):
        if 'leg_feasible(' in line:
            sites.append(('train_sequencer', lineno, line.strip()))
    for module_name, lineno, line in sites:
        print(f"  {module_name}:{lineno}: {line}")
    print(f"call sites found: {len(sites)}")
    return sites


def _buggy_leg_feasible(candidate, mass, require_velocity_match, duty_cycle=sequencer.DUTY_CYCLE):
    """The hypothesized bug: dv1 + dv2 charged unconditionally, flyby or not."""
    cost = candidate['dv1'] + candidate['dv2']
    return cost < duty_cycle * curriculum.delta_v_budget(candidate['tof_days'] * DAY, mass)


@contextlib.contextmanager
def _patched_leg_feasible(fn):
    original = sequencer.leg_feasible
    sequencer.leg_feasible = fn
    try:
        yield
    finally:
        sequencer.leg_feasible = original


def run_greedy(label, pool, launch_epoch):
    start = time.time()
    mission, report = sequencer.greedy_tour(launch_epoch, pool, verbose=False)
    elapsed = time.time() - start
    flyby_legs = report['legs']
    tofs = [leg['tof_days'] for leg in flyby_legs]
    rendezvous = report['rendezvous']
    print(f"\n--- {label} (wall clock {elapsed:.0f} s) ---")
    print(f"  flyby legs planned         : {len(flyby_legs)}")
    print(f"  flyby leg ToFs (days)       : {tofs}")
    if tofs:
        print(f"  median flyby ToF            : {float(np.median(tofs)):.1f} d")
        print(f"  mean flyby ToF              : {float(np.mean(tofs)):.1f} d")
    else:
        print("  median flyby ToF            : n/a (no flyby legs accepted)")
    print(f"  scorable_after_flybys       : {report.get('scorable_after_flybys')}")
    print(f"  rendezvous achieved          : {rendezvous.get('achieved')}")
    if rendezvous.get('achieved'):
        print(f"  rendezvous duration_days    : {rendezvous.get('duration_days')}")
        print(f"  rendezvous mass_kg          : {rendezvous.get('mass_kg'):.3f}")
    return dict(label=label, tofs=tofs, report=report, mission=mission, elapsed=elapsed)


def main():
    static_audit()

    print("\n--- live audit: full 1436-asteroid pool, launch MJD", LAUNCH_MJD, "---")
    pool = catalog.parse_asteroids(CATALOG_PATH)
    launch_epoch = (LAUNCH_MJD - catalog.MJD_J2000) * DAY

    with _patched_leg_feasible(_buggy_leg_feasible):
        before = run_greedy("BEFORE (counterfactual: dv1+dv2 charged on every leg, flyby included)",
                             pool, launch_epoch)

    after = run_greedy("AFTER (current code: dv1-only on flyby legs, dv1+dv2 on rendezvous)",
                        pool, launch_epoch)

    print("\n--- comparison ---")
    before_median = float(np.median(before['tofs'])) if before['tofs'] else None
    after_median = float(np.median(after['tofs'])) if after['tofs'] else None
    print(f"  before: n={len(before['tofs'])} legs, median={before_median}")
    print(f"  after : n={len(after['tofs'])} legs, median={after_median}")

    if before_median is not None and after_median is not None:
        drop = 1.0 - after_median / before_median
        print(f"  median ToF change (after vs before): {drop*100:+.1f} %")
    elif before['tofs'] == [] and after['tofs'] != []:
        print("  before produced 0 feasible flyby legs; after produced "
              f"{len(after['tofs'])} -- charging dv2 on flybys made every candidate infeasible")

    assert True  # this script reports; it does not gate on a pass/fail threshold
    print("\nPhase 5 acceptance data collected")


if __name__ == '__main__':
    main()
