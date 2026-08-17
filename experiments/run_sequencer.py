# Phase 4.3 of the legality track: the greedy sequencing baseline.
#
# Cheapest-next by the leg oracle, each leg verified by flying it with the thrust-limited guidance
# before it is accepted, terminating on a rendezvous. The tour is written out through the Phase 3
# solution writer and handed to the independent checker; the reported J is the one the checker
# re-derives from the file, not the one the planner thought it had.
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

import catalog
import mission as mission_module
import sequencer
from constants import scape_velocity_max

CATALOG_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'gtoc4_problem_data.txt')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'legality_tours')
DAY = 86400.0
DEFAULT_LAUNCH_MJD = 58430.0   # the best launch epoch found by the Phase 2 scan

def emit(mission, pool, out_dir, label):
    """Writes the mission and re-verifies it from the file alone. Returns (path, checker result)."""
    if mission is None or mission.rendezvous_target is None:
        print("\n  no scorable mission: the tour does not end in a rendezvous, so it has no J")
        return None, None
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{label}_solution.txt')
    mission_module.write_solution(mission, path)
    result = mission_module.check_solution(path, ephemerides={a['name']: a for a in pool})
    print(f"\n  solution file        : {path}")
    print(f"  samples              : {len(mission.samples)}")
    print(f"  checker violations   : {result['violations'] if result['violations'] else 'none'}")
    print(f"  sequence             : {' -> '.join(mission.visited)}")
    print(f"  tau                  : {mission.duration/DAY:.2f} d = "
          f"{mission.duration/(365.25*DAY):.3f} yr")
    print(f"  J (from the file)    : {result['J']}")
    print(f"  K = m_f (from file)  : {result['K']:.3f} kg")
    return path, result

def run(launch_mjd=DEFAULT_LAUNCH_MJD, max_legs=20, reserve_days=sequencer.RENDEZVOUS_RESERVE_DAYS,
        top_k=6, buckets=sequencer.TOF_BUCKETS_DAYS, pool_size=None, prescreen=None,
        out_dir=RESULTS_DIR, label='greedy', verbose=True):
    pool = catalog.parse_asteroids(CATALOG_PATH)
    launch_epoch = (launch_mjd - catalog.MJD_J2000) * DAY
    if pool_size is not None:
        pool = restrict_pool(pool, launch_epoch, pool_size)
    print(f"greedy sequencer: launch MJD {launch_mjd}, pool {len(pool)} asteroids, "
          f"ToF buckets {tuple(buckets)} d, duty cycle {sequencer.DUTY_CYCLE}, "
          f"rendezvous reserve {reserve_days:.0f} d, top_k {top_k}")

    mission, report = sequencer.greedy_tour(launch_epoch, pool, max_legs=max_legs,
                                             reserve_days=reserve_days, top_k=top_k,
                                             buckets=buckets, prescreen=prescreen, verbose=verbose)
    print(f"  flyby chain stopped: {report['stopped_because']}")
    if not report['rendezvous'].get('achieved'):
        print(f"  rendezvous not achieved: {report['rendezvous'].get('reason')}")
    path, result = emit(mission, pool, out_dir, label)
    return dict(mission=mission, report=report, path=path, result=result, pool=pool)

def restrict_pool(pool, launch_epoch, size):
    """The `size` asteroids cheapest to reach from Earth at `launch_epoch`, scored as the smallest
    Lambert dv1 + dv2 over the rendezvous aim times of flight, with the 4 km/s launch v_inf
    credited. Used to make the RL sequencer's per-step ranking affordable, and applied identically
    to the greedy baseline it is compared against.

    Phase 2's closed-form screen (curriculum._delta_v_needed) was tried first and is not usable
    here: it ranks the one asteroid this launch epoch can actually rendezvous with, 2008TS10, at
    132nd, so a 120-asteroid pool built from it contains no rendezvous target at all and no tour
    over it is scorable. The Lambert screen ranks the same asteroid 1st. The closed-form estimate
    ignores inclination and treats phasing as a single angle, which is enough to say whether a
    candidate set is empty (what Phase 2 asked of it) and not enough to order one."""
    state = sequencer.launch_state(launch_epoch)
    pool_index = list(range(len(pool)))
    best = {}
    for aim_tof in sequencer.RENDEZVOUS_AIM_TOFS_DAYS:
        for candidate in sequencer.leg_candidates(state, launch_epoch, pool, pool_index, aim_tof,
                                                   set(), scape_velocity_max):
            cost = candidate['dv1'] + candidate['dv2']
            if candidate['name'] not in best or cost < best[candidate['name']]:
                best[candidate['name']] = cost
    keep = set(sorted(best, key=best.get)[:size])
    return [ast for ast in pool if ast['name'] in keep]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch-mjd', type=float, default=DEFAULT_LAUNCH_MJD)
    parser.add_argument('--max-legs', type=int, default=20)
    parser.add_argument('--reserve-days', type=float, default=sequencer.RENDEZVOUS_RESERVE_DAYS)
    parser.add_argument('--top-k', type=int, default=6)
    parser.add_argument('--pool-size', type=int, default=None)
    parser.add_argument('--prescreen', type=int, default=None)
    parser.add_argument('--buckets', type=int, nargs='+', default=list(sequencer.TOF_BUCKETS_DAYS))
    parser.add_argument('--label', type=str, default='greedy')
    args = parser.parse_args()
    run(launch_mjd=args.launch_mjd, max_legs=args.max_legs, reserve_days=args.reserve_days,
        top_k=args.top_k, buckets=tuple(args.buckets), pool_size=args.pool_size,
        prescreen=args.prescreen, label=args.label)
