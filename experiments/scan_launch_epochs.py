# Phase 4 addendum: the greedy tour is run at several legal launch epochs, because Phase 2 showed
# the reachable set varies by a factor of ~1.5 across the window and the tour's whole structure
# hangs on which target the launch v_inf can be aimed at. One line per epoch: how many flyby legs
# the greedy chains, how many of them it can still stop after, and the resulting J and K.
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

import catalog
import run_sequencer
import sequencer

EPOCHS_MJD = (57224, 57727, 58028, 58128, 58430, 58631, 59133, 59635, 60238, 60338)

def main(pool_size=120):
    rows = []
    for launch_mjd in EPOCHS_MJD:
        out = run_sequencer.run(launch_mjd=launch_mjd, pool_size=pool_size, reserve_days=600.0,
                                 label=f'greedy_mjd{launch_mjd}', verbose=False)
        report, result = out['report'], out['result']
        rows.append(dict(launch_mjd=launch_mjd, flybys_chained=len(report['legs']),
                          scorable_after=report.get('scorable_after_flybys'),
                          J=None if result is None else result['J'],
                          K=None if result is None else result['K'],
                          rendezvous=report['rendezvous'].get('name')))
        print(f"\n=== MJD {launch_mjd}: chained {rows[-1]['flybys_chained']} flybys, "
              f"scorable after {rows[-1]['scorable_after']}, J = {rows[-1]['J']}, "
              f"K = {rows[-1]['K']}\n")

    print(f"\n{'launch MJD':>11}{'flybys chained':>16}{'scorable after':>16}{'J':>4}"
          f"{'K (kg)':>11}  rendezvous")
    for row in rows:
        mass = '-' if row['K'] is None else f"{row['K']:.1f}"
        print(f"{row['launch_mjd']:>11}{row['flybys_chained']:>16}"
              f"{str(row['scorable_after']):>16}{str(row['J']):>4}"
              f"{mass:>11}  {row['rendezvous']}")
    return rows

if __name__ == '__main__':
    main()
