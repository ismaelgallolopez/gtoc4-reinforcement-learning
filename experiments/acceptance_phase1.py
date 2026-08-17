# Phase 1 acceptance tests for the legality track: analytic closest approach (1a), rocket-equation
# delta-v budget (1b), constants transcription audit (1c). Run with no arguments; prints every
# measured number so the output can be pasted verbatim into NOTES_legality_track.md.
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

import catalog
import constants
import curriculum
import dynamics
from gtoc4_env import closest_approach_on_segments

CATALOG_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'gtoc4_problem_data.txt')

def test_closest_approach():
    """Rectilinear pass with a known 500 km miss distance: no gravity, no thrust, target fixed, so
    the relative motion is exactly dr0 + dv*t and the true answer is known analytically. The
    encounter is placed halfway between two samples, the worst case for a grid minimum."""
    miss_km = 500.0
    rel_speed = 5000.0                  # m/s, typical heliocentric relative speed
    n_substeps = 10
    control_interval = 86400.0
    dt = control_interval / n_substeps  # 8640 s = 2.4 h, the stage-1/flyby sampling
    t_encounter = 5.5 * dt              # exactly midway between sample 5 and sample 6

    times = np.arange(n_substeps + 1) * dt
    rel_vel = np.tile([rel_speed, 0.0, 0.0], (n_substeps + 1, 1))
    rel_pos = np.stack([rel_speed * (times - t_encounter),
                        np.full_like(times, miss_km * 1e3),
                        np.zeros_like(times)], axis=1)

    grid_min = np.min(np.linalg.norm(rel_pos, axis=1))
    analytic, analytic_epoch = closest_approach_on_segments(rel_pos, rel_vel, times)
    error = abs(analytic - miss_km * 1e3) / (miss_km * 1e3)

    print("--- 1a: analytic closest approach (rectilinear pass, true miss = 500.0 km) ---")
    print(f"sample spacing            : {dt:.0f} s ({dt/3600:.1f} h)")
    print(f"relative speed            : {rel_speed:.0f} m/s")
    print(f"old grid np.min           : {grid_min/1e3:.1f} km")
    print(f"new analytic minimum      : {analytic/1e3:.4f} km  (error {100*error:.4f} %)")
    print(f"true encounter epoch      : {t_encounter:.1f} s")
    print(f"recovered encounter epoch : {analytic_epoch:.1f} s")
    print(f"grid overestimates by     : {grid_min/analytic:.1f} x")
    assert error < 0.05, f"analytic closest approach off by {100*error:.2f} %, tolerance 5 %"
    print("PASS: recovered within 5 %\n")

def test_delta_v_budget():
    """The rocket-equation budget must hit the three reference points and then saturate."""
    year = 365.25 * 86400.0
    mdot = constants.thrust_max / (constants.Isp_engine * dynamics.g0)
    t_exhaust = constants.spacecraft_propellant_mass / mdot

    cases = [("600 days", 600 * 86400.0, 5.08), ("5 years", 5 * year, 19.39),
             ("7 years", 7 * year, 32.32), ("10 years", 10 * year, 32.32)]

    print("--- 1b: rocket-equation delta-v budget ---")
    print(f"mdot                      : {mdot:.6e} kg/s")
    print(f"propellant exhausted at   : {t_exhaust/86400:.1f} days = {t_exhaust/year:.3f} years")
    print(f"{'window':<10} {'old a_max*t (km/s)':>20} {'new rocket eq (km/s)':>22} {'expected':>10}")
    ok = True
    a_max = constants.thrust_max / constants.spacecraft_wet_mass
    for label, t, expected in cases:
        old = a_max * t / 1e3
        new = curriculum.delta_v_budget(t) / 1e3
        print(f"{label:<10} {old:>20.2f} {new:>22.2f} {expected:>10.2f}")
        if abs(new - expected) > 0.01:
            ok = False
    assert ok, "delta-v budget does not match the expected reference values"
    print("PASS: 5.08 / 19.39 km/s reproduced and saturation at 32.32 km/s confirmed\n")

def test_reachable_counts():
    """Reachable-asteroid counts at a 1.5x margin, old budget vs new, over the whole catalog."""
    year = 365.25 * 86400.0
    windows = [("600 days", 600 * 86400.0), ("5 years", 5 * year),
               ("7 years", 7 * year), ("10 years", 10 * year)]
    a_max = constants.thrust_max / constants.spacecraft_wet_mass
    asteroids = catalog.parse_asteroids(CATALOG_PATH)

    print("--- 1b: reachable-asteroid counts at a 1.5x margin (whole catalog, "
          f"{len(asteroids)} asteroids) ---")
    print(f"{'window':<10} {'old budget':>12} {'old count':>10} {'new budget':>12} {'new count':>10}")
    for label, t in windows:
        dv_needed = np.array([curriculum._delta_v_needed(ast, t) for ast in asteroids])
        old_budget, new_budget = a_max * t, curriculum.delta_v_budget(t)
        old_count = int(np.sum(dv_needed < old_budget / 1.5))
        new_count = int(np.sum(dv_needed < new_budget / 1.5))
        print(f"{label:<10} {old_budget/1e3:>11.2f}k {old_count:>10} "
              f"{new_budget/1e3:>11.2f}k {new_count:>10}")
    print(f"cheapest target in catalog: {np.min(dv_needed)/1e3:.2f} km/s\n")

def test_constants_audit():
    """Reports the transcribed Earth elements and confirms the epoch comment is consistent."""
    print("--- 1c: constants transcription audit ---")
    print(f"eccentricity_earth   : {constants.eccentricity_earth!r}  (was 1.671681164160e-02)")
    print(f"mean_anomaly_earth   : {constants.mean_anomaly_earth!r} deg  (was 257.606837077535)")
    print(f"epoch                : {constants.epoch} interpreted as MJD")
    print(f"catalog.MJD_J2000    : {catalog.MJD_J2000}")
    print(f"mjd_to_et(epoch)     : {catalog.mjd_to_et(constants.epoch):.1f} s after J2000 "
          f"= {catalog.mjd_to_et(constants.epoch)/(365.25*86400):.3f} yr")
    assert abs(catalog.mjd_to_et(constants.epoch) / (365.25 * 86400.0) - 6.71) < 0.05, \
        "MJD 54000 should be ~6.7 years after J2000; the epoch is not being read as an MJD"
    print("PASS: MJD 54000 -> 2006-09-22, consistent with MJD_J2000 = 51544.5 (i.e. plain MJD, "
          "not MJD2000)\n")

    # positional effect of the two corrected digits, at the mission start epoch
    state = catalog.earth_initial_state(curriculum.START_EPOCH, constants.sun_gravitational_parameter)
    print(f"Earth state at START_EPOCH with corrected elements:")
    print(f"  r = {state[:3]/1.495978707e11} AU")
    print(f"  v = {state[3:6]/1e3} km/s\n")

if __name__ == '__main__':
    test_closest_approach()
    test_delta_v_budget()
    test_reachable_counts()
    test_constants_audit()
    print("all Phase 1 acceptance tests passed")
