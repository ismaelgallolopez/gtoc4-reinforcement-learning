# WP3.3: verify stage 1 (and 2) are within the spacecraft's delta-v budget before spending days
# tuning PPO against them, and produce the baseline table for coast / tangential thrust.
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import curriculum
import baselines
import catalog
from constants import sun_gravitational_parameter as mu, thrust_max, spacecraft_wet_mass, a_earth

CATALOG_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'gtoc4_problem_data.txt')

def check_delta_v_budget():
    """Delta-v needed vs. delta-v available from continuous max thrust over the stage's time
    window, both evaluated 'by hand' (closed-form, not simulated). Two additive contributions:
    - energy/shape: da = 2 a^2 v dv / mu, tangential burn on a near-circular orbit
    - phasing: two points on similar orbits separated by delta_theta of true anomaly have velocity
      *vectors* differing mainly in direction, not magnitude -- chord length ~ v * delta_theta for
      small angles. This dominates unless the phase offset is kept small, since v ~ 30 km/s means
      even a few degrees costs hundreds to thousands of m/s.
    Uses the *initial* (heaviest) mass, so it's a conservative estimate of available delta_v --
    actual acceleration only grows as propellant burns."""
    v_circ = np.sqrt(mu / a_earth)
    a_max = thrust_max / spacecraft_wet_mass

    for stage in (1, 2):
        cfg = curriculum.STAGES[stage]
        target = cfg['target']
        delta_a = target['a'] - a_earth
        delta_v_energy = abs(delta_a) * np.sqrt(mu) / (2 * a_earth**1.5)

        earth_at_start = catalog.earth_initial_state(curriculum.START_EPOCH, mu)
        target_at_start = catalog.target_states([target], curriculum.START_EPOCH, mu)[0]
        cos_angle = np.dot(earth_at_start[:3], target_at_start[:3]) / (
            np.linalg.norm(earth_at_start[:3]) * np.linalg.norm(target_at_start[:3]))
        delta_theta = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        delta_v_phasing = v_circ * delta_theta

        delta_v_needed = delta_v_energy + delta_v_phasing
        delta_v_budget = a_max * cfg['time_limit']
        margin = delta_v_budget / delta_v_needed
        print(f"stage {stage}: delta_v needed ~{delta_v_needed:.1f} m/s "
              f"(energy {delta_v_energy:.1f} + phasing {delta_v_phasing:.1f}, "
              f"delta_theta={np.degrees(delta_theta):.2f} deg), "
              f"budget ~{delta_v_budget:.1f} m/s over {cfg['time_limit']/86400:.0f} d "
              f"(margin {margin:.1f}x)")
        # stage 1 is the Day-7 PPO checkpoint -- it needs to be genuinely easy, so demand a
        # comfortable margin. Stage 2 is meant to be meaningfully harder; just confirm it's
        # reachable in principle (margin > 1), not that it's easy.
        required_margin = 3.0 if stage == 1 else 1.0
        assert margin > required_margin, f"stage {stage} may not be reachable within its time window"

def baseline_table():
    rng = np.random.default_rng(0)
    print(f"\n{'stage':<7}{'policy':<12}{'|delta_r| (km)':<16}{'|delta_v| (m/s)':<16}{'mass (kg)':<11}success")
    for stage in (1, 2, 3):
        env = curriculum.make_env(stage, catalog_path=CATALOG_PATH, rng=rng)
        for name, policy in [('coast', baselines.coast), ('tangential', baselines.tangential_thrust)]:
            info = policy(env)
            dr_km = np.linalg.norm(info['delta_r']) / 1e3
            dv = np.linalg.norm(info['delta_v'])
            print(f"{stage:<7}{name:<12}{dr_km:<16.1f}{dv:<16.2f}{info['mass']:<11.1f}{info['success']}")

if __name__ == "__main__":
    check_delta_v_budget()
    baseline_table()
