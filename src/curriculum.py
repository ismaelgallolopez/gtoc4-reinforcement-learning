# curriculum stage definitions: two synthetic near-Earth targets of increasing difficulty,
# then real GTOC4 asteroids sampled from a target pool.
import numpy as np
from tudatpy import constants as tudat_constants

import catalog
from gtoc4_env import Gtoc4ControlEnv
from constants import (a_earth, eccentricity_earth, inclination_earth, lan_earth,
                        arg_periapsis_earth, mean_anomaly_earth, epoch,
                        sun_gravitational_parameter as mu, spacecraft_wet_mass,
                        accuracy_position, accuracy_velocity, launch_interval)

AU = tudat_constants.ASTRONOMICAL_UNIT
START_EPOCH = launch_interval[0]

def _earth_like_target(delta_a, delta_i, delta_M):
    """A synthetic target on a near-Earth orbit, offset in semi-major axis, inclination and phase
    at the mission start epoch. M0 is anchored at START_EPOCH (not at the far-past reference
    epoch used for mean_anomaly_earth): the target's semi-major axis differs from Earth's, so its
    mean motion does too, and propagating a shared M0 from the ~8-year-old reference epoch would
    turn even a tiny delta_a into tens of degrees of accumulated phase drift by the mission start
    -- nowhere near the intended "small phase offset" at the moment the episode actually begins."""
    n_earth = np.sqrt(mu / a_earth**3)
    mean_anomaly_earth_at_start = (np.deg2rad(mean_anomaly_earth) +
                                    n_earth * (START_EPOCH - catalog.mjd_to_et(epoch)))
    return {
        'name': 'synthetic',
        'epoch': START_EPOCH,
        'a': a_earth + delta_a,
        'e': eccentricity_earth,
        'i': np.deg2rad(inclination_earth) + delta_i,
        'lan': np.deg2rad(lan_earth),
        'omega': np.deg2rad(arg_periapsis_earth),
        'M0': mean_anomaly_earth_at_start + delta_M,
    }

STAGES = {
    # phase offset (delta_M) is kept small for both stages: at ~30 km/s orbital speed, even a few
    # degrees of phase mismatch costs hundreds to thousands of m/s (v * delta_theta, from the
    # velocity *direction* difference between two points on similar orbits) -- comparable to or
    # larger than the delta_v spent correcting delta_a. The plan's own stage spec only names
    # delta_a/delta_i as the difficulty drivers, so phase is kept minor rather than dominant here.
    1: dict(
        target=_earth_like_target(delta_a=0.02 * AU, delta_i=0.0, delta_M=np.deg2rad(1.0)),
        position_tolerance=1e5 * 1e3,  # 1e5 km
        velocity_tolerance=50.0,       # m/s
        time_limit=400 * 86400.0,
    ),
    2: dict(
        target=_earth_like_target(delta_a=0.1 * AU, delta_i=np.deg2rad(2.0), delta_M=0.0),
        position_tolerance=1e4 * 1e3,  # 1e4 km
        velocity_tolerance=10.0,       # m/s
        time_limit=400 * 86400.0,
    ),
}

def initial_state():
    """Spacecraft starts exactly on Earth's heliocentric orbit (no departure kick). The 4 km/s
    dummy excess velocity used elsewhere (verify_tudat.py, validate_dynamics.py) was only ever a
    placeholder direction for validating dynamics -- adding it here would put the spacecraft on an
    a~0.8 AU, fairly eccentric orbit, hundreds of millions of km from a "near-identical" stage-1
    target, defeating the point of an easy first curriculum stage."""
    earth_state = catalog.earth_initial_state(START_EPOCH, mu)
    return np.concatenate([earth_state, [spacecraft_wet_mass]])

def sample_asteroid_target(catalog_path, rng, pool_size=50):
    asteroids = catalog.parse_asteroids(catalog_path, n_asteroids=pool_size)
    return asteroids[rng.integers(len(asteroids))]

def make_env(stage, catalog_path=None, rng=None):
    """Builds a Gtoc4ControlEnv for curriculum stage 1, 2 (fixed synthetic targets) or 3
    (a GTOC4 asteroid sampled from the target pool, with a randomised time window)."""
    if stage in STAGES:
        cfg = STAGES[stage]
        target, position_tolerance, velocity_tolerance, time_limit = (
            cfg['target'], cfg['position_tolerance'], cfg['velocity_tolerance'], cfg['time_limit'])
    elif stage == 3:
        rng = rng or np.random.default_rng()
        target = sample_asteroid_target(catalog_path, rng)
        position_tolerance = accuracy_position
        velocity_tolerance = accuracy_velocity
        time_limit = rng.uniform(300, 600) * 86400.0
    else:
        raise ValueError(f"unknown curriculum stage: {stage}")

    return Gtoc4ControlEnv(initial_state(), target, START_EPOCH, time_limit,
                            position_tolerance=position_tolerance, velocity_tolerance=velocity_tolerance)
