# sanity checks for Gtoc4ControlEnv: not pytest, just asserts on physical/numerical invariants.
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import dynamics
import catalog
from gtoc4_env import Gtoc4ControlEnv, AU, V_REF
from constants import (sun_gravitational_parameter as mu, Isp_engine, thrust_max,
                        spacecraft_dry_mass, spacecraft_wet_mass, spacecraft_propellant_mass,
                        launch_interval)

START_EPOCH = launch_interval[0]

def _initial_state():
    earth_state = catalog.earth_initial_state(START_EPOCH, mu)
    excess_velocity = np.array([4.0e3, 0.0, 0.0])
    return np.concatenate([earth_state + np.hstack((np.zeros(3), excess_velocity)), [spacecraft_wet_mass]])

def _target():
    return catalog.parse_asteroids(
        os.path.join(os.path.dirname(__file__), '..', 'data', 'gtoc4_problem_data.txt'), n_asteroids=1)[0]

ZERO_ACTION = np.array([1.0, 0.0, 0.0, -1.0])  # direction set, throttle -1 -> 0 thrust
FULL_THRUST_ACTION = np.array([1.0, 0.0, 0.0, 1.0])  # fixed direction, throttle 1 -> thrust_max

def check_zero_thrust_is_keplerian():
    env = Gtoc4ControlEnv(_initial_state(), _target(), START_EPOCH, time_limit=300 * 86400.0)
    env.reset()

    r0, v0 = env.state[:3], env.state[3:6]
    energy0 = np.linalg.norm(v0)**2 / 2 - mu / np.linalg.norm(r0)
    h0 = np.linalg.norm(np.cross(r0, v0))

    n_steps = 300
    for _ in range(n_steps):
        _, _, terminated, truncated, _ = env.step(ZERO_ACTION)
        if terminated or truncated:
            break

    r1, v1 = env.state[:3], env.state[3:6]
    energy1 = np.linalg.norm(v1)**2 / 2 - mu / np.linalg.norm(r1)
    h1 = np.linalg.norm(np.cross(r1, v1))

    energy_drift = abs(energy1 - energy0) / abs(energy0)
    h_drift = abs(h1 - h0) / h0
    print(f"zero-thrust coast, {n_steps} days: energy drift {energy_drift:.2e}, "
          f"angular momentum drift {h_drift:.2e}, mass unchanged: {env.state[6] == spacecraft_wet_mass}")
    assert energy_drift < 1e-8, "specific energy not conserved under zero thrust"
    assert h_drift < 1e-8, "angular momentum not conserved under zero thrust"
    assert env.state[6] == spacecraft_wet_mass, "mass changed under zero thrust"

def check_full_thrust_exhausts_propellant():
    burn_time_analytic = spacecraft_propellant_mass / (thrust_max / (Isp_engine * dynamics.g0))
    time_limit = burn_time_analytic * 1.1

    env = Gtoc4ControlEnv(_initial_state(), _target(), START_EPOCH, time_limit=time_limit,
                           control_interval=86400.0)
    env.reset()
    terminated = truncated = False
    n_steps = 0
    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(FULL_THRUST_ACTION)
        n_steps += 1

    burn_time_sim = env.elapsed_time
    rel_err = abs(burn_time_sim - burn_time_analytic) / burn_time_analytic
    print(f"full-thrust burn time: analytic {burn_time_analytic/86400:.2f} d, "
          f"simulated {burn_time_sim/86400:.2f} d, rel. error {rel_err:.2e}")
    assert terminated and not truncated, "propellant exhaustion should terminate, not truncate"
    assert info['mass'] == spacecraft_dry_mass, "mass should clamp exactly to dry mass"
    assert rel_err < 0.01, "simulated burn time too far from the analytic prediction"

def check_random_actions():
    n_episodes = 1000
    time_limit = 50 * 86400.0  # short episodes, still exercises many random-action transitions
    rng = np.random.default_rng(0)
    target = _target()
    env = Gtoc4ControlEnv(_initial_state(), target, START_EPOCH, time_limit=time_limit)

    max_abs_obs = 0.0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            action = rng.uniform(-1.0, 1.0, size=4).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            assert np.all(np.isfinite(obs)), "non-finite observation"
            assert np.isfinite(reward), "non-finite reward"
            assert info['mass'] >= spacecraft_dry_mass, "mass went below dry mass"
            max_abs_obs = max(max_abs_obs, np.max(np.abs(obs)))

    print(f"{n_episodes} random-action episodes: max |obs| = {max_abs_obs:.3f}")
    assert max_abs_obs < 10.0, "observation exceeded the |obs| < 10 sanity bound"

if __name__ == "__main__":
    check_zero_thrust_is_keplerian()
    check_full_thrust_exhausts_propellant()
    check_random_actions()
    print("\nall sanity checks passed")
