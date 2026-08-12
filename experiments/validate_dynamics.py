# validates the hand-written RK4 dynamics against Tudat: same initial state, same constant
# thrust vector, 300 days. Also checks the RK4 propagator is fast enough for RL training.
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import dynamics
import catalog
import verify_tudat
from constants import sun_gravitational_parameter as mu, Isp_engine, spacecraft_wet_mass, launch_interval

DURATION = 300 * 86400.0  # s
THRUST_DIRECTION = np.array([1.0, 0.0, 0.0])
THRUST_MAGNITUDE = 0.135  # N

def validate_propagation():
    start_epoch = launch_interval[0]

    # RK4: same Earth-departure initial state as verify_tudat.run_propagation, ~0.1 day steps
    earth_state = catalog.earth_initial_state(start_epoch, mu)
    excess_velocity = np.array([4.0e3, 0.0, 0.0])
    initial_state = np.concatenate([earth_state + np.hstack((np.zeros(3), excess_velocity)),
                                     [spacecraft_wet_mass]])
    thrust_force = THRUST_MAGNITUDE * THRUST_DIRECTION

    # Tudat, same constant thrust vector. Its adaptive-step integrator overshoots the requested
    # end epoch, so we compare against the actual saved epoch closest to start_epoch + DURATION,
    # not the nominal duration itself.
    results = verify_tudat.run_propagation(
        n_asteroids=1, duration=DURATION,
        thrust_direction_function=lambda t: THRUST_DIRECTION.tolist(),
        thrust_magnitude_function=lambda t: THRUST_MAGNITUDE,
    )
    epochs = results.state_history.keys()
    matched_epoch = min(epochs, key=lambda e: abs((e - start_epoch) - DURATION))
    duration_matched = matched_epoch - start_epoch
    final_state_tudat = results.state_history[matched_epoch][:6]
    final_mass_tudat = results.dependent_variable_history[matched_epoch][0]

    n_substeps = round(duration_matched / 8640.0)
    final_rk4 = dynamics.propagate(initial_state, thrust_force, duration_matched, n_substeps, mu, Isp_engine)

    pos_err = np.linalg.norm(final_rk4[:3] - final_state_tudat[:3])
    vel_err = np.linalg.norm(final_rk4[3:6] - final_state_tudat[3:6])
    mass_err = abs(final_rk4[6] - final_mass_tudat)

    print(f"RK4 vs Tudat after {duration_matched / 86400:.1f} days, constant thrust:")
    print(f"  position error: {pos_err:.3f} m")
    print(f"  velocity error: {vel_err * 1e3:.3f} mm/s")
    print(f"  mass error:     {mass_err * 1e3:.3f} g")
    # residual position error (~hundreds of m) doesn't shrink with finer RK4 steps -- it isn't
    # truncation error. About half of it is explained by a mu mismatch: Tudat's SPICE-derived
    # solar GM differs from the problem-statement constant (constants.sun_gravitational_parameter)
    # by ~2e-10 relative; using Tudat's exact value roughly halves the error (760 m -> 358 m).
    # The rest sits within the noise floor of comparing two independent integrators -- pinning
    # Tudat's Sun as its own frame origin (ruling out SSB barycentric motion) made no difference.
    assert pos_err < 2000.0, "position error exceeds 2 km"
    assert vel_err < 1.0, "velocity error exceeds 1 m/s"
    assert mass_err < 1.0, "mass error exceeds 1 g"

def check_speed():
    state = np.concatenate([catalog.earth_initial_state(launch_interval[0], mu), [spacecraft_wet_mass]])
    thrust_force = THRUST_MAGNITUDE * THRUST_DIRECTION
    control_interval = 86400.0  # s, 1 day
    n_substeps = 10

    n_steps = 1000
    t0 = time.perf_counter()
    for _ in range(n_steps):
        state = dynamics.propagate(state, thrust_force, control_interval, n_substeps, mu, Isp_engine)
    elapsed = time.perf_counter() - t0

    ms_per_step = elapsed / n_steps * 1e3
    print(f"\n{n_steps} control steps ({n_substeps} RK4 substeps each): {ms_per_step:.4f} ms/step")
    assert ms_per_step < 1.0, "RK4 propagation is too slow for RL training"

def check_execution_time():
    """RK4 vs Tudat wall-clock time for one RL control step (1 day), with a ~50-asteroid target
    pool matching the curriculum's randomised-target stage (WP5). Tudat has no cheap way to
    integrate a single control step from an arbitrary state with a new thrust command -- using it
    inside the RL loop means rebuilding bodies/accelerations/integrator from scratch every step,
    so that reconstruction cost is included here; it's the real cost Tudat-in-the-loop would pay."""
    start_epoch = launch_interval[0]
    earth_state = catalog.earth_initial_state(start_epoch, mu)
    excess_velocity = np.array([4.0e3, 0.0, 0.0])
    initial_state = np.concatenate([earth_state + np.hstack((np.zeros(3), excess_velocity)),
                                     [spacecraft_wet_mass]])
    thrust_force = THRUST_MAGNITUDE * THRUST_DIRECTION
    control_interval = 86400.0  # s, 1 day

    n_reps = 100
    t0 = time.perf_counter()
    for _ in range(n_reps):
        dynamics.propagate(initial_state, thrust_force, control_interval, 10, mu, Isp_engine)
    rk4_time = (time.perf_counter() - t0) / n_reps

    n_reps = 20
    t0 = time.perf_counter()
    for _ in range(n_reps):
        verify_tudat.run_propagation(
            n_asteroids=50, duration=control_interval,
            thrust_direction_function=lambda t: THRUST_DIRECTION.tolist(),
            thrust_magnitude_function=lambda t: THRUST_MAGNITUDE,
        )
    tudat_time = (time.perf_counter() - t0) / n_reps

    print(f"\nExecution time, one control step (1 day, 50-asteroid pool):")
    print(f"  RK4 (warm, no setup):            {rk4_time * 1e3:.4f} ms")
    print(f"  Tudat (fresh simulator per call): {tudat_time * 1e3:.4f} ms")
    print(f"  speedup: {tudat_time / rk4_time:.1f}x")

if __name__ == "__main__":
    validate_propagation()
    check_speed()
    check_execution_time()
