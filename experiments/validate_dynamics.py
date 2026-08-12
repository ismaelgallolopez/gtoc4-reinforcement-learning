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
    # this used to show ~760 m / 0.16 mm/s against Tudat's old default tolerance (1e-10).
    # Traced it down: ~640 m of that was Tudat's own integration not being converged at 1e-10 for
    # this eccentric dummy departure orbit (confirmed by bridging a tol=1e-13 run forward with this
    # same RK4 propagator and comparing to the raw tol=1e-10 output -- they differed by ~640 m,
    # essentially the whole gap). The remaining ~172 m is a real, small effect: the problem
    # statement's solar GM (constants.sun_gravitational_parameter) differs from Tudat's SPICE value
    # by ~2e-10 relative, which is enough to shift this near-resonant orbit's phase measurably over
    # 300 days. verify_tudat.run_propagation now defaults to tolerance=1e-13. With that plus
    # Tudat's exact mu, RK4 agrees with Tudat to 1.9 m / 0.0003 mm/s. Floating-point rounding was
    # checked separately (check_precision, below) and ruled out at the millimetre level.
    assert pos_err < 300.0, "position error exceeds 300 m"
    assert vel_err < 0.1, "velocity error exceeds 0.1 m/s"
    assert mass_err < 1.0, "mass error exceeds 1 g"

def validate_propagation_matched_mu():
    """Same comparison as validate_propagation, but Tudat's Sun is forced to use our own mu
    (constants.sun_gravitational_parameter) instead of its default SPICE value. This isolates
    whether the RK4 *implementation* is correct, with the mu-source difference removed entirely."""
    start_epoch = launch_interval[0]

    earth_state = catalog.earth_initial_state(start_epoch, mu)
    excess_velocity = np.array([4.0e3, 0.0, 0.0])
    initial_state = np.concatenate([earth_state + np.hstack((np.zeros(3), excess_velocity)),
                                     [spacecraft_wet_mass]])
    thrust_force = THRUST_MAGNITUDE * THRUST_DIRECTION

    results = verify_tudat.run_propagation(
        n_asteroids=1, duration=DURATION, sun_gravitational_parameter_override=mu,
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

    print(f"\nRK4 vs Tudat (matched mu) after {duration_matched / 86400:.1f} days, constant thrust:")
    print(f"  position error: {pos_err:.3f} m")
    print(f"  velocity error: {vel_err * 1e3:.3f} mm/s")
    print(f"  mass error:     {mass_err * 1e3:.3f} g")
    # with the mu-source difference removed, this is a check on the RK4 implementation alone
    assert pos_err < 10.0, "position error exceeds 10 m with mu matched -- suggests an RK4 bug"
    assert vel_err < 0.01, "velocity error exceeds 1 cm/s with mu matched -- suggests an RK4 bug"

def check_precision():
    """Confirms the RK4 vs Tudat residual isn't floating-point rounding: reruns one propagation at
    extended (80-bit) precision and checks the shift is negligible next to the residuals above."""
    start_epoch = launch_interval[0]
    earth_state = catalog.earth_initial_state(start_epoch, mu)
    excess_velocity = np.array([4.0e3, 0.0, 0.0])
    initial_state = np.concatenate([earth_state + np.hstack((np.zeros(3), excess_velocity)),
                                     [spacecraft_wet_mass]])
    thrust_force = THRUST_MAGNITUDE * THRUST_DIRECTION
    n_substeps = round(DURATION / 8640.0)

    final_f64 = dynamics.propagate(initial_state, thrust_force, DURATION, n_substeps, mu, Isp_engine)
    final_ld = dynamics.propagate(initial_state.astype(np.longdouble), thrust_force.astype(np.longdouble),
                                   np.longdouble(DURATION), n_substeps, np.longdouble(mu), np.longdouble(Isp_engine))

    shift = np.linalg.norm(np.array(final_ld[:3], dtype=np.float64) - final_f64[:3])
    print(f"\nposition shift from float64 -> extended precision: {shift * 1e3:.3f} mm")
    assert shift < 1.0, "precision-driven shift is unexpectedly large -- rounding may matter after all"

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
    validate_propagation_matched_mu()
    check_precision()
    check_speed()
    check_execution_time()
