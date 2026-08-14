# reference Tudat propagation, used only to validate the hand-written RK4 dynamics against
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from tudatpy import constants, util
from tudatpy.astro import element_conversion
from tudatpy.dynamics import environment_setup, propagation_setup, simulator
from tudatpy.data import save2txt
from tudatpy.interface import spice

from constants import *
from catalog import parse_asteroids, mjd_to_et

spice.load_standard_kernels()

ASTEROIDS_FILEPATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'gtoc4_problem_data.txt')
RESULTS_DIRECTORY = os.path.join(os.path.dirname(__file__), '..', 'results')

def build_bodies(asteroids: list[dict], sun_gravitational_parameter_override=None):
    """sun_gravitational_parameter_override lets a caller replace Tudat's default SPICE-derived
    solar GM with a custom value (e.g. constants.sun_gravitational_parameter), to compare RK4
    against Tudat with the exact same gravity model instead of two independently-sourced GMs."""
    body_settings = environment_setup.get_default_body_settings(['Sun'])
    if sun_gravitational_parameter_override is not None:
        body_settings.get('Sun').gravity_field_settings = environment_setup.gravity_field.central(
            sun_gravitational_parameter_override)

    for ast in asteroids:
        name = f"ast_{ast['name']}"
        body_settings.add_empty_settings(name)
        true_anomaly = element_conversion.mean_to_true_anomaly(ast['e'], ast['M0'])
        body_settings.get(name).ephemeris_settings = (
            environment_setup.ephemeris.keplerian(
                initial_keplerian_state=np.array([
                    ast['a'], ast['e'], ast['i'],
                    ast['omega'], ast['lan'], true_anomaly,
                ]),
                initial_state_epoch=ast['epoch'],
                central_body_gravitational_parameter=sun_gravitational_parameter,
                frame_origin='Sun',
                frame_orientation='ECLIPJ2000',
            )
        )

    body_settings.add_empty_settings('spacecraft')
    body_settings.get('spacecraft').constant_mass = spacecraft_wet_mass

    return environment_setup.create_system_of_bodies(body_settings)

def default_thrust_direction(time):
    return [1.0, 0.0, 0.0]  # dummy, pass a real one in when validating a learned/baseline profile

def default_thrust_magnitude(time):
    return thrust_max

def run_propagation(n_asteroids=None, thrust_direction_function=default_thrust_direction,
                     thrust_magnitude_function=default_thrust_magnitude, duration=None,
                     tolerance=1e-13, sun_gravitational_parameter_override=None, plot=False, save=False):
    asteroids = parse_asteroids(ASTEROIDS_FILEPATH, n_asteroids)
    bodies = build_bodies(asteroids, sun_gravitational_parameter_override)
    names = [f"ast_{a['name']}" for a in asteroids]

    bodies_to_propagate = ['spacecraft'] + names
    central_bodies = ['Sun'] * len(bodies_to_propagate)

    start_epoch = launch_interval[0]
    end_epoch = start_epoch + (duration if duration is not None else time_mission_max)

    # thrust and rotation models
    thrust_magnitude_settings = propagation_setup.thrust.custom_thrust_magnitude_fixed_isp(
        thrust_magnitude_function=thrust_magnitude_function,
        specific_impulse=Isp_engine
    )
    environment_setup.add_engine_model('spacecraft', 'LowThrustEngine', thrust_magnitude_settings, bodies)
    environment_setup.add_rotation_model(
        bodies, 'spacecraft',
        environment_setup.rotation_model.custom_inertial_direction_based(
            thrust_direction_function, 'ECLIPJ2000', 'VehicleFixed'
        )
    )

    # acceleration models
    acceleration_settings = {
        'spacecraft': dict(
            Sun=[propagation_setup.acceleration.point_mass_gravity()],
            spacecraft=[propagation_setup.acceleration.thrust_from_all_engines()],
        ),
        **{name: dict(Sun=[propagation_setup.acceleration.point_mass_gravity()]) for name in names},
    }
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings, bodies_to_propagate, central_bodies
    )

    # spacecraft initial state: Earth at departure + hyperbolic excess velocity (dummy direction)
    delta_t_earth = start_epoch - mjd_to_et(epoch)
    n_earth = np.sqrt(sun_gravitational_parameter / a_earth**3)
    mean_anomaly_earth_at_start_epoch = (np.deg2rad(mean_anomaly_earth) + n_earth * delta_t_earth) % (2 * np.pi)
    true_anomaly_earth = element_conversion.mean_to_true_anomaly(
        eccentricity_earth, mean_anomaly_earth_at_start_epoch)
    earth_state_keplerian = np.array([
        a_earth, eccentricity_earth, np.deg2rad(inclination_earth),
        np.deg2rad(arg_periapsis_earth), np.deg2rad(lan_earth), true_anomaly_earth,
    ])
    earth_state_cartesian = element_conversion.keplerian_to_cartesian(
        earth_state_keplerian, sun_gravitational_parameter)
    excess_velocity_vector = np.array([scape_velocity_max, 0, 0])
    spacecraft_initial_state = earth_state_cartesian + np.hstack((np.zeros(3), excess_velocity_vector))

    # asteroid initial states, propagated from their catalog epoch to start_epoch
    asteroids_initial_states = []
    for ast in asteroids:
        delta_t = start_epoch - ast['epoch']
        n = np.sqrt(sun_gravitational_parameter / ast['a']**3)
        M_at_start = (ast['M0'] + n * delta_t) % (2 * np.pi)
        true_anomaly = element_conversion.mean_to_true_anomaly(ast['e'], M_at_start)
        keplerian_state = np.array([ast['a'], ast['e'], ast['i'], ast['omega'], ast['lan'], true_anomaly])
        asteroids_initial_states.append(
            element_conversion.keplerian_to_cartesian(keplerian_state, sun_gravitational_parameter))

    system_initial_state = np.hstack((spacecraft_initial_state, *asteroids_initial_states))

    # integrator: RKF78, variable step, blockwise tolerance on position and velocity of every body.
    # tolerance defaults to 1e-13, not 1e-10: for the eccentric dummy departure orbit used here,
    # 1e-10 doesn't fully converge (its result still moves ~600 m under further tightening, checked
    # by bridging a 1e-13 run forward with the validated RK4 propagator) -- this module is meant as
    # ground truth, so it should actually be converged.
    n_bodies = len(bodies_to_propagate)
    block_indices = [(6 * i + offset, 0, 3, 1) for i in range(n_bodies) for offset in (0, 3)]
    integrator_settings = propagation_setup.integrator.runge_kutta_variable_step(
        initial_time_step=1e4,
        coefficient_set=propagation_setup.integrator.CoefficientSets.rkf_78,
        step_size_control_settings=propagation_setup.integrator.step_size_control_blockwise_scalar_tolerance(
            block_indices=block_indices, relative_error_tolerance=tolerance, absolute_error_tolerance=tolerance
        ),
        step_size_validation_settings=propagation_setup.integrator.step_size_validation(
            minimum_step=1.0, maximum_step=np.inf
        ),
    )

    termination_settings = propagation_setup.propagator.time_termination(end_epoch)
    translational_settings = propagation_setup.propagator.translational(
        central_bodies, acceleration_models, bodies_to_propagate, system_initial_state,
        start_epoch, integrator_settings, termination_settings,
    )

    mass_rate_models = propagation_setup.create_mass_rate_models(
        bodies, dict(spacecraft=[propagation_setup.mass_rate.from_thrust()]), acceleration_models
    )
    mass_settings = propagation_setup.propagator.mass(
        ['spacecraft'], mass_rate_models, [spacecraft_wet_mass],
        start_epoch, integrator_settings, termination_settings,
    )

    propagator_settings = propagation_setup.propagator.multitype(
        [translational_settings, mass_settings], integrator_settings, start_epoch, termination_settings,
        [propagation_setup.dependent_variable.body_mass('spacecraft')],
    )

    dynamics_simulator = simulator.create_dynamics_simulator(bodies, propagator_settings)
    results = dynamics_simulator.propagation_results

    if save:
        save2txt(solution=results.state_history, filename="PropagationHistory.dat", directory=RESULTS_DIRECTORY)
        save2txt(solution=results.dependent_variable_history,
                 filename="PropagationHistory_DependentVariables.dat", directory=RESULTS_DIRECTORY)

    if plot:
        _plot_trajectory(results)

    return results

def run_policy_verification(initial_state, start_epoch, duration, thrust_direction_function,
                             thrust_magnitude_function, tolerance=1e-13):
    """Replays a piecewise thrust profile (e.g. from an RL rollout) through Tudat's RKF78
    integrator -- WP8's check of whether a policy trained against the hand-written RK4
    propagator still works under an independent, higher-fidelity integrator. Only the
    spacecraft is propagated: the target's ephemeris is the same analytic two-body Kepler
    formula already cross-checked against Tudat in WP1 (validate_dynamics.py), so it doesn't
    need re-verifying here."""
    bodies = build_bodies([])
    end_epoch = start_epoch + duration

    thrust_magnitude_settings = propagation_setup.thrust.custom_thrust_magnitude_fixed_isp(
        thrust_magnitude_function=thrust_magnitude_function, specific_impulse=Isp_engine)
    environment_setup.add_engine_model('spacecraft', 'LowThrustEngine', thrust_magnitude_settings, bodies)
    environment_setup.add_rotation_model(
        bodies, 'spacecraft',
        environment_setup.rotation_model.custom_inertial_direction_based(
            thrust_direction_function, 'ECLIPJ2000', 'VehicleFixed'))

    acceleration_settings = {'spacecraft': dict(
        Sun=[propagation_setup.acceleration.point_mass_gravity()],
        spacecraft=[propagation_setup.acceleration.thrust_from_all_engines()])}
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings, ['spacecraft'], ['Sun'])

    block_indices = [(0, 0, 3, 1), (3, 0, 3, 1)]
    integrator_settings = propagation_setup.integrator.runge_kutta_variable_step(
        initial_time_step=1e4,
        coefficient_set=propagation_setup.integrator.CoefficientSets.rkf_78,
        step_size_control_settings=propagation_setup.integrator.step_size_control_blockwise_scalar_tolerance(
            block_indices=block_indices, relative_error_tolerance=tolerance, absolute_error_tolerance=tolerance),
        step_size_validation_settings=propagation_setup.integrator.step_size_validation(
            minimum_step=1.0, maximum_step=np.inf))

    termination_settings = propagation_setup.propagator.time_termination(end_epoch)
    translational_settings = propagation_setup.propagator.translational(
        ['Sun'], acceleration_models, ['spacecraft'], np.asarray(initial_state[:6], dtype=np.float64),
        start_epoch, integrator_settings, termination_settings)

    mass_rate_models = propagation_setup.create_mass_rate_models(
        bodies, dict(spacecraft=[propagation_setup.mass_rate.from_thrust()]), acceleration_models)
    mass_settings = propagation_setup.propagator.mass(
        ['spacecraft'], mass_rate_models, [float(initial_state[6])],
        start_epoch, integrator_settings, termination_settings)

    propagator_settings = propagation_setup.propagator.multitype(
        [translational_settings, mass_settings], integrator_settings, start_epoch, termination_settings,
        [propagation_setup.dependent_variable.body_mass('spacecraft')])

    dynamics_simulator = simulator.create_dynamics_simulator(bodies, propagator_settings)
    return dynamics_simulator.propagation_results

def _plot_trajectory(results):
    position = util.result2array(results.state_history)[:, 1:4] / constants.ASTRONOMICAL_UNIT
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(position[:, 0], position[:, 1], position[:, 2], 'b-')
    ax.scatter(0, 0, 0, color='yellow', s=200, marker='*', label='Sun')
    ax.scatter(*position[0], color='green', s=50, label='Start')
    ax.scatter(*position[-1], color='red', s=50, label='End')
    ax.set_xlabel('X (AU)')
    ax.set_ylabel('Y (AU)')
    ax.set_zlabel('Z (AU)')
    ax.set_title('Spacecraft Trajectory')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    t0 = time.time()
    run_propagation(plot=True, save=True)
    print(f"Simulation completed in {time.time() - t0:.2f} seconds.")
