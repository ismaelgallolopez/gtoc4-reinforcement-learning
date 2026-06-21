import os
import numpy as np

from tudatpy import constants
from tudatpy.astro import element_conversion, two_body_dynamics, time_representation
from tudatpy.dynamics import environment_setup, environment, propagation_setup, simulator
from tudatpy.data import save2txt
from tudatpy.interface import spice
from tudatpy.math import interpolators

from problem_parameters import * # the problem parameters from the description
from auxiliary_functions import * 

ASTEROIDS_FILEPATH = "../data/gtoc4_problem_data.txt"
n_asteroids = None # set to None to load all asteroids
n_asteroids = 1 # optional, to limit the number of asteroids when testing

# fixed epoch for first iterations, later will work on implementing the variable one (TODO)
start_epoch = launch_interval[0]
end_epoch = start_epoch + time_mission_max

asteroids = parse_asteroids(ASTEROIDS_FILEPATH, n_asteroids)
# print(asteroids) # debugging

bodies    = build_bodies(asteroids)
names     = [f"ast_{a['name']}" for a in asteroids]

bodies_to_propagate = ['spacecraft'] + names
# print("Bodies to propagate:", bodies_to_propagate) # debugging

central_bodies = ['Sun'] * len(bodies_to_propagate)


# setup the thrust model
thrust_magnitude_settings = (propagation_setup.thrust.custom_thrust_magnitude_fixed_isp(
    thrust_magnitude_function= thrust_magnitude_function,
    specific_impulse=Isp_engine # TODO: CHECK IF THE MODEL USES s AS UNIT FOR THIS
))

environment_setup.add_engine_model(
    'spacecraft', 'LowThrustEngine', thrust_magnitude_settings, bodies )

environment_setup.add_rotation_model(
    bodies, 
    'spacecraft', 
    environment_setup.rotation_model.custom_inertial_direction_based(
        lambda time : np.array([1,0,0] ), # inertial_body_axis_direction
        'ECLIPJ2000', # base_frame
        'VehcleFixed' # target_frame
        ) 
    )


# setup the acceleration model
acceleration_settings_spacecraft = dict(
    Sun=[propagation_setup.acceleration.point_mass_gravity()], 
    spacecraft=[    propagation_setup.acceleration.thrust_from_all_engines()]
    )

acceleration_settings_asteroids = dict(
    Sun=[propagation_setup.acceleration.point_mass_gravity()]
)

# thrust_direction_settings = propagation_setup.thrust.custom_thrust_orientation(
#     thrust_direction_function
# )


# computing the initial state of the spacecraft
true_anomaly_earth = element_conversion.mean_to_true_anomaly(eccentricity_earth, mean_anomaly_earth) # convertion needed for the tudat conversion function
earth_initial_state_keplerian = np.array([a_earth, eccentricity_earth, inclination_earth, lan_earth, arg_periapsis_earth, true_anomaly_earth])

earth_initial_state_cartesian = element_conversion.keplerian_to_cartesian(earth_initial_state_keplerian, sun_gravitational_parameter)

excess_velocity_vector = np.array([scape_velocity_max, 0, 0]) # dummy direction
delta_state_departure = np.hstack((np.zeros(3), excess_velocity_vector)) # only velocity change, no position change
spacecraft_initial_state_cartesian = earth_initial_state_cartesian + delta_state_departure

# initial state of the asteroids 
true_anomalies_asteroids = [element_conversion.mean_to_true_anomaly(ast['e'], ast['M0']) for ast in asteroids]
asteroids_initial_states_keplerian = [np.array([ast['a'], ast['e'], ast['i'], ast['lan'], ast['omega'], true_anomaly]) for ast, true_anomaly in zip(asteroids, true_anomalies_asteroids)]
asteroids_initial_states_cartesian = [element_conversion.keplerian_to_cartesian(keplerian_state, sun_gravitational_parameter) for keplerian_state in asteroids_initial_states_keplerian]

# print(np.shape(asteroids_initial_states_cartesian)) # debugging
# initial state of the system (spacecraft + asteroids)
system_initial_state = np.hstack((spacecraft_initial_state_cartesian,
                                 *asteroids_initial_states_cartesian # need to transpose for consistency
                                ))

# print("Initial state of the Earth (cartesian):", earth_initial_state_cartesian) # debugging
# print("Initial state of the spacecraft (cartesian):", spacecraft_initial_state_cartesian) # debugging

# integrator settings
time_step = 10000.0 # s, dummy value for now, will need to be tuned
# fixed step RK4 integrator used for simplicity, later to be implemented in a more sophisticated method (TODO)
integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step( 
    time_step = time_step,
    coefficient_set = propagation_setup.integrator.CoefficientSets.rk_4
)

# acceleration settings
acceleration_settings = { 'spacecraft': acceleration_settings_spacecraft, 
                         **{name: acceleration_settings_asteroids for name in names} }

# print(bodies, central_bodies, bodies_to_propagate) # debugging

acceleration_models = propagation_setup.create_acceleration_models(
    bodies, acceleration_settings, bodies_to_propagate, central_bodies
)

# Create propagation settings.
termination_settings = propagation_setup.propagator.time_termination(end_epoch)

# translational propagator settings
translational_propagator_settings = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models,
    bodies_to_propagate,
    system_initial_state,
    start_epoch,
    integrator_settings,
    termination_settings,
    # output_variables=dependent_variables_to_save,
)

# mass propagation settings
mass_rate_settings = dict(spacecraft=[propagation_setup.mass_rate.from_thrust()])
mass_rate_models = propagation_setup.create_mass_rate_models(
    bodies, 
    mass_rate_settings, 
    acceleration_models
)

mass_propagator_settings = propagation_setup.propagator.mass(
    ['spacecraft'], # because only the spacecraft has a mass variation
    mass_rate_models,
    [spacecraft_wet_mass],
    start_epoch,
    integrator_settings,
    termination_settings,
    )

# multitype propagator settings (translational + mass)
propagator_settings_list = [
    translational_propagator_settings,
    mass_propagator_settings
]

propagator_settings = propagation_setup.propagator.multitype(
    propagator_settings_list,
    integrator_settings,
    start_epoch,
    termination_settings,
)

propagator_settings.print_settings.print_initial_and_final_conditions = True

# propagate orbit
dynamics_simulator = simulator.create_dynamics_simulator(
    bodies, propagator_settings
)

# Retrieve all data produced by simulation
propagation_results = dynamics_simulator.propagation_results

# Extract numerical solution for states and dependent variables
state_history = propagation_results.state_history

