import os
import numpy as np
import matplotlib.pyplot as plt

from tudatpy import constants, util
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
        thrust_direction_function, # inertial_body_axis_direction
        'ECLIPJ2000', # base_frame
        'VehicleFixed' # target_frame
        ) 
    )


# setup the acceleration model
acceleration_settings_spacecraft = dict(
    Sun=[propagation_setup.acceleration.point_mass_gravity()], 
    spacecraft=[propagation_setup.acceleration.thrust_from_all_engines()]
    )

acceleration_settings_asteroids = dict(
    Sun=[propagation_setup.acceleration.point_mass_gravity()]
)

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
time_step = 1e4 # s, dummy value for now, will need to be tuned
block_indices=[(0, 0, 3, 1), (3, 0, 3, 1)]
tolerance=1e-10 # both absolute and relative

step_size_control_settings = propagation_setup.integrator.step_size_control_blockwise_scalar_tolerance(
    block_indices = block_indices,
    relative_error_tolerance = tolerance,
    absolute_error_tolerance = tolerance
)

step_size_validation_settings = propagation_setup.integrator.step_size_validation(
    minimum_step = 1.0, # s
    maximum_step = np.inf, # s
)

# fixed step RK4 integrator used for simplicity, later to be implemented in a more sophisticated method (TODO)
integrator_settings = propagation_setup.integrator.runge_kutta_variable_step( 
    initial_time_step = time_step,
    coefficient_set = propagation_setup.integrator.CoefficientSets.rkf_78,
    step_size_control_settings = step_size_control_settings, 
    step_size_validation_settings = step_size_validation_settings,
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

mass_variables_to_save = [propagation_setup.dependent_variable.body_mass('spacecraft')]

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

variables_to_save_list = mass_variables_to_save
# print("Variables to save:", variables_to_save_list) # debugging

propagator_settings = propagation_setup.propagator.multitype(
    propagator_settings_list,
    integrator_settings,
    start_epoch,
    termination_settings,
    variables_to_save_list
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

dependent_variables_dict = propagation_results.dependent_variable_history
# print("Dependent variables:", dependent_variables) # debugging

# saving the results
results_directory = "./results"

save2txt(
    solution=state_history, filename="PropagationHistory.dat", directory=results_directory
)

save2txt(
    solution=dependent_variables_dict,
    filename="PropagationHistory_DependentVariables.dat",
    directory=results_directory,
)

# extracting dependent variables
dep_var_ids = propagation_results.dependent_variable_ids

# for start_index, settings in dep_var_ids.items(): print(f"Index {start_index} -> {settings}") # debugging 

dependent_variables = util.result2array(dependent_variables_dict)

mass_history = dependent_variables[:, 1] # because only one dependent variable (the mass of the spacecraft)
# print("Mass history:", mass_history) # debugging


# plotting the results
plt.plot(mass_history)
# plt.show()

spacecraft_position = util.result2array(state_history)[:, 1:4] / constants.ASTRONOMICAL_UNIT 

# Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(spacecraft_position[:, 0], spacecraft_position[:, 1], spacecraft_position[:, 2], 'b-')
ax.scatter(0, 0, 0, color='yellow', s=200, marker='*', label='Sun')
ax.scatter(spacecraft_position[0, 0], spacecraft_position[0, 1], spacecraft_position[0, 2], 
           color='green', s=50, label='Start')
ax.scatter(spacecraft_position[-1, 0], spacecraft_position[-1, 1], spacecraft_position[-1, 2], 
           color='red', s=50, label='End')
ax.set_xlabel('X (AU)')
ax.set_ylabel('Y (AU)')
ax.set_zlabel('Z (AU)')
ax.set_title('Spacecraft Trajectory')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
