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

central_bodies = ['Sun']


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

# thrust_direction_settings = propagation_setup.thrust.custom_thrust_orientation(
#     thrust_direction_function
# )

acceleration_settings = { 'spacecraft': acceleration_settings_spacecraft }

# computing the initial state of the spacecraft
true_anomaly_earth = element_conversion.mean_to_true_anomaly(eccentricity_earth, mean_anomaly_earth) # convertion needed for the tudat conversion function
earth_initial_state_keplerian = np.array([a_earth, eccentricity_earth, inclination_earth, lan_earth, arg_periapsis_earth, true_anomaly_earth])

earth_initial_state_cartesian = element_conversion.keplerian_to_cartesian(earth_initial_state_keplerian, sun_gravitational_parameter)

excess_velocity_vector = np.array([scape_velocity_max, 0, 0]) # dummy direction
delta_state_departure = np.hstack((np.zeros(3), excess_velocity_vector)) # only velocity change, no position change
spacecraft_initial_state_cartesian = earth_initial_state_cartesian + delta_state_departure

# print("Initial state of the Earth (cartesian):", earth_initial_state_cartesian) # debugging
# print("Initial state of the spacecraft (cartesian):", spacecraft_initial_state_cartesian) # debugging

# integrator settings
time_step = 100.0 # s, dummy value for now, will need to be tuned
# fixed step RK4 integrator used for simplicity, later to be implemented in a more sophisticated method (TODO)
integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step( 
    time_step = time_step,
    coefficients_set = propagation_setup.integrator.CoefficientSets.rk_4
)


