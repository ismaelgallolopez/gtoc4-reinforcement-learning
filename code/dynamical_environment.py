import os
import numpy as np

from tudatpy import constants
from tudatpy.astro import element_conversion, two_body_dynamics
from tudatpy.dynamics import environment_setup, environment, propagation_setup, simulator
from tudatpy.data import save2txt
from tudatpy.interface import spice
from tudatpy.math import interpolators

from problem_parameters import * # the problem parameters from the description
from auxiliary_functions import * 

ASTEROIDS_FILEPATH = "../data/gtoc4_problem_data.txt"
# n_asteroids = 1 # optional, to limit the number of asteroids when testing
n_asteroids = None # set to None to load all asteroids

asteroids = parse_asteroids(ASTEROIDS_FILEPATH, n_asteroids)
# print(asteroids) # debugging

bodies    = build_bodies(asteroids)
names     = [f"ast_{a['name']}" for a in asteroids]
