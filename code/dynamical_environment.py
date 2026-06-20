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

spice.load_standard_kernels()

asteroids = parse_asteroids("../data/gtoc4_problem_data.txt")
# print(asteroids[0]) # debugging

bodies    = build_bodies(asteroids)
names     = [f"ast_{a['name']}" for a in asteroids]
