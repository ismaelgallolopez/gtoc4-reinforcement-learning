import re
import numpy as np
from tudatpy.interface import spice
from tudatpy.astro import element_conversion
from tudatpy.dynamics import environment_setup
from tudatpy import constants

from problem_parameters import *

spice.load_standard_kernels()

MJD_J2000 = 51544.5  # MJD of J2000 epoch

def mjd_to_et(mjd: float):
    return (mjd - MJD_J2000) * 86400.0

def parse_asteroids(filepath: str, n_asteroids: int = None):
    asteroids = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            m = re.match(
                r"'([^']+)'\s+([\d.]+)\s+([\S]+)\s+([\S]+)\s+([\S]+)\s+([\S]+)\s+([\S]+)\s+([\S]+)",
                line
            )
            if not m:
                continue
            asteroids.append({
                'name':  m.group(1),
                'epoch': mjd_to_et(float(m.group(2))),
                'a':     float(m.group(3)) * constants.ASTRONOMICAL_UNIT,
                'e':     float(m.group(4)),
                'i':     np.deg2rad(float(m.group(5))),
                'lan':   np.deg2rad(float(m.group(6))),
                'omega': np.deg2rad(float(m.group(7))),
                'M0':    np.deg2rad(float(m.group(8))),
            })
            if n_asteroids and len(asteroids) >= n_asteroids:
                break
    return asteroids

def build_bodies(asteroids: list[dict]):
    from tudatpy.dynamics import environment_setup
    
    mu_sun = spice.get_body_gravitational_parameter('Sun')

    body_settings = environment_setup.get_default_body_settings(['Sun'])

    for ast in asteroids:
        name = f"ast_{ast['name']}"
        body_settings.add_empty_settings(name)
        body_settings.get(name).ephemeris_settings = (
            environment_setup.ephemeris.keplerian(
                initial_keplerian_state=np.array([
                    ast['a'], ast['e'], ast['i'],
                    ast['lan'], ast['omega'], ast['M0'],
                ]),
                initial_state_epoch=ast['epoch'],
                central_body_gravitational_parameter=mu_sun,
                frame_origin='Sun',
                frame_orientation='ECLIPJ2000',
            )
        )

    body_settings.add_empty_settings('spacecraft')
    body_settings.get('spacecraft').constant_mass = spacecraft_wet_mass
    
    # ========================================================================
    # THRUST ROTATION MODEL (spacecraft body orientation)
    # ========================================================================
    # Define rotation model: spacecraft body-fixed frame aligned with velocity
    # Can modify later to implement RL steering, costate guidance, etc.
    rotation_model_settings = environment_setup.rotation_model.orbital_state_direction_based(
        central_body='Sun',
        is_colinear_with_velocity=True,  # Thrust direction = velocity direction
        direction_is_opposite_to_vector=False,
        base_frame='ECLIPJ2000',
        target_frame='Spacecraft_Fixed'
    )
    
    # Assign rotation model to spacecraft BEFORE creating system_of_bodies
    body_settings.get('spacecraft').rotation_model_settings = rotation_model_settings

    return environment_setup.create_system_of_bodies(body_settings)

def thrust_direction_function(time):
    """
    Thrust direction as a unit vector in J2000 inertial frame.
    Modify this later to implement guidance laws.
    TODO: Implement RL-based steering, or costate guidance, etc.
    """
    # For now: dummy direction (along X-axis)
    # Later: return computed direction from RL policy or guidance
    return [1.0, 0.0, 0.0]

def thrust_magnitude_function(time):
    """
    Thrust magnitude as a function of time.
    Modify this later to implement variable thrust profiles.
    """
    # For now: constant thrust magnitude
    return thrust_max