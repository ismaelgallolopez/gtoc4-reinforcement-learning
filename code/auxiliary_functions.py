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

def parse_asteroids(filepath: str):
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
    return asteroids

def build_bodies(asteroids: list[dict]):
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

    return environment_setup.create_system_of_bodies(body_settings)
