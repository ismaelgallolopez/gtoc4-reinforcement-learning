import re
import numpy as np
from tudatpy import constants

import dynamics
from constants import (a_earth, eccentricity_earth, inclination_earth,
                        lan_earth, arg_periapsis_earth, mean_anomaly_earth, epoch)

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

def earth_initial_state(t, mu):
    """Earth's Cartesian state at epoch t (seconds), propagated from the reference epoch (MJD 54000)."""
    epoch0 = mjd_to_et(epoch)
    n = np.sqrt(mu / a_earth**3)
    M = np.deg2rad(mean_anomaly_earth) + n * (t - epoch0)
    return dynamics.keplerian_to_cartesian(
        a_earth, eccentricity_earth, np.deg2rad(inclination_earth),
        np.deg2rad(arg_periapsis_earth), np.deg2rad(lan_earth), M, mu)

def target_states(asteroids, t, mu):
    """Vectorised Cartesian state of every asteroid in `asteroids` at epoch t (seconds).
    Returns an array of shape (len(asteroids), 6)."""
    a     = np.array([ast['a'] for ast in asteroids])
    e     = np.array([ast['e'] for ast in asteroids])
    i     = np.array([ast['i'] for ast in asteroids])
    omega = np.array([ast['omega'] for ast in asteroids])
    lan   = np.array([ast['lan'] for ast in asteroids])
    M0    = np.array([ast['M0'] for ast in asteroids])
    epoch0 = np.array([ast['epoch'] for ast in asteroids])

    n = np.sqrt(mu / a**3)
    M = M0 + n * (t - epoch0)
    return dynamics.keplerian_to_cartesian(a, e, i, omega, lan, M, mu)
