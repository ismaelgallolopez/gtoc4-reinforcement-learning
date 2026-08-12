import re
import numpy as np
from tudatpy import constants

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
