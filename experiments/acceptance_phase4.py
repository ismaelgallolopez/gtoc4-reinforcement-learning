# Phase 4 acceptance test for the legality track: the Lambert solver must recover the endpoint
# velocities of a Keplerian arc it did not see.
#
# The test is self-consistent and needs no external reference data: an orbit is sampled at two
# epochs with dynamics.keplerian_to_cartesian, giving (r1, v1) and (r2, v2) and an exactly known
# time of flight; Lambert is then handed only (r1, r2, tof) and must reproduce v1 and v2.
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

import dynamics
import lambert
from constants import sun_gravitational_parameter as mu

AU = 1.495978707e11
DAY = 86400.0

def test_lambert():
    rng = np.random.default_rng(0)
    cases = []
    # a spread of orbit shapes and transfer angles, including angles above pi (retrograde-looking
    # geometry for a prograde transfer), which is where the sign of A matters
    for _ in range(200):
        a = rng.uniform(0.8, 3.0) * AU
        e = rng.uniform(0.01, 0.7)
        i = rng.uniform(0.01, 1.2)
        omega, lan, M1 = rng.uniform(0, 2 * np.pi, 3)
        period = 2 * np.pi * np.sqrt(a**3 / mu)
        tof = rng.uniform(0.05, 0.85) * period

        n = np.sqrt(mu / a**3)
        s1 = dynamics.keplerian_to_cartesian(a, e, i, omega, lan, M1, mu)
        s2 = dynamics.keplerian_to_cartesian(a, e, i, omega, lan, M1 + n * tof, mu)
        v1, v2 = lambert.solve(s1[:3], s2[:3], tof, mu)
        cases.append((np.linalg.norm(v1 - s1[3:]), np.linalg.norm(v2 - s2[3:]),
                      np.degrees(np.arccos(np.clip(np.dot(s1[:3], s2[:3]) /
                                                   (np.linalg.norm(s1[:3]) * np.linalg.norm(s2[:3])),
                                                   -1, 1)))))

    errors = np.array([[c[0], c[1]] for c in cases])
    angles = np.array([c[2] for c in cases])
    worst = int(np.argmax(errors.max(axis=1)))

    print("--- 4.1: universal-variable Lambert vs a propagated Keplerian arc (200 random cases) ---")
    print(f"cases                   : {len(cases)}, a in [0.8, 3.0] AU, e in [0.01, 0.7], "
          f"tof in [0.05, 0.85] period")
    print(f"transfer angle range    : {angles.min():.1f} - {angles.max():.1f} deg "
          f"({int(np.sum(angles > 90))} cases above 90 deg)")
    print(f"median |v1| error       : {np.median(errors[:, 0])*1e3:.6f} mm/s")
    print(f"median |v2| error       : {np.median(errors[:, 1])*1e3:.6f} mm/s")
    print(f"worst |v1| error        : {errors[:, 0].max()*1e3:.6f} mm/s")
    print(f"worst |v2| error        : {errors[:, 1].max()*1e3:.6f} mm/s")
    print(f"worst case              : transfer angle {angles[worst]:.1f} deg")
    assert errors.max() < 1e-3, f"worst velocity error {errors.max()*1e3:.4f} mm/s exceeds 1 mm/s"
    print("PASS: both endpoint velocities recovered to better than 1 mm/s in every case\n")

if __name__ == '__main__':
    test_lambert()
    print("Phase 4.1 acceptance test passed")
