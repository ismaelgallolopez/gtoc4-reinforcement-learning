# Minimal zero-revolution universal-variable Lambert solver (Bate/Mueller/White, in the bisection
# form given by Vallado). Given two position vectors and a time of flight, returns the two
# velocities of the unique zero-revolution conic joining them.
#
# Used by the legality track's Phase 4 as the leg-feasibility oracle: the impulsive delta-v of the
# Lambert arc is a lower-bound proxy for what a low-thrust leg costs, cheap enough to evaluate for
# every (target, time-of-flight) pair in the pruning loop.
import numpy as np

MAX_ITERATIONS = 200
TOLERANCE = 1e-8          # relative, on the time of flight

class LambertError(Exception):
    """Raised when no zero-revolution solution is found for the requested geometry."""

def _stumpff(psi):
    """C(psi), S(psi). The series form is used near psi = 0 where both closed forms are 0/0."""
    if psi > 1e-6:
        sqrt_psi = np.sqrt(psi)
        return (1 - np.cos(sqrt_psi)) / psi, (sqrt_psi - np.sin(sqrt_psi)) / sqrt_psi**3
    if psi < -1e-6:
        sqrt_psi = np.sqrt(-psi)
        return ((1 - np.cosh(sqrt_psi)) / psi,
                (np.sinh(sqrt_psi) - sqrt_psi) / sqrt_psi**3)
    return (0.5 - psi / 24.0 + psi**2 / 720.0,
            1.0 / 6.0 - psi / 120.0 + psi**2 / 5040.0)

def solve(r1, r2, time_of_flight, mu, prograde=True):
    """Returns (v1, v2), the departure and arrival velocities of the zero-revolution transfer from
    r1 to r2 in `time_of_flight` seconds. `prograde` selects the direction of motion, which is what
    disambiguates a transfer angle of dnu from one of 2*pi - dnu."""
    r1 = np.asarray(r1, dtype=np.float64)
    r2 = np.asarray(r2, dtype=np.float64)
    if time_of_flight <= 0:
        raise LambertError(f"non-positive time of flight {time_of_flight}")

    r1_norm, r2_norm = np.linalg.norm(r1), np.linalg.norm(r2)
    cos_dnu = np.clip(np.dot(r1, r2) / (r1_norm * r2_norm), -1.0, 1.0)
    # sign of the out-of-plane component of r1 x r2 tells us whether the prograde transfer angle is
    # below or above pi; A carries that sign
    direction = 1.0 if (np.cross(r1, r2)[2] >= 0.0) == prograde else -1.0
    A = direction * np.sqrt(r1_norm * r2_norm * (1.0 + cos_dnu))
    if abs(A) < 1e-9:
        raise LambertError("transfer angle is 0 or pi; the transfer plane is undefined")

    def evaluate(psi):
        """(time of flight, y) at this psi, or None where the geometry makes y negative."""
        c2, c3 = _stumpff(psi)
        if c2 <= 0.0:
            return None
        y = r1_norm + r2_norm + A * (psi * c3 - 1.0) / np.sqrt(c2)
        if y < 0.0:
            return None
        chi = np.sqrt(y / c2)
        return (chi**3 * c3 + A * np.sqrt(y)) / np.sqrt(mu), y

    # dt(psi) increases monotonically with psi, so the solve is a bisection -- but the bracket has
    # to be established first. A fixed [-4*pi^2, 4*pi^2] is not enough: for A > 0 the low end is
    # cut off where y goes negative, and for A < 0 (transfer angle past pi) a long time of flight
    # needs a psi well below -4*pi^2. Both are handled by walking the lower bound until it is both
    # valid and short enough, which is what a fixed bracket silently failed to do -- it produced
    # non-convergence roughly once per 1500 calls inside the rendezvous pursuit.
    # dt -> infinity as psi -> 4*pi^2 (the one-revolution boundary), where c2 -> 0 exactly, so the
    # upper bound has to sit just inside it rather than on it
    psi_high = 4.0 * np.pi**2 - 1e-6
    for _ in range(MAX_ITERATIONS):
        if evaluate(psi_high) is not None:
            break
        psi_high -= 1e-2
    else:
        raise LambertError("could not find a valid upper bracket")

    psi_low = -4.0 * np.pi**2
    for _ in range(MAX_ITERATIONS):
        low = evaluate(psi_low)
        if low is None:
            psi_low += 0.05 * (psi_high - psi_low)
        elif low[0] > time_of_flight:
            psi_low -= max(1.0, abs(psi_low))
        else:
            break
    else:
        raise LambertError(f"could not bracket the solution below (tof {time_of_flight:.1f} s)")

    high = evaluate(psi_high)
    if high is None or high[0] < time_of_flight:
        raise LambertError(f"no zero-revolution solution for tof {time_of_flight:.1f} s")

    psi = 0.5 * (psi_low + psi_high)
    for _ in range(MAX_ITERATIONS):
        current = evaluate(psi)
        if current is None:
            psi_low = psi
            psi = 0.5 * (psi_low + psi_high)
            continue
        dt, y = current
        if abs(dt - time_of_flight) < TOLERANCE * time_of_flight:
            break
        if dt <= time_of_flight:
            psi_low = psi
        else:
            psi_high = psi
        psi = 0.5 * (psi_low + psi_high)
    else:
        raise LambertError(f"did not converge in {MAX_ITERATIONS} iterations "
                           f"(tof {time_of_flight:.1f} s, last dt {dt:.1f} s)")
    c2, _ = _stumpff(psi)

    f = 1.0 - y / r1_norm
    g = A * np.sqrt(y / mu)
    g_dot = 1.0 - y / r2_norm
    return (r2 - f * r1) / g, (g_dot * r2 - r1) / g
