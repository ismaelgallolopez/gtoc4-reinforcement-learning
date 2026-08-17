# GTOC4 legality bookkeeping and solution-file I/O.
#
# Holds a mission as (epoch, r, v, m, thrust) samples plus the encounters recorded against them,
# enforces the competition rules as the mission is built, and exposes the performance index
# J (distinct asteroids flown by before the final rendezvous) and the tiebreak K = m_f.
#
# Rules enforced (from the problem statement):
#   - launch from Earth, |v_inf| <= 4.0 km/s
#   - launch epoch in MJD 57023-61041
#   - mission duration tau <= 10 years
#   - final mass m_f >= 500 kg
#   - thrust magnitude T <= 0.135 N at every point, Isp = 3000 s
#   - flyby: position match within 1000 km
#   - rendezvous: position within 1000 km AND velocity within 1 m/s
#   - each asteroid visited at most once; the rendezvous target not previously visited
#   - no gravity assists
#
# An encounter that violates a rule is refused (LegalityError), never silently recorded. The
# separate check_solution() re-derives every one of these from a written file alone, trusting
# nothing from the Mission object that produced it.
#
# Solution-file layout: the GTOC4 problem statement document is not present in this repository
# (data/gtoc4_problem_data.txt is the asteroid ephemeris table only), so the column layout below is
# an assumption, documented here and in NOTES_legality_track.md, to be checked against the
# statement before any real submission:
#
#   MJD  x[km]  y[km]  z[km]  vx[km/s]  vy[km/s]  vz[km/s]  m[kg]  Tx[N]  Ty[N]  Tz[N]  body
#
# in the J2000 heliocentric ecliptic frame, one row per sample. `body` is EARTH on the launch row,
# the asteroid's catalogue name on an encounter row, and '-' on an ordinary trajectory row. The
# thrust on a row is the constant thrust applied from that row's epoch until the next row's epoch;
# it is zero on the final row. Lines starting with '#' are comments and carry no information the
# checker relies on.
import numpy as np

import catalog
import dynamics
from constants import (sun_gravitational_parameter as mu, Isp_engine, thrust_max,
                        spacecraft_dry_mass, spacecraft_wet_mass, accuracy_position,
                        accuracy_velocity, scape_velocity_max, time_mission_max)

LAUNCH_WINDOW_MJD = (57023.0, 61041.0)
KM = 1e3
DAY = 86400.0
# tolerances used when re-verifying a file: floating-point slack on the rule limits themselves, and
# how far a row may drift from a two-body + constant-thrust propagation of the previous row before
# the trajectory is called dynamically inconsistent (a gravity assist or an injected impulse).
RULE_EPS = 1e-9
DYNAMICS_POSITION_TOLERANCE = 10.0 * KM
DYNAMICS_VELOCITY_TOLERANCE = 1.0

class LegalityError(Exception):
    """Raised when an operation would put the mission in an illegal state."""

def _mjd(epoch_seconds):
    return epoch_seconds / DAY + catalog.MJD_J2000

def _epoch(mjd):
    return (mjd - catalog.MJD_J2000) * DAY

class Mission:
    """A GTOC4 mission under construction.

    `ephemerides` maps a body name to a Keplerian element dict in catalog.parse_asteroids' format;
    it defaults to the parsed GTOC4 catalogue but is injectable so tests can register synthetic
    bodies without touching the catalogue file."""

    def __init__(self, launch_epoch, v_infinity=None, ephemerides=None, launch_mass=None):
        launch_mjd = _mjd(launch_epoch)
        if not (LAUNCH_WINDOW_MJD[0] - RULE_EPS <= launch_mjd <= LAUNCH_WINDOW_MJD[1] + RULE_EPS):
            raise LegalityError(
                f"launch epoch MJD {launch_mjd:.4f} outside the legal window {LAUNCH_WINDOW_MJD}")

        v_infinity = np.zeros(3) if v_infinity is None else np.asarray(v_infinity, dtype=np.float64)
        if np.linalg.norm(v_infinity) > scape_velocity_max + RULE_EPS:
            raise LegalityError(
                f"launch |v_inf| = {np.linalg.norm(v_infinity):.1f} m/s exceeds the "
                f"{scape_velocity_max:.1f} m/s limit")

        self.ephemerides = ephemerides
        self.launch_epoch = float(launch_epoch)
        self.v_infinity = v_infinity

        earth = catalog.earth_initial_state(launch_epoch, mu)
        launch_mass = spacecraft_wet_mass if launch_mass is None else launch_mass
        # samples[i] = (epoch, r, v, m, thrust applied from samples[i] to samples[i+1])
        self.samples = [(float(launch_epoch), earth[:3].copy(), earth[3:] + v_infinity,
                         float(launch_mass), np.zeros(3))]
        self.bodies = ['EARTH']          # body label per sample
        self.visited = []                # asteroid names, in encounter order
        self.rendezvous_target = None

    # -- state ---------------------------------------------------------------------------------

    @property
    def epoch(self):
        return self.samples[-1][0]

    @property
    def r(self):
        return self.samples[-1][1]

    @property
    def v(self):
        return self.samples[-1][2]

    @property
    def m(self):
        return self.samples[-1][3]

    @property
    def duration(self):
        return self.epoch - self.launch_epoch

    @property
    def J(self):
        """Performance index: distinct asteroids flown by before the final rendezvous."""
        return len([n for n in self.visited if n != self.rendezvous_target])

    @property
    def K(self):
        """Tiebreak: final spacecraft mass, kg."""
        return self.m

    def body_state(self, name, epoch):
        if self.ephemerides is None:
            raise LegalityError("no ephemerides registered; cannot verify an encounter")
        if name not in self.ephemerides:
            raise LegalityError(f"unknown body {name!r}")
        return catalog.target_states([self.ephemerides[name]], epoch, mu)[0]

    # -- building ------------------------------------------------------------------------------

    def advance(self, epoch, r, v, m, thrust=None, body='-'):
        """Records one trajectory sample, refusing anything that breaks a rule."""
        thrust = np.zeros(3) if thrust is None else np.asarray(thrust, dtype=np.float64)
        if self.rendezvous_target is not None:
            raise LegalityError("the mission already ended with a rendezvous; cannot advance further")
        if epoch <= self.epoch - RULE_EPS:
            raise LegalityError(f"epoch {_mjd(epoch):.6f} is not after the previous sample "
                                f"{_mjd(self.epoch):.6f}")
        if epoch - self.launch_epoch > time_mission_max + RULE_EPS:
            raise LegalityError(f"mission duration {(epoch - self.launch_epoch)/DAY:.2f} d exceeds "
                                f"the {time_mission_max/DAY:.2f} d limit")
        if m < spacecraft_dry_mass - RULE_EPS:
            raise LegalityError(f"mass {m:.3f} kg is below the {spacecraft_dry_mass:.1f} kg dry mass")
        if m > self.m + RULE_EPS:
            raise LegalityError(f"mass increased from {self.m:.3f} to {m:.3f} kg")
        previous_thrust = np.linalg.norm(self.samples[-1][4])
        if previous_thrust > thrust_max + RULE_EPS:
            raise LegalityError(f"thrust {previous_thrust:.4f} N exceeds the {thrust_max} N limit")

        self.samples.append((float(epoch), np.asarray(r, dtype=np.float64).copy(),
                             np.asarray(v, dtype=np.float64).copy(), float(m), thrust))
        self.bodies.append(body)

    def _check_encounter(self, name, epoch, r, v, require_velocity_match):
        if name in self.visited:
            raise LegalityError(f"asteroid {name!r} has already been visited; "
                                f"each asteroid may be visited at most once")
        target = self.body_state(name, epoch)
        position_error = np.linalg.norm(np.asarray(r) - target[:3])
        if position_error > accuracy_position + RULE_EPS:
            raise LegalityError(f"{name!r}: position miss {position_error/KM:.3f} km exceeds the "
                                f"{accuracy_position/KM:.0f} km tolerance")
        if require_velocity_match:
            velocity_error = np.linalg.norm(np.asarray(v) - target[3:])
            if velocity_error > accuracy_velocity + RULE_EPS:
                raise LegalityError(f"{name!r}: velocity miss {velocity_error:.4f} m/s exceeds the "
                                    f"{accuracy_velocity:.1f} m/s tolerance")

    def flyby(self, name, epoch, r, v, m, thrust=None):
        """Records a flyby of `name`. Refused if the position match, the visit-once rule or any
        rule that applies to an ordinary sample is violated."""
        self._check_encounter(name, epoch, r, v, require_velocity_match=False)
        self.advance(epoch, r, v, m, thrust, body=name)
        self.visited.append(name)

    def rendezvous(self, name, epoch, r, v, m):
        """Records the final rendezvous with `name` and ends the mission. Refused if the position
        or velocity match fails, if `name` was previously visited, or if m_f < 500 kg."""
        self._check_encounter(name, epoch, r, v, require_velocity_match=True)
        if m < spacecraft_dry_mass - RULE_EPS:
            raise LegalityError(f"final mass {m:.3f} kg is below the "
                                f"{spacecraft_dry_mass:.1f} kg minimum")
        self.advance(epoch, r, v, m, None, body=name)
        self.visited.append(name)
        self.rendezvous_target = name

def write_solution(mission, path):
    """Writes `mission` in the GTOC4 solution format (layout documented at the top of this module).

    The sample history is written as-is, so the caller is responsible for having sampled it at
    one-day increments within each inter-body phase, with an extra partial-day sample at each flyby
    and at the final rendezvous; this is asserted, not silently accepted."""
    if mission.rendezvous_target is None:
        raise LegalityError("mission does not end with a rendezvous; nothing scorable to write")
    for (t0, *_), (t1, *_) in zip(mission.samples, mission.samples[1:]):
        if t1 - t0 > DAY + RULE_EPS:
            raise LegalityError(f"sample spacing {(t1 - t0)/DAY:.4f} d exceeds one day; the "
                                f"trajectory must be sampled at one-day increments")

    with open(path, 'w') as f:
        f.write("# GTOC4 solution, J2000 heliocentric ecliptic frame\n")
        f.write(f"# launch MJD {_mjd(mission.launch_epoch):.6f}, "
                f"|v_inf| {np.linalg.norm(mission.v_infinity):.3f} m/s, "
                f"tau {mission.duration/DAY:.3f} d\n")
        f.write(f"# J = {mission.J}, K = m_f = {mission.K:.6f} kg, "
                f"sequence: {' '.join(mission.visited)}\n")
        f.write("# MJD x[km] y[km] z[km] vx[km/s] vy[km/s] vz[km/s] m[kg] Tx[N] Ty[N] Tz[N] body\n")
        for (epoch, r, v, m, thrust), body in zip(mission.samples, mission.bodies):
            f.write(f"{_mjd(epoch):.10f} "
                    f"{r[0]/KM:.6f} {r[1]/KM:.6f} {r[2]/KM:.6f} "
                    f"{v[0]/KM:.9f} {v[1]/KM:.9f} {v[2]/KM:.9f} "
                    f"{m:.6f} {thrust[0]:.9f} {thrust[1]:.9f} {thrust[2]:.9f} {body}\n")
    return path

def read_solution(path):
    """Parses a solution file back into (epochs, positions, velocities, masses, thrusts, bodies),
    all in SI units and seconds-since-J2000. Comment lines are discarded unread."""
    epochs, positions, velocities, masses, thrusts, bodies = [], [], [], [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            fields = line.split()
            if len(fields) != 12:
                raise ValueError(f"expected 12 columns, got {len(fields)}: {line!r}")
            values = [float(x) for x in fields[:11]]
            epochs.append(_epoch(values[0]))
            positions.append(np.array(values[1:4]) * KM)
            velocities.append(np.array(values[4:7]) * KM)
            masses.append(values[7])
            thrusts.append(np.array(values[8:11]))
            bodies.append(fields[11])
    return (np.array(epochs), np.array(positions), np.array(velocities), np.array(masses),
            np.array(thrusts), bodies)

def check_solution(path, ephemerides=None):
    """Independently re-verifies every rule from a solution file alone, and re-derives J and K.

    Returns dict(J=..., K=..., violations=[...]). All violations are collected rather than raising
    on the first, so a mutated file is reported against every rule it actually breaks -- a test can
    then assert on the specific reason it expects.

    `ephemerides` maps body name -> Keplerian element dict; the GTOC4 catalogue is not loaded
    implicitly, because a checker that silently guesses which ephemeris a name refers to is not
    checking anything."""
    epochs, positions, velocities, masses, thrusts, bodies = read_solution(path)
    violations = []
    if len(epochs) < 2:
        return dict(J=0, K=float('nan'), violations=['solution has fewer than two samples'])

    # -- launch
    launch_mjd = _mjd(epochs[0])
    if not (LAUNCH_WINDOW_MJD[0] - RULE_EPS <= launch_mjd <= LAUNCH_WINDOW_MJD[1] + RULE_EPS):
        violations.append(f"launch epoch MJD {launch_mjd:.4f} outside {LAUNCH_WINDOW_MJD}")
    if bodies[0] != 'EARTH':
        violations.append(f"first sample is at {bodies[0]!r}, not EARTH: launch must be from Earth")
    else:
        earth = catalog.earth_initial_state(epochs[0], mu)
        launch_position_error = np.linalg.norm(positions[0] - earth[:3])
        v_infinity = np.linalg.norm(velocities[0] - earth[3:])
        if launch_position_error > accuracy_position:
            violations.append(f"launch position is {launch_position_error/KM:.1f} km from Earth")
        if v_infinity > scape_velocity_max + RULE_EPS:
            violations.append(f"launch |v_inf| = {v_infinity:.1f} m/s exceeds {scape_velocity_max:.1f} m/s")

    # -- duration, sampling, mass, thrust
    duration = epochs[-1] - epochs[0]
    if duration > time_mission_max + RULE_EPS:
        violations.append(f"mission duration tau = {duration/(365.25*DAY):.4f} yr exceeds "
                          f"{time_mission_max/(365.25*DAY):.1f} yr")
    if np.any(np.diff(epochs) <= 0):
        violations.append("epochs are not strictly increasing")
    if np.any(np.diff(epochs) > DAY + RULE_EPS):
        violations.append(f"sample spacing reaches {np.max(np.diff(epochs))/DAY:.4f} d, "
                          f"exceeding the one-day increment")
    if masses[-1] < spacecraft_dry_mass - RULE_EPS:
        violations.append(f"final mass {masses[-1]:.3f} kg is below the "
                          f"{spacecraft_dry_mass:.1f} kg minimum")
    if np.any(np.diff(masses) > RULE_EPS):
        violations.append("mass increases somewhere along the trajectory")
    thrust_magnitudes = np.linalg.norm(thrusts, axis=1)
    if np.any(thrust_magnitudes > thrust_max + RULE_EPS):
        violations.append(f"peak thrust {thrust_magnitudes.max():.4f} N exceeds {thrust_max} N")

    # -- dynamical consistency: every step must be two-body + the declared constant thrust. This is
    # what rules out a gravity assist (no third body may bend the trajectory) and any delta-v not
    # paid for out of the thrust and mass columns.
    for k in range(len(epochs) - 1):
        state = np.concatenate([positions[k], velocities[k], [masses[k]]])
        predicted = dynamics.propagate(state, thrusts[k], epochs[k + 1] - epochs[k],
                                       10, mu, Isp_engine)
        position_error = np.linalg.norm(predicted[:3] - positions[k + 1])
        velocity_error = np.linalg.norm(predicted[3:6] - velocities[k + 1])
        mass_error = abs(predicted[6] - masses[k + 1])
        if (position_error > DYNAMICS_POSITION_TOLERANCE or
                velocity_error > DYNAMICS_VELOCITY_TOLERANCE or mass_error > 1e-3):
            violations.append(
                f"sample {k}->{k+1} (MJD {_mjd(epochs[k]):.4f}) is not consistent with two-body "
                f"motion under the declared thrust: {position_error/KM:.1f} km, "
                f"{velocity_error:.4f} m/s, {mass_error:.6f} kg off")
            break  # one report is enough; the rest would cascade

    # -- encounters
    encounters = [(k, bodies[k]) for k in range(1, len(bodies)) if bodies[k] != '-']
    if not encounters:
        violations.append("no encounters recorded; the mission is not scorable")
        return dict(J=0, K=float(masses[-1]), violations=violations)
    if encounters[-1][0] != len(bodies) - 1:
        violations.append("the final sample is not an encounter; the mission must end with a rendezvous")

    seen = []
    for k, name in encounters:
        if name in seen:
            violations.append(f"asteroid {name!r} is visited more than once")
        seen.append(name)
        if ephemerides is None or name not in ephemerides:
            violations.append(f"no ephemeris available for {name!r}; encounter cannot be verified")
            continue
        target = catalog.target_states([ephemerides[name]], epochs[k], mu)[0]
        position_error = np.linalg.norm(positions[k] - target[:3])
        if position_error > accuracy_position + RULE_EPS:
            violations.append(f"{name!r} at MJD {_mjd(epochs[k]):.4f}: position miss "
                              f"{position_error/KM:.3f} km exceeds {accuracy_position/KM:.0f} km")
        if (k, name) == encounters[-1]:
            velocity_error = np.linalg.norm(velocities[k] - target[3:])
            if velocity_error > accuracy_velocity + RULE_EPS:
                violations.append(f"rendezvous with {name!r}: velocity miss {velocity_error:.4f} m/s "
                                  f"exceeds {accuracy_velocity:.1f} m/s")

    rendezvous_name = encounters[-1][1]
    if rendezvous_name in [name for _, name in encounters[:-1]]:
        violations.append(f"rendezvous target {rendezvous_name!r} was already visited earlier")

    flybys = {name for _, name in encounters[:-1]}
    return dict(J=len(flybys), K=float(masses[-1]), violations=violations)
