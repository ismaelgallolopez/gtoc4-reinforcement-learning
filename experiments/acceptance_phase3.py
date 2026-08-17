# Phase 3 acceptance tests for the legality track: a hand-built trivial legal mission is written
# out, read back and re-verified by the independent checker (positive), then mutated four ways and
# confirmed rejected with the correct reason (negative).
#
# The trivial mission is a pure coast: launch from Earth with a legal v_inf, then drift on the
# resulting heliocentric orbit for ~300 days with the engine off. Two synthetic bodies are placed
# on that coast arc:
#   A -- shares the spacecraft's *position* at the flyby epoch but not its velocity (a real flyby)
#   B -- shares the spacecraft's whole state at the rendezvous epoch, i.e. sits on the same orbit
#        (so position and velocity both match; B and the spacecraft are coincident at every epoch)
# Real catalogue asteroids are not used because putting the spacecraft within 1000 km of one
# requires solving a transfer, which is Phase 4's job; these tests exercise the bookkeeping, the
# writer and the checker, not the trajectory design.
import os
import shutil
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

import catalog
import dynamics
from mission import Mission, LegalityError, write_solution, check_solution
from constants import sun_gravitational_parameter as mu, Isp_engine, spacecraft_wet_mass

SCRATCH_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'legality_phase3')
DAY = 86400.0
LAUNCH_MJD = 58430.0            # the best launch epoch found by the Phase 2 scan
V_INFINITY = np.array([1500.0, -2000.0, 800.0])   # |v_inf| = 2.65 km/s, within the 4 km/s limit
FLYBY_OFFSET = 100.37 * DAY     # deliberately not on a control-step boundary
RENDEZVOUS_OFFSET = 300.62 * DAY

def _body_from_state(name, state, epoch, velocity_perturbation=None):
    """A Keplerian element dict (catalog.parse_asteroids' format) for a body sitting exactly on
    `state` at `epoch`, optionally on a different orbit through the same point."""
    state = np.asarray(state, dtype=np.float64).copy()
    if velocity_perturbation is not None:
        state[3:6] = state[3:6] + velocity_perturbation
    a, e, i, omega, lan, M = dynamics.cartesian_to_keplerian(state, mu)
    return {'name': name, 'epoch': epoch, 'a': a, 'e': e, 'i': i, 'lan': lan, 'omega': omega, 'M0': M}

def _coast(state, epoch, until):
    """Zero-thrust propagation to `until`, returning the samples at one-day increments plus a
    final partial-day sample landing exactly on `until`."""
    samples = []
    while epoch < until - 1e-6:
        step = min(DAY, until - epoch)
        state = dynamics.propagate(state, np.zeros(3), step, 10, mu, Isp_engine)
        epoch += step
        samples.append((epoch, state.copy()))
    return samples

def build_legal_mission():
    launch_epoch = (LAUNCH_MJD - catalog.MJD_J2000) * DAY
    flyby_epoch = launch_epoch + FLYBY_OFFSET
    rendezvous_epoch = launch_epoch + RENDEZVOUS_OFFSET

    # first pass with no ephemerides: propagate the coast arc to find where the two bodies must sit
    earth = catalog.earth_initial_state(launch_epoch, mu)
    state = np.concatenate([earth[:3], earth[3:] + V_INFINITY, [spacecraft_wet_mass]])
    to_flyby = _coast(state, launch_epoch, flyby_epoch)
    state_at_flyby = to_flyby[-1][1]
    after_flyby = _coast(state_at_flyby, flyby_epoch, rendezvous_epoch)
    state_at_rendezvous = after_flyby[-1][1]

    ephemerides = {
        'A': _body_from_state('A', state_at_flyby, flyby_epoch,
                              velocity_perturbation=np.array([2000.0, 1000.0, -500.0])),
        'B': _body_from_state('B', state_at_rendezvous, rendezvous_epoch),
    }

    # second pass: replay the same arc through the Mission, which validates every step
    m = Mission(launch_epoch, v_infinity=V_INFINITY, ephemerides=ephemerides)
    for epoch, s in to_flyby[:-1]:
        m.advance(epoch, s[:3], s[3:6], s[6])
    m.flyby('A', flyby_epoch, state_at_flyby[:3], state_at_flyby[3:6], state_at_flyby[6])
    for epoch, s in after_flyby[:-1]:
        m.advance(epoch, s[:3], s[3:6], s[6])
    m.rendezvous('B', rendezvous_epoch, state_at_rendezvous[:3], state_at_rendezvous[3:6],
                 state_at_rendezvous[6])
    return m, ephemerides

def test_positive():
    print("--- positive: hand-built legal mission, written out and re-verified from the file ---")
    m, ephemerides = build_legal_mission()

    print(f"launch                 : MJD {LAUNCH_MJD}, |v_inf| = {np.linalg.norm(V_INFINITY):.1f} m/s")
    print(f"samples                : {len(m.samples)}")
    print(f"duration tau           : {m.duration/DAY:.3f} d = {m.duration/(365.25*DAY):.4f} yr")
    print(f"sequence               : {' -> '.join(m.visited)}")
    print(f"J (Mission)            : {m.J}")
    print(f"K = m_f (Mission)      : {m.K:.3f} kg")

    os.makedirs(SCRATCH_DIR, exist_ok=True)
    path = os.path.join(SCRATCH_DIR, 'solution.txt')
    write_solution(m, path)
    print(f"written                : {path} ({os.path.getsize(path)} bytes)")

    result = check_solution(path, ephemerides=ephemerides)
    print(f"checker violations     : {result['violations']}")
    print(f"J (checker)            : {result['J']}")
    print(f"K (checker)            : {result['K']:.3f} kg")
    assert result['violations'] == [], f"legal mission was rejected: {result['violations']}"
    assert result['J'] == m.J == 1, f"expected J = 1, got {result['J']} / {m.J}"
    assert abs(result['K'] - m.K) < 1e-6
    print("PASS: legal mission accepted, J and K re-derived from the file\n")
    return path, ephemerides

def test_encounter_refusal(ephemerides):
    """An encounter that breaks a rule must be refused by Mission, not silently recorded."""
    print("--- refusal: Mission rejects illegal encounters at construction time ---")
    launch_epoch = (LAUNCH_MJD - catalog.MJD_J2000) * DAY
    cases = []

    m, _ = build_legal_mission()
    try:
        m.advance(m.epoch + DAY, m.r, m.v, m.m)
        cases.append(('advance after the final rendezvous', None))
    except LegalityError as exc:
        cases.append(('advance after the final rendezvous', str(exc)))

    try:
        Mission(launch_epoch, v_infinity=[5000.0, 0.0, 0.0], ephemerides=ephemerides)
        cases.append(('|v_inf| = 5 km/s', None))
    except LegalityError as exc:
        cases.append(('|v_inf| = 5 km/s', str(exc)))

    try:
        Mission((50000.0 - catalog.MJD_J2000) * DAY, ephemerides=ephemerides)
        cases.append(('launch at MJD 50000', None))
    except LegalityError as exc:
        cases.append(('launch at MJD 50000', str(exc)))

    m2 = Mission(launch_epoch, v_infinity=V_INFINITY, ephemerides=ephemerides)
    try:
        m2.flyby('A', launch_epoch + DAY, m2.r, m2.v, m2.m)
        cases.append(('flyby of A from 1e8 km away', None))
    except LegalityError as exc:
        cases.append(('flyby of A from 1e8 km away', str(exc)))

    for label, reason in cases:
        print(f"  {label:<38} -> {reason}")
        assert reason is not None, f"{label} was accepted but should have been refused"
    print("PASS: all four illegal operations refused\n")

def _mutate(path, out_name, transform):
    with open(path) as f:
        lines = f.readlines()
    out_path = os.path.join(SCRATCH_DIR, out_name)
    with open(out_path, 'w') as f:
        f.writelines(transform(lines))
    return out_path

def test_negative(path, ephemerides):
    print("--- negative: four mutations of that same file, each must be rejected ---")
    data_indices = [k for k, line in enumerate(lines_of(path)) if not line.startswith('#')]

    def relabel(index, name):
        def transform(lines):
            lines = list(lines)
            fields = lines[index].split()
            fields[-1] = name
            lines[index] = ' '.join(fields) + '\n'
            return lines
        return transform

    def shift_final_epoch(lines):
        lines = list(lines)
        fields = lines[data_indices[-1]].split()
        launch_mjd = float(lines[data_indices[0]].split()[0])
        fields[0] = f"{launch_mjd + 10.5 * 365.25:.10f}"
        lines[data_indices[-1]] = ' '.join(fields) + '\n'
        return lines

    def starve_mass(lines):
        out = []
        for k, line in enumerate(lines):
            if k in data_indices:
                fields = line.split()
                fields[7] = '450.000000'
                out.append(' '.join(fields) + '\n')
            else:
                out.append(line)
        return out

    flyby_index = next(k for k in data_indices
                       if lines_of(path)[k].split()[-1] == 'A')
    midcoast_index = data_indices[len(data_indices) // 4]

    cases = [
        ('revisit an asteroid', 'mutated_revisit.txt', relabel(midcoast_index, 'A'),
         'visited more than once'),
        ('tau = 10.5 years', 'mutated_tau.txt', shift_final_epoch,
         'exceeds 10.0 yr'),
        ('m_f below 500 kg', 'mutated_mass.txt', starve_mass,
         'below the 500.0 kg minimum'),
        ('rendezvous target already flown by', 'mutated_rendezvous.txt', relabel(flyby_index, 'B'),
         'was already visited earlier'),
    ]

    for label, out_name, transform, expected in cases:
        out_path = _mutate(path, out_name, transform)
        result = check_solution(out_path, ephemerides=ephemerides)
        print(f"  {label}")
        for violation in result['violations']:
            print(f"      - {violation}")
        assert result['violations'], f"{label}: mutated file was accepted"
        assert any(expected in v for v in result['violations']), \
            f"{label}: expected a violation containing {expected!r}, got {result['violations']}"
        print(f"      => rejected, expected reason {expected!r} present\n")
    print("PASS: all four mutations rejected with the correct reason\n")

def lines_of(path):
    with open(path) as f:
        return f.readlines()

if __name__ == '__main__':
    if os.path.isdir(SCRATCH_DIR):
        shutil.rmtree(SCRATCH_DIR)
    path, ephemerides = test_positive()
    test_encounter_refusal(ephemerides)
    test_negative(path, ephemerides)
    print("all Phase 3 acceptance tests passed")
