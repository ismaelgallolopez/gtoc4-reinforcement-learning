# Phase 4 of the legality track: turn a pool of asteroids into an ordered tour.
#
# Two layers, deliberately separated:
#
#   PLANNING (impulsive)   Lambert arcs with impulsive delta-v charged against the Phase 1b
#                          thrust-limited budget at a 0.6 duty cycle. Fast enough (~130 us per
#                          Lambert solve) to rank hundreds of candidates per decision, which is
#                          what both the greedy baseline and the RL sequencer need.
#
#   RENDERING (thrust)     the chosen sequence is then flown for real with a guidance law that
#                          never commands more than thrust_max, producing the per-day
#                          (state, thrust) samples a legal solution file needs. An impulsive plan
#                          is not a legal trajectory; only the rendered one is, and only the
#                          rendered one is scored.
#
# Two guidance laws, both built on the same Lambert solver:
#
#   fly_flyby_leg      shrinking aim: each day, solve Lambert from the current position to the
#                      target's position at the fixed arrival epoch and thrust along the required
#                      velocity correction. The correction drives itself to zero -- once it does,
#                      the spacecraft is on the ballistic arc that arrives exactly -- so the miss
#                      distance collapses to metres. Position only; the arrival velocity is
#                      whatever the arc gives, which is all a flyby needs.
#
#   fly_rendezvous_leg sliding aim: the same law, but aiming at where the target will be a *lead
#                      time* ahead rather than at a fixed arrival epoch, with the lead ramped down
#                      over the leg. A fixed lead makes this a pursuit that converges onto the
#                      target's own orbit, so the velocity error falls with the position error;
#                      ramping the lead to a few days then closes the last of it. This is what
#                      makes a rendezvous reachable at all: matching ~1.8 km/s of arrival velocity
#                      at 0.135 N takes ~230 days of continuous braking, so the arrival cannot be
#                      a terminal correction on a ballistic arc -- it has to be the whole last part
#                      of the leg, which is exactly what the pursuit produces.
import numpy as np

import catalog
import curriculum
import dynamics
import lambert
from constants import (sun_gravitational_parameter as mu, Isp_engine, thrust_max,
                        spacecraft_dry_mass, spacecraft_wet_mass, accuracy_position,
                        accuracy_velocity, scape_velocity_max, time_mission_max)

DAY = 86400.0
TOF_BUCKETS_DAYS = (60, 150, 300, 600, 1200)
DUTY_CYCLE = 0.6
# Lambert is run on at most this many pre-screened candidates per bucket. Default: no pruning.
# A vectorised pre-screen was tried first -- rank by |(r_target - r)/tof - v|, the rectilinear
# free-flight departure velocity, at one array operation for the whole pool against ~130 us per
# Lambert solve. Measured against the full pool one leg into a tour, it is far too weak to prune
# with: keeping its best 60 raised the cheapest reachable dv1 at a 300-day time of flight from
# 398 m/s to 12880 m/s, and its best 600 still missed the 398 m/s option entirely, because the
# proxy ignores the gravitational turning that dominates a real transfer. Kept as a knob for
# callers that must cap the per-decision cost (the RL sequencer), off by default.
PRESCREEN = None
RENDEZVOUS_LEAD_DAYS = (200.0, 10.0)   # pursuit lead, start -> end
RENDEZVOUS_RESERVE_DAYS = 600.0        # minimum mission time left before another flyby is added
RENDEZVOUS_AIM_TOFS_DAYS = (200, 300, 400, 600, 900, 1200)  # arcs used to rank and aim
RENDEZVOUS_DURATIONS_DAYS = (1200, 1800, 2400)  # candidate lengths for the final pursuit leg
DIVERGENCE_LIMIT = 20 * 1.495978707e11  # 20 AU; a pursuit past this has diverged, abort it early
# The launch v_inf is backed off 1 mm/s from the 4.0 km/s legal maximum. Sitting exactly on the
# boundary is legal in exact arithmetic but not after a round trip through the solution file: the
# checker recomputes |v_inf| from the written columns, and at that precision an exactly-4000.0 m/s
# departure comes back as 4000.000000001 m/s and is reported as a violation. 1 mm/s out of 4 km/s
# is 2.5e-7 of the allowance, far below anything the rest of the model resolves.
V_INFINITY_LIMIT = scape_velocity_max - 1e-3

def leg_candidates(state, epoch, pool, pool_index, tof_days, visited,
                    v_infinity_credit=0.0, prescreen=PRESCREEN):
    """Ranked leg options from `state` at `epoch` for one time-of-flight bucket.

    Returns a list of dicts sorted by dv1, each with the impulsive costs of the Lambert arc:
    dv1 (departure, after crediting any launch v_inf) and dv2 (arrival velocity match, needed
    only for a rendezvous).

    `prescreen`, if given, caps how many candidates Lambert is actually run on; see the PRESCREEN
    comment at the top of this module for what that proxy costs in solution quality."""
    tof = tof_days * DAY
    arrival = epoch + tof
    targets = catalog.target_states(pool, arrival, mu)

    proxy = np.linalg.norm((targets[:, :3] - state[:3]) / tof - state[3:6], axis=1)
    excluded = [k for k, ast in enumerate(pool) if ast['name'] in visited]
    proxy[excluded] = np.inf
    order = np.argsort(proxy)
    if prescreen is not None:
        order = order[:prescreen]

    out = []
    for k in order:
        if not np.isfinite(proxy[k]):
            break
        try:
            v1, v2 = lambert.solve(state[:3], targets[k][:3], tof, mu)
        except lambert.LambertError:
            continue
        dv1 = max(0.0, np.linalg.norm(v1 - state[3:6]) - v_infinity_credit)
        dv2 = np.linalg.norm(targets[k][3:] - v2)
        out.append(dict(index=pool_index[k], pool_slot=int(k), name=pool[k]['name'],
                        tof_days=tof_days, arrival=arrival, dv1=dv1, dv2=dv2,
                        v_departure=v1, v_arrival=v2, r_arrival=targets[k][:3].copy(),
                        v_target=targets[k][3:].copy()))
    out.sort(key=lambda c: c['dv1'])
    return out

def leg_feasible(candidate, mass, require_velocity_match, duty_cycle=DUTY_CYCLE):
    """Does the leg's impulsive cost fit the thrust-limited delta-v budget over its own time of
    flight, derated by the duty cycle?

    A flyby is charged dv1 only. The brief's oracle charges dv1 + dv2 unconditionally; that is
    the right cost for a rendezvous, but a flyby never performs the arrival impulse -- the
    spacecraft physically continues on the Lambert arrival velocity, and that velocity is exactly
    the state the next leg departs from. Charging for an impulse that is never applied is not
    conservatism, it would simply price a different mission than the one being flown."""
    cost = candidate['dv1'] + (candidate['dv2'] if require_velocity_match else 0.0)
    return cost < duty_cycle * curriculum.delta_v_budget(candidate['tof_days'] * DAY, mass)

def burn_mass(mass, delta_v):
    """Rocket-equation mass after spending `delta_v`, or None if it would go below the dry mass."""
    final = mass * np.exp(-delta_v / (Isp_engine * dynamics.g0))
    return None if final < spacecraft_dry_mass else final

def apply_leg(state, candidate, require_velocity_match):
    """Impulsive-model state transition: arrive at the target's position with the Lambert arrival
    velocity (or the target's own, for a rendezvous), having paid the corresponding delta-v."""
    delta_v = candidate['dv1'] + (candidate['dv2'] if require_velocity_match else 0.0)
    mass = burn_mass(state[6], delta_v)
    if mass is None:
        return None
    velocity = candidate['v_target'] if require_velocity_match else candidate['v_arrival']
    return np.concatenate([candidate['r_arrival'], velocity, [mass]])

# -- rendering: guidance laws that respect the thrust limit ------------------------------------

def _guided_step(state, epoch, step, correction):
    """One control step under a velocity correction: thrust along it, at the magnitude that would
    deliver it over this step, never above thrust_max. Returns (thrust, next_state)."""
    magnitude = np.linalg.norm(correction)
    if magnitude <= 0.0:
        thrust = np.zeros(3)
    else:
        thrust = min(thrust_max, magnitude * state[6] / step) * correction / magnitude
    return thrust, dynamics.propagate(state, thrust, step, 10, mu, Isp_engine)

def fly_flyby_leg(state, epoch, target, arrival):
    """Shrinking-aim Lambert guidance. Returns (samples, final_state, miss_distance) where samples
    is [(epoch, state, thrust_applied_from_here)], or None if the leg cannot be flown."""
    state = np.asarray(state, dtype=np.float64).copy()
    samples = []
    while epoch < arrival - 1e-6:
        step = min(DAY, arrival - epoch)
        target_at_arrival = catalog.target_states([target], arrival, mu)[0]
        try:
            v_required, _ = lambert.solve(state[:3], target_at_arrival[:3], arrival - epoch, mu)
        except lambert.LambertError:
            return None
        thrust, next_state = _guided_step(state, epoch, step, v_required - state[3:6])
        samples.append((epoch, state.copy(), thrust))
        state, epoch = next_state, epoch + step
        if state[6] < spacecraft_dry_mass:
            return None
    final_target = catalog.target_states([target], arrival, mu)[0]
    return samples, state, float(np.linalg.norm(state[:3] - final_target[:3]))

def fly_rendezvous_leg(state, epoch, target, duration, lead_days=RENDEZVOUS_LEAD_DAYS):
    """Sliding-aim Lambert pursuit. Returns (samples, final_state, miss_distance, velocity_miss),
    or None if the leg cannot be flown."""
    state = np.asarray(state, dtype=np.float64).copy()
    start, end = epoch, epoch + duration
    samples = []
    while epoch < end - 1e-6:
        step = min(DAY, end - epoch)
        fraction = (epoch - start) / duration
        lead = max(2.0, lead_days[0] + (lead_days[1] - lead_days[0]) * fraction) * DAY
        aim = catalog.target_states([target], epoch + lead, mu)[0]
        try:
            v_required, _ = lambert.solve(state[:3], aim[:3], lead, mu)
        except lambert.LambertError:
            return None
        thrust, next_state = _guided_step(state, epoch, step, v_required - state[3:6])
        samples.append((epoch, state.copy(), thrust))
        state, epoch = next_state, epoch + step
        if state[6] < spacecraft_dry_mass or np.linalg.norm(state[:3]) > DIVERGENCE_LIMIT:
            return None
    final_target = catalog.target_states([target], end, mu)[0]
    return (samples, state, float(np.linalg.norm(state[:3] - final_target[:3])),
            float(np.linalg.norm(state[3:6] - final_target[3:])))

def _append_leg(mission, samples, launch_epoch, v_infinity, ephemerides):
    """Appends one flown leg's per-day samples to `mission`, creating it on the first leg (the
    launch v_inf is only known once the first leg has been chosen). samples[0] is the leg's start
    state, which is already the mission's last sample, so it is skipped."""
    from mission import Mission
    if mission is None:
        mission = Mission(launch_epoch, v_infinity=v_infinity, ephemerides=ephemerides)
    for k in range(1, len(samples)):
        mission.advance(samples[k][0], samples[k][1][:3], samples[k][1][3:6], samples[k][1][6],
                        samples[k - 1][2])
    return mission

def finish_with_rendezvous(mission, state, epoch, launch_epoch, pool, visited, top_k=15,
                            duty_cycle=DUTY_CYCLE, prescreen=None,
                            aim_tofs_days=RENDEZVOUS_AIM_TOFS_DAYS,
                            durations_days=RENDEZVOUS_DURATIONS_DAYS, verbose=False):
    """Terminal rendezvous, shared by the greedy baseline and the RL sequencer so the two are
    scored on the same procedure. Returns (mission, report).

    Two independent knobs are searched, because they turn out not to be the same quantity:

    * the *aim* time of flight sets the Lambert arc used to rank candidates by dv1 + dv2 and, when
      the mission has not launched yet, the direction the launch v_inf is pointed in;
    * the *duration* is how long the pursuit guidance is then flown for.

    Collapsing the two -- ranking and flying at the same time of flight, which was the first thing
    tried -- fails badly. For one target reachable from the Phase 2 launch epoch, aiming the v_inf
    at its 400-day arc and pursuing for 1200 days lands 0.336 km away with a 0.000 m/s velocity
    error and 1263 kg left, while aiming at its 300-day or 600-day arc and pursuing for the same
    1200 days diverges to 2.5e8 km. The aim sets which orbit the spacecraft is thrown onto; the
    duration sets how long it has to converge onto the target's.

    Candidates are flown, not trusted: the top `top_k` by impulsive cost are each pursued over each
    duration, and the first that lands inside 1000 km and 1 m/s is taken. A pursuit costs one
    Lambert solve per day, about 0.15 s, so this is cheaper than the ranking pass that produced the
    candidates, and it removes any need to believe an impulsive estimate for a leg that is not
    flown impulsively."""
    remaining = time_mission_max - (epoch - launch_epoch)
    if remaining < 30 * DAY:
        return mission, dict(achieved=False, reason='no mission time left')
    pool_index = list(range(len(pool)))
    ephemerides = {ast['name']: ast for ast in pool}

    # with no flyby flown yet the mission has not launched, so the launch v_inf is still available
    # and is credited against -- and aimed at -- the rendezvous leg's own departure
    from_launch = mission is None
    credit = scape_velocity_max if from_launch else 0.0

    durations = [d for d in durations_days if d * DAY <= remaining] or [int(remaining / DAY)]
    aim_tofs = [t for t in aim_tofs_days if t * DAY <= remaining]
    if not aim_tofs:
        aim_tofs = [int(remaining / DAY)]

    options = []
    for aim_tof in aim_tofs:
        for candidate in leg_candidates(state, epoch, pool, pool_index, aim_tof, visited,
                                         credit, prescreen):
            if leg_feasible(candidate, state[6], True, duty_cycle):
                options.append(candidate)
    if not options:
        return mission, dict(achieved=False, reason='no feasible rendezvous candidate')
    options.sort(key=lambda c: c['dv1'] + c['dv2'])

    attempts = []
    for candidate in options[:top_k]:
        start_state = launch_state(launch_epoch, candidate) if from_launch else state
        for duration_days in durations:
            duration = duration_days * DAY
            flown = fly_rendezvous_leg(start_state, epoch, ephemerides[candidate['name']], duration)
            if flown is None:
                attempts.append(dict(name=candidate['name'], aim_tof_days=candidate['tof_days'],
                                     duration_days=duration_days, reason='diverged or ran dry'))
                continue
            samples, final_state, miss, velocity_miss = flown
            attempts.append(dict(name=candidate['name'], aim_tof_days=candidate['tof_days'],
                                 duration_days=duration_days, miss_km=miss / 1e3,
                                 velocity_miss_ms=velocity_miss, mass_kg=float(final_state[6])))
            if verbose:
                print(f"    rendezvous try {candidate['name']:12s} aim {candidate['tof_days']:5d} d "
                      f"pursue {duration_days:5d} d  miss {miss/1e3:12.3f} km  "
                      f"dv {velocity_miss:9.3f} m/s  m {final_state[6]:7.1f} kg")
            if miss > accuracy_position or velocity_miss > accuracy_velocity:
                continue
            v_infinity = (start_state[3:6] - catalog.earth_initial_state(launch_epoch, mu)[3:]
                          if from_launch else None)
            mission = _append_leg(mission, samples, launch_epoch, v_infinity, ephemerides)
            mission.rendezvous(candidate['name'], epoch + duration, final_state[:3],
                               final_state[3:6], final_state[6], samples[-1][2])
            return mission, dict(achieved=True, name=candidate['name'],
                                 aim_tof_days=candidate['tof_days'], duration_days=duration_days,
                                 miss_km=miss / 1e3, velocity_miss_ms=velocity_miss,
                                 mass_kg=float(final_state[6]), attempts=attempts)
    return mission, dict(achieved=False, reason=f'none of the top {top_k} candidates closed',
                         attempts=attempts)

def greedy_tour(launch_epoch, pool, max_legs=20, reserve_days=RENDEZVOUS_RESERVE_DAYS,
                 duty_cycle=DUTY_CYCLE, top_k=6, buckets=TOF_BUCKETS_DAYS, prescreen=None,
                 rendezvous_top_k=20, verbose=False):
    """Cheapest-next greedy, verified as it goes, keeping the longest chain that still ends in a
    legal rendezvous.

    At each decision every feasible (target, time-of-flight) option is ranked by dv1 and the
    cheapest `top_k` are flown for real with the guidance; the first that closes inside the 1000 km
    flyby tolerance is taken and the tour continues from the *flown* state, not the impulsive one.

    A terminal rendezvous is then attempted after *every* flyby (including before the first one),
    and the best mission that actually ends in one is what comes back. Extending the flyby chain
    and still being able to stop are in direct competition -- each flyby leaves the spacecraft on a
    less Earth-like orbit with less propellant, and a rendezvous costs a full velocity match -- so
    the stopping point cannot be fixed in advance by a time reserve. Checking directly costs one
    rendezvous round per leg, a few seconds, and it is the difference between a scorable mission
    and none."""
    import copy
    ephemerides = {ast['name']: ast for ast in pool}
    pool_index = list(range(len(pool)))
    state = launch_state(launch_epoch)
    epoch = launch_epoch
    visited, mission, v_infinity = set(), None, None
    report = dict(legs=[], stopped_because=None, rendezvous_rounds=[])
    best = (None, dict(achieved=False, reason='no rendezvous attempted'), -1)

    for leg in range(max_legs + 1):
        # can the mission stop here, legally? checked before the first flyby too, so a tour that
        # cannot afford any flyby at all still produces a scorable J = 0 mission
        if True:
            finished, rendezvous_report = finish_with_rendezvous(
                copy.deepcopy(mission), state, epoch, launch_epoch, pool, visited,
                top_k=rendezvous_top_k, duty_cycle=duty_cycle, prescreen=prescreen,
                verbose=verbose)
            report['rendezvous_rounds'].append(dict(after_flybys=leg, **{
                k: v for k, v in rendezvous_report.items() if k != 'attempts'}))
            if rendezvous_report.get('achieved') and leg > best[2]:
                best = (finished, rendezvous_report, leg)
                if verbose:
                    print(f"  -> scorable after {leg} flybys: rendezvous "
                          f"{rendezvous_report['name']}, m_f {rendezvous_report['mass_kg']:.1f} kg")

        if leg == max_legs:
            report['stopped_because'] = 'leg limit reached'
            break
        remaining = time_mission_max - (epoch - launch_epoch)
        if remaining <= reserve_days * DAY:
            report['stopped_because'] = 'mission time down to the rendezvous reserve'
            break
        credit = scape_velocity_max if leg == 0 else 0.0
        options = []
        for tof_days in buckets:
            if tof_days * DAY > remaining - reserve_days * DAY:
                continue
            options += [c for c in leg_candidates(state, epoch, pool, pool_index, tof_days,
                                                  visited, credit, prescreen)
                        if leg_feasible(c, state[6], False, duty_cycle)]
        if not options:
            report['stopped_because'] = 'no feasible flyby candidate'
            break
        options.sort(key=lambda c: c['dv1'])

        accepted = None
        for rank, candidate in enumerate(options[:top_k]):
            start_state = launch_state(launch_epoch, candidate) if leg == 0 else state
            flown = fly_flyby_leg(start_state, epoch, ephemerides[candidate['name']],
                                  candidate['arrival'])
            if flown is None or flown[2] > accuracy_position:
                continue
            accepted = (rank, candidate, flown, start_state)
            break
        if accepted is None:
            report['stopped_because'] = f'none of the top {top_k} candidates closed to 1000 km'
            break

        rank, candidate, (samples, final_state, miss), start_state = accepted
        if leg == 0:
            v_infinity = start_state[3:6] - catalog.earth_initial_state(launch_epoch, mu)[3:]
        mission = _append_leg(mission, samples, launch_epoch, v_infinity, ephemerides)
        mission.flyby(candidate['name'], candidate['arrival'], final_state[:3], final_state[3:6],
                      final_state[6], samples[-1][2])
        state, epoch = final_state, candidate['arrival']
        visited.add(candidate['name'])
        report['legs'].append(dict(name=candidate['name'], tof_days=candidate['tof_days'],
                                    dv1=candidate['dv1'], rank=rank, miss_km=miss / 1e3,
                                    mass_kg=float(final_state[6])))
        if verbose:
            print(f"  leg {leg}: flyby {candidate['name']:12s} tof {candidate['tof_days']:5d} d  "
                  f"dv1 {candidate['dv1']:8.1f} m/s  rank {rank}  miss {miss/1e3:8.3f} km  "
                  f"m {final_state[6]:7.1f} kg")

    report['rendezvous'] = best[1]
    report['scorable_after_flybys'] = best[2] if best[0] is not None else None
    return best[0], report

def launch_state(launch_epoch, first_candidate=None):
    """Earth's state at launch, plus the v_inf that best serves the first leg: the launch excess
    is a free impulse of arbitrary direction, so it is pointed straight along the first leg's
    required departure correction and sized at min(4 km/s, what that correction needs)."""
    earth = catalog.earth_initial_state(launch_epoch, mu)
    velocity = earth[3:].copy()
    if first_candidate is not None:
        correction = first_candidate['v_departure'] - earth[3:]
        magnitude = np.linalg.norm(correction)
        if magnitude > 0:
            velocity = velocity + correction / magnitude * min(V_INFINITY_LIMIT, magnitude)
    return np.concatenate([earth[:3], velocity, [spacecraft_wet_mass]])
