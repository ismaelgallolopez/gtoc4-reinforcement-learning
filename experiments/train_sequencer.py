# Phase 4.4 of the legality track: an RL sequencer over the same leg oracle the greedy baseline
# uses.
#
# The MDP is the *sequencing* problem, not the piloting problem: an action picks which asteroid to
# go to next and how long to take, and the trajectory that results is produced by the Lambert
# oracle, not by the policy. This is deliberate -- WP4-WP11 established that direct thrust-vector
# control does not reach GTOC4 tolerances in this training budget, and Phase 4's guidance laws
# already solve the piloting part exactly.
#
# Action (MultiDiscrete([K, n_buckets])): the k-th cheapest feasible candidate at time-of-flight
# bucket b, ranked by dv1 exactly as the greedy baseline ranks. Greedy is the k = 0, argmin-over-
# buckets policy, so the agent's whole job is to learn when *not* to take the cheapest leg.
#
# Reward: +1 per flyby, plus a terminal term. The terminal term is what the greedy baseline has no
# way to express: a flyby leaves the spacecraft on a less Earth-like orbit with less propellant,
# and the mission is only scorable if it can still end in a rendezvous, so flying one more flyby
# can cost the entire score. The agent is paid RENDEZVOUS_BONUS * m_f/m_0 if a rendezvous is still
# feasible when it stops and RENDEZVOUS_PENALTY if not.
#
# Training runs in the impulsive model (a leg costs one Lambert solve per pool member per bucket,
# ~130 us each); the policy's rollout is then rendered with the real thrust-limited guidance and
# scored by the Phase 3 checker, the same way the greedy tour is.
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

import catalog
import mission as mission_module
import sequencer
from constants import (sun_gravitational_parameter as mu, spacecraft_wet_mass, time_mission_max,
                        scape_velocity_max)
import run_sequencer

CATALOG_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'gtoc4_problem_data.txt')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'legality_tours')
DAY = 86400.0
AU = 1.495978707e11
V_REF = np.sqrt(mu / AU)

K_CANDIDATES = 6
RL_BUCKETS = (150, 400, 1000)
RL_AIM_TOFS = (300, 600, 1200)
MAX_LEGS = 8
RENDEZVOUS_BONUS = 5.0
RENDEZVOUS_PENALTY = -3.0

class SequencerEnv(gym.Env):
    """Impulsive-model tour sequencing. One step = one flyby leg."""

    def __init__(self, pool, launch_epoch, buckets=RL_BUCKETS, k=K_CANDIDATES, max_legs=MAX_LEGS):
        super().__init__()
        self.pool = pool
        self.pool_index = list(range(len(pool)))
        self.launch_epoch = launch_epoch
        self.buckets = tuple(buckets)
        self.k = k
        self.max_legs = max_legs
        # observation: r/AU (3), v/V_REF (3), m/m0 (1), time used fraction (1), flybys so far (1),
        # then per (bucket, rank) the leg's dv1 normalised by its own budget (k * n_buckets)
        self.n_features = 9 + k * len(self.buckets)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.n_features,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([k, len(self.buckets)])
        self.state = None

    def _rank(self):
        """Per-bucket ranked feasible candidates from the current state."""
        remaining = time_mission_max - (self.epoch - self.launch_epoch)
        credit = scape_velocity_max if not self.visited else 0.0
        ranked = {}
        for bucket in self.buckets:
            if bucket * DAY > remaining:
                ranked[bucket] = []
                continue
            options = sequencer.leg_candidates(self.state, self.epoch, self.pool, self.pool_index,
                                                bucket, self.visited, credit)
            ranked[bucket] = [c for c in options
                              if sequencer.leg_feasible(c, self.state[6], False)][:self.k]
        return ranked

    def _observation(self):
        features = [self.state[:3] / AU, self.state[3:6] / V_REF,
                    [self.state[6] / spacecraft_wet_mass],
                    [(self.epoch - self.launch_epoch) / time_mission_max],
                    [len(self.visited) / self.max_legs]]
        costs = []
        for bucket in self.buckets:
            budget = sequencer.DUTY_CYCLE * sequencer.curriculum.delta_v_budget(bucket * DAY,
                                                                                self.state[6])
            options = self.ranked[bucket]
            for rank in range(self.k):
                costs.append(options[rank]['dv1'] / budget if rank < len(options) else 2.0)
        features.append(costs)
        return np.concatenate(features).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = sequencer.launch_state(self.launch_epoch)
        self.epoch = self.launch_epoch
        self.visited = set()
        self.sequence = []
        self.ranked = self._rank()
        return self._observation(), {}

    def _rendezvous_feasible(self):
        """Cheap impulsive check: is any unvisited target still reachable as a rendezvous? Used as
        the terminal signal during training; the real (flown) check is far too slow for 10^4
        episodes and is applied once, at render time."""
        remaining = time_mission_max - (self.epoch - self.launch_epoch)
        credit = scape_velocity_max if not self.visited else 0.0
        for aim in RL_AIM_TOFS:
            if aim * DAY > remaining:
                continue
            for candidate in sequencer.leg_candidates(self.state, self.epoch, self.pool,
                                                       self.pool_index, aim, self.visited, credit):
                if sequencer.leg_feasible(candidate, self.state[6], True):
                    return True
        return False

    def step(self, action):
        rank, bucket_index = int(action[0]), int(action[1])
        options = self.ranked[self.buckets[bucket_index]]
        if rank >= len(options):
            # asking for a candidate that does not exist ends the tour where it stands
            return self._observation(), self._terminal_reward(), True, False, self._info()

        candidate = options[rank]
        if not self.visited:
            self.state = sequencer.launch_state(self.launch_epoch, candidate)
        next_state = sequencer.apply_leg(self.state, candidate, False)
        if next_state is None:
            return self._observation(), self._terminal_reward(), True, False, self._info()

        self.state, self.epoch = next_state, candidate['arrival']
        self.visited.add(candidate['name'])
        self.sequence.append(dict(name=candidate['name'], tof_days=candidate['tof_days'],
                                   dv1=candidate['dv1'], rank=rank))
        reward = 1.0
        done = len(self.visited) >= self.max_legs
        self.ranked = self._rank()
        if done or not any(self.ranked.values()):
            done = True
            reward += self._terminal_reward()
        return self._observation(), reward, done, False, self._info()

    def _terminal_reward(self):
        if self._rendezvous_feasible():
            return RENDEZVOUS_BONUS * self.state[6] / spacecraft_wet_mass
        return RENDEZVOUS_PENALTY

    def _info(self):
        return dict(flybys=len(self.visited), mass=float(self.state[6]),
                    sequence=list(self.sequence))

def rollout(model, env):
    """One deterministic episode; returns the chosen flyby sequence."""
    obs, _ = env.reset()
    done = False
    info = {}
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
    return info['sequence']

def render_sequence(sequence, launch_epoch, pool, verbose=True):
    """Flies a chosen flyby sequence with the thrust-limited guidance and finishes it with the
    shared rendezvous procedure, returning (mission, report).

    The rendezvous is attempted after every prefix of the sequence, and the longest prefix that
    ends in one is what comes back -- the same rule the greedy tour is scored under. Without it the
    comparison would be rigged: the greedy is allowed to stop early and the agent would not be,
    even though the agent plans against an impulsive rendezvous check that is optimistic (it asks
    whether a candidate is affordable, not whether the pursuit actually converges on it)."""
    import copy
    ephemerides = {ast['name']: ast for ast in pool}
    pool_index = list(range(len(pool)))
    state = sequencer.launch_state(launch_epoch)
    epoch = launch_epoch
    visited, mission, v_infinity = set(), None, None
    flown_legs = []
    best = (None, dict(achieved=False, reason='no rendezvous attempted'), -1)

    for index, leg in enumerate(sequence + [None]):
        finished, rendezvous_report = sequencer.finish_with_rendezvous(
            copy.deepcopy(mission), state, epoch, launch_epoch, pool, visited, verbose=verbose)
        if rendezvous_report.get('achieved') and len(flown_legs) > best[2]:
            best = (finished, rendezvous_report, len(flown_legs))
            if verbose:
                print(f"  -> scorable after {len(flown_legs)} flybys: rendezvous "
                      f"{rendezvous_report['name']}, m_f {rendezvous_report['mass_kg']:.1f} kg")
        if leg is None:
            break
        arrival = epoch + leg['tof_days'] * DAY
        candidates = sequencer.leg_candidates(
            state, epoch, pool, pool_index, leg['tof_days'], visited,
            scape_velocity_max if index == 0 else 0.0)
        candidate = next((c for c in candidates if c['name'] == leg['name']), None)
        if candidate is None:
            break
        start_state = sequencer.launch_state(launch_epoch, candidate) if index == 0 else state
        flown = sequencer.fly_flyby_leg(start_state, epoch, ephemerides[leg['name']], arrival)
        if flown is None or flown[2] > 1000e3:
            if verbose:
                print(f"  leg {index}: {leg['name']} did not close -- truncating here")
            break
        samples, final_state, miss = flown
        if index == 0:
            v_infinity = start_state[3:6] - catalog.earth_initial_state(launch_epoch, mu)[3:]
        mission = sequencer._append_leg(mission, samples, launch_epoch, v_infinity, ephemerides)
        mission.flyby(leg['name'], arrival, final_state[:3], final_state[3:6], final_state[6],
                      samples[-1][2])
        state, epoch = final_state, arrival
        visited.add(leg['name'])
        flown_legs.append(dict(name=leg['name'], tof_days=leg['tof_days'], miss_km=miss / 1e3,
                                mass_kg=float(final_state[6])))
        if verbose:
            print(f"  leg {index}: flyby {leg['name']:12s} tof {leg['tof_days']:5d} d  "
                  f"miss {miss/1e3:8.3f} km  m {final_state[6]:7.1f} kg")

    return best[0], dict(legs=flown_legs, rendezvous=best[1], scorable_after_flybys=best[2])

def main(timesteps, pool_size, launch_mjd, seed, label, render_only=False):
    pool = catalog.parse_asteroids(CATALOG_PATH)
    launch_epoch = (launch_mjd - catalog.MJD_J2000) * DAY
    pool = run_sequencer.restrict_pool(pool, launch_epoch, pool_size)
    print(f"RL sequencer: launch MJD {launch_mjd}, pool {len(pool)} asteroids, "
          f"buckets {RL_BUCKETS} d, K = {K_CANDIDATES}, max legs {MAX_LEGS}, "
          f"{timesteps} timesteps, seed {seed}")

    run_dir = os.path.join(RESULTS_DIR, label)
    os.makedirs(run_dir, exist_ok=True)
    env = Monitor(SequencerEnv(pool, launch_epoch), filename=os.path.join(run_dir, 'monitor.csv'),
                  info_keywords=('flybys', 'mass'))
    if render_only:
        # re-render an already-trained policy, e.g. after a fix to the writer or the guidance
        model = PPO.load(os.path.join(run_dir, 'ppo_sequencer'), env=env)
    else:
        model = PPO('MlpPolicy', env, verbose=1, seed=seed, n_steps=256, batch_size=64)
        model.learn(total_timesteps=timesteps)
        model.save(os.path.join(run_dir, 'ppo_sequencer'))

    print("\ndeterministic rollout:")
    sequence = rollout(model, env)
    for leg in sequence:
        print(f"  {leg['name']:12s} tof {leg['tof_days']:5d} d  dv1 {leg['dv1']:8.1f} m/s  "
              f"rank {leg['rank']}")
    print(f"  planned flybys: {len(sequence)}")

    print("\nrendering:")
    mission, report = render_sequence(sequence, launch_epoch, pool)
    path, result = run_sequencer.emit(mission, pool, RESULTS_DIR, label)
    return dict(sequence=sequence, report=report, path=path, result=result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=40_000)
    parser.add_argument('--pool-size', type=int, default=200)
    parser.add_argument('--launch-mjd', type=float, default=run_sequencer.DEFAULT_LAUNCH_MJD)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--label', type=str, default='rl')
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    main(args.timesteps, args.pool_size, args.launch_mjd, args.seed, args.label, args.render_only)
