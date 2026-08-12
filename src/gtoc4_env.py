# Gymnasium environment for the low-thrust rendezvous MDP.
#
# Observation (14-D, normalised): r/AU (3), v/v_ref (3), delta_r/AU (3), delta_v/v_ref (3),
# m/m0 (1), t_remaining/t_max (1). Absolute and relative state are both included: gravity
# depends on absolute position, the rendezvous goal depends on the relative state.
#
# Action (4-D, Box(-1, 1)): RTN thrust direction (3) + throttle (1). The direction is
# normalised inside the env; if its norm is below eps, thrust is commanded off regardless of
# throttle. Throttle in [-1, 1] maps to [0, thrust_max]. RTN (radial-transverse-normal) is used
# because it separates energy change (transverse), shape change (radial) and plane change
# (normal), which is what makes the learned-controller analysis in WP7 interpretable.
#
# Reward: potential-based shaping F = Phi(s') - Phi(s), Phi = -(w_r*|delta_r|/AU + w_v*|delta_v|/v_ref),
# with gamma=1 in the shaping term so it telescopes to net progress over the episode (Ng, Harada
# & Russell 1999). A terminal bonus scaled by final mass fraction is added on rendezvous.
#
# Termination: rendezvous within tolerance, propellant exhausted, or divergence (|delta_r| > 5 AU).
# The time limit is truncation, not termination, so PPO/GAE bootstraps across it correctly.
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from tudatpy import constants as tudat_constants

import dynamics
import catalog
from constants import (sun_gravitational_parameter as mu, Isp_engine, thrust_max,
                        spacecraft_dry_mass, spacecraft_wet_mass, accuracy_position, accuracy_velocity)

AU = tudat_constants.ASTRONOMICAL_UNIT
V_REF = np.sqrt(mu / AU)  # characteristic heliocentric velocity (circular speed at 1 AU)
DIVERGENCE_LIMIT = 5 * AU
DIRECTION_EPS = 1e-6

def rtn_frame(r, v):
    """Rotation matrix from RTN to inertial coordinates: columns are R-hat, T-hat, N-hat."""
    r_hat = r / np.linalg.norm(r)
    n_hat = np.cross(r, v)
    n_hat = n_hat / np.linalg.norm(n_hat)
    t_hat = np.cross(n_hat, r_hat)
    return np.column_stack([r_hat, t_hat, n_hat])

class Gtoc4ControlEnv(gym.Env):
    def __init__(self, initial_state, target, start_epoch, time_limit,
                 control_interval=86400.0, integration_substeps=10,
                 position_tolerance=accuracy_position, velocity_tolerance=accuracy_velocity,
                 shaping_weight_position=1.0, shaping_weight_velocity=1.0,
                 propellant_penalty=0.0, rendezvous_bonus=10.0, target_sampler=None):
        """target_sampler, if given, is called with no arguments on every reset() and must return
        (target, time_limit); it overrides the fixed target/time_limit passed above, so a single
        env instance can present a different target each episode (WP5's generalisation step)."""
        super().__init__()
        self.initial_state = np.asarray(initial_state, dtype=np.float64)
        self.target = target
        self.start_epoch = start_epoch
        self.time_limit = time_limit
        self.target_sampler = target_sampler
        self.control_interval = control_interval
        self.integration_substeps = integration_substeps
        self.position_tolerance = position_tolerance
        self.velocity_tolerance = velocity_tolerance
        self.w_r = shaping_weight_position
        self.w_v = shaping_weight_velocity
        self.propellant_penalty = propellant_penalty
        self.rendezvous_bonus = rendezvous_bonus

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        self.state = None
        self.elapsed_time = 0.0
        self.total_impulse = 0.0

    def _target_state(self, t):
        return catalog.target_states([self.target], t, mu)[0]

    def _deltas(self):
        target = self._target_state(self.start_epoch + self.elapsed_time)
        return self.state[:3] - target[:3], self.state[3:6] - target[3:]

    def _potential(self, delta_r, delta_v):
        return -(self.w_r * np.linalg.norm(delta_r) / AU + self.w_v * np.linalg.norm(delta_v) / V_REF)

    def _observation(self, delta_r, delta_v):
        r, v, m = self.state[:3], self.state[3:6], self.state[6]
        t_remaining = (self.time_limit - self.elapsed_time) / self.time_limit
        obs = np.concatenate([
            r / AU, v / V_REF, delta_r / AU, delta_v / V_REF,
            [m / spacecraft_wet_mass], [t_remaining],
        ])
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.target_sampler is not None:
            self.target, self.time_limit = self.target_sampler()
        self.state = self.initial_state.copy()
        self.elapsed_time = 0.0
        self.total_impulse = 0.0
        delta_r, delta_v = self._deltas()
        self._delta_r, self._delta_v = delta_r, delta_v  # cached for the next step()'s "before"
        info = {'delta_r': delta_r, 'delta_v': delta_v, 'mass': self.state[6]}
        return self._observation(delta_r, delta_v), info

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        rtn_raw = action[:3]
        norm = np.linalg.norm(rtn_raw)
        if norm < DIRECTION_EPS:
            rtn_unit = np.zeros(3)
            throttle = 0.0
        else:
            rtn_unit = rtn_raw / norm
            throttle = (action[3] + 1.0) / 2.0

        rotation = rtn_frame(self.state[:3], self.state[3:6])
        thrust_force = throttle * thrust_max * (rotation @ rtn_unit)
        self.total_impulse += np.linalg.norm(thrust_force) * self.control_interval

        # "before" deltas are exactly what the previous step (or reset) computed as "after"
        delta_r_before, delta_v_before = self._delta_r, self._delta_v
        phi_before = self._potential(delta_r_before, delta_v_before)

        self.state = dynamics.propagate(self.state, thrust_force, self.control_interval,
                                         self.integration_substeps, mu, Isp_engine)
        self.elapsed_time += self.control_interval

        propellant_exhausted = self.state[6] < spacecraft_dry_mass
        if propellant_exhausted:
            self.state[6] = spacecraft_dry_mass

        delta_r_after, delta_v_after = self._deltas()
        self._delta_r, self._delta_v = delta_r_after, delta_v_after
        phi_after = self._potential(delta_r_after, delta_v_after)

        reward = (phi_after - phi_before) - self.propellant_penalty * throttle

        rendezvous = (np.linalg.norm(delta_r_after) < self.position_tolerance and
                      np.linalg.norm(delta_v_after) < self.velocity_tolerance)
        diverged = np.linalg.norm(delta_r_after) > DIVERGENCE_LIMIT
        time_up = self.elapsed_time >= self.time_limit

        terminated = bool(rendezvous or propellant_exhausted or diverged)
        truncated = bool(time_up and not terminated)

        if rendezvous:
            reward += self.rendezvous_bonus * (self.state[6] / spacecraft_wet_mass)

        info = {
            'delta_r': delta_r_after, 'delta_v': delta_v_after, 'mass': self.state[6],
            'rtn_action': rtn_unit, 'throttle': throttle, 'success': rendezvous,
            'total_impulse': self.total_impulse,
            # scalar duplicates, for SB3 Monitor(info_keywords=...) which needs scalars per episode
            'delta_r_norm': float(np.linalg.norm(delta_r_after)),
            'delta_v_norm': float(np.linalg.norm(delta_v_after)),
        }
        return self._observation(delta_r_after, delta_v_after), float(reward), terminated, truncated, info
