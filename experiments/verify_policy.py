# WP8: replays the learned stage-1 thrust profile through Tudat's RKF78 integrator (independent
# of the fixed-step RK4 propagator the policy was trained against) and checks whether the
# resulting trajectory still rendezvous within tolerance. This is the real question WP8 asks:
# did the policy learn a control law that works under real orbital dynamics, or did it overfit
# to numerical quirks of the specific RK4 integrator it was trained against?
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

import analyze_policy
import catalog
import curriculum
import verify_tudat
from gtoc4_env import rtn_frame
from constants import sun_gravitational_parameter as mu

def replay_through_tudat(log):
    """log comes from analyze_policy.rollout(): per-control-interval RTN thrust direction and
    throttle, from a deterministic policy rollout under the RK4 training dynamics."""
    start_epoch = log['start_epoch']
    control_interval = 86400.0  # stage 1's control interval

    # spacecraft (r, v) at the START of each interval -- interval 0 starts at the env's fixed
    # initial condition, every later interval starts where the RK4 rollout's previous interval
    # ended, matching how Gtoc4ControlEnv computes its RTN basis right before each step
    r0v0 = curriculum.initial_state()[:6]
    target_states = np.array([catalog.target_states([log['target']], start_epoch + t, mu)[0]
                               for t in log['t']])
    spacecraft_states = target_states.copy()
    spacecraft_states[:, :3] += log['delta_r']
    spacecraft_states[:, 3:] += log['delta_v']
    states_at_interval_start = np.vstack([r0v0, spacecraft_states[:-1]])

    inertial_dirs = np.array([
        rtn_frame(state[:3], state[3:]) @ rtn_unit
        for state, rtn_unit in zip(states_at_interval_start, log['rtn'])
    ])
    throttles = log['throttle']

    def thrust_direction_function(t):
        if np.isnan(t):  # Tudat probes the callback once with NaN before propagation starts
            return inertial_dirs[0]
        idx = min(max(int((t - start_epoch) // control_interval), 0), len(inertial_dirs) - 1)
        return inertial_dirs[idx]

    def thrust_magnitude_function(t):
        if np.isnan(t):
            return throttles[0] * verify_tudat.thrust_max
        idx = min(max(int((t - start_epoch) // control_interval), 0), len(throttles) - 1)
        return throttles[idx] * verify_tudat.thrust_max

    initial_state_with_mass = curriculum.initial_state()
    duration = log['t'][-1]
    return verify_tudat.run_policy_verification(
        initial_state_with_mass, start_epoch, duration,
        thrust_direction_function, thrust_magnitude_function)

def compare(log, results):
    state_history = results.state_history
    final_epoch = max(state_history.keys())
    final_state = state_history[final_epoch]  # multitype propagator: [r(3), v(3), mass(1)]
    tudat_r, tudat_v, tudat_mass = final_state[:3], final_state[3:6], final_state[6]

    target_final = catalog.target_states([log['target']], final_epoch, mu)[0]
    delta_r_tudat = tudat_r - target_final[:3]
    delta_v_tudat = tudat_v - target_final[3:]

    print(f"RK4-trained rollout: success={log['success']}, "
          f"final |delta_r|={np.linalg.norm(log['delta_r'][-1]) / 1e3:.1f} km, "
          f"final |delta_v|={np.linalg.norm(log['delta_v'][-1]):.3f} m/s, "
          f"final mass={log['mass'][-1]:.1f} kg")
    print(f"Tudat replay of the SAME thrust commands: "
          f"final |delta_r|={np.linalg.norm(delta_r_tudat) / 1e3:.1f} km, "
          f"final |delta_v|={np.linalg.norm(delta_v_tudat):.3f} m/s, "
          f"final mass={tudat_mass:.1f} kg")

    tudat_success = (np.linalg.norm(delta_r_tudat) < curriculum.STAGES[1]['position_tolerance'] and
                      np.linalg.norm(delta_v_tudat) < curriculum.STAGES[1]['velocity_tolerance'])
    print(f"still within stage-1 rendezvous tolerance under Tudat: {tudat_success}")

if __name__ == '__main__':
    model, env = analyze_policy.load_policy()
    log = analyze_policy.rollout(model, env)
    results = replay_through_tudat(log)
    compare(log, results)
