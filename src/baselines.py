# baseline guidance laws: coast (no thrust) and constant tangential thrust
import numpy as np

COAST_ACTION = np.array([1.0, 0.0, 0.0, -1.0])       # any direction, throttle -1 -> zero thrust
TANGENTIAL_ACTION = np.array([0.0, 1.0, 0.0, 1.0])   # RTN transverse direction, full throttle

def run_baseline(env, action):
    env.reset()
    terminated = truncated = False
    info = {}
    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(action)
    return info

def coast(env):
    return run_baseline(env, COAST_ACTION)

def tangential_thrust(env):
    return run_baseline(env, TANGENTIAL_ACTION)
