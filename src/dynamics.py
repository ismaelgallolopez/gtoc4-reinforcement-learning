# hand-written RK4 propagator + Kepler solver, standing in for Tudat inside the RL loop
import numpy as np

g0 = 9.80665  # m/s^2, standard gravity (Isp -> mass flow rate)

def state_derivative(state, thrust_force, mu, Isp):
    """state = [rx,ry,rz,vx,vy,vz,m]; thrust_force = [Tx,Ty,Tz] in N, inertial frame."""
    r = state[:3]
    v = state[3:6]
    m = state[6]
    a_grav = -mu * r / np.linalg.norm(r)**3
    a_thrust = thrust_force / m
    dm = -np.linalg.norm(thrust_force) / (Isp * g0)
    return np.concatenate([v, a_grav + a_thrust, [dm]])

def rk4_step(state, thrust_force, dt, mu, Isp):
    k1 = state_derivative(state, thrust_force, mu, Isp)
    k2 = state_derivative(state + 0.5 * dt * k1, thrust_force, mu, Isp)
    k3 = state_derivative(state + 0.5 * dt * k2, thrust_force, mu, Isp)
    k4 = state_derivative(state + dt * k3, thrust_force, mu, Isp)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def propagate(state, thrust_force, duration, n_substeps, mu, Isp, return_trajectory=False):
    """Propagates with a constant thrust_force over duration, using n_substeps RK4 steps.
    Called once per control step by the RL env, and with many substeps for validation runs.
    return_trajectory=True additionally returns the state at every substep (including the
    initial one), for closest-approach checks within a single control interval -- e.g. a
    GTOC4-style flyby, where the spacecraft can pass within tolerance of a fast-moving target
    between two control steps without either endpoint being that close."""
    dt = duration / n_substeps
    trajectory = [state] if return_trajectory else None
    for _ in range(n_substeps):
        state = rk4_step(state, thrust_force, dt, mu, Isp)
        if return_trajectory:
            trajectory.append(state)
    return (state, np.array(trajectory)) if return_trajectory else state

def solve_kepler(M, e, tol=1e-12, max_iter=50):
    """Solves M = E - e*sin(E) for the eccentric anomaly E. M and e may be arrays."""
    M = np.mod(M, 2 * np.pi)
    E = np.where(e < 0.8, M, np.pi * np.ones_like(M))
    for _ in range(max_iter):
        delta = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E = E - delta
        if np.max(np.abs(delta)) < tol:
            break
    return E

def keplerian_to_cartesian(a, e, i, omega, lan, M, mu):
    """Vectorised Keplerian -> Cartesian state. All angle/element args may be arrays of the same shape.
    Returns an array of shape (..., 6): [rx,ry,rz,vx,vy,vz]."""
    E = solve_kepler(M, e)
    true_anomaly = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
    r = a * (1 - e * np.cos(E))
    n = np.sqrt(mu / a**3)

    x_pf = r * np.cos(true_anomaly)
    y_pf = r * np.sin(true_anomaly)
    vx_pf = -a * n * np.sin(E) / (1 - e * np.cos(E))
    vy_pf = a * n * np.sqrt(1 - e**2) * np.cos(E) / (1 - e * np.cos(E))

    cos_O, sin_O = np.cos(lan), np.sin(lan)
    cos_i, sin_i = np.cos(i), np.sin(i)
    cos_w, sin_w = np.cos(omega), np.sin(omega)

    R11 = cos_O * cos_w - sin_O * sin_w * cos_i
    R12 = -cos_O * sin_w - sin_O * cos_w * cos_i
    R21 = sin_O * cos_w + cos_O * sin_w * cos_i
    R22 = -sin_O * sin_w + cos_O * cos_w * cos_i
    R31 = sin_w * sin_i
    R32 = cos_w * sin_i

    x = R11 * x_pf + R12 * y_pf
    y = R21 * x_pf + R22 * y_pf
    z = R31 * x_pf + R32 * y_pf
    vx = R11 * vx_pf + R12 * vy_pf
    vy = R21 * vx_pf + R22 * vy_pf
    vz = R31 * vx_pf + R32 * vy_pf

    return np.stack([x, y, z, vx, vy, vz], axis=-1)
