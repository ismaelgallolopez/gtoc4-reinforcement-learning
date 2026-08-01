"""
Benchmarks for the JAX optimisations in the GTOC4 codebase.
Run from the code/ directory:  python benchmark_jax.py
"""
import time
import math
import numpy as onp
import jax
import jax.numpy as jnp

def bench(fn, n_warmup=3, n_runs=20):
    for _ in range(n_warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        fn()
    return (time.perf_counter() - t0) / n_runs

def report(label, t_old, t_new):
    speedup = t_old / t_new
    verdict = "KEEP" if speedup > 1.0 else "REVERT"
    print(f"  {label}")
    print(f"    old : {t_old*1e3:8.3f} ms")
    print(f"    new : {t_new*1e3:8.3f} ms")
    print(f"    speedup: {speedup:.2f}x  [{verdict}]\n")

N_AST    = 1436
N_FRAMES = 100
MU       = 132712440018e9
AU       = 1.495978707e11
rng      = onp.random.default_rng(42)

# ── Benchmark 1: deg2rad batching ─────────────────────────────────────────────
print("=" * 60)
print("Benchmark 1 — deg2rad batching in parse_asteroids")
print("=" * 60)
print("  WHY REVERTED: jnp.array([4 floats]) allocation overhead")
print("  exceeds the cost of 4 individual scalar dispatches.\n")

i_d     = list(rng.uniform(0,  30, N_AST))
lan_d   = list(rng.uniform(0, 360, N_AST))
omega_d = list(rng.uniform(0, 360, N_AST))
M0_d    = list(rng.uniform(0, 360, N_AST))

def old_deg2rad():
    result = []
    for i in range(N_AST):
        result.append((jnp.deg2rad(i_d[i]), jnp.deg2rad(lan_d[i]),
                       jnp.deg2rad(omega_d[i]), jnp.deg2rad(M0_d[i])))
    jax.block_until_ready(result[-1][0])
    return result

def new_deg2rad():
    result = []
    for i in range(N_AST):
        vals = jnp.deg2rad(jnp.array([i_d[i], lan_d[i], omega_d[i], M0_d[i]]))
        result.append(tuple(vals))
    jax.block_until_ready(result[-1][0])
    return result

report("4 scalar dispatches vs 1 array dispatch per asteroid",
       bench(old_deg2rad), bench(new_deg2rad))


# ── Benchmark 2: initial-state vectorisation ──────────────────────────────────
print("=" * 60)
print("Benchmark 2 — mean-motion / mean-anomaly computation")
print("=" * 60)
print("  WHY REVERTED: vectorised math is faster, but the cost of")
print("  building JAX arrays from Python lists and then iterating")
print("  back element-by-element for the C++ mean_to_true_anomaly")
print("  call dominates. The math alone is shown separately.\n")

a_py     = list(rng.uniform(1.5, 3.5, N_AST) * AU)
M0_py    = list(rng.uniform(0, 2 * math.pi, N_AST))
epoch_py = list(rng.uniform(-1e8, 1e8, N_AST))
e_py     = list(rng.uniform(0.0, 0.3, N_AST))
start_epoch = 0.0

def old_initial_state():
    results = []
    for i in range(N_AST):
        delta_t    = start_epoch - epoch_py[i]
        n          = jnp.sqrt(MU / a_py[i]**3)
        M_at_start = (M0_py[i] + float(n) * delta_t) % (2 * math.pi)
        results.append((e_py[i], M_at_start))
    return results

def new_initial_state_full():
    a_arr          = jnp.array(a_py)
    M0_arr         = jnp.array(M0_py)
    epoch_arr      = jnp.array(epoch_py)
    n_arr          = jnp.sqrt(MU / a_arr**3)
    M_at_start_arr = (M0_arr + n_arr * (start_epoch - epoch_arr)) % (2 * math.pi)
    jax.block_until_ready(M_at_start_arr)
    return [(e_py[i], float(M_at_start_arr[i])) for i in range(N_AST)]

def new_initial_state_math_only():
    """Only the vectorised math — excludes list→array and array→float overhead."""
    a_arr          = jnp.array(a_py)
    M0_arr         = jnp.array(M0_py)
    epoch_arr      = jnp.array(epoch_py)
    n_arr          = jnp.sqrt(MU / a_arr**3)
    M_at_start_arr = (M0_arr + n_arr * (start_epoch - epoch_arr)) % (2 * math.pi)
    jax.block_until_ready(M_at_start_arr)
    return M_at_start_arr

report("full pipeline (list→array + math + array→float loop)",
       bench(old_initial_state), bench(new_initial_state_full))
t_math_old = bench(lambda: [jnp.sqrt(MU / a_py[i]**3) for i in range(N_AST)])
t_math_new = bench(new_initial_state_math_only)
print(f"  math only (no C++ call overhead)")
print(f"    old (1436 scalar sqrt dispatches) : {t_math_old*1e3:.3f} ms")
print(f"    new (1 vectorised sqrt)           : {t_math_new*1e3:.3f} ms")
print(f"    speedup: {t_math_old/t_math_new:.2f}x  [math IS faster, but masked by C++ call loop]\n")


# ── Benchmark 3: animation update (KEPT) ──────────────────────────────────────
print("=" * 60)
print("Benchmark 3 — animation update() array access  [APPLIED]")
print("=" * 60)

ast_fs         = [onp.random.rand(N_FRAMES, 3) for _ in range(N_AST)]
ast_fs_stacked = jnp.stack(ast_fs)

def old_update():
    results = []
    for frame_i in range(N_FRAMES):
        x = jnp.array([ap[frame_i, 0] for ap in ast_fs])
        y = jnp.array([ap[frame_i, 1] for ap in ast_fs])
        z = jnp.array([ap[frame_i, 2] for ap in ast_fs])
        results.append((x, y, z))
    jax.block_until_ready(results[-1][0])
    return results

def new_update():
    results = []
    for frame_i in range(N_FRAMES):
        x = ast_fs_stacked[:, frame_i, 0]
        y = ast_fs_stacked[:, frame_i, 1]
        z = ast_fs_stacked[:, frame_i, 2]
        results.append((x, y, z))
    jax.block_until_ready(results[-1][0])
    return results

report("list comprehension + jnp.array per frame vs pre-stacked index",
       bench(old_update), bench(new_update))
