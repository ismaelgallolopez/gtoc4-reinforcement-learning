# Legality track

Goal: produce a GTOC4-legal, scorable trajectory — one satisfying the competition rules and
assignable a performance index `J`.

Binding rules used throughout (from the problem statement, as restated in the track brief):
launch from Earth with `|v_inf| <= 4.0 km/s`; launch epoch in MJD 57023–61041; `tau <= 10 years`;
`m_f >= 500 kg`; `T <= 0.135 N`; `Isp = 3000 s`; flyby = position match within 1000 km; rendezvous
= position within 1000 km **and** velocity within 1 m/s; each asteroid visited at most once;
rendezvous target not previously visited; no gravity assists.

Python used for every run below: `~/miniconda3/envs/tudat-space/bin/python`.

---

## Phase 1 — Two measurement bugs

### What changed

- `src/gtoc4_env.py`: new module-level `closest_approach_on_segments(rel_positions,
  rel_velocities, times)`. `Gtoc4ControlEnv.step`'s `require_velocity_match=False` branch now calls
  it instead of `np.min` over the RK4 substep grid, and `info` gains `closest_approach` and
  `closest_approach_epoch`.
- `src/curriculum.py`: new `delta_v_budget(time_limit)` — rocket equation with a thrust-limited
  mass history. `build_reachable_pool` uses it instead of `a_max * time_limit`.
- `src/constants.py`: `eccentricity_earth` and `mean_anomaly_earth` corrected; the `epoch = 54000.0`
  comment relabelled from `MJD2000` to `MJD`.
- `experiments/check_curriculum.py`: `check_delta_v_budget` switched to `curriculum.delta_v_budget`
  (it carried a second copy of the same wrong formula), with a dated note left above the original
  docstring text.
- `experiments/acceptance_phase1.py`: new, all Phase 1 acceptance tests.

Also swept into this commit: two pre-existing uncommitted working-tree changes not made by this
track — the `position`/`velocity` keys added to `Gtoc4ControlEnv.step`'s `info` dict, and the
untracked `experiments/planner.py` (WP12/WP13 greedy tour planner).

### Acceptance-test output (verbatim)

```
$ ~/miniconda3/envs/tudat-space/bin/python experiments/acceptance_phase1.py
--- 1a: analytic closest approach (rectilinear pass, true miss = 500.0 km) ---
sample spacing            : 8640 s (2.4 h)
relative speed            : 5000 m/s
old grid np.min           : 21605.8 km
new analytic minimum      : 500.0000 km  (error 0.0000 %)
true encounter epoch      : 47520.0 s
recovered encounter epoch : 47520.0 s
grid overestimates by     : 43.2 x
PASS: recovered within 5 %

--- 1b: rocket-equation delta-v budget ---
mdot                      : 4.588723e-06 kg/s
propellant exhausted at   : 2522.3 days = 6.906 years
window       old a_max*t (km/s)   new rocket eq (km/s)   expected
600 days                   4.67                   5.08       5.08
5 years                   14.20                  19.39      19.39
7 years                   19.88                  32.32      32.32
10 years                  28.40                  32.32      32.32
PASS: 5.08 / 19.39 km/s reproduced and saturation at 32.32 km/s confirmed

--- 1b: reachable-asteroid counts at a 1.5x margin (whole catalog, 1436 asteroids) ---
window       old budget  old count   new budget  new count
600 days          4.67k          0        5.08k          0
5 years          14.20k         17       19.39k         36
7 years          19.88k         37       32.32k        126
10 years         28.40k         91       32.32k        126
cheapest target in catalog: 3.50 km/s

--- 1c: constants transcription audit ---
eccentricity_earth   : 0.0167168116316  (was 1.671681164160e-02)
mean_anomaly_earth   : 257.60683707535 deg  (was 257.606837077535)
epoch                : 54000.0 interpreted as MJD
catalog.MJD_J2000    : 51544.5
mjd_to_et(epoch)     : 212155200.0 s after J2000 = 6.723 yr
PASS: MJD 54000 -> 2006-09-22, consistent with MJD_J2000 = 51544.5 (i.e. plain MJD, not MJD2000)

Earth state at START_EPOCH with corrected elements:
  r = [-1.80655891e-01  9.66547922e-01 -1.46652373e-05] AU
  v = [-2.97671156e+01 -5.58524941e+00  1.22876654e-04] km/s

all Phase 1 acceptance tests passed
```

Effect of the 1c corrections on Earth's state at `START_EPOCH`, measured separately:

```
dr (km) = 0.005869424444885194
dv (m/s)= 1.2006686404402794e-06
```

`check_curriculum.check_delta_v_budget` after the 1b switch:

```
stage 1: delta_v needed ~835.4 m/s (energy 297.9 + phasing 537.6, delta_theta=1.03 deg), budget ~3287.4 m/s over 400 d (margin 3.9x)
stage 2: delta_v needed ~2492.7 m/s (energy 1489.3 + phasing 1003.4, delta_theta=1.93 deg), budget ~5080.0 m/s over 600 d (margin 2.0x)
```

### Numbers

| quantity | old | new |
|---|---|---|
| 500 km rectilinear pass, reported miss | 21605.8 km | 500.0000 km |
| delta-v budget, 600 d | 4.67 km/s | 5.08 km/s |
| delta-v budget, 5 yr | 14.20 km/s | 19.39 km/s |
| delta-v budget, 7 yr | 19.88 km/s | 32.32 km/s (saturated) |
| delta-v budget, 10 yr | 28.40 km/s | 32.32 km/s (saturated) |
| reachable at 1.5x margin, 600 d | 0 | 0 |
| reachable at 1.5x margin, 5 yr | 17 | 36 |
| reachable at 1.5x margin, 7 yr | 37 | 126 |
| reachable at 1.5x margin, 10 yr | 91 | 126 |
| reachable at 1.0x margin, 600 d | 2 | 2 |
| stage-2 margin at 400 d | 1.25x | 1.32x |
| stage-2 margin at 600 d | 1.87x | 2.04x |
| `eccentricity_earth` | 1.671681164160e-2 | 1.671681163160e-2 |
| `mean_anomaly_earth` | 257.606837077535 deg | 257.60683707535 deg |

Propellant exhausts at 2522.3 days = 6.906 years of continuous full thrust, after which the budget
saturates at `Isp*g0*ln(1500/500)` = 32.32 km/s.

### Interpretation

**1a is the load-bearing fix.** The old grid minimum has a hard measurement floor of roughly
`v_rel * dt_substep / 2`. At the flyby configuration (2.4 h substeps, ~5 km/s relative speed) that
floor is ~21,600 km; at the 5-day asteroid control interval it is ~108,000 km. Both are one to two
orders of magnitude above the 1000 km GTOC4 flyby tolerance. The old code therefore could not have
reported a legal flyby under any trajectory whatsoever — a `closest_approach < 1000 km` test
against that estimator was not a test of the trajectory, it was a test of the sampling rate. The
new estimator recovers the synthetic 500 km case exactly (0.0000% error), because the relative
motion over one 2.4 h sub-interval genuinely is near-linear at these scales.

The returned encounter epoch matters for Phase 3: the true closest approach fell at t = 47520 s,
i.e. 5.5 substeps in, which is not a control-step boundary and would not be representable in a
solution file keyed only to control steps.

**1b changes the size of the candidate set, not its existence at short windows.** The constant-
acceleration approximation understates the budget by 9% at 600 days (mass barely changes) but by
35% at 5 years and 63% at 7 years, because it holds the acceleration at its heaviest-mass value for
the whole burn. Correcting it more than doubles the 5-year pool (17 -> 36) and more than triples the
7-year pool (37 -> 126). It does *not* rescue the 600-day window: the cheapest target in the entire
1436-asteroid catalog needs 3.50 km/s against a 5.08 km/s budget, so still 2 targets at a 1.0x
margin and 0 at 1.5x. The old formula also grew without bound, which is why the 10-year number
(28.40 km/s) previously exceeded what the spacecraft can physically deliver; the true 10-year and
7-year budgets are identical, both propellant-limited.

**1c is a correctness fix with no measurable physical consequence.** The two corrected trailing
digits move Earth's position at `START_EPOCH` by 5.9 m and its velocity by 1.2 µm/s — far below
the 1000 km / 1 m/s tolerances, and far below the error of the two-body model itself. It is worth
fixing so the ephemeris matches the statement bit-for-bit, but no recorded result changes because
of it. The `epoch = 54000.0 # MJD2000` comment was simply mislabelled: `catalog.mjd_to_et` subtracts
`MJD_J2000 = 51544.5`, which places 54000 at 6.723 years after J2000 (2006-09-22) — correct for a
plain MJD, and consistent with the asteroid catalog's own MJD 54800 epoch column. The value was
always right; only the comment was wrong. Comment corrected, value untouched.

### Recorded conclusions in `src/curriculum.py` now suspect

Listed by docstring. Nothing has been deleted or rewritten; this is an assessment only, and none of
these were re-run (no training budget in this track for that).

1. **`make_flyby_variant`, the WP11 finding — SUSPECT, and the specific claim this track exists to
   revisit.** "at the true 1000 km tolerance, both a from-scratch policy and one warm-started from
   the stage-1 rendezvous policy get 0/20" was decided by `closest_approach < 1000 km` where
   `closest_approach` was the grid minimum, whose floor on that env (`control_interval=86400`,
   `integration_substeps=10`) is ~21,600 km. A success was arithmetically impossible. The 0/20 is
   consistent with a genuinely bad trajectory *and* with a good one — the experiment cannot tell
   them apart. Note the failure mode narrated in the same docstring (closes to 1.38M km at day 137,
   then flies past and diverges past 20M km) is measured from `delta_r_norm`, the control-step
   endpoint distance, not from the grid minimum, so *that* part of the finding stands. So does the
   "5/5 at 1.5e6 km" re-evaluation: 1.5e6 km is ~70x above the grid floor, where the grid minimum is
   an accurate estimator. What is not supported is the concluding sentence's implied magnitude —
   "the true GTOC4 1000 km tolerance is still ~1000x tighter than what this direct-control PPO setup
   reaches". The measured 1.38M km is real; whether the policy ever passed within 1000 km at some
   instant between substeps was never measured either way.

2. **Module-level stage-3 pool comment (above `ASTEROID_TIME_LIMIT_RANGE`) — PARTLY SUPERSEDED.**
   "only 2 asteroids are reachable at all within 600 days" and "min delta-v needed across the entire
   catalog is 3498 m/s" both still hold exactly. "the 600-day budget of 4666 m/s" is the old formula;
   it is 5080 m/s. "0 clear even a 1.5x margin" still holds. "At a 5-7 year window, 17-37 asteroids
   clear a 1.5x margin" is superseded: it is now 36-126. The conclusion drawn from it — that a 5-7
   year window gives a usable pool where 600 days does not — is unchanged and in fact strengthened.

3. **`STAGES[2]`, the stage-2 failure note — CONCLUSION STANDS, one quoted number superseded.** The
   failure itself (0/5 success, `|delta_r|` ~185M km, plateau at 110-145M km, four attempts) is a
   rendezvous-mode measurement: `require_velocity_match=True` never touches the grid minimum, it
   uses the control-step endpoint `delta_r`, so 1a does not affect it. The parenthetical "budget =
   a_max * time_limit" is the corrected formula, and "Stage 2's delta-v margin was only 1.2x" (that
   figure is at the pre-extension 400-day window) becomes 1.32x; post-extension to 600 days it goes
   from 1.87x to 2.04x. A 2.0x margin is a weaker excuse for the failure than a 1.9x one, but the
   note's own diagnosis — that greedy potential-based shaping locally misleads the policy on a
   combined da+di manoeuvre — is not a budget argument and is untouched.

4. **`STAGES[1]`, the tolerance-loosening note — NOT AFFECTED.** Rendezvous mode throughout, and no
   pool filtering involved. Every number in it is an endpoint measurement.

5. **`_delta_v_needed` docstring — cosmetically stale.** "a first uniform draw needed ~50 km/s
   against a ~3 km/s budget": the ~3 km/s is the old 400-day figure, now 3.29 km/s. The point being
   made (the raw catalog is not sorted by difficulty and a uniform draw is hopeless) is unaffected.

6. **`initial_state` docstring — NOT AFFECTED by Phase 1**, but its premise is what Phase 2
   revisits: it argues against a departure kick on the grounds that 4 km/s would put the spacecraft
   far from a near-identical stage-1 target. That is a curriculum-design argument, and correct for
   the curriculum; it is not an argument about the GTOC4 mission, where the 4 km/s is free delta-v
   the rules grant.

### Choices taken where the brief left it open

- The problem statement document itself is **not in this repo** (`data/gtoc4_problem_data.txt` is
  the asteroid ephemeris table only, and there is no statement PDF anywhere under
  `~/workspace/tud/`). The rules used are those restated in the track brief. Where the brief refers
  to the statement for something it does not itself restate — specifically the `solution.txt`
  column layout in Phase 3 — the layout used is documented in this file at that phase, as an
  assumption to be checked against the statement. This is *not* halt condition 4: no rule bearing
  on whether a trajectory is legal is ambiguous, only an output format.
- `experiments/check_curriculum.py` was edited even though the brief only named
  `build_reachable_pool`. Rejected alternative: leave it, per the minimal-diff rule. Taken because
  it held an independent second copy of the same incorrect budget formula, and leaving it would mean
  two functions in the repo disagreeing about the spacecraft's delta-v.
