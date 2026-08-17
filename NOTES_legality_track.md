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

---

## Phase 2 — Claim the delta-v the rules already grant

### What changed

- `src/curriculum.py`:
  - `initial_state(start_epoch=None, v_infinity=None)` — `v_infinity` is a 3-vector in m/s added to
    Earth's heliocentric velocity; `start_epoch` overrides the module-level `START_EPOCH`. Both
    default to `None`, reproducing the previous behaviour exactly, so no existing caller changes.
  - `_delta_v_needed(target, time_limit, start_epoch=None, v_infinity_magnitude=0.0)` — same.
  - `build_reachable_pool(..., start_epoch=None, v_infinity_magnitude=0.0)` — same, forwarded.
  - Dated notes appended beneath the original docstring text in all three; nothing removed.
- `src/constants.py`: `launch_interval` upper bound corrected from 2025-01-01 to 2026-01-01.
- `experiments/scan_reachability.py`: new, the deliverable scan + acceptance test + figure.
- `figures/legality_reachability_scan.png`, `results/legality_reachability_scan.csv`: outputs.

Nothing was wired into the action or observation space. `START_EPOCH` itself is unchanged, so
`make_env`, `make_stage1_variant`, `make_flyby_variant` and `make_randomized_env` all still build
the identical environments they did before (verified: stage 1, stage 3 and the flyby variant all
reset and step unchanged).

### Acceptance-test output (verbatim)

```
$ ~/miniconda3/envs/tudat-space/bin/python experiments/scan_reachability.py
scanning 41 launch epochs over MJD 57024-61042 (4018 days)
  epoch 1/41 (MJD 57024): cheapest target 3.50 km/s
  epoch 2/41 (MJD 57124): cheapest target 4.12 km/s
  epoch 3/41 (MJD 57224): cheapest target 1.88 km/s
  epoch 4/41 (MJD 57325): cheapest target 3.54 km/s
  epoch 5/41 (MJD 57425): cheapest target 4.21 km/s
  epoch 6/41 (MJD 57526): cheapest target 3.56 km/s
  epoch 7/41 (MJD 57626): cheapest target 3.79 km/s
  epoch 8/41 (MJD 57727): cheapest target 1.42 km/s
  epoch 9/41 (MJD 57827): cheapest target 2.95 km/s
  epoch 10/41 (MJD 57928): cheapest target 2.29 km/s
  epoch 11/41 (MJD 58028): cheapest target 1.43 km/s
  epoch 12/41 (MJD 58128): cheapest target 2.20 km/s
  epoch 13/41 (MJD 58229): cheapest target 1.37 km/s
  epoch 14/41 (MJD 58329): cheapest target 2.34 km/s
  epoch 15/41 (MJD 58430): cheapest target 1.06 km/s
  epoch 16/41 (MJD 58530): cheapest target 3.15 km/s
  epoch 17/41 (MJD 58631): cheapest target 1.66 km/s
  epoch 18/41 (MJD 58731): cheapest target 2.64 km/s
  epoch 19/41 (MJD 58832): cheapest target 3.67 km/s
  epoch 20/41 (MJD 58932): cheapest target 2.28 km/s
  epoch 21/41 (MJD 59032): cheapest target 2.74 km/s
  epoch 22/41 (MJD 59133): cheapest target 1.20 km/s
  epoch 23/41 (MJD 59233): cheapest target 3.85 km/s
  epoch 24/41 (MJD 59334): cheapest target 3.63 km/s
  epoch 25/41 (MJD 59434): cheapest target 3.97 km/s
  epoch 26/41 (MJD 59535): cheapest target 3.77 km/s
  epoch 27/41 (MJD 59635): cheapest target 1.75 km/s
  epoch 28/41 (MJD 59736): cheapest target 1.57 km/s
  epoch 29/41 (MJD 59836): cheapest target 2.38 km/s
  epoch 30/41 (MJD 59937): cheapest target 2.28 km/s
  epoch 31/41 (MJD 60037): cheapest target 3.29 km/s
  epoch 32/41 (MJD 60137): cheapest target 3.87 km/s
  epoch 33/41 (MJD 60238): cheapest target 1.92 km/s
  epoch 34/41 (MJD 60338): cheapest target 1.54 km/s
  epoch 35/41 (MJD 60439): cheapest target 2.91 km/s
  epoch 36/41 (MJD 60539): cheapest target 4.00 km/s
  epoch 37/41 (MJD 60640): cheapest target 2.52 km/s
  epoch 38/41 (MJD 60740): cheapest target 3.00 km/s
  epoch 39/41 (MJD 60841): cheapest target 3.53 km/s
  epoch 40/41 (MJD 60941): cheapest target 2.36 km/s
  epoch 41/41 (MJD 61042): cheapest target 3.19 km/s

--- acceptance: v_inf = 0, epoch = launch_interval[0], 600-day window ---
budget                 : 5.080 km/s
cheapest target        : 3.498 km/s
reachable, margin 1.0x : 2   (recorded value: 2)
reachable, margin 1.5x : 0   (recorded value: 0)
PASS: recorded baseline reproduced

--- reachable-asteroid count at a 1.5x margin (1436-asteroid catalogue) ---
window    |v_inf|    min  median    max   best epoch (MJD)
600 d         0 k      0       1      4              58128
600 d         1 k      1       2      7              58128
600 d         2 k      2       5      8              58128
600 d         3 k      3       8     12              58128
600 d         4 k      6      11     16              58028
5 yr          0 k     32      41     51              57827
5 yr          1 k     33      48     58              57626
5 yr          2 k     42      56     67              57827
5 yr          3 k     45      66     82              58329
5 yr          4 k     50      75     93              58329
10 yr         0 k    100     127    151              59434
10 yr         1 k    120     141    160              57526
10 yr         2 k    129     155    178              59434
10 yr         3 k    141     170    192              59434
10 yr         4 k    159     182    208              58430

saved .../figures/legality_reachability_scan.png
saved .../results/legality_reachability_scan.csv
```

### Numbers

The scan sweeps 41 launch epochs across the full 4018-day legal window, `|v_inf|` in
{0, 1, 2, 3, 4} km/s, and mission windows of 600 d / 5 yr / 10 yr, at a 1.5x delta-v margin.

| configuration | reachable asteroids |
|---|---|
| pinned: 600 d, `v_inf` = 0, epoch = `launch_interval[0]` (curriculum's actual setting) | **0** (2 at a 1.0x margin) |
| 600 d, `v_inf` = 0, best epoch | 4 |
| 600 d, `v_inf` = 4 km/s, best epoch | 16 |
| 5 yr, `v_inf` = 0, best epoch | 51 |
| 5 yr, `v_inf` = 4 km/s, best epoch | 93 |
| 10 yr, `v_inf` = 0, worst / median / best epoch | 100 / 127 / 151 |
| 10 yr, `v_inf` = 4 km/s, worst / median / best epoch | 159 / 182 / **208** |

Cheapest single target across the catalogue, by launch epoch: 1.06 km/s (MJD 58430) at the best
epoch sampled, 4.21 km/s (MJD 57425) at the worst — a factor of 4.0 spread driven purely by
phasing. At the pinned `START_EPOCH` it is 3.50 km/s, in the worst quartile of the window.

Figure: `figures/legality_reachability_scan.png`. Left panel, count vs launch epoch at the 10-year
window, one line per `v_inf`, with the pinned `START_EPOCH` marked. Right panel, count vs `v_inf`,
one solid line per window at its best epoch and a dashed line at the median epoch.

### Interpretation

**Halt condition 3 does not apply; the candidate set is emphatically not empty.** Going from the
curriculum's pinned configuration (600 d, `v_inf` = 0, `launch_interval[0]`) to the best legal
configuration (10 yr, `v_inf` = 4 km/s, MJD 58430) takes the reachable count from 0 to 208 at the
same 1.5x margin. Phase 4's sequencing layer has a large candidate set to work with. That is the
result this phase existed to establish.

**The three pinned quantities are not equally valuable, and they are not what one would guess.**
Ranked by what each buys at the 1.5x margin:

1. *Mission window* is worth by far the most: 600 d -> 10 yr at fixed `v_inf` = 0 takes the median
   epoch from 1 to 127 asteroids, a factor of ~127. This is unsurprising in hindsight — the
   delta-v budget itself rises 5.08 -> 32.32 km/s (Phase 1b), and the requirement distribution has
   a long tail that the extra 27 km/s cuts deep into.
2. *Launch epoch* is worth a factor of ~1.5 at any fixed window and `v_inf` (10 yr, `v_inf` = 0:
   100 at the worst epoch, 151 at the best), and much more at short windows (600 d, `v_inf` = 0:
   0 at the worst, 4 at the best — the difference between an empty candidate set and a non-empty
   one). The pinned `START_EPOCH` sits at 126 on the 10-year curve, essentially the median, so it
   is not a pathological choice — it is simply an arbitrary one, and arbitrary is expensive at
   short windows where it happens to be near-worst (cheapest target 3.50 km/s against a 5.08 km/s
   budget, i.e. margin 1.45x, just under the 1.5x bar; hence the recorded 0).
3. *`v_inf`* is worth ~1.4x at the 10-year window (151 -> 208 at the best epoch) and ~4x at 600
   days (4 -> 16). It matters most exactly where the budget is tightest, which is what one would
   expect from a fixed 4 km/s credit against a budget that ranges from 5 to 32 km/s.

The practical reading for Phase 4: **take the 10-year window first, then choose the launch epoch,
then take the `v_inf` for free.** The reachable-count curve against launch epoch is strongly
non-monotone with a ~200-day quasi-period (visible in the left panel) — that is the synodic
structure of the near-Earth population, and it means epoch selection should be a search, not a
default.

**The `launch_interval` upper bound was wrong.** `constants.launch_interval` was
`[DateTime(2015,1,1), DateTime(2025,1,1)]` = 3653 days, but the legal window is MJD 57023-61041 =
4018 days, i.e. 2015-01-01 through the *end* of 2025. 365 days — 10% of the legal window, and the
half containing three of the five best epochs in the scan — were being discarded. Corrected to
`DateTime(2026,1,1)`. This supplements the Phase 1c transcription audit, which only compared the
Earth ephemeris block; it was found while working out how wide the epoch sweep should be.

A related but harmless detail: tudatpy's `DateTime(y, m, d)` resolves to 12:00, not 00:00
(`DateTime(2015,1,1).to_modified_julian_day()` returns 57023.5). So `launch_interval` is really
MJD 57023.5-61041.5 — half a day inside the stated window at each end. That is conservative and
therefore legal, so it is documented rather than fixed; forcing it to exactly MJD 57023.0 would
require bypassing the `DateTime` helper for no legality gain.

**What this scan is and is not.** It is a closed-form screen — the energy term from the semi-major
axis difference plus a phasing term `v_circ * delta_theta`, compared against a continuous-full-
thrust delta-v budget. It says nothing about whether any *particular* transfer closes, and in
particular it ignores inclination entirely, which for a population with `i` up to 68 deg is a real
omission that will make it optimistic. Its purpose here is to answer one question — is the
candidate set empty at the pinned settings and non-empty at legal ones — and the answer is yes and
no respectively. Phase 4's Lambert-based oracle replaces it with something that actually solves the
transfer.

### Choices taken where the brief left it open

- **How `v_inf` enters `_delta_v_needed`**: credited one-for-one against the requirement, floored
  at zero (`max(0, dv_energy + dv_phasing - v_inf)`). Rationale: the estimator is already a sum of
  two scalar contributions with no vector structure, and the launch excess is a free impulse of
  arbitrary direction, so there is nothing in the model that could distinguish a well-pointed
  4 km/s from a badly-pointed one. Rejected alternative: add the `v_inf` vector to Earth's velocity,
  take the osculating semi-major axis of the resulting departure orbit, and recompute the energy
  term from that while leaving the phasing term alone. That is more faithful, but it makes the
  result depend on a `v_inf` *direction* that would then have to be optimised per target per epoch,
  turning a screen into a small optimisation problem — disproportionate for a filter whose job is
  to decide whether a set is empty. The simple version is optimistic in the same direction for
  every candidate, so the ranking it produces is what Phase 4 consumes, and Phase 4 re-checks every
  leg with Lambert anyway.
- **Grid resolution**: 41 epochs (~100-day spacing) over 4018 days, 5 `v_inf` values. The epoch
  curve has ~200-day structure, so 100-day spacing resolves it but does not pin the optimum
  precisely; Phase 4 can refine locally. Rejected: a finer sweep, which costs linearly (the scan is
  1436 asteroids x 41 epochs of scalar `_delta_v_needed` calls, ~40 s) and was not needed to answer
  the empty-or-not question.
- **`START_EPOCH` left pointing at `launch_interval[0]`.** Rejected alternative: repoint it at the
  best epoch found here. Not taken because every recorded curriculum result in `src/curriculum.py`
  was measured at the current value, and silently moving it would invalidate all of them. Phase 4's
  sequencer passes its chosen epoch explicitly instead.
