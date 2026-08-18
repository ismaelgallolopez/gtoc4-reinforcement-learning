# Legality track

## Summary

**Update (multi-flyby track, halted after Phase 6 of 10 — see `## Multi-flyby track` below):**
the legality-track result immediately below (`J = 0`, `K = 1350.916 kg`) is still the best legal
result either track produced; the multi-flyby track did not reach the point of running a search
that could improve on it. It got as far as auditing and calibrating the leg-cost oracle that any
such search would depend on (Phases 5-6), found the calibration could not be made to meet its own
accuracy bar in three genuine attempts (halt condition 1), and stopped before Phase 7's
element-space pruning or Phase 8's beam search were built. The mechanism identified along the way
is itself the answer to "what currently limits `J`": **short flyby legs (ToF <= 120 days, the range
a 44-flyby, 10-year mission needs to average) produced zero verified successes across 360 sampled
attempts** with the existing thrust-limited guidance, independent of anything a smarter search over
target sequencing could fix, and the flown-vs-estimated cost ratio for the legs that do succeed
depends on delta-v magnitude as well as on time of flight, which a time-of-flight-only correction
(what Phase 6 was scoped to build) cannot capture. Full detail, verbatim acceptance-test output, and
the reasoning behind stopping here rather than pushing through: `## Multi-flyby track`, Phase 6.

**Update (guidance diagnostic track, all 4 phases completed — see `## Guidance diagnostic track`
below):** dug into the specific claim above ("0/360 at ToF <= 120 d") and found it does not mean
what it looks like it means. Across 390 combined sampled attempts at ToF <= 120 d, a candidate
offered by the oracle's own top-3 ranking is essentially always either trivial (free, covered
outright by the launch `v_inf` credit — closes 100% of the time, 174/174) or already beyond the
oracle's own 0.6x-duty-cycle affordability margin (fails 100% of the time, 0/214) — a candidate that
is both realistic *and* affordable appeared **twice** across two independent samples, one succeeding
and one failing. So the guidance law's
short-ToF failures are overwhelmingly a statement about which candidates get offered, not a
guidance-law defect — and a real search already filters by that same margin before ever asking the
guidance to fly anything, so it would rarely hand the guidance what failed here. The part that
*is* a real guidance-law finding: the same saturation/overshoot failure signatures, and a
comparably high failure rate (20-80%, worst at 900 days) on candidates that genuinely are
affordable, persist all the way out to 1200 days — not confined to short legs as the track's
opening hypothesis assumed. A specific, well-motivated fix (spread each day's correction over the
leg's remaining time instead of demanding it all within the next 24 hours) was implemented and
tested against the exact same seeds/candidates already sampled: it made the raw failure rate worse
at every one of five 150-300 d buckets sampled (e.g. 20.0% → 46.7% at 300 d) and pushed the
already-small feasible-and-realistic population's failure rate from a mixed 0-67% to a uniform
100% everywhere a comparison was possible. Reverted; the guidance law is unchanged from the
legality track. Bottom line for whether this guidance law can support a multi-flyby mission: usable
on candidates a real search would actually offer it at short ToF (which are almost all trivially
cheap when they exist at all), but *not* demonstrated reliable at the 20-80% level on genuinely
substantive legs at any ToF sampled from 60 to 1200 days, and the obvious small fix for that does
not work. Full detail: `## Guidance diagnostic track`.

**Best legal `J`: 0. Tiebreak `K = m_f` = 1350.916 kg.**
Solution file: `results/legality_tours/greedy_mjd58128_solution.txt`
(launch MJD 58128, `|v_inf|` = 3999.999 m/s, rendezvous with asteroid 2007DC after 1200 days,
1201 samples, `tau` = 3.285 yr, zero violations from the independent checker).

That is a real, rule-satisfying, scorable GTOC4 trajectory — the first this project has produced.
For scale, the winning GTOC4 entry (Moscow State University) scored `J = 44`. A single-digit result
was the expected outcome here; zero is one below that, and the reason is specific rather than
diffuse, so it is worth stating exactly.

**Flybys are not the obstacle. Stopping is.** The greedy sequencer chains up to **eight consecutive
legal flybys**, every one of them closing to better than **12 metres** against a 1000 km tolerance,
with 968 kg of 1500 kg still aboard at the end. Not one of those chains can be *ended*: GTOC4 scores
nothing unless the mission terminates in a rendezvous, and after every flyby the greedy takes, no
rendezvous target survives being flown. The only point at which the tour can legally stop is before
the first flyby, so `J = 0`. This holds at eight of ten launch epochs tested across the legal window,
with `K` between 1253 and 1351 kg — it is a structural result, not a bad-epoch artefact.

**Greedy vs RL, both scored by the checker on their own emitted file:**

| planner | pool | flyby legs planned | rendered legal | `J` | `K` (kg) |
|---|---|---|---|---|---|
| greedy | 1436 | 2 | 2 | 0 | 1263.332 |
| greedy | 120 | 8 | 8 | 0 | 1263.332 |
| greedy, best of 10 launch epochs | 120 | 1 | 1 | **0** | **1350.916** |
| RL (PPO, 30k steps, seed 0) | 120 | 6 | 0 | 0 | 1263.332 |

The RL sequencer does not beat greedy. It learned the intended trade-off — reward 0.82 -> 9.25,
episode length 2.18 -> 7.0, declining the free first leg greedy always takes in order to keep the
terminal rendezvous bonus — and planned a six-flyby tour that is better than anything greedy found
*in the model it was trained in*. The plan then failed to render: on its very first leg the
thrust-limited guidance spent 8267 m/s against the oracle's impulsive estimate of 3270 m/s and ran
into a geometry with no zero-revolution Lambert solution 73 days from arrival.

### What is still blocking a higher score

1. **The mission can only stop where it starts.** Every flyby the greedy takes spends the launch
   `v_inf` — the mission's only impulsive delta-v — on bending the departure toward *that* target,
   after which every rendezvous has to be bought at 0.135 N and none is affordable. A planner that
   chose the rendezvous target first and the flybys as detours along the way to it would not have
   this problem. This is the single biggest gap between `J = 0` and a single-digit `J`.
2. **The leg oracle understates finite-thrust cost by ~2.5x on long legs.** The impulsive Lambert
   delta-v is a lower bound, and the 0.6 duty-cycle derate does not cover the gap. Everything
   planned against it is optimistic, which is exactly what sank the RL rollout. A cost model
   calibrated against the flown cost — even a fitted correction factor as a function of time of
   flight — would make both planners' feasibility tests mean something.
3. **The RL trained against that same optimistic model.** Training against the flown cost would be
   a fair test of the learning; at ~0.15 s per rendered leg against 10^4 episodes it was not
   affordable in this track's compute budget. The greedy only beats the agent because it flies every
   candidate before accepting it — the comparison measures verification, not search.
4. **Launch epoch and `v_inf` direction are chosen greedily and never revisited.** Ten epochs were
   sampled out of a 4018-day window whose reachable-count curve has ~200-day structure; the `v_inf`
   direction is pinned to the first leg's Lambert arc rather than optimised.
5. **The rendezvous pursuit is a heuristic, not an optimiser.** It converges when the target is
   close to reachable and diverges otherwise, with no way to trade a longer leg against a cheaper
   one. Its aim time of flight and duration are searched over a fixed grid of 6 x 3 combinations.
6. **The problem statement document is not in this repository**, so the `solution.txt` column
   layout is an assumption (documented in Phase 3) rather than a transcription. Nothing about
   legality depends on it, but a real submission would.

### Phase index

| phase | what it produced | commit |
|---|---|---|
| 1 | analytic closest approach (fixes a 21,600 km measurement floor), rocket-equation delta-v budget, constants audit | `d77037e` |
| 2 | launch epoch and `v_inf` parameterised, reachable set scanned over the legal ranges (0 -> 208 asteroids) | `6c5b423` |
| 3 | `src/mission.py` legality bookkeeping, `solution.txt` writer, independent checker | `e8975a9` |
| 4 | Lambert solver, leg oracle, greedy tour, RL sequencer, the legal missions above | `a3a7933` |
| 5 (multi-flyby track) | audited the flyby leg-cost model; hypothesized bug already absent, counterfactual confirms the mechanism | `45dc3bf` |
| 6 (multi-flyby track) | empirical Lambert correction fitted three ways; none met its own accuracy bar in 3 attempts -- **halted per halt condition 1** | this commit |

---

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

---

## Phase 3 — Legality bookkeeping and a scorable output

### What changed

- `src/mission.py`: new. `Mission` holds `(epoch, r, v, m, thrust)` samples plus the encounters
  recorded against them, enforces every rule from the Context list as the mission is built, and
  exposes `J` (distinct asteroids flown by before the final rendezvous) and `K = m_f`. Illegal
  operations raise `LegalityError` — nothing is silently recorded. Also `write_solution`,
  `read_solution` and `check_solution`, the last being an independent re-verifier that trusts
  nothing from the `Mission` object.
- `src/dynamics.py`: new `cartesian_to_keplerian(state, mu)`, the scalar inverse of the existing
  `keplerian_to_cartesian`. Needed to define a synthetic body sitting on a given Cartesian state.
  Round-trip verified over 200 random orbits: worst position error 0.58 mm.
- `experiments/acceptance_phase3.py`: new, the positive and negative acceptance tests.
- `results/legality_phase3/`: the written solution file and its four mutations.

`src/gtoc4_env.py` was not modified.

### Solution-file layout used

The GTOC4 problem statement document is **not in this repository** — `data/gtoc4_problem_data.txt`
is the asteroid ephemeris table only, and there is no statement PDF anywhere under
`~/workspace/tud/`. The column layout below is therefore an assumption, to be checked against the
statement before any real submission:

```
MJD  x[km]  y[km]  z[km]  vx[km/s]  vy[km/s]  vz[km/s]  m[kg]  Tx[N]  Ty[N]  Tz[N]  body
```

J2000 heliocentric ecliptic frame, one row per sample, one-day increments within each inter-body
phase plus a partial-day row at each flyby and at the final rendezvous. `body` is `EARTH` on the
launch row, the asteroid's catalogue name on an encounter row, `-` otherwise. The thrust on a row
is the constant thrust applied from that row's epoch to the next row's epoch, zero on the last row.
`#` lines are comments and the checker reads no information from them.

The `body` column is the part most likely to differ from the statement. It is there because the
brief requires the checker to re-verify every rule *from the file alone*, which is impossible
without knowing which asteroid each encounter claims to be. If the statement specifies encounter
identification some other way (a separate header block, a numeric index), only `write_solution`
and `read_solution` change; nothing in `Mission` or the rule checks depends on it.

First and last lines of the produced file:

```
# GTOC4 solution, J2000 heliocentric ecliptic frame
# launch MJD 58430.000000, |v_inf| 2624.881 m/s, tau 300.620 d
# J = 1, K = m_f = 1500.000000 kg, sequence: A B
# MJD x[km] y[km] z[km] vx[km/s] vy[km/s] vz[km/s] m[kg] Tx[N] Ty[N] Tz[N] body
58430.0000000000 104187826.175601 105436617.954153 -1753.104997 -20.174306121 18.825961904 0.799706019 1500.000000 0.000000000 0.000000000 0.000000000 EARTH
58431.0000000000 102429001.393107 107047055.364259 67338.256472 -20.538285601 18.451645841 0.799590454 1500.000000 0.000000000 0.000000000 0.000000000 -
...
58730.3700000000 89658867.136719 117112036.830828 535825.517703 -22.859196213 15.742554749 0.792288880 1500.000000 0.000000000 0.000000000 0.000000000 -
58730.6200000000 89164244.810262 117450944.698469 552933.731318 -22.939101883 15.637742121 0.791802375 1500.000000 0.000000000 0.000000000 0.000000000 B
```

### Acceptance-test output (verbatim)

```
$ ~/miniconda3/envs/tudat-space/bin/python experiments/acceptance_phase3.py
--- positive: hand-built legal mission, written out and re-verified from the file ---
launch                 : MJD 58430.0, |v_inf| = 2624.9 m/s
samples                : 303
duration tau           : 300.620 d = 0.8231 yr
sequence               : A -> B
J (Mission)            : 1
K = m_f (Mission)      : 1500.000 kg
written                : .../results/legality_phase3/solution.txt (47185 bytes)
checker violations     : []
J (checker)            : 1
K (checker)            : 1500.000 kg
PASS: legal mission accepted, J and K re-derived from the file

--- refusal: Mission rejects illegal encounters at construction time ---
  advance after the final rendezvous     -> the mission already ended with a rendezvous; cannot advance further
  |v_inf| = 5 km/s                       -> launch |v_inf| = 5000.0 m/s exceeds the 4000.0 m/s limit
  launch at MJD 50000                    -> launch epoch MJD 50000.0000 outside the legal window (57023.0, 61041.0)
  flyby of A from 1e8 km away            -> 'A': position miss 14728124.458 km exceeds the 1000 km tolerance
PASS: all four illegal operations refused

--- negative: four mutations of that same file, each must be rejected ---
  revisit an asteroid
      - 'A' at MJD 58505.0000: position miss 4906021.963 km exceeds 1000 km
      - asteroid 'A' is visited more than once
      => rejected, expected reason 'visited more than once' present

  tau = 10.5 years
      - mission duration tau = 10.5000 yr exceeds 10.0 yr
      - sample spacing reaches 3534.7550 d, exceeding the one-day increment
      - sample 301->302 (MJD 58730.3700) is not consistent with two-body motion under the declared thrust: 11201536123.3 km, 31822.4400 m/s, 0.000000 kg off
      - 'B' at MJD 62265.1250: position miss 57347601.433 km exceeds 1000 km
      - rendezvous with 'B': velocity miss 12780.6160 m/s exceeds 1.0 m/s
      => rejected, expected reason 'exceeds 10.0 yr' present

  m_f below 500 kg
      - final mass 450.000 kg is below the 500.0 kg minimum
      => rejected, expected reason 'below the 500.0 kg minimum' present

  rendezvous target already flown by
      - asteroid 'B' is visited more than once
      - rendezvous target 'B' was already visited earlier
      => rejected, expected reason 'was already visited earlier' present

PASS: all four mutations rejected with the correct reason

all Phase 3 acceptance tests passed
```

### Numbers

| quantity | value |
|---|---|
| test mission launch | MJD 58430.0, `\|v_inf\|` = 2624.9 m/s |
| samples written | 303 |
| `tau` | 300.620 d = 0.8231 yr |
| `J` (from `Mission`, and re-derived from the file) | 1 |
| `K = m_f` | 1500.000 kg |
| checker violations on the legal file | 0 |
| `cartesian_to_keplerian` round-trip, worst of 200 random orbits | 0.58 mm |

### Interpretation

**The positive test is a round trip, not a self-check.** `check_solution` re-parses the file and
recomputes the launch `v_inf` against Earth's ephemeris, the duration, the sample spacing, the mass
monotonicity and floor, the peak thrust, every encounter's position (and the last one's velocity)
against the body ephemerides, the visit-once rule, and the rendezvous-not-previously-visited rule.
It re-derives `J` and `K` from the file's own contents and both match the `Mission` object. Nothing
is carried across from the producer.

**"No gravity assists" is enforced dynamically, not by name.** There is no list of forbidden bodies
to check against. Instead the checker re-propagates every consecutive pair of samples under
two-body motion plus the thrust the file itself declares, and rejects the file if the next sample
is more than 10 km / 1 m/s away. A gravity assist bends the trajectory by far more than that, and
so does any impulse not paid for out of the thrust and mass columns. This one check therefore
covers the gravity-assist ban, the `T <= 0.135 N` limit's honesty (a file cannot under-declare its
thrust) and `Isp = 3000 s` (the mass drop must match `T/(Isp*g0)` to within 1 mg per step)
simultaneously. The 10 km threshold corresponds to roughly 0.12 m/s of hidden delta-v per one-day
step; RK4 at 10 substeps per day self-agrees to well under a millimetre, so the threshold is loose
by many orders of magnitude relative to numerical noise and still tight relative to any physically
meaningful cheat.

**The negative tests confirm the checker reports the right reason, not merely that it fails.** The
`tau` mutation is the interesting one: moving the final epoch out by 10.5 years breaks five rules
at once (duration, sample spacing, dynamical consistency, the rendezvous position match and its
velocity match), and all five are reported. That is why `check_solution` collects violations rather
than raising on the first — a fail-fast checker would have reported the spacing violation and
concealed the fact that the mission is also too long. The `m_f` mutation is the cleanest: setting
every mass row to 450 kg keeps `dm = 0` consistent with the zero thrust column, so the *only*
violation is the 500 kg floor, exactly as intended.

**What this phase does not establish.** The test mission's `J = 1` is arithmetic, not
astrodynamics: both "asteroids" were reverse-engineered to sit on a coast arc. It proves the
bookkeeping and the file format are correct, and it gives Phase 4 a scoring path it can trust. It
says nothing about whether a real catalogue asteroid can be reached.

### Choices taken where the brief left it open

- **Synthetic test bodies rather than real catalogue asteroids.** Putting the spacecraft within
  1000 km of a real asteroid requires solving a transfer, which is Phase 4's Lambert solver.
  Rejected alternative: defer the Phase 3 acceptance tests until Phase 4 exists, so they could use
  real targets. Not taken because it would leave the writer and checker unvalidated while Phase 4
  was being built on top of them, and the bug class those tests catch (a mis-signed column, a
  frame slip, an off-by-one in the encounter index) does not care whether the body is real.
- **`check_solution` takes `ephemerides` explicitly and does not fall back to the catalogue.** A
  checker that silently guesses which ephemeris an unrecognised name refers to is not checking
  anything; an unknown name is reported as a violation instead. Phase 4 passes the parsed
  catalogue.
- **Launch mass is not checked against 1500 kg.** The rule list in the brief constrains `m_f >= 500
  kg` and `T <= 0.135 N` but does not restate a fixed launch mass, and `Mission` accepts a
  `launch_mass` argument defaulting to `spacecraft_wet_mass`. If the statement fixes the launch mass
  at 1500 kg, one line in `check_solution` adds it.
- **`J` counts flybys only, excluding the final rendezvous target**, per the brief's wording
  ("distinct asteroids visited before the final rendezvous"). Both `Mission.J` and the checker's
  independently computed `J` use this definition, so they cannot silently disagree.

---

## Phase 4 — Sequencing

### What changed

- `src/lambert.py`: new. Minimal zero-revolution universal-variable Lambert solver.
- `src/sequencer.py`: new. The leg oracle, two thrust-limited guidance laws that render an
  impulsive plan into a real trajectory, the greedy tour, and the shared terminal-rendezvous
  procedure.
- `src/curriculum.py`: `delta_v_budget` gains an optional `initial_mass` (default: the wet mass,
  so the Phase 1b reference values are unchanged) — a leg partway through a tour starts lighter.
- `src/mission.py`: `advance`'s `thrust` argument is now the thrust applied over the interval
  *ending* at the sample, stored against the previous sample, which is where the solution format
  wants it; `rendezvous` gained the same argument. The Phase 3 acceptance tests are unaffected
  (they fly a zero-thrust coast) and still pass.
- `experiments/acceptance_phase4.py`: new, the Lambert acceptance test.
- `experiments/run_sequencer.py`: new, the greedy baseline runner and the pool restriction.
- `experiments/train_sequencer.py`: new, the RL sequencer.
- `results/legality_tours/`: solution files and logs.

### 4.1 Lambert solver — acceptance test (verbatim)

```
$ ~/miniconda3/envs/tudat-space/bin/python experiments/acceptance_phase4.py
--- 4.1: universal-variable Lambert vs a propagated Keplerian arc (200 random cases) ---
cases                   : 200, a in [0.8, 3.0] AU, e in [0.01, 0.7], tof in [0.05, 0.85] period
transfer angle range    : 7.1 - 179.6 deg (98 cases above 90 deg)
median |v1| error       : 0.064660 mm/s
median |v2| error       : 0.066802 mm/s
worst |v1| error        : 0.310532 mm/s
worst |v2| error        : 0.288289 mm/s
worst case              : transfer angle 68.2 deg
PASS: both endpoint velocities recovered to better than 1 mm/s in every case

Phase 4.1 acceptance test passed
```

Throughput: 131 us per solve (4308 solves in 0.56 s), measured on the same machine.

### 4.2 Leg feasibility oracle

`sequencer.leg_candidates` solves Lambert from the current state to every pool member's state at
`t + ToF` and returns `dv1` (departure, after crediting the launch v_inf when it is still
available) and `dv2` (arrival velocity match). `sequencer.leg_feasible` compares that against
`curriculum.delta_v_budget(ToF, current_mass)` derated by a 0.6 duty cycle.

### 4.3 Greedy baseline (verbatim, at the Phase 2 best launch epoch MJD 58430)

Full 1436-asteroid catalogue:

```
greedy sequencer: launch MJD 58430.0, pool 1436 asteroids, ToF buckets (60, 150, 300, 600, 1200) d, duty cycle 0.6, rendezvous reserve 600 d, top_k 6
    rendezvous try 2008TS10     aim   400 d pursue  1200 d  miss        0.336 km  dv     0.000 m/s  m  1263.3 kg
  -> scorable after 0 flybys: rendezvous 2008TS10, m_f 1263.3 kg
  leg 0: flyby 2007YQ56     tof    60 d  dv1      0.0 m/s  rank 0  miss    0.004 km  m  1500.0 kg
  leg 1: flyby 2000WP19     tof   300 d  dv1    398.1 m/s  rank 0  miss    0.004 km  m  1477.5 kg
  flyby chain stopped: no feasible flyby candidate

  solution file        : .../results/legality_tours/greedy_solution.txt
  samples              : 1201
  checker violations   : none
  sequence             : 2008TS10
  tau                  : 1200.00 d = 3.285 yr
  J (from the file)    : 0
  K = m_f (from file)  : 1263.332 kg
```

Restricted 120-asteroid pool (the same pool the RL sequencer is trained on):

```
greedy sequencer: launch MJD 58430.0, pool 120 asteroids, ToF buckets (60, 150, 300, 600, 1200) d, duty cycle 0.6, rendezvous reserve 600 d, top_k 6
    rendezvous try 2008TS10     aim   400 d pursue  1200 d  miss        0.336 km  dv     0.000 m/s  m  1263.3 kg
  -> scorable after 0 flybys: rendezvous 2008TS10, m_f 1263.3 kg
  leg 0: flyby 88254        tof   150 d  dv1      0.0 m/s  rank 0  miss    0.007 km  m  1500.0 kg
  leg 1: flyby 2007VL3      tof   300 d  dv1    688.0 m/s  rank 1  miss    0.009 km  m  1450.8 kg
  leg 2: flyby 2008PW4      tof   300 d  dv1   1263.6 m/s  rank 0  miss    0.002 km  m  1382.1 kg
  leg 3: flyby 2002VX91     tof   300 d  dv1    428.0 m/s  rank 0  miss    0.012 km  m  1357.4 kg
  leg 4: flyby 2003WY153    tof   600 d  dv1   3093.9 m/s  rank 2  miss    0.003 km  m  1205.9 kg
  leg 5: flyby 2006VU2      tof   300 d  dv1   1174.6 m/s  rank 0  miss    0.008 km  m  1159.2 kg
  leg 6: flyby 164215       tof   600 d  dv1   1987.1 m/s  rank 0  miss    0.007 km  m  1018.5 kg
  leg 7: flyby 2006DN       tof   300 d  dv1   1410.9 m/s  rank 0  miss    0.003 km  m   968.3 kg
  flyby chain stopped: no feasible flyby candidate

  solution file        : .../results/legality_tours/greedy_pool120_solution.txt
  samples              : 1201
  checker violations   : none
  sequence             : 2008TS10
  tau                  : 1200.00 d = 3.285 yr
  J (from the file)    : 0
  K = m_f (from file)  : 1263.332 kg
```

First and last data rows of the scorable file:

```
# GTOC4 solution, J2000 heliocentric ecliptic frame
# launch MJD 58430.000000, |v_inf| 3811.411 m/s, tau 1200.000 d
# J = 0, K = m_f = 1263.331711 kg, sequence: 2008TS10
# MJD x[km] y[km] z[km] vx[km/s] vy[km/s] vz[km/s] m[kg] Tx[N] Ty[N] Tz[N] body
58430.0000000000 104187826.175601 105436617.954153 -1753.104997 -20.993455390 24.537535742 0.535910680 1500.000000 -0.110914111 0.002718816 -0.076913380 EARTH
...
59630.0000000000 -224477421.566971 31095539.048323 1362924.824685 -2.847513901 -21.407920909 -0.541882435 1263.331711 0.000000000 0.000000000 0.000000000 2008TS10
```

### Numbers so far

| quantity | value |
|---|---|
| Lambert worst endpoint-velocity error, 200 cases | 0.311 mm/s |
| Lambert solve throughput | 131 us |
| flyby legs flown at <1000 km, greedy on the 120-pool | 8 (worst miss 0.012 km) |
| flyby legs flown at <1000 km, greedy on the full pool | 2 (worst miss 0.004 km) |
| best legal mission, greedy (either pool) | J = 0, K = 1263.332 kg, tau = 3.285 yr |
| launch `\|v_inf\|` used | 3811.4 m/s of the 4000 allowed |
| checker violations on the emitted file | none |

### Interpretation (greedy)

**A legal, checker-verified GTOC4 mission now exists.** `results/legality_tours/greedy_solution.txt`
launches from Earth at MJD 58430 with 3811 m/s of hyperbolic excess, thrusts under 0.135 N
throughout, and rendezvouses with 2008TS10 1200 days later at 0.336 km and 0.000 m/s, with
1263.332 kg of the 1500 kg left. `check_solution` re-derives every rule from the file alone and
returns no violations. That is the first scorable result this project has produced.

**Flybys are not the hard part; stopping is.** On the 120-asteroid pool the greedy chains eight
consecutive flybys, every one of them closing to better than 12 metres against a 1000 km tolerance,
with 968 kg still aboard after 8 legs. None of those chains can be *ended*: after every one of the
eight, no rendezvous candidate survives being flown. So `J = 0` is not a statement that flybys are
unreachable — eight of them were flown and verified. It is a statement that a mission which cannot
end in a rendezvous scores nothing, and that cheapest-next chooses legs with no regard for that.

The mechanism is visible in the leg table. The greedy's first leg is always the one with `dv1 = 0`,
a leg the launch v_inf pays for entirely. But a free leg is free precisely because the launch
excess was spent bending the departure toward *that* target, and the launch v_inf is the mission's
only impulsive delta-v. After it is committed the spacecraft is on a fast, non-Earth-like orbit,
and every subsequent rendezvous has to be paid for out of 0.135 N. The scorable mission spends its
3811 m/s aiming at 2008TS10's 400-day arc instead, and gets a rendezvous.

**Two guidance laws, and the difference between them matters.** Position-only flyby targeting
(shrinking aim) is easy: the required velocity correction drives itself to zero, at which point the
spacecraft is on the ballistic arc that arrives exactly, so the miss collapses to metres. Every one
of the ten flyby legs flown across both pools closed to under 12 m. A rendezvous cannot be done
that way at all: matching ~1.8 km/s of arrival velocity at 0.135 N takes ~230 days of continuous
braking, so the arrival is not a terminal correction on a ballistic arc, it is the last third of
the leg. The sliding-aim pursuit — aim where the target will be a lead time ahead, ramp the lead
down over the leg — converges onto the target's own orbit instead of onto one point of it, and
brings the velocity error down with the position error. That is what made the rendezvous reachable.

**The aim time of flight and the pursuit duration are different quantities, and conflating them
fails loudly.** Aiming the launch v_inf at 2008TS10's 400-day arc and pursuing for 1200 days lands
0.336 km away at 0.000 m/s with 1263 kg left; aiming at its 300-day or 600-day arc and pursuing for
the same 1200 days diverges to 2.5e8 km. The aim decides which orbit the spacecraft is thrown onto;
the duration decides how long it has to converge. Searching them jointly is what turned "no
scorable mission" into a scorable one.

**Two measured facts that cost real time, recorded so they are not rediscovered.**
- The vectorised pre-screen (rank by `|(r_target - r)/tof - v|`) is far too weak to prune with.
  One leg into a tour it raised the cheapest reachable `dv1` at a 300-day time of flight from
  398 m/s to 12880 m/s at its best 60, and its best 600 still missed the 398 m/s option. It ignores
  gravitational turning, which is most of a real transfer. Off by default.
- Phase 2's closed-form screen `curriculum._delta_v_needed` cannot order a pool either, only decide
  whether one is empty. It ranks 2008TS10 — the only asteroid this launch epoch can actually
  rendezvous with — 132nd out of 1436; a 120-asteroid pool built from it contains no rendezvous
  target at all. The Lambert screen ranks it 1st. This does not undermine Phase 2's conclusion,
  which was about whether the candidate set is empty, but it does bound what that estimator is for.
- The Lambert solver's original fixed `[-4*pi^2, 4*pi^2]` bisection bracket failed to converge
  roughly once per 1500 calls inside the pursuit, which silently killed every rendezvous attempt.
  Both ends now walk to a valid bracket before bisecting.

### 4.3 addendum — greedy across ten legal launch epochs

`experiments/scan_launch_epochs.py` reruns the greedy tour at ten epochs spread over the legal
window (the epochs Phase 2's scan flagged as locally good), each with its own 120-asteroid
Lambert-screened pool.

```
 launch MJD  flybys chained  scorable after   J     K (kg)  rendezvous
      57224               1            None None         -  None
      57727               1            None None         -  None
      58028               1               0    0    1270.9  2006UB17
      58128               1               0    0    1350.9  2007DC
      58430               8               0    0    1263.3  2008TS10
      58631               1               0    0    1262.3  2001KW18
      59133               1               0    0    1253.1  2000TE2
      59635               1               0    0    1280.9  2006UQ216
      60238               2               0    0    1291.9  2000TE2
      60338               1               0    0    1281.4  2005ES1
```

"flybys chained" is how many legal flyby legs the greedy strings together; "scorable after" is how
many of those it can still stop after with a legal rendezvous.

Best mission by the tiebreak, since `J` ties at 0 everywhere: **MJD 58128, rendezvous with 2007DC,
`J = 0`, `K = 1350.916 kg`**, launch `|v_inf| = 4000.0 m/s` (the full legal allowance), `tau = 1200`
days, 1201 samples, zero checker violations —
`results/legality_tours/greedy_mjd58128_solution.txt`.

Interpretation: `J = 0` is not an artefact of one launch epoch. Eight of the ten epochs produce a
legal scorable mission and every one of them scores zero, because in every case the only point at
which the tour can legally stop is before the first flyby. Two epochs (MJD 57224, 57727) produce no
scorable mission at all — the greedy's first flyby is taken before the rendezvous check can save it
and nothing afterwards closes. The `K` spread across the eight is narrow, 1253-1351 kg, which is
what one would expect when every mission is the same shape: one aimed launch and one 1200-day
pursuit.

Two serialization fixes went in alongside this rerun, both found by the checker rather than by
inspection, and both of the same kind: a mission legal in exact arithmetic became illegal on the
round trip through the file.

- Thrust and velocity components are now written with 12 decimals instead of 9. At 9, rounding a
  thrust component pushed the magnitude reconstructed from the file 1e-9 N above `thrust_max`.
- The launch `v_inf` is backed off 1 mm/s from the 4.0 km/s maximum (`sequencer.V_INFINITY_LIMIT`).
  Four of the ten epochs want the full allowance, and an exactly-4000.0 m/s departure came back
  from the file as 4000.000000001 m/s: `greedy_mjd59635_solution.txt` was rejected with
  `launch |v_inf| = 4000.0 m/s exceeds 4000.0 m/s`. 1 mm/s is 2.5e-7 of the allowance, far below
  anything else in the model.

All ten files re-verify clean after the fixes:

```
greedy_mjd58028_solution.txt       J=0 K= 1270.880 vinf=1844.313804 peakT=0.135000000001 viol=0
greedy_mjd58128_solution.txt       J=0 K= 1350.916 vinf=3999.999000 peakT=0.135000000001 viol=0
greedy_mjd58430_solution.txt       J=0 K= 1263.332 vinf=3811.410960 peakT=0.135000000001 viol=0
greedy_mjd58631_solution.txt       J=0 K= 1262.308 vinf=3999.999000 peakT=0.135000000001 viol=0
greedy_mjd59133_solution.txt       J=0 K= 1253.108 vinf=3965.887159 peakT=0.135000000001 viol=0
greedy_mjd59635_solution.txt       J=0 K= 1280.871 vinf=3999.999000 peakT=0.135000000001 viol=0
greedy_mjd60238_solution.txt       J=0 K= 1291.933 vinf=3541.500947 peakT=0.135000000001 viol=0
greedy_mjd60338_solution.txt       J=0 K= 1281.372 vinf=3470.657216 peakT=0.135000000001 viol=0
greedy_pool120_solution.txt        J=0 K= 1263.332 vinf=3811.410960 peakT=0.135000000001 viol=0
greedy_solution.txt                J=0 K= 1263.332 vinf=3811.410960 peakT=0.135000000001 viol=0
```

(The residual `peakT = 0.135000000001` is the floating-point norm of the components themselves,
1e-12 N over, well inside the checker's `RULE_EPS = 1e-9`.)

### 4.4 RL sequencer

Setup: `MultiDiscrete([K=6, 3 buckets])` over the same leg oracle the greedy uses — the k-th
cheapest feasible candidate at ToF bucket b, buckets (150, 400, 1000) d. Observation: normalised
`r`, `v`, `m`, mission-time fraction, flybys so far, and the `dv1/budget` of every (bucket, rank)
cell. Reward +1 per flyby plus a terminal term: `5.0 * m_f/m_0` if an impulsive rendezvous is still
feasible when the tour stops, `-3.0` if not. Greedy is exactly the `k = 0`, argmin-over-buckets
policy, so the agent's only job is to learn when *not* to take the cheapest leg.

One run, seed 0, 30,208 timesteps, 120-asteroid pool, 2338 s wall clock (39 min) — inside the
track's budget of two runs at 250k steps each. Learning curve from the run log:

```
| ep_len_mean | ep_rew_mean | time_elapsed | total_timesteps |
|        2.18 |       0.822 |           24 |             256 |
|        2.12 |        1.96 |           48 |             512 |
|        3.23 |        6.23 |          480 |            5120 |
|        5.82 |         8.2 |          928 |           10240 |
|        6.93 |         9.2 |         1325 |           15360 |
|        7    |        9.25 |         1712 |           20480 |
|        6.98 |        9.18 |         2030 |           25600 |
|        7    |        9.25 |         2338 |           30208 |
```

Deterministic rollout and rendering:

```
deterministic rollout:
  2005ES1      tof  1000 d  dv1   3269.5 m/s  rank 1
  2003ND       tof   400 d  dv1   1675.1 m/s  rank 0
  2004TT12     tof   400 d  dv1   1124.8 m/s  rank 0
  2007EK       tof   400 d  dv1   1610.1 m/s  rank 0
  2006VU2      tof   400 d  dv1   2392.4 m/s  rank 0
  1994EK       tof   400 d  dv1   2616.2 m/s  rank 0
  planned flybys: 6

rendering:
    rendezvous try 2008TS10     aim   400 d pursue  1200 d  miss        0.336 km  dv     0.000 m/s  m  1263.3 kg
  -> scorable after 0 flybys: rendezvous 2008TS10, m_f 1263.3 kg
  leg 0: 2005ES1 did not close -- truncating here

  solution file        : .../results/legality_tours/rl_solution.txt
  samples              : 1201
  checker violations   : none
  sequence             : 2008TS10
  tau                  : 1200.00 d = 3.285 yr
  J (from the file)    : 0
  K = m_f (from file)  : 1263.332 kg
```

### Both J values, side by side

| planner | pool | flyby legs planned | flyby legs rendered legal | **J (checker)** | **K (kg)** |
|---|---|---|---|---|---|
| greedy | 1436 | 2 | 2 | **0** | **1263.332** |
| greedy | 120 | 8 | 8 | **0** | **1263.332** |
| greedy, best of 10 launch epochs (MJD 58128) | 120 | 1 | 1 | **0** | **1350.916** |
| RL (PPO, 30k steps, seed 0) | 120 | 6 | 0 | **0** | **1263.332** |

**The agent does not beat greedy.** Both score `J = 0` with the same `K`, and both do it by
stopping before the first flyby. Reported plainly, as the brief asked; it was also the more likely
outcome.

### Interpretation (RL)

**The agent learned what it was asked to learn, and the thing it was asked to learn was wrong.**
Episode length went 2.18 -> 7.0 and reward 0.82 -> 9.25, saturating by ~15k steps. Decomposing the
final 9.25: six flybys at +1 each, plus ~3.25 of terminal bonus, which is `5.0 * 974/1500` — the
agent is collecting the rendezvous bonus, not the penalty. In the model it was trained in, it found
exactly what the reward asked for: a six-flyby tour that still has an affordable rendezvous at the
end, which is strictly better than anything the greedy found. Greedy never chains more than 2 on
the full pool and can stop after none of them.

**It does not survive contact with the guidance.** Rendering the rollout, the very first leg fails.
Instrumenting it: the guidance flies 927 of the leg's 1000 days, spends **8267 m/s** of delta-v
against the oracle's impulsive estimate of **3270 m/s** — 2.5x more — and then, with 73 days to go,
reaches a geometry with no zero-revolution Lambert solution at all and cannot continue. The plan
was never flyable; the oracle only ever said it was affordable.

That gap is the finding, and it is a specific, quantified one. The impulsive Lambert cost is a
*lower bound* on what a 0.135 N spacecraft spends, and the bound is loose in proportion to how long
the leg is and how far off the ballistic arc the guidance starts: the shrinking-aim law keeps
re-correcting toward a receding target, and every correction it makes early is partly undone later.
A 0.6 duty-cycle derate is not enough to cover a 2.5x discrepancy.

**The greedy baseline is not better at planning; it is better at checking.** It evaluates the same
oracle and takes the same kind of leg, but it *flies every candidate before accepting it* and falls
through to the next one when the flight fails — which is affordable at ~0.15 s per attempt for a
ten-leg tour, and completely unaffordable at 10^4 training episodes. The comparison here is
therefore not "greedy search vs learned policy". It is "verified plan vs unverified plan", and the
verification is doing all the work. An RL sequencer that trained against the flown cost rather than
the impulsive one would be a fair test of the learning; this run was not one, and the budget for it
does not exist in this track.

**What the agent's rollout does show.** Its planned sequence — six legs at 1125-3270 m/s each,
mostly at the 400-day bucket, mostly rank 0 but starting with a rank-1 pick at the 1000-day bucket
— is visibly not the greedy's. It deliberately declines the free `dv1 = 0` first leg that greedy
always takes, because in its model that leg destroys the terminal bonus. That is precisely the
trade-off the reward was designed to expose, and the agent found it. The impulsive model just is
not accurate enough for the answer to mean anything.

One bookkeeping loss: `results/legality_tours/rl/monitor.csv` was overwritten by the `--render-only`
rerun (it opens a fresh `Monitor` on the same path), so the per-episode training history is gone.
The learning curve above, from the run log, is what survives.

### Choices taken where the brief left it open (Phase 4)

- **`leg_feasible` charges `dv1` only for a flyby**, `dv1 + dv2` for a rendezvous. The brief's
  oracle charges `dv1 + dv2` unconditionally. Rejected because a flyby never performs the arrival
  impulse — the spacecraft continues on the Lambert arrival velocity, and that velocity is exactly
  the state the next leg departs from — so charging for it prices a different mission than the one
  being flown. It is not conservatism, it is a different mission.
- **The greedy verifies each leg by flying it before accepting it**, and attempts the terminal
  rendezvous after every flyby, keeping the longest chain that still ends in one. Rejected
  alternative: a fixed time reserve for the rendezvous, which is what was tried first and which
  cannot work — how much time the rendezvous needs depends on which orbit the flybys left the
  spacecraft on, not on the clock.
- **`results/` and `figures/` are gitignored by project convention**, so the tour outputs and the
  Phase 2 figure are not tracked. One exception is force-added:
  `results/legality_tours/greedy_mjd58128_solution.txt`, the scored deliverable this track exists
  to produce — the summary points at it, and a pointer to an untracked file is not a deliverable.
  Everything else regenerates from `experiments/run_sequencer.py`,
  `experiments/scan_launch_epochs.py`, `experiments/scan_reachability.py` and
  `experiments/train_sequencer.py --render-only`.
- **Training budget used: one run** (30,208 steps, seed 0, 39 min), of the two allowed. Two earlier
  attempts were aborted inside ten minutes each and are not counted: the first because the pool
  restriction was built on Phase 2's closed-form screen and contained no rendezvous target at all,
  the second on relaunching it as a tracked background job. Neither produced a result.

---

## Multi-flyby track

Continuation of the legality track's Phase 4 finding: a legal mission exists (J = 0, K = 1350.916
kg), but the launch v_inf gets fully spent bending toward the first flyby target, leaving nothing
for a rendezvous afterward. Goal here: chain multiple legal flybys into a legal rendezvous. Not
J = 44 (MSU's winning entry, ~83 days/flyby over 10 years) — a defensible AE4350-scale result,
honestly reported, with the limiting mechanism identified.

### Phase 5 — Audit the leg cost model for flybys

#### What changed

Nothing in `src/`. The static audit found the hypothesized bug already absent: `sequencer.py`'s
`leg_feasible` (written during the legality track's Phase 4, see its docstring) already charges
`dv1` only for a flyby and `dv1 + dv2` only for a rendezvous, and every call site in `sequencer.py`
and `train_sequencer.py` passes `require_velocity_match` correctly. There is nothing to fix.

To still produce the before/after comparison the acceptance test asks for, `experiments/
acceptance_phase5.py` reconstructs the counterfactual "buggy" oracle (dv1 + dv2 charged
unconditionally, flyby included) by monkey-patching `sequencer.leg_feasible` for the "before" run
only, entirely inside the test script, then restores it and reruns for "after" (the current,
unmodified code). No production file was touched.

#### Acceptance-test output (verbatim)

```
$ ~/miniconda3/envs/tudat-space/bin/python experiments/acceptance_phase5.py
--- static audit: every leg_feasible call site ---
  sequencer:246: if leg_feasible(candidate, state[6], True, duty_cycle):
  sequencer:340: if leg_feasible(c, state[6], False, duty_cycle)]
  train_sequencer:84: if sequencer.leg_feasible(c, self.state[6], False)][:self.k]
  train_sequencer:122: if sequencer.leg_feasible(candidate, self.state[6], True):
call sites found: 4

--- live audit: full 1436-asteroid pool, launch MJD 58128.0 ---

--- BEFORE (counterfactual: dv1+dv2 charged on every leg, flyby included) (wall clock 8 s) ---
  flyby legs planned         : 1
  flyby leg ToFs (days)       : [300]
  median flyby ToF            : 300.0 d
  mean flyby ToF              : 300.0 d
  scorable_after_flybys       : 0
  rendezvous achieved          : True
  rendezvous duration_days    : 1200
  rendezvous mass_kg          : 1350.916

--- AFTER (current code: dv1-only on flyby legs, dv1+dv2 on rendezvous) (wall clock 14 s) ---
  flyby legs planned         : 3
  flyby leg ToFs (days)       : [60, 150, 300]
  median flyby ToF            : 150.0 d
  mean flyby ToF              : 170.0 d
  scorable_after_flybys       : 0
  rendezvous achieved          : True
  rendezvous duration_days    : 1200
  rendezvous mass_kg          : 1350.916

--- comparison ---
  before: n=1 legs, median=300.0
  after : n=3 legs, median=150.0
  median ToF change (after vs before): +50.0 %

Phase 5 acceptance data collected
```

#### Numbers

| quantity | before (dv1+dv2 on flybys) | after (dv1-only, current code) |
|---|---|---|
| flyby legs chained | 1 | 3 |
| flyby leg ToFs (days) | [300] | [60, 150, 300] |
| median flyby ToF | 300.0 d | 150.0 d |
| mean flyby ToF | 300.0 d | 170.0 d |
| flybys in the final scored tour | 0 | 0 |
| final rendezvous leg length | 1200 d | 1200 d |
| final J | 0 | 0 |
| final K | 1350.916 kg | 1350.916 kg |

#### Interpretation

**The hypothesized bug does not exist in the current code.** `leg_feasible` has charged `dv1` only
for flybys since it was written in the legality track's Phase 4 — this was already identified and
fixed before this track started, not newly discovered here.

**Reintroducing it as a counterfactual confirms the mechanism is real, in the predicted direction.**
Charging `dv2` on a flyby prices every near-miss as if it needed a full velocity match (order
1-3 km/s for these targets), which is enough to make short-ToF candidates infeasible against the
smaller delta-v budget available over a short window; the greedy is pushed toward longer ToF
buckets, where the budget is larger and can absorb the inflated cost. Measured effect: 3x more
flybys chained (1 -> 3) at half the median duration (300 -> 150 days) once the correct dv1-only
costing is used. Phase 6's plan to sample legs down to 83-day scale is reasonable on this evidence
-- short legs are reachable once flybys are not mispriced.

**But this does not change the final scored mission at all.** With the bug or without it, the tour
the greedy actually returns is identical: zero flybys survive to the winning tour, and the single
rendezvous leg is 1200 days in both cases, because `RENDEZVOUS_DURATIONS_DAYS` starts at 1200 and
`finish_with_rendezvous` is never asked to try anything shorter. So this audit explains why the
*attempted* flyby chain lengthens under the wrong cost model, but it does not explain why *no*
flyby chain -- short or long, buggy costing or correct -- survives into a closeable tour. That is
a downstream effect of the greedy's single-path search: it commits to the cheapest flyby, evaluates
rendezvous feasibility only from the state that leg produces, and never explores an alternative
branch when that state turns out to be unrecoverable. Fixing the cost model was necessary (it
triples the flyby count explored) but not sufficient; the search itself only ever tries one path.

**Reading against halt condition 3.** The condition asks whether fixing the flyby-cost bug (if it
exists) materially shortens *typical leg length*. Read as the length of the legs the search
explores and is willing to chain, the answer is a clear, measured yes: median flyby ToF dropped 50%
and the chain grew 3x. Read as the length of the leg in the final *scored* tour, the answer is no:
that leg is 1200 days either way, because it is the only leg that survives, and it always has been.
Both readings are reported above rather than picked between. The evidence that opening up shorter,
cheaper flybys (Phase 6's calibration) combined with a search that can hold open more than one
branch at a time (Phase 7-8's beam search) addresses the actual mechanism -- a single-path greedy
that abandons a branch the instant it looks unrecoverable, rather than an underpriced or overpriced
leg -- justifies proceeding rather than halting here.

### Phase 6 — Empirical correction for the Lambert oracle

#### What changed

`experiments/acceptance_phase6.py` only. **`src/sequencer.py` was not modified.** Three empirical
correction curves were fitted and evaluated against a common held-out set; none met the phase's own
acceptance bar (within 20% of true flown delta-v for every held-out leg) in three attempts, so per
the standing halt rules no correction was applied to the oracle. This halts the multi-flyby track
at the end of Phase 6 (halt condition 1: "an acceptance test fails and you cannot fix it within
three attempts") -- see Interpretation below for why, and what was learned regardless.

#### Method

Legs were sampled from two kinds of plausible departure state -- `earth` (Earth departure with the
4 km/s `v_inf` credit available, as the first leg of a mission) and `mid` (sitting exactly on a
random real asteroid's own orbital state at a random epoch, a proxy for "the spacecraft is
somewhere in the near-Earth population," not a literal post-flyby state -- see Interpretation) --
crossed with 9 ToF buckets from 60 to 1200 days. For each (kind, ToF), a real `sequencer.
leg_candidates()` call ranked the actual reachable targets from that state, and a candidate was
drawn biased toward the cheapest few ranks (50/30/20% for rank 0/1/2), exactly mirroring what the
greedy sequencer or a beam search would actually consider -- not a uniform-random (state, target)
pair, which is essentially always unreachable (curriculum.py's own finding: a naive draw needs
~50 km/s). Each candidate was then *flown* with the real thrust-limited shrinking-aim guidance
(`fly_flyby_leg`); only attempts that closed within the 1000 km flyby tolerance and had a
non-trivial estimated cost (`dv1_est > 30 m/s`, to exclude the free-by-v_inf-credit legs that carry
no information about the correction) entered the fit. Attempts per bucket were capped at 120, up to
6 usable samples each.

#### Acceptance-test output (verbatim)

```
$ ~/miniconda3/envs/tudat-space/bin/python experiments/acceptance_phase6.py
--- sampling: 9 ToF buckets x up to 6 usable legs each, launch MJD 58128.0 ---
  ToF    60 d: 0/120 usable (missed=25, fly_fail=41, no_candidates=0, trivial=54)
  ToF    90 d: 0/120 usable (missed=14, fly_fail=57, no_candidates=0, trivial=49)
  ToF   120 d: 0/120 usable (missed=21, fly_fail=39, no_candidates=0, trivial=60)
  ToF   180 d: 6/65 usable (missed=12, fly_fail=13, no_candidates=0, trivial=34)
  ToF   300 d: 6/21 usable (missed=1, fly_fail=3, no_candidates=0, trivial=11)
  ToF   450 d: 6/13 usable (missed=1, fly_fail=2, no_candidates=0, trivial=4)
  ToF   600 d: 6/8 usable (missed=1, fly_fail=1, no_candidates=0, trivial=0)
  ToF   900 d: 6/12 usable (missed=1, fly_fail=5, no_candidates=0, trivial=0)
  ToF  1200 d: 6/11 usable (missed=0, fly_fail=5, no_candidates=0, trivial=0)
collected 36 usable samples in 187.2 s

--- per-bucket outcome counts ---
total attempts: 490, fly_fail: 166 (33.9 %), missed: 76 (15.5 %)
short ToF (<=120 d) failure rate: 197/360 (54.7 %)

--- attempt 1: binned mean, combined ---
  ToF   180 d: correction =  1.390  (n=6)
  ToF   300 d: correction =  1.193  (n=6)
  ToF   450 d: correction =  1.357  (n=6)
  ToF   600 d: correction =  1.490  (n=6)
  ToF   900 d: correction =  1.444  (n=6)
  ToF  1200 d: correction =  2.115  (n=6)

--- attempt 2: binned mean, split by kind ---
  earth: 600d=1.553, 1200d=2.465
  mid: 180d=1.390, 300d=1.193, 450d=1.357, 600d=1.364, 900d=1.444, 1200d=1.414

--- attempt 3: log-linear regression ---
  ratio = exp(-0.8367) * tof_days^0.1885

--- held-out test: attempt 1 (binned, combined) ---
  ToF   350 d  kind=mid    est=   967.9  corrected=  1207.3  flown=  1044.5  error= 15.6 %  PASS
  ToF   380 d  kind=mid    est=  1045.2  corrected=  1338.1  flown=  1119.7  error= 19.5 %  PASS
  ToF  1190 d  kind=earth  est=  3571.5  corrected=  7472.9  flown=  8567.5  error= 12.8 %  PASS
  ToF   430 d  kind=mid    est=   246.2  corrected=   328.7  flown=   256.1  error= 28.4 %  FAIL
  ToF   930 d  kind=mid    est=   305.5  corrected=   461.7  flown=   306.8  error= 50.5 %  FAIL
  3/5 within 20 %

--- held-out test: attempt 2 (binned, by kind) ---
  ToF   350 d  kind=mid    est=   967.9  corrected=  1207.3  flown=  1044.5  error= 15.6 %  PASS
  ToF   380 d  kind=mid    est=  1045.2  corrected=  1338.1  flown=  1119.7  error= 19.5 %  PASS
  ToF  1190 d  kind=earth  est=  3571.5  corrected=  8750.5  flown=  8567.5  error=  2.1 %  PASS
  ToF   430 d  kind=mid    est=   246.2  corrected=   328.7  flown=   256.1  error= 28.4 %  FAIL
  ToF   930 d  kind=mid    est=   305.5  corrected=   440.3  flown=   306.8  error= 43.5 %  FAIL
  3/5 within 20 %

--- held-out test: attempt 3 (log-linear) ---
  ToF   350 d  kind=mid    est=   967.9  corrected=  1264.7  flown=  1044.5  error= 21.1 %  FAIL
  ToF   380 d  kind=mid    est=  1045.2  corrected=  1387.1  flown=  1119.7  error= 23.9 %  FAIL
  ToF  1190 d  kind=earth  est=  3571.5  corrected=  5877.6  flown=  8567.5  error= 31.4 %  FAIL
  ToF   430 d  kind=mid    est=   246.2  corrected=   334.5  flown=   256.1  error= 30.6 %  FAIL
  ToF   930 d  kind=mid    est=   305.5  corrected=   480.0  flown=   306.8  error= 56.4 %  FAIL
  0/5 within 20 %

--- summary: 5 held-out legs ---
  attempt 1 (binned, combined) : 3/5 within 20 %
  attempt 2 (binned, by kind)  : 3/5 within 20 %
  attempt 3 (log-linear)       : 0/5 within 20 %

  best of the three: binned_combined (3/5)
```

#### Numbers

| quantity | value |
|---|---|
| total leg attempts across all buckets | 490 |
| overall fly_fail rate | 33.9 % |
| overall missed (closed guidance run, but >1000 km off) rate | 15.5 % |
| **short ToF (<=120 d) failure rate (fly_fail + missed)** | **54.7 % (197/360)** |
| **usable (closed, non-trivial) samples at ToF in {60, 90, 120} d** | **0 / 360 attempts** |
| usable samples at ToF in {180, ..., 1200} d | 36 (6 per bucket) |
| flown/estimated ratio range across buckets | 1.19x - 2.12x |
| best held-out pass rate (3 attempts) | 3/5 (60 %) -- required 5/5 |
| held-out legs that failed in all 3 attempts | ToF 430 d (ratio 1.04, needs ~no correction) and ToF 930 d (ratio 1.00, needs ~no correction) |

#### Interpretation

**No usable short-ToF sample exists at all.** At each of the three shortest buckets (60, 90, 120
days), all 120 attempts landed in exactly three outcomes: the candidate was free (already covered
by the launch `v_inf` credit, so uninformative), or it missed by a wide margin, or the guided
propagation failed outright (typically propellant exhaustion under continuous saturation, since
`_guided_step` commands `thrust_max` whenever the required correction is large). Not one non-trivial
candidate closed. This bears directly on the track's own premise: MSU's winning entry averages one
flyby every ~83 days over ten years, which requires exactly this ToF range to work routinely, and
in this sample it does not work at all -- not "requires a bigger correction," but zero verified
successes out of 360 attempts using the oracle's own top-ranked candidates. Whatever limits
multi-flyby chaining, short legs failing outright at the guidance level is now a directly measured
part of it, prior to and independent of anything the later search-quality phases (element pruning,
beam search) could fix.

**The fitted correction is real but not reliable at the per-leg level, and the reason is
identifiable.** All three fitting attempts -- one global curve, curves split by departure kind, and
a smooth log-linear regression -- converge on similar bucket-level corrections (roughly 1.2x-2.1x,
increasing with ToF, consistent with the legality track's own single-leg measurement of ~2.5x at
ToF=1000 d). But the same two held-out legs fail every single attempt: both have a true
flown/estimated ratio within a couple of percent of 1.0 (1.04 and 1.00 respectively) -- i.e. they
needed essentially *no* correction -- while every fitted curve, keyed on ToF alone, predicts
1.3x-1.5x for legs in that ToF range and so over-corrects them by 28-56%. Both of these legs also
have a small `dv1_est` (246 and 306 m/s) relative to most of the fit sample. The pattern is
consistent with the correction factor depending on the *magnitude* of the required delta-v as well
as on ToF -- a gentle, near-ballistic correction has little of the re-aim-and-undo inefficiency the
shrinking-aim law accumulates on a leg that has to fight hard against the geometry -- which a
ToF-only fit, as Phase 6 was scoped to produce, structurally cannot express.

**This halts the track per the standing rules.** Three genuine, methodologically distinct fitting
attempts were made against one fixed held-out set; the best reached 3/5, not the required 5/5.
Extending the fit to include delta-v magnitude as a second input would very plausibly resolve the
two failures (the mechanism is identified, not mysterious), but that is a materially different
model than "correction as a function of ToF" and is out of Phase 6's stated scope, so it is not
attempted here -- doing so would not be a fourth *attempt* at the specified task, it would be
redefining the task after failing it. No correction was applied to `src/sequencer.py`; the oracle
is exactly as it was at the end of the legality track's Phase 4. Phases 7-10 (element-space
pruning, beam search, `v_inf` placement, the learned value function) were not started: proceeding
to build a beam search on top of a leg-cost oracle whose own calibration attempt failed its
accuracy bar would repeat the mistake the legality track's Phase 4 already made once (searching
against an oracle later found to be badly miscalibrated) with the added problem of using it inside
a wider, more expensive search this time.

#### Choices taken where the brief left it open

- **`kind='mid'` departure states use the target asteroid's exact position *and* velocity**, not a
  velocity carried over from an actual incoming flyby trajectory (a real flyby does not match
  velocity, so a genuine post-flyby state would retain most of the spacecraft's *incoming* velocity
  at the flyby target's position). This is flagged as a proxy in the code and here rather than
  fixed, because building a genuine post-flyby state generator would mean actually flying flyby
  legs first to harvest real terminal states -- circular with what this phase is trying to
  calibrate, and disproportionate for what is meant to be a broad, MVP sample of near-Earth
  orbital geometries rather than a literal mission history.
- **Held-out legs were evaluated against all three fits from one shared draw** rather than drawing
  a fresh held-out set per attempt. Taken deliberately: it is the only way to compare the three
  methodologies on equal footing, and the acceptance test's "for each" bar was checked against the
  same 5 legs every time, so no attempt got an easier test than another.
- **The three attempts were binned-combined, binned-by-kind, and log-linear regression, tried in
  that order.** Rejected alternative, considered but not attempted: a two-input fit (ToF and
  `dv1_est` magnitude) that the held-out failures directly suggest would work. Not attempted because
  it falls outside Phase 6's stated scope (a function of ToF), and the halt rule is meant to trigger
  on exactly this kind of boundary rather than be argued around by redefining the deliverable.

---

## Guidance diagnostic track

Continuation of the multi-flyby track, which halted at the end of Phase 6 after failing to
calibrate the leg-cost oracle -- but Phase 6's own sampling data buried a sharper finding: at
ToF <= 120 days, 0 of 360 candidate legs closed, using the oracle's own top-ranked targets flown
with the existing shrinking-aim guidance (`sequencer.fly_flyby_leg` / `_guided_step`). This matters
because MSU's winning GTOC4 entry averages one flyby every ~83 days over ten years -- squarely
inside the range that failed 0/360. Goal here: diagnose whether that is a delta-v shortage, a
guidance-law instability, or something else -- not fix it by default.

### Phase 1 — Trace one failing leg in full

#### What changed

`src/sequencer.py`: `fly_flyby_leg` gained an optional `trace=None` parameter. When given a list,
it appends one diagnostic dict per control step (elapsed/remaining leg time, thrust magnitude and
unit direction, the aim point, position and velocity error to it before the step, and mass) plus
one terminal dict recording why the leg ended (`lambert_fail`, `propellant_exhausted`, or
`completed`). Purely additive -- confirmed byte-identical output (same miss distance, same final
state) with and without `trace` on a real leg. `_guided_step` itself was not touched; it has no
`arrival`/aim-point context of its own, and `fly_flyby_leg` already holds everything the diagnostic
needs, so instrumenting there avoids touching `_guided_step`'s signature or body at all -- see
Choices below.

`experiments/guidance_trace1.py`: new. Duplicates (does not import) `acceptance_phase6.py`'s
sampling method -- same RNG seed (`np.random.default_rng(2)`), same bucket order, same rank bias
(50/30/20% toward rank 0/1/2) -- so the traced cases are the same *kind* of attempt Phase 6 already
measured failing 54.7% of the time at ToF <= 120 d, not a fresh, possibly unrepresentative resample.
It replays the RNG stream through the exhausted 60-day bucket (Phase 6: 0/120 usable, all 120
attempts consumed) and then scans the 90-day bucket for the first `fly_fail` and first `missed`
case, tracing both.

#### Acceptance-test output (verbatim)

```
$ ~/miniconda3/envs/tudat-space/bin/python experiments/guidance_trace1.py
--- replaying Phase 6's RNG stream up to the ToF=90 d bucket ---
replay complete; RNG stream now matches Phase 6's state at the start of this bucket

--- scanning for a fly_fail case at ToF=90 d ---
found: {'kind': 'mid', 'tof_days': 90, 'rank': 0, 'status': 'fly_fail', 'dv1_est': 760.739903908732, 'target_name': '4183'}

--- fly_fail case: final 10 trace entries ---
  {'elapsed_days': 78.0, 'time_remaining_days': 12.0, 'thrust_magnitude': 0.135, ... 'position_error_km': 42463270.53, 'velocity_error_ms': 3312.17, 'mass_kg': 1316.29}
  {'elapsed_days': 79.0, 'time_remaining_days': 11.0, 'thrust_magnitude': 0.135, ... 'position_error_km': 39207581.51, 'velocity_error_ms': 3596.38, 'mass_kg': 1315.90}
  {'elapsed_days': 80.0, 'time_remaining_days': 10.0, 'thrust_magnitude': 0.135, ... 'position_error_km': 35947640.51, 'velocity_error_ms': 3938.86, 'mass_kg': 1315.50}
  {'elapsed_days': 81.0, 'time_remaining_days':  9.0, 'thrust_magnitude': 0.135, ... 'position_error_km': 32684328.90, 'velocity_error_ms': 4359.10, 'mass_kg': 1315.10}
  {'elapsed_days': 82.0, 'time_remaining_days':  8.0, 'thrust_magnitude': 0.135, ... 'position_error_km': 29418532.47, 'velocity_error_ms': 4886.31, 'mass_kg': 1314.71}
  {'elapsed_days': 83.0, 'time_remaining_days':  7.0, 'thrust_magnitude': 0.135, ... 'position_error_km': 26151143.30, 'velocity_error_ms': 5566.40, 'mass_kg': 1314.31}
  {'elapsed_days': 84.0, 'time_remaining_days':  6.0, 'thrust_magnitude': 0.135, ... 'position_error_km': 22883064.52, 'velocity_error_ms': 6475.87, 'mass_kg': 1313.91}
  {'elapsed_days': 85.0, 'time_remaining_days':  5.0, 'thrust_magnitude': 0.135, ... 'position_error_km': 19615221.58, 'velocity_error_ms': 7752.41, 'mass_kg': 1313.52}
  {'elapsed_days': 86.0, 'time_remaining_days':  4.0, 'thrust_magnitude': 0.135, ... 'position_error_km': 16348588.75, 'velocity_error_ms': 9671.39, 'mass_kg': 1313.12}
  {'event': 'lambert_fail', 'elapsed_days': 87.0, 'time_remaining_days': 3.0}
  saved figures/guidance_trace_flyfail.png

--- scanning for a missed case at ToF=90 d ---
found: {'kind': 'mid', 'tof_days': 90, 'rank': 1, 'status': 'missed', 'dv1_est': 6612.269451785356, 'miss_km': 128120862.17215547, 'target_name': '2005OU2'}

--- missed case: final 10 trace entries ---
  {'elapsed_days': 81.0, 'time_remaining_days':  9.0, 'thrust_magnitude': 0.135, ... 'position_error_km':  93944503.07, 'velocity_error_ms':    259419.48, 'mass_kg': 1375.29}
  {'elapsed_days': 82.0, 'time_remaining_days':  8.0, 'thrust_magnitude': 0.135, ... 'position_error_km':  97847686.89, 'velocity_error_ms':    293772.30, 'mass_kg': 1374.89}
  {'elapsed_days': 83.0, 'time_remaining_days':  7.0, 'thrust_magnitude': 0.135, ... 'position_error_km': 101740213.92, 'velocity_error_ms':    337883.79, 'mass_kg': 1374.49}
  {'elapsed_days': 84.0, 'time_remaining_days':  6.0, 'thrust_magnitude': 0.135, ... 'position_error_km': 105613530.28, 'velocity_error_ms':    396603.02, 'mass_kg': 1374.10}
  {'elapsed_days': 85.0, 'time_remaining_days':  5.0, 'thrust_magnitude': 0.135, ... 'position_error_km': 109460460.49, 'velocity_error_ms':    478653.66, 'mass_kg': 1373.70}
  {'elapsed_days': 86.0, 'time_remaining_days':  4.0, 'thrust_magnitude': 0.135, ... 'position_error_km': 113274997.15, 'velocity_error_ms':    601473.25, 'mass_kg': 1373.30}
  {'elapsed_days': 87.0, 'time_remaining_days':  3.0, 'thrust_magnitude': 0.135, ... 'position_error_km': 117052125.24, 'velocity_error_ms':    805727.33, 'mass_kg': 1372.91}
  {'elapsed_days': 88.0, 'time_remaining_days':  2.0, 'thrust_magnitude': 0.135, ... 'position_error_km': 120787674.91, 'velocity_error_ms':   1213353.99, 'mass_kg': 1372.51}
  {'elapsed_days': 89.0, 'time_remaining_days':  1.0, 'thrust_magnitude': 0.135, ... 'position_error_km': 124478197.74, 'velocity_error_ms':   2433805.01, 'mass_kg': 1372.11}
  {'event': 'completed', 'elapsed_days': 90.0, 'position_error_km': 128120862.17, 'mass_kg': 1371.72}
  saved figures/guidance_trace_missed.png

Phase 1 trace collection complete
```

Both traces are 88-91 entries long (one per elapsed day, ToF=90 d); the excerpts above are the
final 10 as required. First-entry and mid-leg (day ~38-42) values, gathered separately to answer
"where in the leg does it fail":

```
fly_fail, day 0:  position_error_km=236641669.09  velocity_error_ms=760.74   thrust=0.135 N (saturated)
fly_fail, day 40: position_error_km=155507911.34  velocity_error_ms=1032.15  thrust=0.135 N (saturated)
missed,   day 0:  position_error_km=229512242.88  velocity_error_ms=6612.27  thrust=0.135 N (saturated)
missed,   day 40: position_error_km=124945194.85  velocity_error_ms=24346.48 thrust=0.135 N (saturated)
```

Figures: `figures/guidance_trace_flyfail.png`, `figures/guidance_trace_missed.png` -- thrust
magnitude, position error (log scale) and time-to-go, all vs. elapsed leg time.

#### Numbers

| quantity | fly_fail case (target 4183) | missed case (target 2005OU2) |
|---|---|---|
| ToF | 90 d | 90 d |
| rank offered | 0 (cheapest) | 1 |
| `dv1_est` | 760.7 m/s | 6612.3 m/s |
| thrust-limited budget over 90 d (`delta_v_budget`) | 789.7 m/s | 755.5 m/s |
| `dv1_est` vs. 0.6x budget (the oracle's own `leg_feasible` margin) | 760.7 > 473.8 -- **fails** | 6612.3 > 453.3 -- **fails** |
| thrust saturated (0.135 N) | steps 0-86 of 86 (100%) | steps 0-89 of 90 (100%) |
| position error: start -> minimum -> end | 236.6M km -> 16.3M km (day 86) -> n/a (Lambert failed) | 229.5M km -> 45.7M km (day ~64) -> 128.1M km (day 90) |
| outcome | `lambert_fail` at day 87 (3 days remaining) | `completed`, missed by 128.1M km |

#### Interpretation

**Both traced cases fail with thrust saturated at `thrust_max` from the very first control step to
the last -- not something that develops mid-leg.** There is no early period of gentle, unsaturated
correction followed by a late scramble; the very first day already commands 0.135 N, and every day
after it does too. Whatever limits these legs, it is present from day zero.

**The fly_fail case is genuine convergence that runs out of runway, not divergence.** Position error
falls smoothly and monotonically for the whole visible trace (log-scale plot: a clean descending
curve, no reversal) -- 236.6M km at day 0 down to 16.3M km at day 86, a 93% reduction. But the
*velocity* error the guidance keeps re-solving for climbs the entire time -- 760.7 m/s at day 0 to
9671 m/s at day 86 -- because a Lambert transfer that must close a still-large spatial gap in an
ever-shrinking time window requires ever-higher speed; this is a property of the transfer geometry,
not a control artefact. By day 86 the required correction so far exceeds what four more days of
0.135 N thrust from a partially-depleted 1313 kg spacecraft could deliver that the day-87 Lambert
solve cannot bracket a solution at all, and the leg is abandoned. The mechanism reads as: 90 days
does not give the myopic, one-day-lookahead law enough slack to bring position error close to zero
*well before* the end of the leg, and the terminal cost of any residual position error is highly
convex in the time left to close it.

**The missed case shows a different, sharper signature: convergence, then reversal.** Position error
decreases for about two-thirds of the leg (229.5M km -> ~45.7M km around day 64), then turns around
and *increases* for the remaining third, ending at 128.1M km -- worse than at the halfway point.
This is the plot's most visible feature (a clean V shape on the log-scale error trace) and is not
simple failure-to-converge-in-time; it is the trajectory passing its closest approach to the moving
aim point and then being unable to arrest the growing separation with the remaining thrust and time.
Velocity error here starts two orders of magnitude worse than the fly_fail case (6612 vs 761 m/s)
and finishes four orders of magnitude worse (2.43M m/s by day 89) -- consistent with a target that
was never realistically reachable at this ToF.

**Both traced cases would already have been rejected by the oracle's own feasibility margin.**
`dv1_est` for the fly_fail case (760.7 m/s) exceeds the 90-day thrust-limited budget's 0.6x
duty-cycle threshold (473.8 m/s) by 61%; for the missed case (6612.3 m/s) it exceeds it by more
than 14x. Neither is a candidate `greedy_tour` or a real beam search would ever attempt to fly --
both `leg_candidates()` calls here returned their result unfiltered, exactly as Phase 6's own
sampling did, and both traced cases happen to be candidates that fail `leg_feasible` even at
rank 0. This does not undermine the traces themselves (they are genuine, reproducible failures of
the guidance law under the parameters given), but it means Phase 1's two cases alone cannot say
whether the guidance also fails on candidates the oracle considers affordable -- that is exactly
what Phase 2's batch classification, with an oracle-feasibility cross-tab added for this reason,
needs to establish before drawing any conclusion about whether this blocks a real multi-flyby
search.

#### Choices taken where the brief left it open

- **Instrumentation lives in `fly_flyby_leg`, not `_guided_step`.** The brief names `_guided_step`
  specifically. `_guided_step` receives only `(state, epoch, step, correction)` -- it has no access
  to the aim point, the target, or the arrival epoch, all of which the diagnostic needs (position
  error, aim point). `fly_flyby_leg` already computes all of it every iteration and already calls
  `_guided_step` for the thrust; adding the trace append there records everything requested,
  including the thrust `_guided_step` returns, while leaving `_guided_step` completely unmodified
  -- strictly more conservative than threading a trace parameter through it, and satisfies "don't
  change control flow" trivially since nothing about `_guided_step` changed at all.
- **`replay_to_bucket` duplicates ~30 lines of `acceptance_phase6.py` rather than importing it.**
  The standing rule says not to touch anything from the halted Phase 6 calibration; importing and
  calling its functions would not modify that file, but duplicating keeps this track fully
  independent of it and removes any risk of an incidental behavioural coupling if that file is
  ever revisited. The duplicated functions (`_departure`, `_fly_one`) are reproduced verbatim
  except for the added `trace` passthrough.

### Phase 2 — Classify the failure mode across a sample

#### What changed

`experiments/guidance_trace2.py`: new. Two parts, run in one script:

1. Reproduces Phase 6's exact ToF=60/90/120 d samples (seed=2, MJD 58128, n=120 each -- the very
   data that produced the "54.7%" figure), and for every attempt records whether the candidate was
   **oracle-feasible** (`dv1 < 0.6x` the thrust-limited budget, `sequencer.leg_feasible`'s own
   margin) and whether it was **trivial** (`dv1 < 30 m/s`, Phase 6's own threshold for "covered by
   the launch `v_inf` credit, uninformative"), cross-tabbed against whether the leg actually closed.
   This was added after a first pass conflated "oracle-feasible" with "closes reliably": most of
   what looked like a clean 100%-feasible-succeeds / 0%-infeasible-succeeds split turned out to be
   driven almost entirely by trivial, free legs, which trivially succeed and say nothing about the
   guidance under real load.
2. A fresh, fully-traced, 30-attempt batch across two launch epochs (MJD 58128 and 58430, the
   legality track's own 8-flyby-chain epoch) and the three short ToF buckets, using
   `experiments/guidance_trace1.py`'s `_fly_one` (this track's own module) with tracing enabled,
   classifying every failure's primary proximate cause via the rule in `classify()`: `exhaustion`
   if the leg's terminal event is propellant depletion; else `oscillation` if position error
   reaches a minimum well before the leg ends and then grows back by more than 5%; else
   `saturation` if thrust sits at >=99% of `thrust_max` for more than 80% of the leg's steps; else
   `other`.

#### Acceptance-test output (verbatim)

```
$ ~/miniconda3/envs/tudat-space/bin/python experiments/guidance_trace2.py
--- part 1: Phase 6's exact seed=2 short-ToF samples, feasibility x triviality ---
   ToF     n     trivial+feasible    nontrivial+feasible    trivial+infeasible    nontrivial+infeasible
    60   120                54/54                    0/0                   0/0                     0/66
    90   120                49/49                    0/0                   0/0                     0/71
   120   120                60/60                    0/1                   0/0                     0/59

--- sampling 3 ToF buckets x 2 epochs x 5 attempts = 30 ---
  MJD 58128.0 ToF  60 d: 5 attempts collected in 5
  MJD 58128.0 ToF  90 d: 5 attempts collected in 5
  MJD 58128.0 ToF 120 d: 5 attempts collected in 5
  MJD 58430.0 ToF  60 d: 5 attempts collected in 5
  MJD 58430.0 ToF  90 d: 5 attempts collected in 5
  MJD 58430.0 ToF 120 d: 5 attempts collected in 5

--- part 2: fresh two-epoch batch, n=30 ---
outcome counts: {'fly_fail': 14, 'missed': 4, 'ok': 12}

feasibility x triviality (this batch, for comparison with part 1):
  feasible+trivial        : 11/11
  feasible+nontrivial     : 1/1
  infeasible+trivial      : 0/0
  infeasible+nontrivial   : 0/18

--- primary failure-mode classification, by ToF bucket ---
  ToF  60 d (n=8): saturation=5 (62%), oscillation=3 (38%)
  ToF  90 d (n=5): saturation=5 (100%)
  ToF 120 d (n=5): saturation=4 (80%), oscillation=1 (20%)

--- primary failure-mode classification, by outcome type ---
  fly_fail (n=14): saturation=14 (100%)
  missed (n=4): oscillation=4 (100%)

--- overall primary classification (all 18 failures) ---
  saturation  :  14 (77.8%)
  oscillation :   4 (22.2%)

--- representative traces per category actually observed ---
  saturation (14 total, showing up to 3):
    MJD 58128.0 ToF 60d fly_fail  dv1_est=   1737.1 feasible=False trivial=False n_steps= 57 frac_sat=1.0 pos_err(start->min->end km)=1.58e+08->1.32e+07->1.32e+07 terminal=lambert_fail secondary=[]
    MJD 58128.0 ToF 60d fly_fail  dv1_est=  10852.1 feasible=False trivial=False n_steps= 26 frac_sat=1.0 pos_err(start->min->end km)=1.13e+08->8.76e+07->8.76e+07 terminal=lambert_fail secondary=[]
    MJD 58128.0 ToF 90d fly_fail  dv1_est=   3758.2 feasible=False trivial=False n_steps= 79 frac_sat=1.0 pos_err(start->min->end km)=1.81e+08->4.76e+07->4.76e+07 terminal=lambert_fail secondary=[]
  oscillation (4 total, showing up to 3):
    MJD 58128.0 ToF 60d missed    dv1_est=   4341.4 feasible=False trivial=False n_steps= 60 frac_sat=1.0 pos_err(start->min->end km)=8.55e+07->8.67e+06->1.92e+07 terminal=completed secondary=['saturation']
    MJD 58128.0 ToF 60d missed    dv1_est=   4086.2 feasible=False trivial=False n_steps= 60 frac_sat=1.0 pos_err(start->min->end km)=1.98e+08->2.65e+07->3.93e+07 terminal=completed secondary=['saturation']
    MJD 58430.0 ToF 60d missed    dv1_est=   1801.2 feasible=False trivial=False n_steps= 60 frac_sat=1.0 pos_err(start->min->end km)=9.94e+07->4.42e+06->6.43e+06 terminal=completed secondary=['saturation']
  exhaustion: none observed
  other: none observed
```

#### Numbers

| quantity | value |
|---|---|
| Phase 6 exact-seed short-ToF sample, trivial+feasible success rate | 54/54, 49/49, 60/60 (100% at all three buckets) |
| Phase 6 exact-seed short-ToF sample, non-trivial+feasible attempts observed | 1 (out of 360), which failed |
| Phase 6 exact-seed short-ToF sample, non-trivial+infeasible success rate | 0/66, 0/71, 0/59 (0% at all three buckets) |
| fresh two-epoch batch, feasible+trivial | 11/11 (100%) |
| fresh two-epoch batch, feasible+nontrivial | 1/1 (succeeded) |
| fresh two-epoch batch, infeasible+nontrivial | 0/18 (0%) |
| failure classification: saturation | 14/18 (77.8%) -- 100% of all `fly_fail` outcomes |
| failure classification: oscillation | 4/18 (22.2%) -- 100% of all `missed` outcomes |
| failure classification: exhaustion, other | 0 observed of either |

#### Interpretation

**The dominant, best-supported finding is structural, not a guidance instability: at ToF <= 120
days, the oracle's top-3-ranked candidates are essentially bimodal.** Across 360 attempts
reproducing Phase 6's exact sample (n=120 per bucket at 60/90/120 d) plus 30 fresh attempts across
a second epoch, a sampled candidate is almost always either trivial (`dv1 < 30` m/s, covered
outright by the launch `v_inf` credit -- and when it is, it closes 100% of the time: 54/54, 49/49,
60/60, 11/11) or non-trivial *and* already beyond the oracle's own 0.6x-duty-cycle feasibility
margin -- and when it is, it fails 100% of the time (0/66, 0/71, 0/59, 0/18). A candidate that is
both non-trivial *and* within the oracle's own affordability margin -- the only kind of candidate
whose outcome would actually test the guidance law's reliability under realistic conditions --
appeared exactly **once** across 390 combined attempts (at ToF=120 d), and that one case still
failed. Phase 6's 54.7% short-ToF failure figure is overwhelmingly a statement about which
candidates the top-3-ranked, unfiltered sampling offers at short ToF, not a direct measurement of
guidance reliability -- and a real search (`greedy_tour`, a beam search) filters by `leg_feasible`
before ever attempting to fly a candidate, so it would never hand the guidance most of what failed
here in the first place.

**Among the failures that did occur, the proximate mechanism is clean and fully explained by Phase
1's two traced cases, with no ambiguity.** Every `fly_fail` (14/14) is `saturation`: thrust pinned
at `thrust_max` for the whole leg, position error converging but too slowly to reach zero before
the shrinking time-to-go drives the required terminal velocity beyond what remains achievable,
ending in a Lambert bracket failure -- exactly Phase 1's target-4183 case. Every `missed` (4/4) is
`oscillation`: position error reaches a minimum partway through the leg and then grows back by
more than 5% before the leg ends -- exactly Phase 1's target-2005OU2 case, generalised. No
`exhaustion` or `other` case was observed in either the exact-seed reproduction or the fresh batch.
This satisfies Phase 2's acceptance test cleanly (a clear majority in an interpretable split, by
both ToF bucket and outcome type) -- the halt condition for an inconclusive classification does not
apply.

**What this phase does *not* establish, and Phase 3 is now aimed squarely at:** whether the
guidance law is reliable on genuinely affordable, non-trivial short-ToF candidates, since the
sampling method used throughout this diagnostic (mirroring Phase 6's, by design, for comparability)
essentially never produces one to test. A supplementary check at ToF=180-450 d (same method,
n=120 per bucket, not yet part of any phase's formal acceptance test) found the "non-trivial and
feasible" population becoming non-negligible from 180 d onward (14/120 at 180 d, 37/120 at 300 d,
58/120 at 450 d) but *not* closing reliably even once it exists: 5/14 (36%) at 180 d, 24/37 (65%)
at 300 d, 40/58 (69%) at 450 d. Unlike the short-ToF result, these numbers are a genuine, if partial,
signal that the guidance itself is imperfect even on affordable candidates -- just measured on too
small and informal a sample to report as a finding of this phase. Phase 3 repeats this properly,
with the acceptance test's own sampling method and reporting.

#### Choices taken where the brief left it open

- **The oracle-feasibility x triviality cross-tab was added beyond what Phase 2 asked for.**
  Rejected alternative: report only the four-way saturation/oscillation/exhaustion/other split, as
  literally specified. Added because Phase 1 already flagged both traced cases as oracle-infeasible,
  and a first draft of this phase's cross-tab (oracle-feasible vs. not, without excluding trivial
  legs) produced a misleadingly clean 100%/0% split that dissolved under a closer look -- reporting
  only the failure-mode split without this context would have let the "guidance is unstable"
  framing stand uncorrected when the data says something more specific and more interesting.
- **`classify()`'s checking order (exhaustion, then oscillation, then saturation, then other) and
  thresholds (80% of steps for saturation, 5% growth off the minimum for oscillation) are fixed,
  simple, and not tuned against the sample.** Chosen before running the batch, per the brief's own
  suggested numbers (">80%... pick a threshold... say so").

### Phase 3 — Check whether the failure threshold moves with ToF

#### What changed

`experiments/guidance_trace3.py`: new. Sweeps ToF in {150, 180, 210, 250, 300} days, 15 legs each,
MJD 58128, same candidate-sampling method as Phases 1-2 (real `leg_candidates()` call, rank drawn
50/30/20% toward rank 0/1/2). Reports both the raw (fly_fail + missed)/attempts rate -- Phase 6's
own metric, for direct comparability -- and the rate restricted to non-trivial, oracle-feasible
candidates (`dv1 >= 30` m/s and `dv1 < 0.6x` budget), the more meaningful number given Phase 2's
finding. A follow-up (same method, not part of the formal 5-point sweep) additionally sampled
n=15 at ToF in {600, 900, 1200} days, because Phase 6's own log -- read closely rather than taken
on the "near-0% at 300d+" summary the track brief opened with -- does not actually show near-zero
failure at long ToF (see Numbers).

#### Acceptance-test output (verbatim)

```
$ ~/miniconda3/envs/tudat-space/bin/python experiments/guidance_trace3.py
--- ToF sweep: (150, 180, 210, 250, 300) d, 15 legs each, MJD 58128.0 ---
  ToF  150 d (n=15, 15 attempts): raw failure rate =  46.7%  |  trivial= 7  nontrivial+feasible= 1 (failure   0.0%)  nontrivial+infeasible= 7
  ToF  180 d (n=15, 15 attempts): raw failure rate =  33.3%  |  trivial= 9  nontrivial+feasible= 1 (failure   0.0%)  nontrivial+infeasible= 5
  ToF  210 d (n=15, 15 attempts): raw failure rate =  60.0%  |  trivial= 4  nontrivial+feasible= 3 (failure  33.3%)  nontrivial+infeasible= 8
  ToF  250 d (n=15, 15 attempts): raw failure rate =  53.3%  |  trivial= 6  nontrivial+feasible= 2 (failure  50.0%)  nontrivial+infeasible= 7
  ToF  300 d (n=15, 15 attempts): raw failure rate =  20.0%  |  trivial= 8  nontrivial+feasible= 5 (failure  20.0%)  nontrivial+infeasible= 2

--- raw (Phase-6-style) failure rate by ToF ---
  ToF  150 d:  46.7%
  ToF  180 d:  33.3%
  ToF  210 d:  60.0%
  ToF  250 d:  53.3%
  ToF  300 d:  20.0%

raw failure rate first drops below 50% at ToF=150
raw failure rate does not drop below 10% within the sampled range

--- non-trivial, oracle-feasible-only failure rate by ToF ---
  ToF  150 d (n=1):   0.0%
  ToF  180 d (n=1):   0.0%
  ToF  210 d (n=3):  33.3%
  ToF  250 d (n=2):  50.0%
  ToF  300 d (n=5):  20.0%
```

Follow-up (long ToF, same method, seed=5, n=15 each):

```
ToF  600d (n=15, 15 attempts): raw failure=26.7%  trivial=0  nontrivial+feasible=14 (failure=21.4%)
ToF  900d (n=15, 15 attempts): raw failure=80.0%  trivial=0  nontrivial+feasible=15 (failure=80.0%)
ToF 1200d (n=15, 15 attempts): raw failure=40.0%  trivial=0  nontrivial+feasible=15 (failure=40.0%)
```

Phase 6's own log, re-read as a raw (fail/attempts) rate rather than its "N usable" framing (both
are the same underlying data, Phase 6's `experiments/acceptance_phase6.py` output, quoted in this
file's Phase 6 section above -- recomputed here only to express it as a percentage):

| ToF | Phase 6 attempts | Phase 6 fail (fly_fail + missed) | raw failure rate |
|---|---|---|---|
| 180 d | 65 | 25 | 38.5% |
| 300 d | 21 | 4 | 19.0% |
| 450 d | 13 | 3 | 23.1% |
| 600 d | 8 | 2 | 25.0% |
| 900 d | 12 | 6 | 50.0% |
| 1200 d | 11 | 5 | 45.5% |

#### Numbers

| quantity | value |
|---|---|
| raw failure rate range, 150-300 d sweep | 20.0% - 60.0%, non-monotonic |
| raw failure rate crosses below 50% | at ToF=150 d (46.7%) -- but rises back above 50% at 210 d (60.0%) |
| raw failure rate drops below 10% | **not within the 150-300 d sampled range** |
| raw failure rate at ToF=600/900/1200 d (this phase's own resample) | 26.7% / **80.0%** / 40.0% |
| raw failure rate at ToF=600/900/1200 d (Phase 6's own log, independent data) | 25.0% / 50.0% / 45.5% |
| non-trivial+feasible population at ToF<=180 d (this sweep) | 1 of 15 per bucket |
| non-trivial+feasible population at ToF=600-1200 d | 14-15 of 15 per bucket (essentially all of it) |
| non-trivial+feasible failure rate at ToF=600/900/1200 d | 21.4% / 80.0% / 40.0% |

#### Interpretation

**The premise that motivated this phase -- "near-0% failure at ToF>=300 d" -- does not hold up
against Phase 6's own recorded data, and this phase's independent resample confirms it does not.**
Phase 6's log, read as a plain (fail/attempts) percentage rather than through its "N usable out of
M attempts" framing, shows 19-50% raw failure at every ToF from 180 d to 1200 d, with **no**
ToF at or above 10%, let alone near 0%. This phase's own fresh samples at 600/900/1200 d (a
different seed, same method) land in the same range -- 26.7%/80.0%/40.0% -- confirming this is not
an artefact of Phase 6's particular draw. Long legs are not a reliably solved regime for this
guidance law; if anything, 900 d is the single worst point measured in either dataset.

**The 150-300 d sweep itself is too noisy at n=15 per bucket to support a transition-point claim,
and is reported as such rather than smoothed into one.** Raw failure rate goes 46.7% -> 33.3% ->
60.0% -> 53.3% -> 20.0% across 150/180/210/250/300 d -- not monotonic, and the "crosses below 50%"
figure required by the acceptance test (ToF=150 d) is immediately contradicted by a rise back above
60% at 210 d two points later. The non-trivial+feasible population within this range is tiny (1-5
per bucket of 15), too small for its failure-rate column to mean much on its own.

**What the data does support cleanly: the *population* of non-trivial, oracle-feasible candidates
grows with ToF, but the *guidance's reliability on that population*, once it exists in enough
numbers to measure, does not obviously improve with ToF.** At ToF<=180 d there is only ever about
one such candidate per 15 attempts (matching Phase 2's near-total-absence finding); by 600-1200 d,
essentially the whole sample (14-15 of 15) is non-trivial and feasible, because there is no longer
a free `v_inf`-covered option and the larger budget makes most top-ranked candidates affordable --
but the failure rate measured on exactly that population is 21-80%, with no visible trend toward
zero as ToF grows to 1200 days, the longest a flyby leg can legally be.

**This changes what Phase 4 should be gated on.** The track brief's working hypothesis was
"excessive gain or too-aggressive re-aiming *early in a short leg*" -- implying a mechanism that
should weaken or disappear as ToF grows and the guidance has more slack. The data does not show
that: the same saturation/oscillation failure modes Phase 2 characterised at 60-120 d are not
confined to short legs, and 900 d -- about as much slack as a single flyby leg can have short of
the full 10-year mission limit -- is measured worse than every ToF sampled below it. A fix
motivated by "give short legs a gentler early ramp, scaled to how little time they have" would be
solving a ToF-dependence the numbers do not show. Phase 4 is reconsidered below in light of this,
rather than proceeding on the original short-leg-specific framing.

#### Choices taken where the brief left it open

- **The long-ToF (600/900/1200 d) follow-up goes beyond the literal 150-300 d sweep the phase
  specifies.** Taken because Phase 6's own log, read plainly, already contradicted the "near-0% at
  300d+" premise before this phase's own sweep even ran; confirming or refuting that with a direct,
  independent resample at the ToF values in question was necessary before Phase 4's gate (which
  explicitly turns on whether the mechanism is short-leg-specific) could be evaluated honestly.
  Rejected alternative: report only the specified 150-300 d sweep and let Phase 4 proceed on the
  original short-leg framing regardless -- rejected because that framing was demonstrably false
  before this phase started, and building Phase 4 on it anyway would repeat the mistake this whole
  diagnostic track exists to avoid.

### Phase 4 — Propose a fix, and test it narrowly (gated)

#### Gate decision

Phase 3 found the saturation/oscillation failure modes present, and not obviously improving, from
60 through 1200 days -- contradicting the track brief's opening hypothesis of a mechanism confined
to "too-aggressive re-aiming early in a short leg." Rather than treat that as disqualifying
(halt condition 4's "inherently a control-authority limit... do not attempt a fix"), the traced
data itself pointed at a specific, ToF-*independent* candidate mechanism, checked before deciding
whether to proceed: `_guided_step` sizes the thrust needed to deliver the current correction against
`step` -- the physical integration step, fixed at one day except on a leg's last day -- never
against how much time is actually left. Phase 1's fly_fail case needed 760.7 m/s on day 0 of a
90-day leg: 88x what one day of `thrust_max` can deliver, but only 96% of what the *full* 90 days
can deliver. This is a small, local, well-motivated candidate fix squarely inside Phase 4's scope
(a gain/horizon change to `_guided_step`, not a redesign), so it was implemented and tested per the
gate's instructions, rather than stopped on before trying.

#### What changed (implemented, tested, then reverted -- see Numbers)

`src/sequencer.py`: `_guided_step` gained an optional `horizon` parameter sizing the commanded
thrust against a caller-supplied time base instead of `step`; `fly_flyby_leg` called it with
`horizon=arrival - epoch` (the leg's remaining time), spreading each day's correction over every
day actually left rather than demanding it all within the next 24 hours. Confirmed structurally
isolated: `fly_rendezvous_leg`'s call to `_guided_step` was untouched (no `horizon` argument), and
`horizon=None` reproduces the prior formula exactly. **After testing (below), this was reverted in
full** -- `_guided_step` and `fly_flyby_leg` are now byte-for-byte identical to their state at the
end of the legality track's Phase 4 (`git diff 679f930 -- src/sequencer.py` shows only a docstring
addition). The attempt is recorded here rather than kept in the code.

#### Acceptance-test output (verbatim)

Phase 2's exact sample (same seeds, same candidates, same two epochs) re-run with the fix active:

```
$ ~/miniconda3/envs/tudat-space/bin/python experiments/guidance_trace2.py   # with the fix applied
--- part 1: Phase 6's exact seed=2 short-ToF samples, feasibility x triviality ---
   ToF     n     trivial+feasible    nontrivial+feasible    trivial+infeasible    nontrivial+infeasible
    60   120                54/54                    0/0                   0/0                     0/66
    90   120                49/49                    0/0                   0/0                     0/71
   120   120                60/60                    0/1                   0/0                     0/59

--- part 2: fresh two-epoch batch, n=30 ---
outcome counts: {'fly_fail': 14, 'missed': 5, 'ok': 11}

feasibility x triviality (this batch, for comparison with part 1):
  feasible+trivial        : 11/11
  feasible+nontrivial     : 0/1
  infeasible+trivial      : 0/0
  infeasible+nontrivial   : 0/18

--- overall primary classification (all 19 failures) ---
  saturation  :  14 (73.7%)
  oscillation :   4 (21.1%)
  other       :   1 (5.3%)
```

Phase 3's exact sweep (same seed, same launch epoch) re-run with the fix active:

```
$ ~/miniconda3/envs/tudat-space/bin/python experiments/guidance_trace3.py   # with the fix applied
--- ToF sweep: (150, 180, 210, 250, 300) d, 15 legs each, MJD 58128.0 ---
  ToF  150 d (n=15): raw failure rate =  53.3%  |  nontrivial+feasible= 1 (failure 100.0%)
  ToF  180 d (n=15): raw failure rate =  40.0%  |  nontrivial+feasible= 1 (failure 100.0%)
  ToF  210 d (n=15): raw failure rate =  73.3%  |  nontrivial+feasible= 3 (failure 100.0%)
  ToF  250 d (n=15): raw failure rate =  60.0%  |  nontrivial+feasible= 2 (failure 100.0%)
  ToF  300 d (n=15): raw failure rate =  46.7%  |  nontrivial+feasible= 5 (failure 100.0%)

raw failure rate first drops below 50% at ToF=180
raw failure rate does not drop below 10% within the sampled range
```

#### Numbers

| ToF | raw failure rate, before | raw failure rate, after fix | nontrivial+feasible failure rate, before | after fix |
|---|---|---|---|---|
| 60-120 d (Phase 2 batch, n=30 combined) | 18/30 = 60.0% | 19/30 = 63.3% | 0/1 = 0% (n too small) | 0/1 = 0% (same case, unchanged) |
| 150 d | 46.7% | 53.3% | 0/1 = 0% | 1/1 = **100%** |
| 180 d | 33.3% | 40.0% | 0/1 = 0% | 1/1 = **100%** |
| 210 d | 60.0% | 73.3% | 2/3 = 66.7% | 3/3 = **100%** |
| 250 d | 53.3% | 60.0% | 1/2 = 50.0% | 2/2 = **100%** |
| 300 d | 20.0% | 46.7% | 1/5 = 20.0% | 5/5 = **100%** |

**The fix makes the raw failure rate worse at every one of the five 150-300 d buckets, with no
exception, and the non-trivial-feasible failure rate goes from a mixed 0-67% to a uniform 100% at
every bucket where a comparison is possible.** The two large (n=120x3, n=30) short-ToF samples show
no measurable change either way, because -- as Phase 2 established -- they contain almost no
genuinely-affordable, non-trivial candidates for the fix to help or hurt; the one that exists (at
120 d) fails both before and after. The Phase 2 batch's own raw rate ticked up slightly too
(60.0% -> 63.3%), driven by the single regression visible in the acceptance-test output above
(`ok`: 12 -> 11, `missed`: 4 -> 5).

#### Interpretation

**The diagnosis was right about the mechanism; the remedy was wrong, and the data says so
unambiguously.** Phase 1-3 correctly identified that the old sizing formula demands the full
correction within one day regardless of how much time remains, and that this is what drives
saturation. The fix removes exactly that: an unsaturated correction is now spread evenly over every
day left in the leg. It does not help. It hurts, consistently, everywhere it could be measured.

**Why: this is a receding/moving-target intercept, not a resource-budgeting problem, and early
correction has leverage that a later, evenly-paced correction cannot recover.** The old law snaps
onto the exact ballistic arc that would hit the target *today's estimate of where it will be*
whenever that snap is affordable in one day -- and when it is, the spacecraft is, from that point,
exactly on the trajectory the physics says it needs, needing no further correction until the next
day's fresh Lambert solve (constant target position for a flyby leg) finds a new one, typically
small, from natural integration drift. Spreading the same total correction over many days instead
means the spacecraft spends most of the leg *not* on that trajectory -- deliberately under-corrected
relative to what the current geometry calls for -- so the *natural* two-body relative motion between
the partially-corrected spacecraft and the target is free to diverge for longer before the guidance
catches up. Lambert's own required-velocity term is highly convex in time-to-go (Phase 1's plot:
velocity error climbed from 760.7 to 9671 m/s as time-to-go fell from 90 to 4 days, even while
position error was still *shrinking*): delaying correction defers cost into exactly the part of the
leg where it compounds fastest. The 0.6x duty-cycle margin in `leg_feasible` was already meant to
buy slack for guidance imperfection; Phase 1's traced case needed 96% of the full, un-derated
90-day budget, leaving essentially none -- no reallocation of *when* thrust is spent changes that a
leg with 96% of its total budget already spoken for by geometry has almost no room for any
guidance law, however it paces itself, to be imperfect at all.

**This does not fit Phase 4's positive path (a narrow tuning fix that helps), and the disciplined
outcome is the one the halt condition anticipates for that case: report the result, keep nothing.**
The fix is reverted. The guidance law is byte-identical to its state at the end of the legality
track's Phase 4. No behavioural change survives this phase. The negative result -- a specific,
plausible, testable idea, tried and found to make things worse everywhere measured, with a
physical explanation for why -- is itself the deliverable, exactly as the brief anticipated for a
diagnosis that "does not point at something this narrow."

#### Choices taken where the brief left it open

- **The fix was fully implemented and tested (not just proposed and stopped on) before being
  reverted.** The brief's gate technically only requires stopping *if* the diagnosis doesn't point
  at something narrow; here it did, so the fix was built and tested per the "propose... implement
  only that... acceptance test: re-run..." instructions, and the *result* of that test -- not a
  prior judgement about scope -- is what says not to keep it. This is different from, and stronger
  evidence than, declining to try in the first place would have been.
- **Reverted to a byte-identical `_guided_step`/`fly_flyby_leg` rather than leaving the `horizon`
  parameter in place unused.** Rejected alternative: keep the parameter, default it to `None`
  (preserving behaviour), and simply never call it with a non-default value. Rejected because an
  unused, tested-and-found-harmful parameter left in the function signature is exactly the kind of
  thing "minimal diffs" and "MVP quality" argue against -- the finding belongs in the notes, not as
  dead code inviting a future caller to re-enable it without the context of why it was tried and
  abandoned.
