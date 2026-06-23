# Jerk-Limited Motion (standalone)

Status: **implemented, EXPERIMENTAL (unit-tested, not yet hardware-validated).**
OFF by default; when disabled the motion pipeline is byte-for-byte stock. This
branch (`claude/great-pascal-auwn5x`) is cut from `main` and is **independent of
the torque-curve / TOPP-RA work** (`topp-ra-v2`). Jerk limiting and the torque
curve are *siblings*, not parent/child — see "Architecture" below.

## Implementation map

- `klippy/extras/jerk_limiting.py` -- the `[jerk_limiting]` config section, the
  pure trajectory math (`ramp_time`, `dist_jerk`, `reach_v2`, `ramp_slices`,
  `plan_segments`, `distribute_slices`, cubic B-spline `corner_blend`), and the
  three phase policies (`plan_moves`, `round_corner`).
- `klippy/toolhead.py` -- gated hooks: jerk reachability in `Move.calc_junction`
  and `LookAheadQueue.flush`; the slice emission loop in `_process_moves`; the
  Phase 3 corner pre-pass in `LookAheadQueue.add_move`. All no-ops when the
  `[jerk_limiting]` section is absent or `enabled: False`.
- `klippy/kinematics/extruder.py` -- `move_segment`, per-slice extruder trapq
  emission keeping pressure advance synced (written for `main`'s extruder ABI;
  topp-ra's could not be cherry-picked because the ABIs differ).
- `test/test_jerk_limiting.py` -- unit tests incl. an end-to-end check that the
  real C trapq accepts emitted slices and yields continuous motion.
- `config/sample-jerk-limiting.cfg` -- documented sample config.

What is NOT yet validated: behavior on real hardware, and Phase 2/3 interactions
with the full feature set (arcs, bed mesh, pressure advance) under load. Treat
as experimental.

## Motivation

Kalico/Klipper generates motion as **piecewise-constant acceleration**
trapezoids, so acceleration changes *instantaneously* (infinite jerk) at several
points. Input shaping does **not** fix this: convolution is linear and cancels
only the *tuned* resonance(s); an instantaneous accel change is a step in force
with broadband spectrum that still excites un-shaped secondary modes, audible
noise, and demands an instantaneous change in motor torque the motor cannot
physically deliver.

Removing those instantaneous acceleration changes is valuable on its own, for
*any* machine running stock constant `max_accel` -- no torque curve required.

## Architecture: a shared motion core, two sibling policies

The hard, reusable piece is **emitting one planned move as many constant-accel
sub-segments while keeping the extruder / pressure-advance synced**. That core
is policy-agnostic:

```
        multi-segment move emission + PA sync   <- shared core
           /                              \
   TOPP-RA (a_max(v) slices)      Jerk limiting (S-curve slices)
```

- The **torque curve** answers "how hard can I push at this speed?" (`a_max(v)`).
- **Jerk limiting** answers "how fast can I change how hard I push?" (bound on
  `da/dt`).

Both are just policies deciding where slice boundaries fall and what accel each
slice gets. Neither needs the other.

### Getting the core onto `main`

A reference implementation of the multi-segment emitter + PA sync already exists
on `topp-ra-v2` (`klippy/toolhead.py` `_process_moves` slice loop;
`klippy/kinematics/extruder.py` `move_segment`). Two paths:

1. **Cherry-pick the curve-agnostic core** -- lift the slice-emission loop and
   `move_segment` PA sync, drop everything curve-specific (`reach`, `plan_move`'s
   `a_max(v)` lookup, the CSV loader). Fastest; the PA-over-segments sync is the
   experimental part already debugged there.
2. **Rebuild fresh on `main`** -- cleaner history, but re-derives the trickiest
   bit (PA integration across segments).

Recommendation: option 1, then re-home the slice planner under jerk-limit policy.

## Where acceleration jumps today (`main`)

1. **Within a ramp** -- stock emits one accel for the whole accel phase; with a
   slice emitter the within-ramp profile is whatever the policy chooses. Jerk
   policy makes `a` ramp smoothly instead of stepping `0 -> a_max`.

2. **Ramp <-> cruise, and launch from rest** -- `a` steps `a_max -> 0`,
   `0 -> -a_max`, and `0 -> a_max` at the start. These are the real
   infinite-jerk events. The launch step is the "missing bottom of the S-curve."

3. **Move-to-move junction, along-path** -- velocity *magnitude* is already
   continuous (`flush` sets `next_end_v2 = start_v2`), but `a` can flip sign
   across the boundary (move A decelerating, move B accelerating).

4. **Move-to-move junction, geometric (the corner)** -- the velocity *vector*
   rotates instantaneously. `calc_junction` bounds centripetal accel *magnitude*
   via junction deviation but applies it as an instantaneous speed clamp; the
   direction change still happens in zero time.

Classes 1-3 are **retiming** problems; class 4 is a **geometry** problem.

## Phase 1 -- round the ramp ends (cheap, local)

- Introduce a jerk bound `J_max` (mm/s^3). Config it directly, or derive from
  motor/driver bandwidth (`d(tau)/dt`).
- In the slice planner, where `a` would step to/from `0` (ramp<->cruise) and at
  launch, insert short rounding slices that ramp `a` linearly over `T_j = a/J_max`
  (a few constant-accel sub-slices). Result: a genuine **two-sided** S-curve on
  every ramp, including the launch-from-rest bottom that constant accel cannot
  round.
- **Gotcha:** rounding consumes extra distance, so the lookahead's reachability
  (`calc_junction`, `flush`) must account for the rounding overhead or short /
  triangle moves over-run and the planner becomes optimistic.
- **Free bonus:** the slice emitter mirrors every slice to the extruder, so
  rounding slices are pressure-advance-synced automatically.

## Phase 2 -- blend across junctions, along-path (moderate, spans two moves)

- Class 3 straddles the A->B boundary, so it can't live inside a single move's
  planner. Either pass neighbor accel context into the planner (the `flush` loop
  has the whole queue) or add a post-pass over the concatenated slice chains in
  `_process_moves` that splices blend slices where A's final `a` and B's initial
  `a` differ, ramping within `J_max`.
- Velocity is already continuous, so this is a pure accel-rate fixup.
- Subsumes the crude `smooth_delta_v2 = 2*move_d*max_accel_to_decel` heuristic
  (i.e. `minimum_cruise_ratio`).

## Phase 3 -- corner rounding / geometry (hard, architectural; fully standalone)

Class 4 is the only one retiming cannot touch, and it is the *most* independent
of acceleration policy: it changes the **path geometry** (a smoother polyline),
which any timer (stock trapezoid, S-curve, or TOPP-RA) then plays. It could even
run as a pre-pass before the planner.

To bound centripetal *jerk* the velocity vector must rotate over finite time,
i.e. synthesize a smooth blend curve through each corner, inside the
junction-deviation tolerance band, with bounded curvature and curvature-rate.

### Continuity hierarchy (centripetal accel = kappa * v^2)

- **G0** (sharp corner): tangent discontinuous -> accel impulse.
- **G1** (arc blend): tangent continuous, curvature jumps `0 -> 1/R` -> **jerk
  impulse** at arc ends.
- **G2** (clothoid / Euler spiral): curvature continuous (ramps linearly in arc
  length) -> **finite, bounded jerk**. Practical target.
- **G3+** (splines / elastica): curvature-rate continuous -> smooth jerk.

### Discrete differential geometry: what applies

The path *is* a polyline; discrete curvature at a vertex is the concentrated
turning angle (a Dirac). Corner smoothing = spreading that Dirac over a finite
arc inside the tolerance band. In rough order of practicality:

1. **Cubic B-spline / 4-point subdivision (recommended).** Treat the move
   polyline as a control polygon; locally subdivide near corners. Cubic
   B-spline limit is **C2** (no jerk impulse), cheap, local, bounded support;
   output is a refined polyline that drops into the slice emitter. Deviation is
   bounded by control-polygon distance (maps onto `junction_deviation`).
   *Prototype + validation: `scripts/jerk_planning/bspline_corner.py`* -- in the
   practical deviation band (~0.5-2 mm) it lands within a few percent of the
   clothoid (minimum-peak-jerk) optimum.
2. **Clothoid (Euler-spiral) blends.** Canonical CNC answer; G2, constant jerk,
   explicit deviation/curvature bounds. Use for tight-tolerance / high-speed
   corners where subdivision's gap to optimum widens.
3. **Quintic / cubic Bezier transitions.** Closed-form deviation bounds; G2/G3.
4. **Minimum Variation Curves (Moreton-Sequin) / discrete elastica.** The
   variational DDG objects (MVC minimizes integral (dkappa/ds)^2 = jerk content).
   Iterative -> use as an offline reference, not the realtime path.
5. **kappa-Curves (Yan et al. 2017).** Local, G2; middle ground.
6. **Discrete curvature flow / Taubin fairing.** *Not* suitable (global, no hard
   tolerance).

DDG's contribution is the discrete-curvature foundation (turning angle =
concentrated curvature) and the variational objectives that define an *optimal*
low-jerk blend; shipping code uses a **local** scheme (B-spline subdivision)
validated against an MVC/elastica reference.

Note: holonomy is *not* a useful tool here -- the XY plane is flat and the
toolhead is holonomic, so the tangent's rotation is just integrated curvature
(`int kappa ds = turn angle`), not connection holonomy. Holonomy / SO(3)
geometry only earns its keep for **orientation** smoothing on multi-axis
machines, not XY corner rounding.

### New architectural cost specific to curved blends

The slice emitter currently assumes one direction per move (shared `axes_r`). A
curved blend gives **each micro-segment its own direction**, so the emission loop
must pass a *per-segment* `axes_r` to the trapq (and extruder). The trapq already
takes `axes_r` per call, so this is bounded but real. Arc-length
re-parameterization and re-timing over variable curvature is the genuine
complexity of Phase 3.

## Sequencing

- **Establish the shared core on `main`** (cherry-pick from topp-ra-v2).
- **Phase 1** -- highest value-per-effort, self-contained. Do first.
- **Phase 2** -- natural follow-on in the emission loop.
- **Phase 3** -- research-grade; input shaping already notches the resonance the
  corner impulse excites and `junction_deviation` bounds corner accel magnitude,
  so marginal return vs 1-2. If pursued: cubic B-spline subdivision, validated
  against an offline MVC/elastica reference.

## Supporting analysis

- `scripts/jerk_planning/plot_shaper_max_accel.py` -- why input shaping caps
  accel (`a_max ~ f^2`) and cannot remove jerk.
- `scripts/jerk_planning/bspline_corner.py` -- corner-subdivision prototype with
  curvature, jerk proxy, and clothoid frontier.
