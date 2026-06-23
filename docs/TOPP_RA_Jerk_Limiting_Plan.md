# Jerk-Limited Motion on top of the TOPP-RA Torque-Curve Work

Status: **design / plan only** (no implementation yet). Target branch for the
torque-curve work is `topp-ra-v2`; this plan is staged on
`claude/great-pascal-auwn5x`.

## Motivation

Kalico/Klipper generates motion as **piecewise-constant acceleration**
trapezoids. Acceleration therefore changes *instantaneously* (infinite jerk) at
several places. Input shaping does **not** remove this: convolution is linear
and cancels only the *tuned* resonance(s); an instantaneous accel change is a
step in force with broadband spectrum that still excites un-shaped secondary
modes, audible noise, and — critically for the torque-curve work — demands an
instantaneous change in motor torque the motor cannot physically deliver.

A torque-speed curve is only honorable if we never demand a step change in
torque. Smoothing the acceleration transitions is what makes `a_max(v)` real on
hardware, not just in the lookahead. The TOPP-RA slice emitter
(`klippy/extras/torque_curve.py: plan_move`) already approximates a smooth
`a(v)` *along each ramp* by subdivision; this plan extends that to remove the
remaining instantaneous acceleration changes.

## Where acceleration jumps today (code sites on `topp-ra-v2`)

1. **Within a ramp, slice-to-slice** — `plan_move` (`torque_curve.py:346`). Each
   slice uses `a = a_max(v)`; adjacent slices step by
   `|a_max(v_i) - a_max(v_{i+1})|`. Already nearly jerk-limited; shrink
   `dv_slice` and it converges to smooth `a(v)`. **Essentially solved.**

2. **Ramp <-> cruise, and launch from rest** — `a` steps `a_max -> 0`,
   `0 -> -a_max`, and `0 -> a_max(0)` at the start. These are the real
   infinite-jerk events. The launch step is the "missing bottom of the S-curve."

3. **Move-to-move junction, along-path** — velocity *magnitude* is already
   continuous (`flush` sets `next_end_v2 = start_v2`, `toolhead.py:240`), but `a`
   can flip sign across the boundary (A decelerating, B accelerating).

4. **Move-to-move junction, geometric (the corner)** — the velocity *vector*
   rotates instantaneously. `calc_junction` (`toolhead.py:101-124`) bounds the
   centripetal accel *magnitude* via junction deviation, but applies it as an
   instantaneous speed clamp; the direction change still happens in zero time.

Key insight: **classes 1-3 are retiming problems** (fixable in the emitter);
**class 4 is a geometry problem** (needs a smooth blend curve).

## Phase 1 — round the ramp ends (cheap, local to `torque_curve.py`)

- Introduce a jerk bound `J_max` (mm/s^3). Config it, or derive it from the
  torque curve's `d(tau)/dt` / driver bandwidth — the physical limit on how fast
  the motor can change force.
- In `plan_move`, where `a` would step to/from `0` (ramp<->cruise) and at launch,
  insert short rounding slices that ramp `a` linearly over `T_j = a / J_max`,
  approximated as a few constant-accel sub-slices (reuse the existing
  subdivision). Result: a genuine **two-sided** S-curve on every ramp, including
  the launch-from-rest bottom that pure `a_max(v)` cannot round.
- **Gotcha:** rounding consumes extra distance, so `_dist_up` (`:313`),
  `_peak_velocity` (`:329`), and `reach` (`:293`) must all account for the
  rounding overhead, or short/triangle moves over-run and the lookahead becomes
  optimistic (`calc_junction` and `flush` both consult `reach`).
- **Free bonus:** the emission loop already mirrors every slice to
  `extruder.move_segment(...)`, so rounding slices are pressure-advance-synced
  automatically. No extruder changes.

## Phase 2 — blend across junctions, along-path (moderate, spans two moves)

- Class 3 cannot live inside a single `plan_move` (the ramp straddles the A->B
  boundary). Either pass neighbor accel context into `plan_move` (the `flush`
  loop has the whole queue), or add a post-pass in `_process_moves` over the
  concatenated slice chains that splices blend slices across boundaries where
  A's final `a` and B's initial `a` differ, ramping within `J_max`.
- Velocity is already continuous, so this is a pure accel-rate fixup.
- Subsumes/retires the existing crude heuristic
  `smooth_delta_v2 = 2*move_d*max_accel_to_decel` (`Move.__init__`,
  `toolhead.py:59`, i.e. `minimum_cruise_ratio`).

## Phase 3 — corner rounding / the geometry problem (hard, architectural)

Class 4 is the only one retiming cannot touch. To bound centripetal *jerk* the
velocity vector must rotate over finite time, i.e. **synthesize a smooth blend
curve** through each corner, inside the junction-deviation tolerance band, with
bounded curvature and bounded curvature-rate.

### Continuity hierarchy (what each level buys)

centripetal accel = `kappa * v^2`; so:

- **G0** (sharp corner): tangent discontinuous -> accel impulse (infinite).
- **G1** (circular-arc blend): tangent continuous, curvature jumps `0 -> 1/R` ->
  finite accel, but a **jerk impulse** at the arc endpoints.
- **G2** (clothoid / Euler spiral): curvature continuous (ramps linearly in arc
  length) -> **finite, bounded jerk**. This is the practical target.
- **G3+** (splines / elastica): curvature-rate continuous -> smooth jerk.

For a 3D printer, **G2 is almost certainly sufficient.**

### Discrete differential geometry: what is actually applicable

The motion path *is* a polyline (sequence of linear moves), and discrete
curvature at a vertex is exactly the concentrated turning angle (a Dirac).
Corner smoothing = spreading that Dirac over a finite arc inside the tolerance
band. Relevant, in rough order of practicality for real-time on the host:

1. **Cubic B-spline / 4-point subdivision (most architecturally aligned).**
   Treat the move polyline as a control polygon; run a couple of rounds of local
   subdivision near each corner. Cubic B-spline limit curve is **C2** ->
   curvature continuous -> no jerk impulse. Cheap, local, bounded support, and
   its output is a refined polyline that **drops straight into the existing
   slice emitter**. Deviation is bounded by the control-polygon distance, which
   maps naturally onto `junction_deviation`. **Recommended starting point.**

2. **Clothoid (Euler-spiral) corner blends.** The canonical CNC/robotics answer;
   closed-form-ish, G2, constant jerk through the blend, explicit deviation- and
   curvature-bound formulas. Slightly more math (Fresnel integrals / series) but
   directly gives bounded jerk.

3. **Quintic / cubic Bezier corner transitions.** Widely used for CNC corner
   smoothing; closed-form deviation bounds; quintic gives G2 (even G3). Easy to
   sample into micro-segments.

4. **Minimum Variation Curves (Moreton-Sequin) / discrete elastica.** The
   variational DDG objects: elastica minimizes `int kappa^2 ds`; MVC minimizes
   `int (d kappa / ds)^2 ds` — the latter is *literally* minimizing jerk content.
   Principled and optimal, but iterative -> better as an offline reference / for
   validating cheaper local schemes than as the realtime path.

5. **kappa-Curves (Yan et al., SIGGRAPH 2017).** Local, G2, curvature-controlled
   interpolation; a middle ground between Bezier blends and full MVC.

6. **Discrete curvature flow / Laplacian (Taubin) fairing.** *Not* suitable:
   global, no hard deviation tolerance.

**DDG's real contribution here** is (a) the discrete-curvature foundation
(turning angle = concentrated curvature, the object we must spread) and (b) the
variational objectives (elastica / MVC) that define what an *optimal* low-jerk
blend even is. For shipping code, the pragmatic choice is a **local** scheme —
cubic B-spline subdivision (most aligned with the slice architecture) or
clothoid/quintic-Bezier blends — validated against an offline MVC/elastica
reference.

### New architectural cost specific to curved blends

The current slice emitter assumes a single straight direction per move: every
slice in `plan_move` shares `move.axes_r`. A curved blend means **each
micro-segment has its own direction**, so the emission loop in `_process_moves`
must pass a *per-segment* `axes_r` to `trapq_append` (and to
`extruder.move_segment`). `trapq_append` already takes `axes_r` per call, so this
is bounded but real. Arc-length parameterization and re-timing `a(v)` over a
variable-curvature curve is the genuine complexity of Phase 3.

## Recommendation / sequencing

- **Phase 1** is high-value and self-contained (rounding slices + reach/peak
  distance feedback). Do first.
- **Phase 2** is the natural follow-on in the emission loop.
- **Phase 3** is a research-grade effort. Given input shaping already notches the
  resonance the corner impulse would excite, and `junction_deviation` bounds the
  corner accel magnitude, Phase 3 has marginal return versus 1-2. If pursued,
  start with **cubic B-spline subdivision** for G2 corner blends and validate
  against an offline **MVC/elastica** reference.
