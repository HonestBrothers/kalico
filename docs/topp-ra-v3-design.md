# topp-ra-v3 — unified phase-plane motion pipeline

## Principle
Keep every decision and every constraint in the path phase plane `(s, ṡ)`.
Render to step times exactly **once**, by a single monotonic integration of the
final feasible `s(t)`. **Never edit the trajectory in the time domain after
that.** The hardware forces a time-domain rendering (step pulses are events in
time); the sin is *re-editing* the trajectory once we've left the phase plane.

## What this replaces (the four stacked rewriters on v2)
1. stock trapezoid planner
2. `jerk_limiting` slice chains (`ramp_slices` → constant-accel micro-slices)
3. `topp_ra` per-move slice emission
4. `input_shaper` (time-domain FIR convolution in `kin_shaper.c`)

The interaction (2/3 emit ~2 ms slices; 4 convolves them at ~11 ms impulse
spacing) is the source of the intermittent `stepcompress i=0 Invalid sequence`
shutdown. Collapsing to one pipeline removes the failure class by construction.

## Target pipeline
1. **One path `x(s)`** — geometry (incl. corner blends) baked into `s`. Single
   source of truth.
2. **One reachability pass → `u(s) = ṡ²(s)`** (TOPP-RA). All constraints are
   affine in `(u, u')` and already live here:
   - velocity → bound on `u`
   - `torque_curve` `a_max(v)` → bound on the `(u, u')` combination
   - melt-flow (from the MPC MeltLimiter) → `v_max(s)` → bound on `u`
   - jerk → see below (3rd-order wrinkle)
3. **One monotonic emission** — integrate `dt = ds/√u` once → `s(t)`; generate
   steps directly from `x(s(t))`. No slicing, no second filter.

## Vibration: model-inverse feedforward (replaces input_shaper)
Enabled because Klipper plans offline — we have the full future, so the acausal
stable inverse is legal (feedback control can't do this).

Axis flexible mode (measured Y ≈ 75–79 Hz, ζ ≈ 1.9%):
```
ÿ + 2ζωₙ ẏ + ωₙ² y = ωₙ² x     (x = motor, y = tool tip)
```
Pure resonant mode → no finite zeros → inverse is a stable differentiator:
```
x_motor = y_des + (2ζ/ωₙ)·ẏ_des + (1/ωₙ²)·ÿ_des
```
In phase-plane terms (`ẏ = y'(s)ṡ`, `ÿ = y'(s)s̈ + y''(s)ṡ²`):
```
x_motor(s) = y(s) + (2ζ/ωₙ)·y'(s)·ṡ + (1/ωₙ²)·[y'(s)·s̈ + y''(s)·ṡ²]
```
- Pure function of path state → no time-domain convolution.
- **Speed-aware** (depends on ṡ) → fixes the "auto-shaper mistunes at speed"
  problem; a fixed shaper's impulse timing is speed-independent.
- Needs `y_des ∈ C²` → jerk limiting is what makes the FF well-posed (the two
  are the same programme, not separate features).
- Structurally cannot cause the shaper crash: reads instantaneous v,a at the
  *current* time, no time-shift/reordering. (May command small smooth reversals
  to pre-cancel ringing — bounded by jerk-limited a; that's the physics.)

## Klipper insertion points
- Emission seam: `toolhead.py _process_moves` (today loops jerk/topp slices →
  `trapq_append`). Becomes: emit the single feasible profile.
- FF seam: `kin_shaper.c` already modifies the stepper *kinematic position
  function* (not a naive command convolution) — reuse that exact hook, replace
  the impulse-sum with `x += (2ζ/ωₙ)·v + (1/ωₙ²)·a` from the trapq's pos/vel/
  accel at time t. Cheap in the C hot path.

## Robustness knob (Stage 4, built)
`model_inverse_ff.py` `FFParams` interpolates the inverse between two forms:
- `r=0`: exact single-zero cancellation `F1(s)=1+c1 s+c2 s^2`
  (`c1=2ζ/ωₙ`, `c2=1/ωₙ²`) — deep but fragile ("pole peeks out" on ωₙ drift).
- `r=1`: derivative-matched robust inverse `F2=F1²` — enforces `F(jωₙ)=0` AND
  `dF/dω|ωₙ=0` (flat-bottomed notch, ZVD-style), tolerant to ωₙ error.

`F_r=(1-r)F1+r F2` → `x = y + b1 y' + b2 y'' + b3 y''' + b4 y''''` with
`b1=c1(1+r)`, `b2=c2(1+r)+r c1²`, `b3=2r c1 c2`, `b4=r c2²`. So `r>0`
introduces jerk (`b3`) and snap (`b4`) content. Verified: worst-case residual
over a ±15% ωₙ band falls 0.322→0.104 (r=0→1); on-axis notch depth = `2ζ`.

**Ties to the acceleration constraint (why this is NOT the shaper's max_accel
rule):** input_shaping derates `max_accel` to bound convolution smoothing. The
FF does no smoothing (sharp features preserved), so that derate is dropped.
Instead the torque-curve `a_max(v)` now bounds the *motor* trajectory
`x=y+corrections`; the FF inflates motor accel by `b1·jerk+b2·snap+…`
(`motor_accel_extra`). A `headroom` fraction reserves `a_max(v)` for this — the
tool planner runs against `plan_accel_scale()·a_max(v)`. And the accel term is
discontinuous on coarse trapq: an accel jump `Δa` makes an
`b2·Δa` motor-position jump (measured 0.085 mm at 20000 mm/s² → stepcompress).
Hence **the C application of the accel term (and all of `r>0`) requires the
continuous-accel/C³ trajectory from Stage 3.** At `r=0`, only `b1·v` (velocity
lead, continuous) + `b2·a` are used; `r=0` is safe once accel is continuous.

Delivered as a tested pure-math library + `[model_inverse_ff]` Klipper object
(SET_MODEL_FF live tuning, get_status, best-effort C push; planner-only until
the C seam lands). The C seam (`ff_*_calc_position` in kin_shaper.c, smoothed
v/a for continuity) is the next step and rides Stage 3.

## Jerk (the honest wrinkle)
TOPP-RA is 2nd-order (`u`, `u'`); jerk is 3rd-order (`s⃛`). Options:
- (A) approximate: curvature / rate limit on `u(s)` (cheap, in-plane; what the
  smoothing already approaches).
- (B) exact: 3rd-order reachability carrying `s̈` as a state (one extra dim,
  more compute).
Decide by required fidelity. Start with (A).

## Migration stages
1. **Unify emission** — one feasible `u(s)` → one monotonic step pass; retire
   the slice+shaper composition. Biggest structural win; kills the crash family.
   The `jl-stepguard` invariant becomes something this stage *guarantees*, so
   the guard retires into a regression test.
2. **Fold constraints** into the single reachability pass (velocity,
   torque_curve, melt-flow). ~90% already phase-plane-native.
3. **Jerk** — pick (A) or (B).
4. **Model-inverse FF** in the `kin_shaper` seam; delete `input_shaper` usage.
5. **Validate on hardware** — ringing tower vs shaper; flow-ramp for melt-flow.

## Preserved from v2 (do not re-port from clean Kalico — keep the fixes)
- `torque_curve` `a_max(v)` + `torque_curve_calibrate`
- ring-down `(ωₙ, ζ)` estimator (`_ringdown_decay`) — the FF parameter source
- Phase-2 decel-coalescing clamp; adaptive corner blend
- melt-flow MeltLimiter (MPC-driven) — folds into the reachability velocity cap

## Retires
- `jl-stepguard` runtime guard → regression test
- `input_shaper` → model-inverse FF
