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
- Structurally cannot cause the shaper crash: no time-shift or reordering of the
  trajectory. (The `±hst` velocity difference is *symmetric* — it derives `a`, it
  does not delay or re-order the output the way a convolution shaper does.) May
  command small smooth reversals to pre-cancel ringing — bounded by jerk-limited
  a; that's the physics.

## Klipper insertion points
- Emission seam: `toolhead.py _process_moves` (today loops jerk/topp slices →
  `trapq_append`). Becomes: emit the single feasible profile. **DONE** (Stage
  1+2, `unified_planner` flag).
- FF seam: `kin_shaper.c` modifies the stepper *kinematic position function*.
  **DONE** (Stage 4 C): `ff_calc_axis` computes `y + c1·v + c2·a` per axis, gated
  behind `ff_x/ff_y.enabled`.
  `input_shaper_set_ff_params(sk, axis, enabled, c1, c2, hst)` sets it and clears
  any pulses/smoother on that axis (FF replaces the shaper).
  `model_inverse_ff.py` wraps the steppers (same input_shaper struct) and pushes
  the 2nd-order coeffs. Verified: `test/test_ff_seam.c` calls the compiled
  `calc_position_cb` — legacy exactness, mid-move exactness, and boundary
  continuity (jump raw 0.016 mm → smoothed 0.0).
- **Accel is NOT read pointwise off the trapq (changed 2026-07-22).** The trapq
  exposes `a = 2·half_accel`, a *piecewise-constant staircase*, so `c2·a` stepped
  by `c2·Δa` at every move boundary — and because the jerk ramp is built from many
  short constant-accel slices, at *every slice boundary*, with magnitude
  `c2·J·dt₀`. Since `c2 = 1/ωₙ²` this explodes at low `freq_y`: 0.134 mm (~11
  microsteps demanded instantaneously) at `freq_y=21` → `MCU shutdown: Timer too
  close` (the MCU was *not* loaded — `mcu_awake=0.008`; it was an impossible
  instantaneous step *rate*). `a` is now derived from a `±hst` **central
  difference of velocity** (`ff_axis_velocity`, projected per-move — a neighbour
  can have a different `axes_r`, so differencing raw scalar speeds across a corner
  is silently wrong). Velocity is continuous across trapq boundaries, so the
  result is continuous and each accel step becomes a linear ramp. Cost:
  `sinc(ωₙ·hst)` attenuation at the cancelled mode (~0.3% at 21 Hz, hst=1 ms).
  Config `smooth_time` (default 0.5 ms); `0` = legacy staircase, bit-for-bit.
- **⚠️ Step-gen window is no longer 0.** Reading `v(t±hst)` crosses into
  neighbouring moves, so `shaper_note_generation_time()` sets
  `gen_steps_pre_active = gen_steps_post_active = hst`. Omit that and the
  move-list walk touches *freed* moves → intermittent garbage/segfault.

## FF realization: 2nd-order pointwise vs 4th-order (robustness)
The C seam realizes the **2nd-order** exact inverse `x = y + p1·v + p2·a`
(`p1=2ζ_eff/ωₙ`, `p2=1/ωₙ²`) because v and a are the only derivatives the
constant-accel trapq exposes pointwise. Robustness widens the zero damping
`ζ_eff = ζ + r·(ζ_wide−ζ)` — a clean, wider 2nd-order notch at ωₙ (notch depth
on jω = 2ζ_eff), realizable from v,a alone. This does NOT give the ZVD
frequency-insensitivity (`dF/dω|ωₙ=0`), which needs the 4th-order `b3·jerk +
b4·snap` terms and hence a C³ trajectory carrying jerk & snap — deferred; ωₙ
drift is instead handled by re-identifying the mode from ring-down. The
4th-order `b1..b4` remain in FFParams as the analysis/target model.
- Mutually exclusive with `[input_shaper]` on the same axis (FF replaces it).
- Off by default (`freq_*`=0). Enabling wraps steppers + pushes coeffs.

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
tool planner runs against `plan_accel_scale()·a_max(v)`. The accel term *was*
discontinuous on coarse trapq: an accel jump `Δa` made a `b2·Δa` motor-position
jump (0.085 mm at 20000 mm/s² → stepcompress). **Superseded 2026-07-22:** the
seam now derives `a` from smoothed velocity, so `c2·a` is continuous regardless
of trapq granularity. The accel term therefore **no longer depends on Stage 3
for crash-safety** — Stage 3 remains valuable for the physics (and for keeping
the *commanded* motion sane), but it is no longer the precondition. `r>0` still
requires the C³/C⁴ trajectory, because `b3·jerk + b4·snap` are derivatives the
trapq does not carry at all.

Delivered as a tested pure-math library + `[model_inverse_ff]` Klipper object
(SET_MODEL_FF live tuning, get_status, C push; falls back to planner-only if the
C symbol is absent). The C seam (`ff_calc_axis` in kin_shaper.c) is **BUILT**,
including the smoothed-v/a continuity work it was waiting on.

## Jerk (the honest wrinkle) — approach (A) BUILT
TOPP-RA is 2nd-order (`u`, `u'`); jerk is 3rd-order (`s⃛`). Options:
- (A) approximate: rate-limit the acceleration inside the emitter (cheap,
  in-plane). **CHOSEN + built.**
- (B) exact: 3rd-order reachability carrying `s̈` as a state (one extra dim).

Approach (A) in `pathplan._ramp_up_jerk`: integrate the accel phase at fixed
`jerk_dt`, `a_new = min(a_max(v), a_brake, a_prev + J·dt)`. The brake cap
`a_brake = sqrt(2J·(v1−v))` has `da/dt = −J` along it, so following it makes
`a` rise ≤`J·dt` per step and taper to ~0 at the phase boundary — `a(t)` is
continuous. Decel = time-reversed accel ramp. Feasibility stays a_max-based:
a move too short to jerk-limit falls back per-move to the sharp profile (which
always fits what the lookahead approved). `max_jerk=None` = byte-for-byte sharp.

**Discrete taper floor & why it matters here:** the last brake slice lands with
residual accel ≤ `2·J·dt0` (partial slice triggers at `rem ≤ a·dt0`
⟹ `a ≤ 2J·dt0`). That residual is the *only* accel discontinuity left, and it
is FF-safe by design: mapped through the FF accel coefficient `b2 = 1/ωₙ²`, the
motor-position jump is `b2·2J·dt0` — measured **0.0008 mm** (vs a 0.0125 mm
step) for the 77 Hz mode at `J=1e5, dt0=1ms`.

**Correction 2026-07-22 — the claim above that the taper residual is the *only*
accel discontinuity left is WRONG, and it misdirected a debugging session.** The
jerk ramp is emitted as many short constant-accel slices, so accel steps by
`J·dt₀` at *every slice boundary*, not just the final one; each was a `b2·J·dt₀`
FF position jump. That is why shrinking `dt₀` appeared to "fix" things (it scales
every jump linearly, pushing them under the one-microstep quantization floor)
while the real defect was structural. The seam-level fix (smoothed accel, above)
removes the whole class. Stage 3 is consequently **no longer the precondition**
for a crash-safe Stage-4 accel term — useful for the motion physics, not load-
bearing for stepcompress safety.
Config: `[printer] unified_max_jerk` (0=off), `unified_jerk_dt` (default 1ms).

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
