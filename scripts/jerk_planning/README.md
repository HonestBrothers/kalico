# Jerk-planning analysis scripts

Standalone analysis/prototype scripts supporting
`docs/TOPP_RA_Jerk_Limiting_Plan.md`. Pure `numpy`/`matplotlib`, no Kalico
imports — run them directly. Each writes a PNG next to itself.

## `plot_shaper_max_accel.py`

Reproduces Kalico's `_get_shaper_smoothing` math and plots the recommended
`max_accel` ceiling vs. resonant frequency for every input shaper (zv, mzv,
zvd, ei, 2hump_ei, 3hump_ei), holding smoothing at `TARGET_SMOOTHING = 0.12`
mm. Shows the `a_max ~ f^2` scaling and the robustness/smoothing ranking.

## `bspline_corner.py`

Prototype of cubic B-spline corner subdivision (Phase 3 of the plan): polyline
in -> C2 refined curve out. Plots geometry, curvature `kappa(s)` (= accel/v^2),
curvature rate `dkappa/ds` (= jerk/v^3), and the deviation<->jerk trade. Panel D
overlays the clothoid (minimum-peak-jerk) frontier as the reference subdivision
is judged against. In the practical junction-deviation band (~0.5-2 mm),
subdivision sits within a few percent of the clothoid optimum.

Output PNGs are git-ignored regeneration artifacts; commit only the scripts.
