"""
Standalone prototype: cubic B-spline corner subdivision for motion-path
smoothing. Polyline in -> refined C2 curve out, with curvature, jerk-proxy,
and the deviation<->jerk trade-off plotted. No Kalico imports; pure numpy.

Geometric quantities only (curvature kappa, dkappa/ds). Physical mapping:
  centripetal accel = kappa * v^2
  jerk magnitude   ~ v^3 * |dkappa/ds|   (constant-speed traversal)
so peak kappa scales the accel ceiling and peak |dkappa/ds| scales jerk.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "bspline_corner.png")


# --- cubic B-spline subdivision (masks (1,6,1)/8 vertex, (4,4)/8 edge) -------
def subdivide_cubic(P, rounds):
    P = np.asarray(P, float)
    for _ in range(rounds):
        # repeat-endpoint padding so the open curve ends stay put-ish
        Pp = np.vstack([P[0], P, P[-1]])
        n = len(Pp)
        out = []
        for i in range(1, n - 1):
            out.append((Pp[i - 1] + 6 * Pp[i] + Pp[i + 1]) / 8.0)  # vertex
            out.append((Pp[i] + Pp[i + 1]) / 2.0)                  # edge
        P = np.array(out)
    return P


# --- densify each segment to localize rounding (deviation knob) --------------
def densify(P, h):
    P = np.asarray(P, float)
    out = [P[0]]
    for a, b in zip(P[:-1], P[1:]):
        L = np.linalg.norm(b - a)
        k = max(1, int(np.ceil(L / h)))
        for j in range(1, k + 1):
            out.append(a + (b - a) * (j / k))
    return np.array(out)


# --- discrete curvature (Menger / circumradius of consecutive triples) ------
def curvature(C):
    C = np.asarray(C, float)
    seg = np.diff(C, axis=0)
    ds = np.linalg.norm(seg, axis=1)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    kap = np.zeros(len(C))
    for i in range(1, len(C) - 1):
        a = C[i] - C[i - 1]
        b = C[i + 1] - C[i]
        c = C[i + 1] - C[i - 1]
        la, lb, lc = (np.linalg.norm(v) for v in (a, b, c))
        area2 = abs(a[0] * b[1] - a[1] * b[0])  # 2*triangle area
        denom = la * lb * lc
        kap[i] = area2 / denom if denom > 1e-12 else 0.0
    return s, kap


def dev_from_polyline(C, P):
    # max distance from refined points C to the original polyline P
    P = np.asarray(P, float)
    worst = 0.0
    for pt in C:
        d = min(_pt_seg(pt, P[k], P[k + 1]) for k in range(len(P) - 1))
        worst = max(worst, d)
    return worst


def _pt_seg(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-18), 0.0, 1.0)
    return np.linalg.norm(p - (a + t * ab))


# --- clothoid (Euler-spiral) corner blend: the min-PEAK-jerk reference -------
# A symmetric biclothoid has a TRIANGULAR curvature profile -> constant
# |dkappa/ds| = 4*theta/S^2, which is the minimum achievable peak jerk for a
# curvature-continuous blend of support length S and turn angle theta. (The MVC
# that minimizes integral (dkappa/ds)^2 is in the same family; the clothoid is
# the right reference when the binding constraint is a peak jerk limit J_max.)
def _line_intersect(p1, d1, p2, d2):
    A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
    t = np.linalg.solve(A, p2 - p1)
    return p1 + t[0] * d1


def clothoid_corner(theta, S, n=600):
    s = np.linspace(0.0, S, n)
    half = 0.5 * S
    kpeak = 2.0 * theta / S
    kap = np.where(s <= half, kpeak * (s / half), kpeak * ((S - s) / half))
    ds = np.diff(s)
    phi = np.concatenate([[0.0], np.cumsum(0.5 * (kap[1:] + kap[:-1]) * ds)])
    phi -= 0.5 * theta  # symmetric about heading 0
    x = np.concatenate([[0.0],
                        np.cumsum(0.5 * (np.cos(phi[1:]) + np.cos(phi[:-1]))
                                  * ds)])
    y = np.concatenate([[0.0],
                        np.cumsum(0.5 * (np.sin(phi[1:]) + np.sin(phi[:-1]))
                                  * ds)])
    C = np.column_stack([x, y])
    # rotate so the incoming tangent is +x and the outgoing tangent is +y
    rot = 0.5 * theta
    R = np.array([[np.cos(rot), -np.sin(rot)],
                  [np.sin(rot), np.cos(rot)]])
    C = C @ R.T
    d_in = R @ np.array([np.cos(-0.5 * theta), np.sin(-0.5 * theta)])
    d_out = R @ np.array([np.cos(0.5 * theta), np.sin(0.5 * theta)])
    V = _line_intersect(C[0], d_in, C[-1], d_out)
    C = C - V  # apex at origin
    peak_dk = 4.0 * theta / (S * S)
    return C, peak_dk


# --- test polyline: lead-in, 90 deg corner, sharp ~135 deg turn, lead-out ---
poly = np.array([
    [0.0, 0.0],
    [40.0, 0.0],
    [40.0, 40.0],   # 90 deg corner
    [10.0, 55.0],   # sharp turn
    [10.0, 90.0],
])

ROUNDS = 5

fig, ax = plt.subplots(2, 2, figsize=(13, 10))

# Panel A: geometry for a few densification spacings -------------------------
axA = ax[0, 0]
axA.plot(poly[:, 0], poly[:, 1], "k--o", lw=1, ms=5, label="original polyline")
trade = []
for h, col in [(40.0, "tab:red"), (12.0, "tab:orange"), (4.0, "tab:green")]:
    P = densify(poly, h)
    C = subdivide_cubic(P, ROUNDS)
    s, kap = curvature(C)
    dev = dev_from_polyline(C, poly)
    peak_k = kap.max()
    dk = np.abs(np.gradient(kap, s))
    peak_dk = dk[2:-2].max()
    trade.append((h, dev, peak_k, peak_dk))
    axA.plot(C[:, 0], C[:, 1], color=col, lw=2,
             label="blend h=%.0fmm  dev=%.2fmm" % (h, dev))
axA.set_aspect("equal")
axA.set_title("Cubic B-spline corner blend (C2)\nsmaller support h -> tighter, "
              "less deviation")
axA.legend(fontsize=8)
axA.grid(alpha=0.3)

# Panel B: curvature vs arc length (the spread-out 'Dirac') ------------------
axB = ax[0, 1]
for h, col in [(40.0, "tab:red"), (12.0, "tab:orange"), (4.0, "tab:green")]:
    P = densify(poly, h)
    C = subdivide_cubic(P, ROUNDS)
    s, kap = curvature(C)
    axB.plot(s, kap, color=col, lw=2, label="h=%.0fmm" % h)
axB.set_xlabel("arc length s (mm)")
axB.set_ylabel(r"curvature $\kappa$ (1/mm)  ~  accel/$v^2$")
axB.set_title("Curvature is continuous & bounded\n(sharp corner = a Dirac here)")
axB.legend(fontsize=8)
axB.grid(alpha=0.3)

# Panel C: curvature rate = jerk proxy --------------------------------------
axC = ax[1, 0]
for h, col in [(40.0, "tab:red"), (12.0, "tab:orange"), (4.0, "tab:green")]:
    P = densify(poly, h)
    C = subdivide_cubic(P, ROUNDS)
    s, kap = curvature(C)
    dk = np.gradient(kap, s)
    axC.plot(s, dk, color=col, lw=1.5, label="h=%.0fmm" % h)
axC.set_xlabel("arc length s (mm)")
axC.set_ylabel(r"$d\kappa/ds$ (1/mm$^2$)  ~  jerk/$v^3$")
axC.set_title("Jerk proxy: finite everywhere (no impulse)")
axC.legend(fontsize=8)
axC.grid(alpha=0.3)

# Panel D: jerk frontier, B-spline vs clothoid on ONE isolated 90deg corner --
# Fair apples-to-apples: same corner geometry, sweep blend size, compare peak
# jerk (peak |dkappa/ds|) at each deviation. Clothoid = min-peak-jerk frontier.
axD = ax[1, 1]
theta = np.pi / 2.0
L = 40.0
corner = np.array([[-L, 0.0], [0.0, 0.0], [0.0, L]])  # apex at origin

bs_dev, bs_jerk = [], []
for h in np.linspace(2.0, 38.0, 16):
    P = densify(corner, h)
    C = subdivide_cubic(P, ROUNDS)
    s, kap = curvature(C)
    bs_dev.append(dev_from_polyline(C, corner))
    bs_jerk.append(np.abs(np.gradient(kap, s))[2:-2].max())

cl_dev, cl_jerk = [], []
for S in np.linspace(6.0, 90.0, 24):
    C, peak_dk = clothoid_corner(theta, S)
    cl_dev.append(dev_from_polyline(C, corner))
    cl_jerk.append(peak_dk)

order = np.argsort(cl_dev)
cl_dev = np.array(cl_dev)[order]
cl_jerk = np.array(cl_jerk)[order]
axD.plot(bs_dev, bs_jerk, "o-", color="tab:green",
         label="cubic B-spline subdivision")
axD.plot(cl_dev, cl_jerk, "s--", color="tab:red",
         label="clothoid (min-peak-jerk frontier)")
axD.set_xlabel("max deviation from corner (mm)")
axD.set_ylabel(r"peak $d\kappa/ds$ (1/mm$^2$)  ~  jerk/$v^3$")
axD.set_yscale("log")
axD.set_title("Jerk frontier on a 90$\\degree$ corner\n"
              "(how much subdivision leaves on the table)")
axD.legend(fontsize=8)
axD.grid(alpha=0.3, which="both")

fig.tight_layout()
fig.savefig(OUT, dpi=125)

print("h(mm)  dev(mm)  peak_kappa(1/mm)  peak_dk(1/mm^2)")
for h, dev, pk, pdk in trade:
    print("%5.0f  %7.3f  %15.4f  %14.4f" % (h, dev, pk, pdk))

print("\nJerk frontier (90deg corner), B-spline vs clothoid at matched dev:")
print("  dev(mm)  bspline_jerk   clothoid_jerk   ratio")
for d_t in [0.5, 1.0, 2.0, 4.0]:
    bj = float(np.interp(d_t, bs_dev, bs_jerk))
    cj = float(np.interp(d_t, cl_dev, cl_jerk))
    print("  %6.1f  %12.5f  %13.5f  %5.2fx" % (d_t, bj, cj, bj / cj))
print("\nNote: at v mm/s, peak centripetal accel = peak_kappa * v^2,")
print("peak jerk ~ peak_dk * v^3. Junction-deviation tolerance sets the")
print("operating point; clothoid is the lower bound subdivision is judged by.")
