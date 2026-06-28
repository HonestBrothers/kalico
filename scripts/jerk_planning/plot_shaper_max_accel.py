import math
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DR = 0.1  # default damping ratio
SCV = 5.0
TARGET_SMOOTHING = 0.12
VR = 20.0  # SHAPER_VIBRATION_REDUCTION


# --- classic FIR impulse shapers (from shaper_defs.py) ---
def zv(f, z):
    df = math.sqrt(1 - z * z); K = math.exp(-z * math.pi / df); td = 1 / (f * df)
    return [1.0, K], [0.0, 0.5 * td]

def zvd(f, z):
    df = math.sqrt(1 - z * z); K = math.exp(-z * math.pi / df); td = 1 / (f * df)
    return [1.0, 2 * K, K * K], [0.0, 0.5 * td, td]

def mzv(f, z):
    df = math.sqrt(1 - z * z); K = math.exp(-0.75 * z * math.pi / df); td = 1 / (f * df)
    a1 = 1 - 1 / math.sqrt(2); a2 = (math.sqrt(2) - 1) * K; a3 = a1 * K * K
    return [a1, a2, a3], [0.0, 0.375 * td, 0.75 * td]

def ei(f, z):
    v = 1 / VR; df = math.sqrt(1 - z * z); K = math.exp(-z * math.pi / df); td = 1 / (f * df)
    a1 = 0.25 * (1 + v); a2 = 0.5 * (1 - v) * K; a3 = a1 * K * K
    return [a1, a2, a3], [0.0, 0.5 * td, td]

def ei2(f, z):
    v = 1 / VR; df = math.sqrt(1 - z * z); K = math.exp(-z * math.pi / df); td = 1 / (f * df)
    V2 = v * v; X = pow(V2 * (math.sqrt(1 - V2) + 1), 1 / 3)
    a1 = (3 * X * X + 2 * X + 3 * V2) / (16 * X); a2 = (0.5 - a1) * K; a3 = a2 * K; a4 = a1 * K ** 3
    return [a1, a2, a3, a4], [0.0, 0.5 * td, td, 1.5 * td]

def ei3(f, z):
    v = 1 / VR; df = math.sqrt(1 - z * z); K = math.exp(-z * math.pi / df); td = 1 / (f * df)
    K2 = K * K
    a1 = 0.0625 * (1 + 3 * v + 2 * math.sqrt(2 * (v + 1) * v))
    a2 = 0.25 * (1 - v) * K; a3 = (0.5 * (1 + v) - 2 * a1) * K2; a4 = a2 * K2; a5 = a1 * K2 * K2
    return [a1, a2, a3, a4, a5], [0.0, 0.5 * td, td, 1.5 * td, 2 * td]


def smoothing(A, T, accel, scv=SCV):
    # exact port of _get_shaper_smoothing
    half = accel * 0.5
    invD = 1.0 / sum(A); n = len(T)
    ts = sum(A[i] * T[i] for i in range(n)) * invD
    o90 = o180 = 0.0
    for i in range(n):
        if T[i] >= ts:
            o90 += A[i] * (scv + half * (T[i] - ts)) * (T[i] - ts)
        o180 += A[i] * half * (T[i] - ts) ** 2
    return max(o90 * invD * math.sqrt(2), o180 * invD)


def max_accel(shaper_fn, f):
    A, T = shaper_fn(f, DR)
    lo, hi = 100.0, 100000.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if smoothing(A, T, mid) <= TARGET_SMOOTHING:
            lo = mid
        else:
            hi = mid
    return lo


shapers = [
    ("zv", zv, "tab:blue"), ("mzv", mzv, "tab:orange"),
    ("zvd", zvd, "tab:green"), ("ei", ei, "tab:red"),
    ("2hump_ei", ei2, "tab:purple"), ("3hump_ei", ei3, "tab:brown"),
]

freqs = np.linspace(20, 100, 400)
plt.figure(figsize=(10, 6.5))
for name, fn, c in shapers:
    amax = [max_accel(fn, f) for f in freqs]
    plt.plot(freqs, amax, label=name, color=c, lw=2)

# reference f^2 scaling guide anchored to zv at 50 Hz
zv50 = max_accel(zv, 50.0)
plt.plot(freqs, zv50 * (freqs / 50.0) ** 2, "k--", lw=1, alpha=0.5,
         label=r"$\propto f^2$ guide")

plt.xlabel("Resonant frequency (Hz)")
plt.ylabel("Recommended max_accel (mm/s$^2$)")
plt.title("Input-shaper accel ceiling vs. resonance\n"
          "(smoothing = ½·σ²·a held at TARGET_SMOOTHING = 0.12 mm, scv = 5 mm/s)")
plt.grid(True, alpha=0.3)
plt.legend(title="shaper")
plt.ylim(0, 20000)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "amax_vs_freq.png"), dpi=130)

# print a small table at common frequencies
print("freq |  " + "  ".join("%9s" % n for n, _, _ in shapers))
for f in [30, 40, 50, 60, 80]:
    print("%4d | " % f + "  ".join("%9.0f" % max_accel(fn, f) for _, fn, _ in shapers))
