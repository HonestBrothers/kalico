#!/usr/bin/env python3
# Plot a torque-curve CSV (speed,max_accel) produced by TORQUE_CURVE_CALIBRATE.
#
# Usage: plot_torque_curve.py <input.csv> [output.png]
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_curve(path):
    """Return (speeds, accels) from a speed,max_accel CSV.

    Skips '#' comment lines and a non-numeric header row, matching what
    [torque_curve] itself accepts.
    """
    speeds, accels = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            try:
                speed = float(parts[0])
                accel = float(parts[1])
            except ValueError:
                continue  # header line such as "speed,max_accel"
            speeds.append(speed)
            accels.append(accel)
    return speeds, accels


def main():
    if len(sys.argv) < 2:
        sys.stderr.write(
            "usage: plot_torque_curve.py <input.csv> [output.png]\n"
        )
        return 1
    csv_path = sys.argv[1]
    if len(sys.argv) >= 3:
        png_path = sys.argv[2]
    else:
        png_path = os.path.splitext(csv_path)[0] + ".png"

    speeds, accels = load_curve(csv_path)
    if len(speeds) < 1:
        sys.stderr.write("no data points in %s\n" % csv_path)
        return 1

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(speeds, accels, "-o", color="tab:blue", label="max accel")
    ax.fill_between(speeds, accels, 0, color="tab:blue", alpha=0.12)
    ax.set_xlabel("Speed (mm/s)")
    ax.set_ylabel("Max acceleration (mm/s^2)")
    ax.set_title("Torque curve: %s" % os.path.basename(csv_path))
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(png_path, dpi=100)
    sys.stdout.write("wrote %s\n" % png_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
