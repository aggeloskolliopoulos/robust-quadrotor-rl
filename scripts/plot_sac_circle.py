import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


CSV_PATH = Path(
    "results/evaluation/"
    "circle_zero_shot_seed_1100_trajectory.csv"
)

OUTPUT_DIR = Path("results/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load trajectory data
# ============================================================

with CSV_PATH.open("r", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))


def column(name):
    return np.array(
        [float(row[name]) for row in rows],
        dtype=np.float64,
    )


time = column("time_s")

actual_x = column("actual_x_m")
actual_y = column("actual_y_m")

target_x = column("target_x_m")
target_y = column("target_y_m")

position_error = column("position_error_m")
velocity_error = column("velocity_error_m_s")


# ============================================================
# Plot 1: Actual vs desired XY trajectory
# ============================================================

plt.figure(figsize=(8, 7))

plt.plot(
    target_x,
    target_y,
    "--",
    linewidth=2,
    label="Desired trajectory",
)

plt.plot(
    actual_x,
    actual_y,
    linewidth=2,
    label="Actual SAC trajectory",
)

plt.scatter(
    target_x[0],
    target_y[0],
    marker="o",
    s=70,
    label="Target start",
)

plt.scatter(
    actual_x[0],
    actual_y[0],
    marker="x",
    s=90,
    label="Actual start",
)

plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")

plt.title(
    "SAC Zero-Shot Circular Tracking\n"
    "R = 0.30 m, omega = 0.40 rad/s"
)

plt.axis("equal")
plt.grid(True)
plt.legend()

plt.tight_layout()

xy_path = OUTPUT_DIR / "sac_circle_xy.png"
plt.savefig(xy_path, dpi=200)
plt.close()


# ============================================================
# Plot 2: Position tracking error vs time
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    time,
    position_error,
    linewidth=1.8,
    label="Position error",
)

plt.axvline(
    5.0,
    linestyle="--",
    label="Late-tracking interval",
)

plt.xlabel("Time (s)")
plt.ylabel("3D Position error (m)")
plt.title("SAC Circular Tracking: Position Error")

plt.grid(True)
plt.legend()
plt.tight_layout()

error_path = OUTPUT_DIR / "sac_circle_position_error.png"
plt.savefig(error_path, dpi=200)
plt.close()


# ============================================================
# Plot 3: Velocity tracking error vs time
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    time,
    velocity_error,
    linewidth=1.8,
)

plt.xlabel("Time (s)")
plt.ylabel("3D Velocity error (m/s)")
plt.title("SAC Circular Tracking: Velocity Error")

plt.grid(True)
plt.tight_layout()

velocity_path = OUTPUT_DIR / "sac_circle_velocity_error.png"
plt.savefig(velocity_path, dpi=200)
plt.close()


print("Plots saved:")
print(xy_path)
print(error_path)
print(velocity_path)