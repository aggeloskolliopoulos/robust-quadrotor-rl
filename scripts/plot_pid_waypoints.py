from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CSV_PATH = Path("results/pid_waypoints.csv")
PLOTS_DIR = Path("results/plots")


def main():
    # Load experiment data
    data = np.genfromtxt(
        CSV_PATH,
        delimiter=",",
        names=True,
    )

    time = data["time"]

    x = data["x"]
    y = data["y"]
    z = data["z"]

    target_x = data["target_x"]
    target_y = data["target_y"]
    target_z = data["target_z"]

    rpm_1 = data["rpm_1"]
    rpm_2 = data["rpm_2"]
    rpm_3 = data["rpm_3"]
    rpm_4 = data["rpm_4"]

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Position tracking
    # ---------------------------------------------------------

    plt.figure()

    plt.plot(time, x, label="x")
    plt.plot(time, target_x, "--", label="target x")

    plt.plot(time, y, label="y")
    plt.plot(time, target_y, "--", label="target y")

    plt.plot(time, z, label="z")
    plt.plot(time, target_z, "--", label="target z")

    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.title("PID Waypoint Position Tracking")
    plt.legend()
    plt.grid()

    position_plot_path = PLOTS_DIR / "pid_waypoints_position.png"

    plt.savefig(
        position_plot_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved position plot to: {position_plot_path}")

    # ---------------------------------------------------------
    # 2. Motor RPMs
    # ---------------------------------------------------------

    plt.figure()

    plt.plot(time, rpm_1, label="Motor 1")
    plt.plot(time, rpm_2, label="Motor 2")
    plt.plot(time, rpm_3, label="Motor 3")
    plt.plot(time, rpm_4, label="Motor 4")

    plt.xlabel("Time [s]")
    plt.ylabel("Motor speed [RPM]")
    plt.title("PID Waypoint Motor Commands")
    plt.legend()
    plt.grid()

    rpm_plot_path = PLOTS_DIR / "pid_waypoints_rpms.png"

    plt.savefig(
        rpm_plot_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved RPM plot to: {rpm_plot_path}")

    # ---------------------------------------------------------
    # 3. Zoomed motor RPMs around first waypoint transition
    # ---------------------------------------------------------

    zoom_start = 3.8
    zoom_end = 5.5

    zoom_mask = (time >= zoom_start) & (time <= zoom_end)

    plt.figure()

    plt.plot(time[zoom_mask], rpm_1[zoom_mask], label="Motor 1")
    plt.plot(time[zoom_mask], rpm_2[zoom_mask], label="Motor 2")
    plt.plot(time[zoom_mask], rpm_3[zoom_mask], label="Motor 3")
    plt.plot(time[zoom_mask], rpm_4[zoom_mask], label="Motor 4")

    plt.axvline(
        4.0,
        linestyle="--",
        label="Waypoint change",
    )

    plt.xlabel("Time [s]")
    plt.ylabel("Motor speed [RPM]")
    plt.title("PID Motor Commands Around First Waypoint Change")
    plt.legend()
    plt.grid()

    rpm_zoom_plot_path = PLOTS_DIR / "pid_waypoints_rpms_zoom.png"

    plt.savefig(
        rpm_zoom_plot_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved zoomed RPM plot to: {rpm_zoom_plot_path}")

    # ---------------------------------------------------------
    # 4. XY trajectory
    # ---------------------------------------------------------

    # Find the samples where the waypoint changes
    waypoint_change = np.r_[
        True,
        (np.diff(target_x) != 0) | (np.diff(target_y) != 0),
    ]

    waypoint_x = target_x[waypoint_change]
    waypoint_y = target_y[waypoint_change]

    plt.figure()

    plt.plot(
        x,
        y,
        label="Actual trajectory",
    )

    plt.plot(
        waypoint_x,
        waypoint_y,
        "--o",
        label="Waypoints",
    )

    for i, (wx, wy) in enumerate(zip(waypoint_x, waypoint_y)):
        plt.annotate(
            f"W{i}",
            (wx, wy),
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("PID Waypoint XY Trajectory")
    plt.legend()
    plt.grid()
    plt.axis("equal")

    trajectory_plot_path = PLOTS_DIR / "pid_waypoints_xy.png"

    plt.savefig(
        trajectory_plot_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved XY trajectory plot to: {trajectory_plot_path}")


if __name__ == "__main__":
    main()