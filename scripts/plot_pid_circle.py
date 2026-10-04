from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CSV_PATH = Path("results/pid_circle.csv")
PLOTS_DIR = Path("results/plots")


def main():

    # ---------------------------------------------------------
    # Load experiment data
    # ---------------------------------------------------------

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

    error_x = data["error_x"]
    error_y = data["error_y"]
    error_z = data["error_z"]

    rpm_1 = data["rpm_1"]
    rpm_2 = data["rpm_2"]
    rpm_3 = data["rpm_3"]
    rpm_4 = data["rpm_4"]

    # Magnitude of the 3D position error
    position_error_norm = np.sqrt(
        error_x**2
        + error_y**2
        + error_z**2
    )

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

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
    plt.title("PID Circle Position Tracking")

    plt.legend()
    plt.grid()

    position_plot_path = (
        PLOTS_DIR / "pid_circle_position.png"
    )

    plt.savefig(
        position_plot_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved position plot to: "
        f"{position_plot_path}"
    )

    # ---------------------------------------------------------
    # 2. XY trajectory
    # ---------------------------------------------------------

    plt.figure()

    plt.plot(
        target_x,
        target_y,
        "--",
        label="Target trajectory",
    )

    plt.plot(
        x,
        y,
        label="Actual trajectory",
    )

    # Mark start point
    plt.plot(
        target_x[0],
        target_y[0],
        "o",
        label="Start",
    )

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("PID Circle XY Trajectory")

    plt.legend()
    plt.grid()
    plt.axis("equal")

    xy_plot_path = (
        PLOTS_DIR / "pid_circle_xy.png"
    )

    plt.savefig(
        xy_plot_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved XY trajectory plot to: "
        f"{xy_plot_path}"
    )

    # ---------------------------------------------------------
    # 3. Position error magnitude
    # ---------------------------------------------------------

    plt.figure()

    plt.plot(
        time,
        position_error_norm,
        label="3D position error",
    )

    plt.xlabel("Time [s]")
    plt.ylabel("Position error [m]")
    plt.title("PID Circle Position Error")

    plt.legend()
    plt.grid()

    error_plot_path = (
        PLOTS_DIR / "pid_circle_error.png"
    )

    plt.savefig(
        error_plot_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved error plot to: "
        f"{error_plot_path}"
    )

    # ---------------------------------------------------------
    # 4. Motor RPMs
    # ---------------------------------------------------------

    plt.figure()

    plt.plot(
        time,
        rpm_1,
        label="Motor 1",
    )

    plt.plot(
        time,
        rpm_2,
        label="Motor 2",
    )

    plt.plot(
        time,
        rpm_3,
        label="Motor 3",
    )

    plt.plot(
        time,
        rpm_4,
        label="Motor 4",
    )

    plt.xlabel("Time [s]")
    plt.ylabel("Motor speed [RPM]")
    plt.title("PID Circle Motor Commands")

    plt.legend()
    plt.grid()

    rpm_plot_path = (
        PLOTS_DIR / "pid_circle_rpms.png"
    )

    plt.savefig(
        rpm_plot_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved RPM plot to: "
        f"{rpm_plot_path}"
    )


if __name__ == "__main__":
    main()