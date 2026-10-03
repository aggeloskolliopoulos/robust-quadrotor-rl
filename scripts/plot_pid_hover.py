from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CSV_PATH = Path("results/pid_hover_transient.csv")


def main():
    # Load experiment data from CSV
    data = np.genfromtxt(
        CSV_PATH,
        delimiter=",",
        names=True,
    )

    time = data["time"]
    z = data["z"]
    target_z = data["target_z"]

    plots_dir = Path("results/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Altitude plot
    # ---------------------------------------------------------

    plt.figure()

    plt.plot(time, z, label="Actual z")
    plt.plot(time, target_z, "--", label="Target z")

    plt.xlabel("Time [s]")
    plt.ylabel("Altitude z [m]")
    plt.title("PID Hover Transient Response")
    plt.legend()
    plt.grid(True)

    altitude_plot_path = plots_dir / "pid_hover_altitude.png"

    plt.savefig(
        altitude_plot_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved plot to: {altitude_plot_path}")

    # ---------------------------------------------------------
    # Motor RPM plot
    # ---------------------------------------------------------

    rpm_1 = data["rpm_1"]
    rpm_2 = data["rpm_2"]
    rpm_3 = data["rpm_3"]
    rpm_4 = data["rpm_4"]

    plt.figure()

    plt.plot(time, rpm_1, label="Motor 1")
    plt.plot(time, rpm_2, label="Motor 2")
    plt.plot(time, rpm_3, label="Motor 3")
    plt.plot(time, rpm_4, label="Motor 4")

    plt.xlabel("Time [s]")
    plt.ylabel("Motor speed [RPM]")
    plt.title("PID Motor Commands")
    plt.legend()
    plt.grid(True)

    rpm_plot_path = plots_dir / "pid_hover_rpms.png"

    plt.savefig(
        rpm_plot_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved plot to: {rpm_plot_path}")


if __name__ == "__main__":
    main()