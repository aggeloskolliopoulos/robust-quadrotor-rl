
"""
Diagnose steady-state tracking differences between
Original Scratch SAC and Curriculum 100k.

No training. No checkpoint modifications.

Run:
    python -u scripts/diagnose_late_rmse.py
"""

from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC

from eval_radius_phase import PhaseCircleTrackingEnv


# ============================================================
# 1. Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATHS = {
    "Original": ROOT / (
        "results/checkpoints/"
        "sac_pos5_scratch_mixed_seed42_20261030_190357/"
        "checkpoints/sac_pos5_77000_steps.zip"
    ),

    "Curriculum": ROOT / (
        "results/checkpoints/"
        "pos5_radius_phase_curriculum_seed42_20261106_173230/"
        "checkpoints/sac_curriculum_102000_steps.zip"
    ),
}

EXISTING_RESULTS = ROOT / (
    "results/evaluation/"
    "curriculum_checkpoint_development_20261106_183253/"
    "episodes.csv"
)

SCENARIOS = (
    ("A_nominal", 0.30, 0.0, 0.6),
    ("B_phase", 0.30, math.pi / 2, 0.6),
    ("C_radius", 0.45, math.pi / 2, 0.6),
)

LATE_START_S = 5.0


# ============================================================
# 2. Select a common successful evaluation seed
# ============================================================

def survived_in_csv(value):
    return str(value).strip().lower() in ("true", "1")


def episode_key(name, radius, phase, omega, seed):
    return (
        name,
        round(float(radius), 4),
        round(float(phase), 6),
        round(float(omega), 4),
        int(seed),
    )


def choose_common_seed():
    """
    Select a previously evaluated seed where both SAC
    policies survived all three scenarios.
    """

    with EXISTING_RESULTS.open(
        newline="",
        encoding="utf-8",
    ) as stream:

        old_rows = list(csv.DictReader(stream))

    success = {
        episode_key(
            row["controller"],
            row["radius_m"],
            row["phase_rad"],
            row["omega_rad_s"],
            row["seed"],
        ): survived_in_csv(row["survived"])

        for row in old_rows
    }

    for seed in range(6000, 6020):

        if all(
            success.get(
                episode_key(
                    name,
                    radius,
                    phase,
                    omega,
                    seed,
                ),
                False,
            )
            for _, radius, phase, omega in SCENARIOS
            for name in (
                "Scratch_original",
                "Curriculum_100k",
            )
        ):
            return seed

    raise RuntimeError(
        "No seed survives all three scenarios "
        "for both SAC controllers."
    )


# ============================================================
# 3. Extract roll and pitch from quaternion
# ============================================================

def roll_pitch_degrees(q):
    """
    PyBullet quaternion order:
        [qx, qy, qz, qw]
    """

    x, y, z, w = map(float, q)

    roll = math.atan2(
        2 * (w * x + y * z),
        1 - 2 * (x * x + y * y),
    )

    pitch = math.asin(
        float(
            np.clip(
                2 * (w * y - z * x),
                -1,
                1,
            )
        )
    )

    return np.degrees([roll, pitch])


# ============================================================
# 4. Run one deterministic SAC evaluation episode
# ============================================================

def run_episode(model, radius, phase, omega, seed):

    env = PhaseCircleTrackingEnv(
        gui=False,
        physics_freq_hz=240,
        control_freq_hz=48,
        radius_m=radius,
        angular_speed_rad_s=omega,
        initial_phase_rad=phase,
    )

    keys = (
        "time",
        "position_error",
        "velocity_error",
        "actual_position",
        "target_position",
        "roll_pitch",
        "action",
        "rpm",
        "radial_error",
        "phase_error_deg",
    )

    trace = {
        key: []
        for key in keys
    }

    try:

        observation, _ = env.reset(seed=seed)

        center = np.asarray(
            env.circle_center,
            dtype=float,
        )

        hover_rpm = float(env.hover_rpm)

        terminated = False
        truncated = False

        for _ in range(env.max_steps):

            action, _ = model.predict(
                observation,
                deterministic=True,
            )

            action = np.asarray(
                action,
                dtype=np.float32,
            ).reshape(4)

            (
                observation,
                _,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            position_error = np.asarray(
                observation[0:3],
                dtype=float,
            )

            velocity_error = np.asarray(
                observation[3:6],
                dtype=float,
            )

            target = np.asarray(
                info["target_pos"],
                dtype=float,
            )

            # observation[0:3] = target - actual
            actual = target - position_error

            desired_angle = math.atan2(
                target[1] - center[1],
                target[0] - center[0],
            )

            actual_angle = math.atan2(
                actual[1] - center[1],
                actual[0] - center[0],
            )

            angle_difference = (
                actual_angle - desired_angle
            )

            phase_error = math.atan2(
                math.sin(angle_difference),
                math.cos(angle_difference),
            )

            radial_error = (
                np.linalg.norm(
                    actual[:2] - center[:2]
                )
                - radius
            )

            trace["time"].append(
                float(info["reference_time_s"])
            )

            trace["position_error"].append(
                position_error
            )

            trace["velocity_error"].append(
                velocity_error
            )

            trace["actual_position"].append(
                actual
            )

            trace["target_position"].append(
                target
            )

            trace["roll_pitch"].append(
                roll_pitch_degrees(
                    observation[9:13]
                )
            )

            trace["action"].append(
                action.astype(float)
            )

            trace["rpm"].append(
                np.asarray(
                    info["rpm"],
                    dtype=float,
                ).copy()
            )

            trace["radial_error"].append(
                float(radial_error)
            )

            trace["phase_error_deg"].append(
                math.degrees(phase_error)
            )

            if terminated or truncated:
                break

        # Convert lists into numpy arrays.

        for key in keys:
            trace[key] = np.asarray(
                trace[key],
                dtype=float,
            )

        trace["hover_rpm"] = hover_rpm

        trace["survived"] = bool(
            truncated and not terminated
        )

        trace["duration_s"] = float(
            trace["time"][-1]
        )

        return trace

    finally:
        env.close()


# ============================================================
# 5. Numerical diagnostics
# ============================================================

def rms(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    return float(
        np.sqrt(
            np.mean(values ** 2)
        )
    )


def summarize(trace):

    t = trace["time"]

    mask = t >= LATE_START_S

    print(
        f"Survived: {trace['survived']}, "
        f"duration={trace['duration_s']:.3f}s"
    )

    full_error = np.linalg.norm(
        trace["position_error"],
        axis=1,
    )

    print(
        f"Full position RMSE: "
        f"{100 * rms(full_error):.3f} cm"
    )

    if not np.any(mask):
        print(
            "No late window: episode ended before 5s."
        )
        return

    ep = trace["position_error"][mask]
    ev = trace["velocity_error"][mask]

    rpy = trace["roll_pitch"][mask]
    rpm = trace["rpm"][mask]

    hover_rpm = trace["hover_rpm"]

    # Position and velocity RMSE

    position_rmse = rms(
        np.linalg.norm(ep, axis=1)
    )

    velocity_rmse = rms(
        np.linalg.norm(ev, axis=1)
    )

    per_axis_rmse = np.sqrt(
        np.mean(ep ** 2, axis=0)
    )

    mean_error = np.mean(
        ep,
        axis=0,
    )

    # Fraction of squared position error
    # attributed to the altitude axis.

    z_contribution = (
        100
        * per_axis_rmse[2] ** 2
        / position_rmse ** 2
    )

    # Signed radial error

    radial_error = trace[
        "radial_error"
    ][mask]

    # Circular phase bias

    phase_angle = np.deg2rad(
        trace["phase_error_deg"][mask]
    )

    phase_bias = math.degrees(
        np.angle(
            np.mean(
                np.exp(1j * phase_angle)
            )
        )
    )

    # RPM-based action effort proxy

    relative_rpm = (
        (rpm - hover_rpm) / hover_rpm
    )

    effort = float(
        np.mean(
            np.sum(
                relative_rpm ** 2,
                axis=1,
            )
        )
    )

    # RPM smoothness proxy

    if len(rpm) > 1:

        rpm_differences = np.diff(
            rpm,
            axis=0,
        )

        smoothness = float(
            np.mean(
                np.sum(
                    (
                        rpm_differences / hover_rpm
                    ) ** 2,
                    axis=1,
                )
            )
        )

    else:
        smoothness = float("nan")

    print(
        f"Late position RMSE: "
        f"{100 * position_rmse:.3f} cm"
    )

    print(
        f"Late velocity RMSE: "
        f"{velocity_rmse:.4f} m/s"
    )

    print(
        "Late per-axis RMSE [x,y,z] cm:",
        np.round(
            100 * per_axis_rmse,
            3,
        ),
    )

    print(
        "Late mean position error [x,y,z] cm:",
        np.round(
            100 * mean_error,
            3,
        ),
    )

    print(
        f"Fraction of late squared error "
        f"due to z: {z_contribution:.1f}%"
    )

    print(
        f"Late mean radial error: "
        f"{100 * np.mean(radial_error):.3f} cm"
    )

    print(
        f"Late phase bias: "
        f"{phase_bias:.3f} degrees"
    )

    print(
        f"Late maximum |roll/pitch|: "
        f"{np.max(np.abs(rpy)):.2f} degrees"
    )

    # Scientific notation to avoid rounding to zero.

    print(
        f"Late RPM effort proxy: "
        f"{effort:.8e}"
    )

    print(
        f"Late RPM smoothness proxy: "
        f"{smoothness:.8e}"
    )


# ============================================================
# 6. Save the complete time-series data to CSV
# ============================================================

def save_trace_csv(path, trace):

    columns = [
        "time_s",
        "ex_m", "ey_m", "ez_m",
        "evx_m_s", "evy_m_s", "evz_m_s",
        "x_m", "y_m", "z_m",
        "target_x_m", "target_y_m", "target_z_m",
        "roll_deg", "pitch_deg",
        "radial_error_m", "phase_error_deg",
        "action_1", "action_2", "action_3", "action_4",
        "rpm_1", "rpm_2", "rpm_3", "rpm_4",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:

        writer = csv.writer(stream)

        writer.writerow(columns)

        for k in range(len(trace["time"])):

            writer.writerow(
                [trace["time"][k]]
                + list(
                    trace["position_error"][k]
                )
                + list(
                    trace["velocity_error"][k]
                )
                + list(
                    trace["actual_position"][k]
                )
                + list(
                    trace["target_position"][k]
                )
                + list(
                    trace["roll_pitch"][k]
                )
                + [
                    trace["radial_error"][k],
                    trace["phase_error_deg"][k],
                ]
                + list(
                    trace["action"][k]
                )
                + list(
                    trace["rpm"][k]
                )
            )


# ============================================================
# 7. Plot tracking diagnostics
# ============================================================

def save_plots(out, label, traces):

    # --------------------------------------------------------
    # Figure 1: Tracking diagnostics (3 rows x 2 columns)
    # --------------------------------------------------------

    fig_tracking, tracking_axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(13, 11),
        squeeze=False,
    )

    for name, tr in traces.items():

        t = tr["time"]
        p = tr["actual_position"]

        tracking_axes[0, 0].plot(
            t,
            100 * np.linalg.norm(
                tr["position_error"],
                axis=1,
            ),
            label=name,
        )

        tracking_axes[0, 1].plot(
            t,
            np.linalg.norm(
                tr["velocity_error"],
                axis=1,
            ),
            label=name,
        )

        tracking_axes[1, 0].plot(
            p[:, 0],
            p[:, 1],
            label=name,
        )

        tracking_axes[1, 1].plot(
            t,
            tr["roll_pitch"][:, 0],
            label=f"{name} roll",
        )

        tracking_axes[1, 1].plot(
            t,
            tr["roll_pitch"][:, 1],
            "--",
            label=f"{name} pitch",
        )

        tracking_axes[2, 0].plot(
            t,
            100 * tr["radial_error"],
            label=name,
        )

        tracking_axes[2, 1].plot(
            t,
            tr["phase_error_deg"],
            label=name,
        )

    # Plot desired trajectory.

    reference = next(iter(traces.values()))

    target = reference["target_position"]

    tracking_axes[1, 0].plot(
        target[:, 0],
        target[:, 1],
        "k--",
        label="Desired trajectory",
    )

    titles = (
        "Position error norm (cm)",
        "Velocity error norm (m/s)",
        "XY trajectory (m)",
        "Roll / pitch (degrees)",
        "Signed radial error (cm)",
        "Angular phase error (degrees)",
    )

    for panel, panel_title in zip(
        tracking_axes.flat,
        titles,
    ):

        panel.set_title(panel_title)
        panel.grid(alpha=0.3)
        panel.legend(fontsize=8)

    tracking_axes[1, 0].set_aspect(
        "equal",
        adjustable="box",
    )

    for r, c in (
        (0, 0),
        (0, 1),
        (1, 1),
        (2, 0),
        (2, 1),
    ):

        panel = tracking_axes[r, c]

        panel.axvspan(
            LATE_START_S,
            10,
            alpha=0.07,
        )

        panel.set_xlabel("Time (s)")

    fig_tracking.suptitle(label)
    fig_tracking.tight_layout()

    fig_tracking.savefig(
        out / f"{label}_tracking.png",
        dpi=150,
    )

    plt.close(fig_tracking)

    # --------------------------------------------------------
    # Figure 2: Normalized motor commands
    # --------------------------------------------------------

    # squeeze=False guarantees an ndarray with shape (4,1).

    fig_motors, motor_axes_grid = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(12, 9),
        sharex=True,
        squeeze=False,
    )

    # Explicitly create a list of individual Axes objects.

    motor_panels = [
        motor_axes_grid[i, 0]
        for i in range(4)
    ]

    for name, tr in traces.items():

        for motor_id, motor_panel in enumerate(
            motor_panels
        ):

            motor_panel.plot(
                tr["time"],
                tr["action"][:, motor_id],
                label=name,
            )

    for motor_id, motor_panel in enumerate(
        motor_panels
    ):

        motor_panel.set_ylabel(
            f"Motor {motor_id + 1}"
        )

        motor_panel.set_ylim(
            -1.05,
            1.05,
        )

        motor_panel.grid(alpha=0.3)
        motor_panel.legend(fontsize=8)

    motor_panels[-1].set_xlabel("Time (s)")

    fig_motors.suptitle(
        f"{label} - normalized motor commands"
    )

    fig_motors.tight_layout()

    fig_motors.savefig(
        out / f"{label}_motors.png",
        dpi=150,
    )

    plt.close(fig_motors)

    # --------------------------------------------------------
    # Figure 3: Altitude tracking
    # --------------------------------------------------------

    fig_z, altitude_panel = plt.subplots(
        figsize=(11, 4)
    )

    for name, tr in traces.items():

        altitude_panel.plot(
            tr["time"],
            tr["actual_position"][:, 2],
            label=f"{name} actual z",
        )

    reference = next(iter(traces.values()))

    altitude_panel.plot(
        reference["time"],
        reference["target_position"][:, 2],
        "k--",
        label="Target z",
    )

    altitude_panel.axvspan(
        LATE_START_S,
        10,
        alpha=0.07,
    )

    altitude_panel.set_xlabel("Time (s)")
    altitude_panel.set_ylabel("Altitude (m)")

    altitude_panel.set_title(
        f"{label} - altitude tracking"
    )

    altitude_panel.legend()
    altitude_panel.grid(alpha=0.3)

    fig_z.tight_layout()

    fig_z.savefig(
        out / f"{label}_altitude.png",
        dpi=150,
    )

    plt.close(fig_z)


# ============================================================
# 8. Main
# ============================================================

def main():

    # Verify checkpoints.

    for name, path in MODEL_PATHS.items():

        if not path.is_file():
            raise FileNotFoundError(
                f"{name}: {path}"
            )

    if not EXISTING_RESULTS.is_file():
        raise FileNotFoundError(
            EXISTING_RESULTS
        )

    # Reuse a known successful seed for diagnosis.

    seed = choose_common_seed()

    # Load frozen SAC policies.

    policies = {
        name: SAC.load(
            str(path),
            device="cpu",
        )
        for name, path in MODEL_PATHS.items()
    }

    # Unique output directory.

    output = (
        ROOT
        / "results/evaluation"
        / (
            "late_rmse_diagnostic_"
            + datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
        )
    )

    output.mkdir(
        parents=True,
        exist_ok=False,
    )

    print(
        f"Common successful seed: {seed}"
    )

    print(
        f"Output folder: {output}",
        flush=True,
    )

    # --------------------------------------------------------
    # Run all three scenarios.
    # --------------------------------------------------------

    for label, radius, phase, omega in SCENARIOS:

        print("\n" + "=" * 62)

        print(
            f"{label} | "
            f"R={radius:.2f} | "
            f"phase={phase/math.pi:.2f}pi | "
            f"omega={omega}"
        )

        print("=" * 62, flush=True)

        traces = {}

        for name, policy in policies.items():

            trace = run_episode(
                policy,
                radius,
                phase,
                omega,
                seed,
            )

            traces[name] = trace

            print(f"\n--- {name} ---")

            summarize(trace)

            save_trace_csv(
                output / f"{label}_{name}.csv",
                trace,
            )

        # Save plots only after collecting both policies.

        save_plots(
            output,
            label,
            traces,
        )

        print(
            f"[SAVED] Plots for {label}",
            flush=True,
        )

    print("\n[FINISHED] Diagnostic complete.")

    print(
        "Results:",
        output,
    )


if __name__ == "__main__":
    main()

