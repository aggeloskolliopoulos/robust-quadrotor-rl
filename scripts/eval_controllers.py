"""Nominal, paired PID vs SAC benchmark on circular references."""

import csv
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from gym_pybullet_drones.control.DSLPIDControl import (
    DSLPIDControl,
)
from gym_pybullet_drones.utils.enums import DroneModel

from quadrotor_rl.envs.circle_tracking_env import (
    CircleTrackingEnv,
)


# ============================================================
# Configuration
# ============================================================

SAC_PATH = Path(
    "results/checkpoints/sac_hover_stage1/"
    "sac_hover_80000_steps.zip"
)

OUTPUT = Path(
    "results/evaluation/pid_vs_sac_circle.csv"
)

ANGULAR_SPEEDS = (0.2, 0.4, 0.6, 0.8, 1.0)

SEEDS = range(1200, 1220)

CONTROLLERS = ("SAC_Hover80k", "PID")

CONTROL_HZ = 48
MAX_STEPS = 480
LATE_START = 240


# ============================================================
# RPM -> normalized action
# ============================================================

def pid_rpm_to_action(rpm, env):
    """
    Inverse of QuadrotorTrackingEnv._action_to_rpm().

    Maps PID motor RPMs into [-1, 1]^4.
    """

    rpm = np.clip(
        np.asarray(rpm, dtype=np.float64).reshape(4),
        0.0,
        env.max_rpm,
    )

    action = np.where(
        rpm <= env.hover_rpm,
        rpm / env.hover_rpm - 1.0,
        (rpm - env.hover_rpm)
        / (env.max_rpm - env.hover_rpm),
    )

    return np.clip(
        action,
        -1.0,
        1.0,
    ).astype(np.float32)


# ============================================================
# PID control action
# ============================================================

def pid_action(pid, env):
    """
    Compute the PID command using the CURRENT state
    and the CURRENT reference.

    The environment then advances one control interval.
    """

    state = env.sim._getDroneStateVector(0)

    rpm, _, _ = pid.computeControl(
        control_timestep=1.0 / CONTROL_HZ,

        cur_pos=state[0:3],
        cur_quat=state[3:7],
        cur_vel=state[10:13],
        cur_ang_vel=state[13:16],

        target_pos=env.target_pos.copy(),
        target_vel=env.target_vel.copy(),

        target_rpy=np.zeros(3),
    )

    action = pid_rpm_to_action(rpm, env)

    commanded_rpm = np.clip(
        rpm,
        0.0,
        env.max_rpm,
    )

    return action, commanded_rpm


# ============================================================
# Evaluate one episode
# ============================================================

def run_episode(env, controller_name, sac, seed):

    obs, _ = env.reset(seed=seed)

    # Create a fresh PID for each episode.
    # This resets integral and derivative memory.

    pid = (
        DSLPIDControl(drone_model=DroneModel.CF2X)
        if controller_name == "PID"
        else None
    )

    pos_sq = []
    vel_sq = []
    effort_sq = []
    smooth_sq = []

    last_rpm = None

    terminated = False
    truncated = False

    for step in range(MAX_STEPS):

        # ----------------------------------------------------
        # Controller action
        # ----------------------------------------------------

        if pid is not None:

            action, commanded_rpm = pid_action(
                pid,
                env,
            )

        else:

            action, _ = sac.predict(
                obs,
                deterministic=True,
            )

            commanded_rpm = None

        # ----------------------------------------------------
        # SAME environment step for PID and SAC
        # ----------------------------------------------------

        obs, _, terminated, truncated, info = (
            env.step(action)
        )

        applied_rpm = np.asarray(
            info["rpm"],
            dtype=np.float64,
        )

        # Verify the PID inverse mapping.
        if commanded_rpm is not None and step == 0:

            assert np.allclose(
                applied_rpm,
                commanded_rpm,
                atol=0.1,
            ), (
                applied_rpm,
                commanded_rpm,
            )

        # ----------------------------------------------------
        # Tracking errors
        # ----------------------------------------------------

        position_error = np.asarray(
            obs[0:3],
            dtype=np.float64,
        )

        velocity_error = np.asarray(
            obs[3:6],
            dtype=np.float64,
        )

        pos_sq.append(
            float(np.sum(position_error ** 2))
        )

        vel_sq.append(
            float(np.sum(velocity_error ** 2))
        )

        # ----------------------------------------------------
        # Control effort
        #
        # RMS relative motor-RPM deviation from hover.
        # Both controllers use the same physical units.
        # ----------------------------------------------------

        relative_rpm = (
            (applied_rpm - env.hover_rpm)
            / env.hover_rpm
        )

        effort_sq.append(
            float(np.mean(relative_rpm ** 2))
        )

        # ----------------------------------------------------
        # Control smoothness
        #
        # RMS change in normalized motor RPM between steps.
        # ----------------------------------------------------

        if last_rpm is not None:

            relative_change = (
                (applied_rpm - last_rpm)
                / env.hover_rpm
            )

            smooth_sq.append(
                float(np.mean(relative_change ** 2))
            )

        last_rpm = applied_rpm

        if terminated or truncated:
            break

    # ========================================================
    # Episode metrics
    # ========================================================

    survived = bool(
        truncated
        and not terminated
        and len(pos_sq) == MAX_STEPS
    )

    return {
        "controller": controller_name,

        "seed": seed,

        "survived": int(survived),

        "duration_s":
            len(pos_sq) / CONTROL_HZ,

        "position_rmse_m":
            float(np.sqrt(np.mean(pos_sq))),

        "late_position_rmse_m":
            (
                float(
                    np.sqrt(
                        np.mean(pos_sq[LATE_START:])
                    )
                )
                if survived
                else None
            ),

        "velocity_rmse_m_s":
            float(np.sqrt(np.mean(vel_sq))),

        "rpm_effort_rms":
            float(np.sqrt(np.mean(effort_sq))),

        "rpm_step_change_rms":
            (
                float(np.sqrt(np.mean(smooth_sq)))
                if smooth_sq
                else 0.0
            ),
    }


# ============================================================
# Print results
# ============================================================

def print_summary(rows, name):

    survivors = [
        row
        for row in rows
        if row["survived"]
    ]

    def mean_of(key, selected):

        if not selected:
            return float("nan")

        return float(
            np.mean([
                row[key]
                for row in selected
            ])
        )

    survival_rate = (
        100.0 * len(survivors) / len(rows)
    )

    print(
        f"{name:13s} | "
        f"survival={survival_rate:5.1f}% | "
        f"duration={mean_of('duration_s', rows):5.2f}s | "
        f"RMSE={mean_of('position_rmse_m', survivors):.4f}m | "
        f"late={mean_of('late_position_rmse_m', survivors):.4f}m | "
        f"vel={mean_of('velocity_rmse_m_s', survivors):.4f}m/s | "
        f"effort={mean_of('rpm_effort_rms', survivors):.4f}"
    )


# ============================================================
# Main benchmark
# ============================================================

def main():

    if not SAC_PATH.is_file():
        raise FileNotFoundError(SAC_PATH)

    sac = SAC.load(
        str(SAC_PATH),
        device="cpu",
    )

    all_rows = []

    print()
    print("=== Nominal PID vs SAC Circle Benchmark ===")
    print(f"Seeds per configuration: {len(SEEDS)}")
    print(f"Angular speeds: {ANGULAR_SPEEDS}")

    for omega in ANGULAR_SPEEDS:

        print(
            f"\n=== Circle omega = {omega:.2f} rad/s ===",
            flush=True,
        )

        env = CircleTrackingEnv(
            gui=False,
            physics_freq_hz=240,
            control_freq_hz=CONTROL_HZ,
            radius_m=0.30,
            angular_speed_rad_s=omega,
        )

        try:

            for controller_name in CONTROLLERS:

                group = []

                for seed in SEEDS:

                    result = run_episode(
                        env,
                        controller_name,
                        sac,
                        seed,
                    )

                    result["omega_rad_s"] = omega

                    group.append(result)
                    all_rows.append(result)

                print_summary(
                    group,
                    controller_name,
                )

        finally:
            env.close()

    # ========================================================
    # Save detailed results
    # ========================================================

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(all_rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(all_rows)

    # ========================================================
    # Overall comparison
    # ========================================================

    print()
    print("=== All speeds combined ===")

    for controller_name in CONTROLLERS:

        controller_rows = [
            row
            for row in all_rows
            if row["controller"] == controller_name
        ]

        print_summary(
            controller_rows,
            controller_name,
        )

    print()
    print(f"Results saved to: {OUTPUT}")


if __name__ == "__main__":
    main()