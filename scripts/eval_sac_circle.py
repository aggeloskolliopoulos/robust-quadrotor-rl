"""Zero-shot circular tracking evaluation for the trained SB3 hover SAC."""

import csv
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from quadrotor_rl.envs.circle_tracking_env import CircleTrackingEnv


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = Path(
    "results/checkpoints/sac_hover_stage1/sac_hover_80000_steps.zip"
)

EVAL_SEEDS = range(1100, 1110)

CONTROL_FREQ_HZ = 48
EPISODE_SECONDS = 10.0
LATE_START_SECONDS = 5.0

OUTPUT_DIR = Path("results/evaluation")

SUMMARY_CSV = (
    OUTPUT_DIR / "circle_zero_shot_episode_metrics.csv"
)

TRAJECTORY_CSV = (
    OUTPUT_DIR / "circle_zero_shot_seed_1100_trajectory.csv"
)


# ============================================================
# Evaluate one episode
# ============================================================

def evaluate_episode(env, model, seed):

    obs, _ = env.reset(seed=seed)

    position_errors = []
    velocity_errors = []
    rewards = []
    trajectory = []

    terminated = False
    truncated = False

    max_steps = round(
        EPISODE_SECONDS * CONTROL_FREQ_HZ
    )

    for step in range(1, max_steps + 1):

        # ----------------------------------------------------
        # 1. Deterministic SAC action
        # ----------------------------------------------------

        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        # ----------------------------------------------------
        # 2. Environment step
        # ----------------------------------------------------

        obs, reward, terminated, truncated, info = (
            env.step(action)
        )

        # ----------------------------------------------------
        # 3. Position and velocity tracking errors
        #
        # obs[0:3] = desired_pos - actual_pos
        # obs[3:6] = desired_vel - actual_vel
        # ----------------------------------------------------

        pos_error = np.asarray(
            obs[0:3],
            dtype=np.float64,
        ).copy()

        vel_error = np.asarray(
            obs[3:6],
            dtype=np.float64,
        ).copy()

        desired_pos = np.asarray(
            info["target_pos"],
            dtype=np.float64,
        )

        desired_vel = np.asarray(
            info["target_vel"],
            dtype=np.float64,
        )

        actual_pos = desired_pos - pos_error

        position_errors.append(pos_error)
        velocity_errors.append(vel_error)

        rewards.append(float(reward))

        # ----------------------------------------------------
        # 4. Store trajectory for the first seed
        #
        # We will use this CSV later to plot:
        #
        # Actual XY path vs Desired XY path
        # ----------------------------------------------------

        if seed == EVAL_SEEDS.start:

            trajectory.append({
                "time_s":
                    step / CONTROL_FREQ_HZ,

                "actual_x_m":
                    float(actual_pos[0]),

                "actual_y_m":
                    float(actual_pos[1]),

                "actual_z_m":
                    float(actual_pos[2]),

                "target_x_m":
                    float(desired_pos[0]),

                "target_y_m":
                    float(desired_pos[1]),

                "target_z_m":
                    float(desired_pos[2]),

                "target_vx_m_s":
                    float(desired_vel[0]),

                "target_vy_m_s":
                    float(desired_vel[1]),

                "position_error_m":
                    float(np.linalg.norm(pos_error)),

                "velocity_error_m_s":
                    float(np.linalg.norm(vel_error)),
            })

        if terminated or truncated:
            break

    # ========================================================
    # Compute metrics
    # ========================================================

    pos = np.asarray(position_errors)
    vel = np.asarray(velocity_errors)

    # 3D position tracking RMSE:
    #
    # sqrt(mean(ex^2 + ey^2 + ez^2))

    pos_rmse = float(
        np.sqrt(
            np.mean(np.sum(pos ** 2, axis=1))
        )
    )

    # 3D velocity tracking RMSE.

    vel_rmse = float(
        np.sqrt(
            np.mean(np.sum(vel ** 2, axis=1))
        )
    )

    # Position RMSE after the first five seconds.
    # This helps us distinguish transient and later tracking.

    late_start_step = round(
        LATE_START_SECONDS * CONTROL_FREQ_HZ
    )

    late_pos = pos[late_start_step:]

    late_rmse = (
        float(
            np.sqrt(
                np.mean(np.sum(late_pos ** 2, axis=1))
            )
        )
        if len(late_pos)
        else None
    )

    result = {
        "seed":
            seed,

        "survived":
            int(truncated and not terminated),

        "duration_s":
            len(pos) / CONTROL_FREQ_HZ,

        "position_rmse_m":
            pos_rmse,

        "velocity_rmse_m_s":
            vel_rmse,

        "late_position_rmse_m":
            late_rmse,

        "max_position_error_m":
            float(
                np.max(
                    np.linalg.norm(pos, axis=1)
                )
            ),

        "final_position_error_m":
            float(np.linalg.norm(pos[-1])),

        "return":
            float(np.sum(rewards)),
    }

    return result, trajectory


# ============================================================
# CSV helper
# ============================================================

def write_csv(path, rows):

    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Main
# ============================================================

def main():

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)

    # --------------------------------------------------------
    # 1. Load the previously trained hover SAC
    # --------------------------------------------------------

    model = SAC.load(
        str(MODEL_PATH),
        device="cpu",
    )

    # --------------------------------------------------------
    # 2. Create circle environment
    # --------------------------------------------------------

    env = CircleTrackingEnv(
        gui=False,
        physics_freq_hz=240,
        control_freq_hz=CONTROL_FREQ_HZ,
        radius_m=0.30,
        angular_speed_rad_s=0.40,
    )

    results = []
    first_trajectory = []

    try:

        # ----------------------------------------------------
        # 3. Evaluate on fixed seeds
        # ----------------------------------------------------

        for seed in EVAL_SEEDS:

            result, trajectory = evaluate_episode(
                env,
                model,
                seed,
            )

            results.append(result)

            if trajectory:
                first_trajectory = trajectory

            print(
                f"Seed {seed}: "
                f"duration={result['duration_s']:.2f}s, "
                f"RMSE={result['position_rmse_m']:.3f}m, "
                f"velocity RMSE="
                f"{result['velocity_rmse_m_s']:.3f}m/s, "
                f"survived={bool(result['survived'])}"
            )

    finally:
        env.close()

    # --------------------------------------------------------
    # 4. Save results
    # --------------------------------------------------------

    write_csv(
        SUMMARY_CSV,
        results,
    )

    write_csv(
        TRAJECTORY_CSV,
        first_trajectory,
    )

    # --------------------------------------------------------
    # 5. Aggregate metrics
    # --------------------------------------------------------

    def avg(key):

        values = [
            r[key]
            for r in results
            if r[key] is not None
        ]

        return (
            float(np.mean(values))
            if values
            else float("nan")
        )

    print()
    print("=== Hover SAC -> Circle Zero-Shot ===")

    print(f"Episodes: {len(results)}")

    print(
        f"Survival rate: "
        f"{100 * avg('survived'):.1f}%"
    )

    print(
        f"Mean duration: "
        f"{avg('duration_s'):.2f} s"
    )

    print(
        f"Position RMSE: "
        f"{avg('position_rmse_m'):.4f} m"
    )

    print(
        f"Velocity RMSE: "
        f"{avg('velocity_rmse_m_s'):.4f} m/s"
    )

    print(
        f"Position RMSE (last 5 s): "
        f"{avg('late_position_rmse_m'):.4f} m"
    )

    print(
        f"Mean max error: "
        f"{avg('max_position_error_m'):.4f} m"
    )

    print(
        f"Mean final error: "
        f"{avg('final_position_error_m'):.4f} m"
    )

    print(
        f"Mean return: "
        f"{avg('return'):.2f}"
    )

    print()
    print(f"Metrics CSV: {SUMMARY_CSV}")
    print(f"Trajectory CSV: {TRAJECTORY_CSV}")


if __name__ == "__main__":
    main()