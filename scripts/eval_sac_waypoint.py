"""
Evaluate SB3 SAC waypoint checkpoints.

Metrics:
    - Survival rate
    - Waypoint reach rate
    - Stable at episode end
    - Position RMSE
    - Maximum position error
    - Time to reach
    - Episode return

Evaluation uses deterministic actions.
"""

import csv
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from quadrotor_rl.envs.waypoint_tracking_env import (
    WaypointTrackingEnv,
)


# ============================================================
# Configuration
# ============================================================

CHECKPOINT_DIR = Path(
    "results/checkpoints/sac_waypoint_stage2"
)

CHECKPOINT_STEPS = (
    5_000,
    10_000,
    15_000,
    20_000,
)

EVAL_SEEDS = range(1000, 1010)

CONTROL_FREQ_HZ = 48
MAX_EPISODE_STEPS = 600

POSITION_TOLERANCE = 0.10
VELOCITY_TOLERANCE = 0.15
HOLD_SECONDS = 0.5

HOLD_STEPS = round(
    HOLD_SECONDS * CONTROL_FREQ_HZ
)

OUTPUT_DIR = Path(
    "results/evaluation"
)

EPISODES_CSV = (
    OUTPUT_DIR
    / "sac_waypoint_stage2_validation_episodes.csv"
)

SUMMARY_CSV = (
    OUTPUT_DIR
    / "sac_waypoint_stage2_validation_summary.csv"
)


# ============================================================
# Evaluate one episode
# ============================================================

def evaluate_episode(
    env,
    model,
    seed,
):

    observation, _ = env.reset(
        seed=seed
    )

    total_reward = 0.0

    position_errors = []

    stable_steps = 0
    reached = False
    time_to_reach = None

    terminated = False
    truncated = False

    episode_length = 0

    for step in range(
        1,
        MAX_EPISODE_STEPS + 1,
    ):

        # ----------------------------------------------------
        # 1. Deterministic SAC action
        # ----------------------------------------------------

        action, _ = model.predict(
            observation,
            deterministic=True,
        )

        # ----------------------------------------------------
        # 2. Environment step
        # ----------------------------------------------------

        (
            observation,
            reward,
            terminated,
            truncated,
            _,
        ) = env.step(action)

        episode_length = step
        total_reward += float(reward)

        # ----------------------------------------------------
        # 3. Position error
        #
        # Observation:
        #   [0:3] = desired_pos - current_pos
        # ----------------------------------------------------

        position_error = observation[0:3]

        distance = float(
            np.linalg.norm(position_error)
        )

        position_errors.append(distance)

        # ----------------------------------------------------
        # 4. Velocity magnitude
        #
        # Observation:
        #   [3:6] = desired_vel - current_vel
        #
        # For a stationary waypoint:
        #   desired_vel = 0
        #
        # Therefore:
        #   ||current_vel|| = ||velocity_error||
        # ----------------------------------------------------

        velocity_error = observation[3:6]

        speed = float(
            np.linalg.norm(velocity_error)
        )

        # ----------------------------------------------------
        # 5. Check waypoint stabilization
        # ----------------------------------------------------

        inside_tolerance = (
            distance < POSITION_TOLERANCE
            and speed < VELOCITY_TOLERANCE
        )

        if inside_tolerance:
            stable_steps += 1
        else:
            stable_steps = 0

        # Record the first successful 0.5-second hold.
        if (
            stable_steps >= HOLD_STEPS
            and not reached
        ):

            reached = True

            time_to_reach = (
                step / CONTROL_FREQ_HZ
            )

        # ----------------------------------------------------
        # 6. Episode termination
        # ----------------------------------------------------

        if terminated or truncated:
            break

    else:
        raise RuntimeError(
            f"Episode with seed {seed} exceeded "
            f"{MAX_EPISODE_STEPS} steps."
        )

    # ========================================================
    # Episode metrics
    # ========================================================

    errors = np.asarray(
        position_errors,
        dtype=np.float64,
    )

    rmse = float(
        np.sqrt(np.mean(errors ** 2))
    )

    max_error = float(
        np.max(errors)
    )

    survival = int(
        truncated and not terminated
    )

    stable_at_end = int(
        stable_steps >= HOLD_STEPS
        and survival == 1
    )

    return {
        "seed": seed,
        "survival": survival,
        "reached": int(reached),
        "stable_at_end": stable_at_end,
        "episode_length": episode_length,
        "duration_s": (
            episode_length / CONTROL_FREQ_HZ
        ),
        "rmse_m": rmse,
        "max_error_m": max_error,
        "final_error_m": float(errors[-1]),
        "time_to_reach_s": time_to_reach,
        "return": total_reward,
    }


# ============================================================
# CSV helper
# ============================================================

def save_csv(
    path,
    rows,
):

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
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Main
# ============================================================

def main():

    env = WaypointTrackingEnv(
        gui=False,
        physics_freq_hz=240,
        control_freq_hz=CONTROL_FREQ_HZ,
    )

    episode_rows = []
    summary_rows = []

    print()
    print("=== SAC Waypoint Checkpoint Evaluation ===")
    print(f"Validation episodes: {len(EVAL_SEEDS)}")
    print(f"Position tolerance: {POSITION_TOLERANCE} m")
    print(f"Velocity tolerance: {VELOCITY_TOLERANCE} m/s")
    print(f"Required hold: {HOLD_SECONDS} s")
    print()

    try:

        for steps in CHECKPOINT_STEPS:

            # ------------------------------------------------
            # 1. Load checkpoint
            # ------------------------------------------------

            checkpoint_path = (
                CHECKPOINT_DIR
                / f"sac_waypoint_{steps}_steps.zip"
            )

            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    checkpoint_path
                )

            model = SAC.load(
                str(checkpoint_path),
                device="cpu",
            )

            results = []

            # ------------------------------------------------
            # 2. Run evaluation episodes
            # ------------------------------------------------

            for seed in EVAL_SEEDS:

                result = evaluate_episode(
                    env=env,
                    model=model,
                    seed=seed,
                )

                results.append(result)

                episode_rows.append(
                    {
                        "checkpoint_steps": steps,
                        **result,
                    }
                )

            # ------------------------------------------------
            # 3. Aggregate metrics
            # ------------------------------------------------

            reach_times = [
                result["time_to_reach_s"]
                for result in results
                if result["time_to_reach_s"] is not None
            ]

            mean_reach_time = (
                float(np.mean(reach_times))
                if reach_times
                else None
            )

            summary = {
                "checkpoint_steps": steps,

                "episodes": len(results),

                "survival_rate_pct": float(
                    100 * np.mean([
                        r["survival"]
                        for r in results
                    ])
                ),

                "reach_rate_pct": float(
                    100 * np.mean([
                        r["reached"]
                        for r in results
                    ])
                ),

                "stable_at_end_pct": float(
                    100 * np.mean([
                        r["stable_at_end"]
                        for r in results
                    ])
                ),

                "mean_duration_s": float(
                    np.mean([
                        r["duration_s"]
                        for r in results
                    ])
                ),

                "mean_rmse_m": float(
                    np.mean([
                        r["rmse_m"]
                        for r in results
                    ])
                ),

                "mean_max_error_m": float(
                    np.mean([
                        r["max_error_m"]
                        for r in results
                    ])
                ),

                "mean_final_error_m": float(
                    np.mean([
                        r["final_error_m"]
                        for r in results
                    ])
                ),

                "mean_time_to_reach_s":
                    mean_reach_time,

                "mean_return": float(
                    np.mean([
                        r["return"]
                        for r in results
                    ])
                ),
            }

            summary_rows.append(summary)

            # ------------------------------------------------
            # 4. Print checkpoint results
            # ------------------------------------------------

            print(f"Checkpoint: {steps} steps")

            print(
                f"  Survival: "
                f"{summary['survival_rate_pct']:.1f}%"
            )

            print(
                f"  Waypoint reached: "
                f"{summary['reach_rate_pct']:.1f}%"
            )

            print(
                f"  Stable at end: "
                f"{summary['stable_at_end_pct']:.1f}%"
            )

            print(
                f"  Mean duration: "
                f"{summary['mean_duration_s']:.2f} s"
            )

            print(
                f"  Position RMSE: "
                f"{summary['mean_rmse_m']:.3f} m"
            )

            print(
                f"  Final position error: "
                f"{summary['mean_final_error_m']:.3f} m"
            )

            if mean_reach_time is not None:

                print(
                    f"  Time to reach: "
                    f"{mean_reach_time:.2f} s"
                )

            else:

                print(
                    "  Time to reach: N/A"
                )

            print()

    finally:
        env.close()

    # ========================================================
    # Save results
    # ========================================================

    save_csv(
        EPISODES_CSV,
        episode_rows,
    )

    save_csv(
        SUMMARY_CSV,
        summary_rows,
    )

    print("Evaluation completed.")

    print(
        f"Episode results: {EPISODES_CSV}"
    )

    print(
        f"Summary results: {SUMMARY_CSV}"
    )


if __name__ == "__main__":
    main()