"""
Evaluate Conservative SAC Circle Fine-Tuning.

Compare:
    - Original Hover SAC 80k
    - Conservative FT +2.5k
    - Conservative FT +5k
    - Conservative FT +7.5k
    - Conservative FT +10k
    - Conservative FT +12.5k
    - Conservative FT +15k
    - Conservative FT +17.5k
    - Conservative FT +20k

Evaluation:
    - 5 angular speeds
    - 20 common validation seeds
    - Deterministic actions

Total:
    9 models * 5 speeds * 20 seeds = 900 episodes.
"""

import csv
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from quadrotor_rl.envs.circle_tracking_env import (
    CircleTrackingEnv,
)

# Both files are inside scripts/
from eval_sac_circle import evaluate_episode


# ============================================================
# 1. Configuration
# ============================================================

HOVER_MODEL_PATH = Path(
    "results/checkpoints/sac_hover_stage1/"
    "sac_hover_80000_steps.zip"
)

CONSERVATIVE_CHECKPOINT_DIR = Path(
    "results/checkpoints/sac_circle_conservative"
)

# The first 1000 steps were replay-buffer warm-up.
#
# SB3 checkpoint filenames include these 1000 steps.

PRECOLLECT_STEPS = 1000

CHECKPOINT_STEPS = [
    3500,
    6000,
    8500,
    11000,
    13500,
    16000,
    18500,
    21000,
]

# Build the model dictionary.

MODEL_PATHS = {
    "Hover_80k": HOVER_MODEL_PATH,
}

for step in CHECKPOINT_STEPS:

    finetune_steps = step - PRECOLLECT_STEPS

    model_name = f"FT_{finetune_steps}"

    model_path = (
        CONSERVATIVE_CHECKPOINT_DIR
        / f"sac_circle_safe_{step}_steps.zip"
    )

    MODEL_PATHS[model_name] = model_path


# Evaluation configuration.

ANGULAR_SPEEDS = [
    0.20,
    0.40,
    0.60,
    0.80,
    1.00,
]

EVAL_SEEDS = range(1200, 1220)

RADIUS_M = 0.30

CONTROL_FREQ_HZ = 48
PHYSICS_FREQ_HZ = 240


# Save to a NEW file.
#
# Do not overwrite the previous standard fine-tuning
# validation results.

OUTPUT_PATH = Path(
    "results/evaluation/"
    "circle_conservative_validation.csv"
)


# ============================================================
# 2. Metric helpers
# ============================================================

def mean_metric(results, key):
    """
    Calculate the mean of a metric.

    Returns NaN if the list is empty.
    """

    if not results:
        return float("nan")

    return float(
        np.mean([
            result[key]
            for result in results
        ])
    )


def mean_available(results, key):
    """
    Calculate mean while ignoring None values.

    Useful for metrics that are not available
    when an episode terminates early.
    """

    values = [
        result[key]
        for result in results
        if result[key] is not None
    ]

    if not values:
        return float("nan")

    return float(np.mean(values))


# ============================================================
# 3. Load models
# ============================================================

def load_models():

    models = {}

    print("Loading SAC models...\n")

    for name, path in MODEL_PATHS.items():

        if not path.is_file():

            raise FileNotFoundError(
                f"Model not found: {path}"
            )

        model = SAC.load(
            str(path),
            device="cpu",
        )

        models[name] = model

        print(f"Loaded: {name}")

    return models


# ============================================================
# 4. Evaluate one model at one angular speed
# ============================================================

def evaluate_model_at_speed(
    env,
    model,
    model_name,
    omega,
):

    results = []

    # All models use the same validation seeds.
    for seed in EVAL_SEEDS:

        result, _ = evaluate_episode(
            env,
            model,
            seed,
        )

        results.append(result)

    # --------------------------------------------------------
    # Split successful and failed episodes.
    # --------------------------------------------------------

    survivors = [
        result
        for result in results
        if result["survived"] == 1
    ]

    num_episodes = len(results)
    num_survivors = len(survivors)

    survival_rate = (
        100.0 * num_survivors / num_episodes
    )

    # --------------------------------------------------------
    # Position RMSE across all episodes.
    #
    # WARNING:
    # Failed episodes may terminate early.
    # Their RMSE is measured over a shorter interval.
    # --------------------------------------------------------

    full_rmse_all = mean_metric(
        results,
        "position_rmse_m",
    )

    # --------------------------------------------------------
    # Position RMSE over SURVIVING episodes only.
    #
    # All these episodes lasted the full 10 seconds.
    # --------------------------------------------------------

    full_rmse_survivors = mean_metric(
        survivors,
        "position_rmse_m",
    )

    # --------------------------------------------------------
    # Late RMSE: final 5 seconds.
    #
    # Use survivors only for fair comparison.
    # --------------------------------------------------------

    late_rmse_survivors = mean_available(
        survivors,
        "late_position_rmse_m",
    )

    # --------------------------------------------------------
    # Velocity tracking RMSE.
    # --------------------------------------------------------

    velocity_rmse_survivors = mean_metric(
        survivors,
        "velocity_rmse_m_s",
    )

    # --------------------------------------------------------
    # Episode duration and final position error.
    # --------------------------------------------------------

    mean_duration = mean_metric(
        results,
        "duration_s",
    )

    final_error_survivors = mean_metric(
        survivors,
        "final_position_error_m",
    )

    # --------------------------------------------------------
    # Construct summary.
    # --------------------------------------------------------

    summary = {
        "model": model_name,

        "omega_rad_s": omega,

        "num_episodes": num_episodes,

        "survived_episodes": num_survivors,

        "survival_rate_pct": survival_rate,

        "mean_duration_s": mean_duration,

        "full_rmse_all_m": full_rmse_all,

        "full_rmse_survivors_m":
            full_rmse_survivors,

        "late_rmse_survivors_m":
            late_rmse_survivors,

        "velocity_rmse_survivors_m_s":
            velocity_rmse_survivors,

        "final_error_survivors_m":
            final_error_survivors,
    }

    return summary


# ============================================================
# 5. Main evaluation
# ============================================================

def main():

    print()
    print("=" * 65)
    print("SAC Conservative Fine-Tuning Validation")
    print("=" * 65)

    print(f"Models: {len(MODEL_PATHS)}")
    print(f"Angular speeds: {ANGULAR_SPEEDS}")
    print(f"Seeds per configuration: {len(EVAL_SEEDS)}")

    total_episodes = (
        len(MODEL_PATHS)
        * len(ANGULAR_SPEEDS)
        * len(EVAL_SEEDS)
    )

    print(f"Total evaluation episodes: {total_episodes}")
    print()

    # --------------------------------------------------------
    # 1. Load the original and fine-tuned policies.
    # --------------------------------------------------------

    models = load_models()

    summary_rows = []

    # --------------------------------------------------------
    # 2. Evaluate at each angular speed.
    # --------------------------------------------------------

    for omega in ANGULAR_SPEEDS:

        print()
        print("=" * 75)
        print(f"Angular speed: {omega:.2f} rad/s")
        print("=" * 75)

        env = CircleTrackingEnv(
            gui=False,
            physics_freq_hz=PHYSICS_FREQ_HZ,
            control_freq_hz=CONTROL_FREQ_HZ,
            radius_m=RADIUS_M,
            angular_speed_rad_s=omega,
        )

        try:

            for model_name, model in models.items():

                summary = evaluate_model_at_speed(
                    env=env,
                    model=model,
                    model_name=model_name,
                    omega=omega,
                )

                summary_rows.append(summary)

                print(
                    f"{model_name:12s} | "
                    f"survival="
                    f"{summary['survival_rate_pct']:5.1f}% | "
                    f"RMSE="
                    f"{summary['full_rmse_survivors_m']:.4f}m | "
                    f"late="
                    f"{summary['late_rmse_survivors_m']:.4f}m | "
                    f"vel="
                    f"{summary['velocity_rmse_survivors_m_s']:.4f}m/s",
                    flush=True,
                )

        finally:
            env.close()

    # ========================================================
    # 6. Save CSV
    # ========================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(summary_rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    # ========================================================
    # 7. Aggregate metrics across angular speeds
    # ========================================================

    print()
    print("=" * 75)
    print("OVERALL COMPARISON")
    print("=" * 75)

    overall_rows = []

    for model_name in MODEL_PATHS:

        model_rows = [
            row
            for row in summary_rows
            if row["model"] == model_name
        ]

        # Each angular speed has the same number
        # of episodes, so survival can be pooled.

        total_survivors = sum(
            row["survived_episodes"]
            for row in model_rows
        )

        total_tested = sum(
            row["num_episodes"]
            for row in model_rows
        )

        overall_survival = (
            100.0 * total_survivors / total_tested
        )

        # Mean across the five angular speeds.
        #
        # Each speed receives equal weight.
        #
        # RMSE is conditional on survival.

        overall_rmse = float(
            np.nanmean([
                row["full_rmse_survivors_m"]
                for row in model_rows
            ])
        )

        overall_late_rmse = float(
            np.nanmean([
                row["late_rmse_survivors_m"]
                for row in model_rows
            ])
        )

        overall_velocity_rmse = float(
            np.nanmean([
                row["velocity_rmse_survivors_m_s"]
                for row in model_rows
            ])
        )

        overall_rows.append({
            "model": model_name,
            "survival": overall_survival,
            "rmse": overall_rmse,
            "late_rmse": overall_late_rmse,
            "velocity_rmse": overall_velocity_rmse,
        })

    for row in overall_rows:

        print(
            f"{row['model']:12s} | "
            f"survival={row['survival']:5.1f}% | "
            f"RMSE={row['rmse']:.4f}m | "
            f"late={row['late_rmse']:.4f}m | "
            f"vel={row['velocity_rmse']:.4f}m/s"
        )

    # ========================================================
    # 8. Final information
    # ========================================================

    print()
    print("Validation completed successfully.")
    print(f"Detailed results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()