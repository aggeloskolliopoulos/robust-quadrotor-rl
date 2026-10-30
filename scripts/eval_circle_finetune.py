"""
Evaluate Long Mixed-Task SAC Training.

Compare:
    - Original Hover SAC 80k
    - Mixed-Task SAC +25k
    - Mixed-Task SAC +50k
    - Mixed-Task SAC +75k
    - Mixed-Task SAC +100k

Evaluation:
    - 5 angular speeds
    - 20 common validation seeds
    - Deterministic policy actions
    - Same simulator and initial conditions

Total:
    5 models * 5 speeds * 20 seeds = 500 episodes.
"""

import csv
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from quadrotor_rl.envs.circle_tracking_env import (
    CircleTrackingEnv,
)

# Both files are inside scripts/.
from eval_sac_circle import evaluate_episode


# ============================================================
# 1. Configuration
# ============================================================

MODEL_PATHS = {
    "Hover_80k": Path(
        "results/checkpoints/sac_hover_stage1/"
        "sac_hover_80000_steps.zip"
    ),

    "Mixed_25k": Path(
        "results/checkpoints/sac_mixed_tracking/"
        "sac_mixed_27000_steps.zip"
    ),

    "Mixed_50k": Path(
        "results/checkpoints/sac_mixed_tracking/"
        "sac_mixed_52000_steps.zip"
    ),

    "Mixed_75k": Path(
        "results/checkpoints/sac_mixed_tracking/"
        "sac_mixed_77000_steps.zip"
    ),

    "Mixed_100k": Path(
        "results/checkpoints/sac_mixed_tracking/"
        "sac_mixed_102000_steps.zip"
    ),
}


# ------------------------------------------------------------
# Evaluation configuration
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Output path
#
# Save results in a NEW CSV file.
# Do not overwrite previous fine-tuning results.
# ------------------------------------------------------------

OUTPUT_PATH = Path(
    "results/evaluation/"
    "circle_mixed_100k_validation.csv"
)


# ============================================================
# 2. Metric helpers
# ============================================================

def mean_metric(results, key):
    """
    Calculate mean of a metric.

    Returns NaN if no results are available.
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
# 3. Load all SAC models
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

    # --------------------------------------------------------
    # Evaluate using the SAME seeds for every model.
    # --------------------------------------------------------

    for seed in EVAL_SEEDS:

        result, _ = evaluate_episode(
            env,
            model,
            seed,
        )

        results.append(result)

    # --------------------------------------------------------
    # Separate successful and failed episodes.
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
    # Position RMSE across ALL episodes.
    #
    # Failed episodes may terminate early.
    # Therefore this metric must be interpreted carefully.
    # --------------------------------------------------------

    full_rmse_all = mean_metric(
        results,
        "position_rmse_m",
    )

    # --------------------------------------------------------
    # Position RMSE over successful episodes only.
    #
    # All survivors complete the full 10-second episode.
    # --------------------------------------------------------

    full_rmse_survivors = mean_metric(
        survivors,
        "position_rmse_m",
    )

    # --------------------------------------------------------
    # Late position RMSE: final 5 seconds.
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
# 5. Overall comparison helpers
# ============================================================

def overall_model_summary(summary_rows, model_name):
    """
    Aggregate performance across all five angular speeds.

    Survival is pooled across all evaluation episodes.

    Tracking metrics are averaged across angular speeds,
    giving each speed equal weight.

    RMSE metrics are conditional on survival.
    """

    model_rows = [
        row
        for row in summary_rows
        if row["model"] == model_name
    ]

    total_survivors = sum(
        row["survived_episodes"]
        for row in model_rows
    )

    total_episodes = sum(
        row["num_episodes"]
        for row in model_rows
    )

    overall_survival = (
        100.0 * total_survivors / total_episodes
    )

    def speed_mean(key):

        values = [
            row[key]
            for row in model_rows
            if np.isfinite(row[key])
        ]

        if not values:
            return float("nan")

        return float(np.mean(values))

    return {
        "model": model_name,

        "survival": overall_survival,

        "rmse": speed_mean(
            "full_rmse_survivors_m"
        ),

        "late_rmse": speed_mean(
            "late_rmse_survivors_m"
        ),

        "velocity_rmse": speed_mean(
            "velocity_rmse_survivors_m_s"
        ),

        "duration": speed_mean(
            "mean_duration_s"
        ),
    }


# ============================================================
# 6. Main evaluation
# ============================================================

def main():

    print()
    print("=" * 70)
    print("SAC Long Mixed-Task Training Validation")
    print("=" * 70)

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
    # 1. Load models.
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
                    f"duration="
                    f"{summary['mean_duration_s']:5.2f}s | "
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
    # 3. Save detailed validation results.
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
    # 4. Overall comparison across angular speeds.
    # ========================================================

    print()
    print("=" * 75)
    print("OVERALL COMPARISON")
    print("=" * 75)

    overall_rows = []

    for model_name in MODEL_PATHS:

        summary = overall_model_summary(
            summary_rows,
            model_name,
        )

        overall_rows.append(summary)

        print(
            f"{summary['model']:12s} | "
            f"survival={summary['survival']:5.1f}% | "
            f"duration={summary['duration']:5.2f}s | "
            f"RMSE={summary['rmse']:.4f}m | "
            f"late={summary['late_rmse']:.4f}m | "
            f"vel={summary['velocity_rmse']:.4f}m/s"
        )

    # ========================================================
    # 5. Identify the best checkpoint.
    #
    # For checkpoint selection:
    #   1. Prefer 100% survival.
    #   2. Minimize mean position RMSE.
    #
    # This is only DEVELOPMENT checkpoint selection.
    # Final testing must use independent seeds.
    # ========================================================

    reliable_models = [
        row
        for row in overall_rows
        if np.isclose(row["survival"], 100.0)
    ]

    if reliable_models:

        best_model = min(
            reliable_models,
            key=lambda row: row["rmse"],
        )

        print()
        print("=" * 75)
        print("BEST MODEL WITH 100% VALIDATION SURVIVAL")
        print("=" * 75)

        print(f"Model: {best_model['model']}")
        print(f"RMSE: {best_model['rmse']:.4f} m")
        print(f"Late RMSE: {best_model['late_rmse']:.4f} m")

    else:

        print()
        print(
            "No model achieved 100% validation survival."
        )

    # ========================================================
    # 6. Final information.
    # ========================================================

    print()
    print("Validation completed successfully.")
    print(f"Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()