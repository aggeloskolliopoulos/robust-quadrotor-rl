import csv
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from quadrotor_rl.envs.circle_tracking_env import (
    CircleTrackingEnv,
)

from eval_sac_circle import evaluate_episode


MODEL_PATH = (
    "results/checkpoints/sac_hover_stage1/"
    "sac_hover_80000_steps.zip"
)

ANGULAR_SPEEDS = [0.2, 0.4, 0.6, 0.8, 1.0]
EVAL_SEEDS = range(1100, 1120)

RADIUS_M = 0.30

OUTPUT_PATH = Path(
    "results/evaluation/circle_speed_sweep.csv"
)


def main():

    model = SAC.load(
        MODEL_PATH,
        device="cpu",
    )

    summary_rows = []

    print("\n=== SAC Circle Angular-Speed Sweep ===")

    for omega in ANGULAR_SPEEDS:

        env = CircleTrackingEnv(
            gui=False,
            radius_m=RADIUS_M,
            angular_speed_rad_s=omega,
        )

        results = []

        try:
            for seed in EVAL_SEEDS:

                result, _ = evaluate_episode(
                    env,
                    model,
                    seed,
                )

                results.append(result)

        finally:
            env.close()

        def mean(key):
            return float(np.mean([
                r[key] for r in results
            ]))

        late_values = [
            r["late_position_rmse_m"]
            for r in results
            if r["late_position_rmse_m"] is not None
        ]

        late_rmse = (
            float(np.mean(late_values))
            if late_values
            else float("nan")
        )

        summary = {
            "omega_rad_s": omega,
            "speed_m_s": RADIUS_M * omega,
            "centripetal_acc_m_s2": RADIUS_M * omega**2,
            "survival_rate_pct": 100 * mean("survived"),
            "mean_duration_s": mean("duration_s"),
            "position_rmse_m": mean("position_rmse_m"),
            "velocity_rmse_m_s": mean("velocity_rmse_m_s"),
            "late_position_rmse_m": late_rmse,
            "late_episode_count": len(late_values),
            "mean_final_error_m": mean(
                "final_position_error_m"
            ),
        }

        summary_rows.append(summary)

        print(
            f"omega={omega:.2f} | "
            f"survival={summary['survival_rate_pct']:.1f}% | "
            f"RMSE={summary['position_rmse_m']:.4f}m | "
            f"late_RMSE={late_rmse:.4f}m | "
            f"velocity_RMSE="
            f"{summary['velocity_rmse_m_s']:.4f}m/s"
        )

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

    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()