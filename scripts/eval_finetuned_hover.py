import numpy as np
from pathlib import Path
from stable_baselines3 import SAC

from quadrotor_rl.envs.quadrotor_tracking_env import (
    QuadrotorTrackingEnv,
)


MODELS = {
    "Hover_80k": (
        "results/checkpoints/sac_hover_stage1/"
        "sac_hover_80000_steps.zip"
    ),
    "FT_2500": (
        "results/checkpoints/sac_circle_conservative/"
        "sac_circle_safe_3500_steps.zip"
    ),
    "FT_20000": (
        "results/checkpoints/sac_circle_conservative/"
        "sac_circle_safe_21000_steps.zip"
    ),
}

SEEDS = range(1300, 1320)

CONTROL_FREQ_HZ = 48
MAX_STEPS = 480
LATE_START_STEP = 240


def evaluate_model(model, env):

    results = []

    for seed in SEEDS:

        obs, _ = env.reset(seed=seed)

        squared_errors = []
        terminated = False
        truncated = False

        for _ in range(MAX_STEPS):

            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            obs, reward, terminated, truncated, info = (
                env.step(action)
            )

            pos_error = np.asarray(
                obs[:3],
                dtype=np.float64,
            )

            squared_errors.append(
                float(np.sum(pos_error ** 2))
            )

            if terminated or truncated:
                break

        survived = (
            truncated
            and not terminated
            and len(squared_errors) == MAX_STEPS
        )

        results.append({
            "survived": bool(survived),
            "duration_s": (
                len(squared_errors) / CONTROL_FREQ_HZ
            ),
            "rmse_m": float(
                np.sqrt(np.mean(squared_errors))
            ),
            "late_rmse_m": (
                float(
                    np.sqrt(
                        np.mean(
                            squared_errors[LATE_START_STEP:]
                        )
                    )
                )
                if survived
                else None
            ),
        })

    return results


def main():

    env = QuadrotorTrackingEnv(
        gui=False,
        physics_freq_hz=240,
        control_freq_hz=CONTROL_FREQ_HZ,
    )

    try:

        print("\n=== Hover Retention Evaluation ===")
        print(f"Seeds: {len(SEEDS)}\n")

        for name, path in MODELS.items():

            if not Path(path).is_file():
                raise FileNotFoundError(path)

            model = SAC.load(
                path,
                device="cpu",
            )

            results = evaluate_model(model, env)

            survivors = [
                r for r in results
                if r["survived"]
            ]

            survival = (
                100 * len(survivors) / len(results)
            )

            duration = np.mean([
                r["duration_s"]
                for r in results
            ])

            if survivors:

                rmse = np.mean([
                    r["rmse_m"]
                    for r in survivors
                ])

                late_rmse = np.mean([
                    r["late_rmse_m"]
                    for r in survivors
                ])

            else:
                rmse = float("nan")
                late_rmse = float("nan")

            print(
                f"{name:12s} | "
                f"survival={survival:5.1f}% | "
                f"duration={duration:5.2f}s | "
                f"RMSE={rmse:.4f}m | "
                f"late_RMSE={late_rmse:.4f}m"
            )

    finally:
        env.close()


if __name__ == "__main__":
    main()