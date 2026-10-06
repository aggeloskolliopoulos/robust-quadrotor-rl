import csv
from pathlib import Path

import numpy as np

from quadrotor_rl.envs.quadrotor_tracking_env import QuadrotorTrackingEnv


# ---------------------------------------------------------------------
# Exactly the same evaluation initial conditions as SAC.
# ---------------------------------------------------------------------
EVAL_SEEDS = list(range(1000, 1010))


# ---------------------------------------------------------------------
# Evaluation output
# ---------------------------------------------------------------------
EVAL_DIR = Path("results/evaluation")

OUTPUT_CSV = EVAL_DIR / "zero_hover_stage1_episodes.csv"


def evaluate_episode(env, seed):
    """Evaluate one zero-action baseline episode."""

    obs, info = env.reset(seed=seed)

    terminated = False
    truncated = False

    episode_return = 0.0
    step_count = 0

    position_errors = []

    # -----------------------------------------------------------------
    # Zero normalized action corresponds to HOVER_RPM on every motor.
    #
    # This controller applies equal nominal hover thrust but performs
    # no active position or attitude correction.
    # -----------------------------------------------------------------
    action = np.zeros(
        4,
        dtype=np.float32,
    )

    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(action)

        episode_return += reward
        step_count += 1

        position_errors.append(
            obs[0:3].copy()
        )

    position_errors = np.asarray(
        position_errors,
        dtype=np.float32,
    )

    error_norms = np.linalg.norm(
        position_errors,
        axis=1,
    )

    position_rmse = np.sqrt(
        np.mean(error_norms**2)
    )

    max_position_error = np.max(
        error_norms
    )

    full_episode = (
        truncated and not terminated
    )

    return {
        "seed": seed,
        "return": float(episode_return),
        "length": step_count,
        "time": step_count / env.control_freq_hz,
        "rmse_position": float(position_rmse),
        "max_position_error": float(max_position_error),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "full_episode": bool(full_episode),
    }


def main():
    EVAL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    env = QuadrotorTrackingEnv(gui=False)

    results = []

    print("\nZero-action Stage-1 baseline")
    print("============================")
    print(f"Evaluation seeds: {EVAL_SEEDS}\n")

    try:
        for seed in EVAL_SEEDS:
            result = evaluate_episode(
                env=env,
                seed=seed,
            )

            results.append(result)

            print(
                f"Seed {seed}: "
                f"time={result['time']:.2f}s, "
                f"RMSE={result['rmse_position']:.3f}m, "
                f"max_err={result['max_position_error']:.3f}m, "
                f"return={result['return']:.2f}, "
                f"success={result['full_episode']}"
            )

        mean_return = np.mean(
            [r["return"] for r in results]
        )

        mean_length = np.mean(
            [r["length"] for r in results]
        )

        mean_time = np.mean(
            [r["time"] for r in results]
        )

        mean_rmse = np.mean(
            [r["rmse_position"] for r in results]
        )

        mean_max_error = np.mean(
            [r["max_position_error"] for r in results]
        )

        success_rate = np.mean(
            [r["full_episode"] for r in results]
        )

        print("\nSummary")
        print("----------------------------")
        print(f"Mean return:       {mean_return:.3f}")
        print(f"Mean length:       {mean_length:.1f} steps")
        print(f"Mean duration:     {mean_time:.2f}s")
        print(f"Mean RMSE:         {mean_rmse:.3f} m")
        print(f"Mean max error:    {mean_max_error:.3f} m")
        print(f"10 s success rate: {100 * success_rate:.1f}%")

        # -------------------------------------------------------------
        # Save individual episode results.
        # -------------------------------------------------------------
        fieldnames = [
            "seed",
            "return",
            "length",
            "time",
            "rmse_position",
            "max_position_error",
            "terminated",
            "truncated",
            "full_episode",
        ]

        with OUTPUT_CSV.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for result in results:
                writer.writerow(result)

        print(f"\nResults saved to: {OUTPUT_CSV}")

    finally:
        env.close()


if __name__ == "__main__":
    main()