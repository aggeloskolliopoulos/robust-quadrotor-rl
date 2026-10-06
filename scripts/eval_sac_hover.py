import csv
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from quadrotor_rl.envs.quadrotor_tracking_env import QuadrotorTrackingEnv


# ---------------------------------------------------------------------
# Checkpoints to evaluate
# ---------------------------------------------------------------------
CHECKPOINT_STEPS = list(
    range(10_000, 100_001, 10_000)
)

CHECKPOINT_DIR = Path(
    "results/checkpoints/sac_hover_stage1"
)

# ---------------------------------------------------------------------
# Fixed evaluation initial conditions.
#
# These seeds are intentionally different from the training seed.
# Every checkpoint sees exactly the same ten initial conditions.
# ---------------------------------------------------------------------
EVAL_SEEDS = list(range(1000, 1010))


# ---------------------------------------------------------------------
# Evaluation output
# ---------------------------------------------------------------------
EVAL_DIR = Path("results/evaluation")

OUTPUT_CSV = EVAL_DIR / "sac_hover_stage1_episodes.csv"


def evaluate_episode(model, env, seed):
    """Evaluate one deterministic SAC episode."""

    obs, info = env.reset(seed=seed)

    terminated = False
    truncated = False

    episode_return = 0.0
    step_count = 0

    position_errors = []

    while not (terminated or truncated):
        # -------------------------------------------------------------
        # During evaluation we use the deterministic policy.
        #
        # Training uses stochastic SAC actions for exploration.
        # Evaluation asks:
        #
        #   "What does the learned controller itself do?"
        # -------------------------------------------------------------
        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        obs, reward, terminated, truncated, info = env.step(action)

        episode_return += reward
        step_count += 1

        # First three observation components are position error.
        position_errors.append(
            obs[0:3].copy()
        )

    position_errors = np.asarray(
        position_errors,
        dtype=np.float32,
    )

    # Euclidean 3D position error at each timestep.
    error_norms = np.linalg.norm(
        position_errors,
        axis=1,
    )

    # Tracking RMSE over the episode.
    position_rmse = np.sqrt(
        np.mean(error_norms**2)
    )

    # Largest tracking excursion during the episode.
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

    all_results = []

    print("\nStage-1 SAC checkpoint evaluation")
    print("=================================")
    print(f"Evaluation seeds: {EVAL_SEEDS}\n")

    try:
        for step in CHECKPOINT_STEPS:
            model_path = (
                CHECKPOINT_DIR
                / f"sac_hover_{step}_steps.zip"
            )

            # ---------------------------------------------------------
            # Make missing checkpoints obvious instead of crashing
            # with an unclear file-loading error.
            # ---------------------------------------------------------
            if not model_path.exists():
                print(
                    f"[WARNING] Missing checkpoint: {model_path}"
                )
                continue

            # ---------------------------------------------------------
            # We only need the trained policy for prediction, so an
            # environment does not need to be attached to SAC.load().
            # ---------------------------------------------------------
            model = SAC.load(
                str(model_path),
                device="cpu",
            )

            checkpoint_results = []

            print(f"\nCheckpoint: {step:,} steps")

            for seed in EVAL_SEEDS:
                result = evaluate_episode(
                    model=model,
                    env=env,
                    seed=seed,
                )

                result["training_steps"] = step

                checkpoint_results.append(result)
                all_results.append(result)

                print(
                    f"  Seed {seed}: "
                    f"time={result['time']:.2f}s, "
                    f"RMSE={result['rmse_position']:.3f}m, "
                    f"max_err={result['max_position_error']:.3f}m, "
                    f"return={result['return']:.2f}, "
                    f"success={result['full_episode']}"
                )

            # ---------------------------------------------------------
            # Aggregate metrics for this checkpoint.
            #
            # Each evaluation seed gets equal weight.
            # ---------------------------------------------------------
            mean_return = np.mean(
                [r["return"] for r in checkpoint_results]
            )

            mean_length = np.mean(
                [r["length"] for r in checkpoint_results]
            )

            mean_time = np.mean(
                [r["time"] for r in checkpoint_results]
            )

            mean_rmse = np.mean(
                [r["rmse_position"] for r in checkpoint_results]
            )

            mean_max_error = np.mean(
                [r["max_position_error"] for r in checkpoint_results]
            )

            success_rate = np.mean(
                [r["full_episode"] for r in checkpoint_results]
            )

            print("  --------------------------------")
            print(f"  Mean return:       {mean_return:.3f}")
            print(f"  Mean length:       {mean_length:.1f} steps")
            print(f"  Mean duration:     {mean_time:.2f}s")
            print(f"  Mean RMSE:         {mean_rmse:.3f} m")
            print(f"  Mean max error:    {mean_max_error:.3f} m")
            print(f"  10 s success rate: {100 * success_rate:.1f}%")

        # -------------------------------------------------------------
        # Save per-episode results.
        #
        # Keeping individual seed results instead of only aggregate
        # means lets us perform more detailed analysis later.
        # -------------------------------------------------------------
        fieldnames = [
            "training_steps",
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

            for result in all_results:
                writer.writerow(result)

        print("\n=================================")
        print("Evaluation completed.")
        print(f"Results saved to: {OUTPUT_CSV}")

    finally:
        env.close()


if __name__ == "__main__":
    main()