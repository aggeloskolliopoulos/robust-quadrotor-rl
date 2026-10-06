import csv
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from quadrotor_rl.envs.quadrotor_tracking_env import QuadrotorTrackingEnv


# ---------------------------------------------------------------------
# Selected Stage-1 model
#
# We selected the 80k checkpoint using the development/evaluation
# seeds 1000-1009.
# ---------------------------------------------------------------------
MODEL_PATH = Path(
    "results/checkpoints/sac_hover_stage1/"
    "sac_hover_80000_steps.zip"
)


# ---------------------------------------------------------------------
# Unseen Stage-1 test set
#
# These initial conditions have not been used for checkpoint selection.
# Once we inspect these results, we should NOT use these seeds for
# further hyperparameter tuning.
# ---------------------------------------------------------------------
TEST_SEEDS = list(range(2000, 2100))


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------
OUTPUT_DIR = Path("results/evaluation")

OUTPUT_CSV = (
    OUTPUT_DIR
    / "sac_hover_stage1_test_80k.csv"
)


def evaluate_episode(model, env, seed):
    """Evaluate one deterministic SAC episode."""

    obs, info = env.reset(seed=seed)

    terminated = False
    truncated = False

    episode_return = 0.0
    step_count = 0

    position_errors = []
    rpm_history = []

    while not (terminated or truncated):
        # -------------------------------------------------------------
        # Deterministic evaluation:
        # no exploration sampling during testing.
        # -------------------------------------------------------------
        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        obs, reward, terminated, truncated, info = env.step(action)

        episode_return += reward
        step_count += 1

        # Position error = first 3 observation components.
        position_errors.append(
            obs[0:3].copy()
        )

        # Store the ACTUAL motor commands applied to the simulator.
        rpm_history.append(
            info["rpm"].copy()
        )

    position_errors = np.asarray(
        position_errors,
        dtype=np.float32,
    )

    rpm_history = np.asarray(
        rpm_history,
        dtype=np.float32,
    )

    # -----------------------------------------------------------------
    # Position tracking metrics
    # -----------------------------------------------------------------
    error_norms = np.linalg.norm(
        position_errors,
        axis=1,
    )

    position_rmse = np.sqrt(
        np.mean(error_norms**2)
    )

    rmse_x = np.sqrt(
        np.mean(position_errors[:, 0] ** 2)
    )

    rmse_y = np.sqrt(
        np.mean(position_errors[:, 1] ** 2)
    )

    rmse_z = np.sqrt(
        np.mean(position_errors[:, 2] ** 2)
    )

    max_position_error = np.max(
        error_norms
    )

    # -----------------------------------------------------------------
    # Control effort
    #
    # Normalize every motor RPM relative to nominal hover RPM.
    # -----------------------------------------------------------------
    normalized_rpm_deviation = (
        rpm_history - env.hover_rpm
    ) / env.hover_rpm

    control_effort = np.mean(
        np.sum(
            normalized_rpm_deviation**2,
            axis=1,
        )
    )

    # -----------------------------------------------------------------
    # Action smoothness
    #
    # Difference between consecutive applied RPM commands,
    # normalized by hover RPM.
    # -----------------------------------------------------------------
    if len(rpm_history) > 1:
        delta_rpm = np.diff(
            rpm_history,
            axis=0,
        )

        normalized_delta_rpm = (
            delta_rpm / env.hover_rpm
        )

        action_smoothness = np.mean(
            np.sum(
                normalized_delta_rpm**2,
                axis=1,
            )
        )
    else:
        action_smoothness = 0.0

    full_episode = (
        truncated and not terminated
    )

    return {
        "seed": seed,
        "return": float(episode_return),
        "length": step_count,
        "time": step_count / env.control_freq_hz,
        "rmse_position": float(position_rmse),
        "rmse_x": float(rmse_x),
        "rmse_y": float(rmse_y),
        "rmse_z": float(rmse_z),
        "max_position_error": float(max_position_error),
        "control_effort": float(control_effort),
        "action_smoothness": float(action_smoothness),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "full_episode": bool(full_episode),
    }


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    # -----------------------------------------------------------------
    # Load the already trained SAC policy.
    #
    # No environment is needed inside SAC.load() because we only use
    # model.predict() during evaluation.
    # -----------------------------------------------------------------
    model = SAC.load(
        str(MODEL_PATH),
        device="cpu",
    )

    env = QuadrotorTrackingEnv(
        gui=False,
    )

    results = []

    print("\nStage-1 SAC unseen test")
    print("=======================")
    print(f"Model: {MODEL_PATH}")
    print(
        f"Test seeds: "
        f"{TEST_SEEDS[0]}-{TEST_SEEDS[-1]}"
    )
    print(
        f"Number of episodes: {len(TEST_SEEDS)}\n"
    )

    try:
        for index, seed in enumerate(
            TEST_SEEDS,
            start=1,
        ):
            result = evaluate_episode(
                model=model,
                env=env,
                seed=seed,
            )

            results.append(result)

            # Print every 10 episodes so the terminal output
            # remains readable.
            if index % 10 == 0:
                print(
                    f"Completed {index:3d}/"
                    f"{len(TEST_SEEDS)} episodes"
                )

        # -------------------------------------------------------------
        # Aggregate statistics
        # -------------------------------------------------------------
        returns = np.array(
            [r["return"] for r in results]
        )

        durations = np.array(
            [r["time"] for r in results]
        )

        rmses = np.array(
            [r["rmse_position"] for r in results]
        )

        max_errors = np.array(
            [r["max_position_error"] for r in results]
        )

        efforts = np.array(
            [r["control_effort"] for r in results]
        )

        smoothness = np.array(
            [r["action_smoothness"] for r in results]
        )

        successes = np.array(
            [r["full_episode"] for r in results],
            dtype=np.float32,
        )

        print("\nTest summary")
        print("--------------------------------------")
        print(
            f"Success rate:       "
            f"{100 * np.mean(successes):.1f}%"
        )

        print(
            f"Mean duration:      "
            f"{np.mean(durations):.3f} s"
        )

        print(
            f"Position RMSE:      "
            f"{np.mean(rmses):.4f} "
            f"± {np.std(rmses):.4f} m"
        )

        print(
            f"Mean max error:     "
            f"{np.mean(max_errors):.4f} m"
        )

        print(
            f"Mean return:        "
            f"{np.mean(returns):.3f}"
        )

        print(
            f"Control effort:     "
            f"{np.mean(efforts):.6f}"
        )

        print(
            f"Action smoothness:  "
            f"{np.mean(smoothness):.6f}"
        )

        # -------------------------------------------------------------
        # Per-axis RMSE averaged across episodes
        # -------------------------------------------------------------
        print(
            f"RMSE x:             "
            f"{np.mean([r['rmse_x'] for r in results]):.4f} m"
        )

        print(
            f"RMSE y:             "
            f"{np.mean([r['rmse_y'] for r in results]):.4f} m"
        )

        print(
            f"RMSE z:             "
            f"{np.mean([r['rmse_z'] for r in results]):.4f} m"
        )

        # -------------------------------------------------------------
        # Save all 100 individual episode results.
        # -------------------------------------------------------------
        fieldnames = [
            "seed",
            "return",
            "length",
            "time",
            "rmse_position",
            "rmse_x",
            "rmse_y",
            "rmse_z",
            "max_position_error",
            "control_effort",
            "action_smoothness",
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

        print(
            f"\nPer-episode results saved to:"
            f"\n{OUTPUT_CSV}"
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()