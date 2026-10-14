"""Evaluate custom SAC hover checkpoints on fixed validation seeds."""

import csv
from pathlib import Path

import numpy as np
import torch

from quadrotor_rl.agents.sac_agent import SACAgent
from quadrotor_rl.envs.quadrotor_tracking_env import QuadrotorTrackingEnv


# ============================================================
# Configuration
# ============================================================

CHECKPOINT_DIR = Path(
    "results/checkpoints/custom_sac_hover_stage1"
)

CHECKPOINT_STEPS = (15_000,)

EVAL_SEEDS = range(2000, 2100)

OUTPUT_DIR = Path("results/evaluation")

EPISODES_CSV = (
    OUTPUT_DIR / "custom_sac_hover_stage1_test_15k_episodes.csv"
)

SUMMARY_CSV = (
    OUTPUT_DIR / "custom_sac_hover_stage1_test_15k_summary.csv"
)


# ============================================================
# Load a trained SAC checkpoint
# ============================================================

def load_agent(checkpoint_path: Path) -> SACAgent:

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    # These are our own trusted checkpoints.
    # They contain NumPy replay-buffer arrays as well as tensors.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    config = checkpoint["config"]

    agent = SACAgent(
        observation_dim=int(config["observation_dim"]),
        action_dim=int(config["action_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        gamma=float(config["gamma"]),
        tau=float(config["tau"]),
        initial_alpha=float(config["initial_alpha"]),
        device="cpu",
    )

    # Only the actor is needed for evaluation.
    agent.actor.load_state_dict(
        checkpoint["actor_state_dict"]
    )

    agent.actor.eval()

    return agent


# ============================================================
# Evaluate one episode
# ============================================================

def evaluate_episode(
    env: QuadrotorTrackingEnv,
    agent: SACAgent,
    seed: int,
) -> dict:

    observation, _ = env.reset(seed=seed)

    rewards = []
    squared_position_errors = []
    position_error_norms = []

    terminated = False
    truncated = False

    while not (terminated or truncated):

        # Deterministic evaluation:
        #
        # action = tanh(mean(s))
        #
        # No stochastic exploration noise.
        action = agent.select_action(
            observation,
            deterministic=True,
        )

        (
            observation,
            reward,
            terminated,
            truncated,
            _,
        ) = env.step(action)

        # The first three observation components are
        # the 3D position tracking error.
        position_error = observation[:3]

        squared_position_errors.append(
            float(np.sum(position_error ** 2))
        )

        position_error_norms.append(
            float(np.linalg.norm(position_error))
        )

        rewards.append(float(reward))

    episode_length = len(rewards)

    # 3D position RMSE:
    #
    # sqrt(mean(ex^2 + ey^2 + ez^2))
    rmse = float(
        np.sqrt(np.mean(squared_position_errors))
    )

    max_error = float(
        np.max(position_error_norms)
    )

    success = int(
        truncated and not terminated
    )

    return {
        "seed": seed,
        "return": float(np.sum(rewards)),
        "length": episode_length,
        "duration_s": episode_length / env.control_freq_hz,
        "rmse_m": rmse,
        "max_error_m": max_error,
        "success": success,
    }


# ============================================================
# Save results to CSV
# ============================================================

def write_csv(
    path: Path,
    rows: list[dict],
) -> None:

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
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Main evaluation
# ============================================================

def main() -> None:

    env = QuadrotorTrackingEnv(
        gui=False,
    )

    episode_rows = []
    summary_rows = []

    try:

        # Evaluate all four checkpoints.
        for steps in CHECKPOINT_STEPS:

            checkpoint_path = (
                CHECKPOINT_DIR
                / f"custom_sac_hover_{steps}_steps.pt"
            )

            agent = load_agent(
                checkpoint_path
            )

            results = []

            # Same validation seeds for every checkpoint.
            for seed in EVAL_SEEDS:

                result = evaluate_episode(
                    env=env,
                    agent=agent,
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
            # Aggregate results over the 10 validation episodes
            # ------------------------------------------------

            summary = {
                "checkpoint_steps": steps,

                "episodes": len(results),

                "success_rate_pct": 100.0 * np.mean(
                    [r["success"] for r in results]
                ),

                "mean_duration_s": float(np.mean(
                    [r["duration_s"] for r in results]
                )),

                "mean_length": float(np.mean(
                    [r["length"] for r in results]
                )),

                "mean_rmse_m": float(np.mean(
                    [r["rmse_m"] for r in results]
                )),

                "mean_max_error_m": float(np.mean(
                    [r["max_error_m"] for r in results]
                )),

                "mean_return": float(np.mean(
                    [r["return"] for r in results]
                )),
            }

            summary_rows.append(summary)

            print(
                f"{steps:>5} steps | "
                f"success={summary['success_rate_pct']:>5.1f}% | "
                f"duration={summary['mean_duration_s']:>5.2f}s | "
                f"RMSE={summary['mean_rmse_m']:.3f}m | "
                f"max error={summary['mean_max_error_m']:.3f}m | "
                f"return={summary['mean_return']:.2f}",
                flush=True,
            )

    finally:
        env.close()

    # ========================================================
    # Save results
    # ========================================================

    write_csv(
        EPISODES_CSV,
        episode_rows,
    )

    write_csv(
        SUMMARY_CSV,
        summary_rows,
    )

    print()
    print(f"Episode results: {EPISODES_CSV}")
    print(f"Summary results: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()