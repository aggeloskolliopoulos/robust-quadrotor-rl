from pathlib import Path

import numpy as np
import torch

from quadrotor_rl.agents.sac_agent import SACAgent
from quadrotor_rl.envs.quadrotor_tracking_env import (
    QuadrotorTrackingEnv,
)


MODEL_PATH = Path(
    "results/models/custom_sac_hover_smoke.pt"
)

EVAL_SEEDS = range(1000, 1010)


def main() -> None:

    device = "cpu"

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    config = checkpoint["config"]

    env = QuadrotorTrackingEnv(
        gui=False,
    )

    agent = SACAgent(
        observation_dim=config["observation_dim"],
        action_dim=config["action_dim"],
        hidden_dim=config["hidden_dim"],
        gamma=config["gamma"],
        tau=config["tau"],
        device=device,
    )

    agent.actor.load_state_dict(
        checkpoint["actor_state_dict"]
    )

    agent.actor.eval()

    episode_returns = []
    episode_lengths = []
    episode_rmses = []
    episode_max_errors = []
    successes = []

    for seed in EVAL_SEEDS:

        observation, _ = env.reset(
            seed=seed
        )

        episode_return = 0.0
        position_errors = []

        terminated = False
        truncated = False

        step = 0

        while not (terminated or truncated):

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
            ) = env.step(
                action
            )

            position_error = observation[0:3]

            position_errors.append(
                position_error.copy()
            )

            episode_return += reward
            step += 1

        position_errors = np.asarray(
            position_errors
        )

        error_norms = np.linalg.norm(
            position_errors,
            axis=1,
        )

        rmse = np.sqrt(
            np.mean(
                np.sum(
                    position_errors ** 2,
                    axis=1,
                )
            )
        )

        max_error = np.max(
            error_norms
        )

        success = (
            truncated
            and not terminated
        )

        episode_returns.append(
            episode_return
        )

        episode_lengths.append(
            step
        )

        episode_rmses.append(
            rmse
        )

        episode_max_errors.append(
            max_error
        )

        successes.append(
            float(success)
        )

        print(
            f"Seed {seed}: "
            f"return={episode_return:.3f}, "
            f"length={step}, "
            f"rmse={rmse:.3f} m, "
            f"max_error={max_error:.3f} m, "
            f"success={success}"
        )

    print()
    print("=== Custom SAC 5k evaluation ===")

    print(
        f"Mean return: "
        f"{np.mean(episode_returns):.3f}"
    )

    print(
        f"Mean episode length: "
        f"{np.mean(episode_lengths):.1f}"
    )

    print(
        f"Mean duration: "
        f"{np.mean(episode_lengths) / env.control_freq_hz:.2f} s"
    )

    print(
        f"Mean position RMSE: "
        f"{np.mean(episode_rmses):.3f} m"
    )

    print(
        f"Mean max error: "
        f"{np.mean(episode_max_errors):.3f} m"
    )

    print(
        f"Success rate: "
        f"{100.0 * np.mean(successes):.1f}%"
    )

    env.close()


if __name__ == "__main__":
    main()