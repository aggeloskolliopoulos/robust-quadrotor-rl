from collections import deque
from pathlib import Path
import random

import numpy as np
import torch

from quadrotor_rl.agents.replay_buffer import ReplayBuffer
from quadrotor_rl.agents.sac_agent import SACAgent
from quadrotor_rl.envs.quadrotor_tracking_env import QuadrotorTrackingEnv


# =====================================================================
# Experiment configuration
# =====================================================================

SEED = 42

TOTAL_TIMESTEPS = 20_000
LEARNING_STARTS = 500

BUFFER_SIZE = 50_000
BATCH_SIZE = 256

HIDDEN_DIM = 256

GAMMA = 0.99
TAU = 0.005

ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
ALPHA_LR = 3e-4

INITIAL_ALPHA = 1.0

LOG_INTERVAL = 500
CHECKPOINT_INTERVAL = 5_000

CHECKPOINT_DIR = Path(
    "results/checkpoints/custom_sac_hover_stage1"
)

MODEL_PATH = Path(
    "results/models/custom_sac_hover_stage1_final.pt"
)


def main() -> None:

    # =================================================================
    # Reproducibility
    # =================================================================

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    rng = np.random.default_rng(SEED)

    device = "cpu"

    print(f"Device: {device}")
    print(f"Seed: {SEED}")

    # =================================================================
    # Environment
    # =================================================================

    env = QuadrotorTrackingEnv(
        gui=False,
    )

    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print(
        f"Observation dimension: {observation_dim}"
    )

    print(
        f"Action dimension: {action_dim}"
    )

    # =================================================================
    # Replay buffer
    # =================================================================

    replay_buffer = ReplayBuffer(
        observation_dim=observation_dim,
        action_dim=action_dim,
        capacity=BUFFER_SIZE,
        device=device,
        seed=SEED,
    )

    # =================================================================
    # SAC agent
    # =================================================================

    agent = SACAgent(
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_dim=HIDDEN_DIM,
        gamma=GAMMA,
        tau=TAU,
        initial_alpha=INITIAL_ALPHA,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR,
        alpha_lr=ALPHA_LR,
        device=device,
    )

    # =================================================================
    # Checkpoint helper
    # =================================================================

    def save_checkpoint(
        path: Path,
        step: int,
    ) -> None:
        """Save the current SAC training state."""

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            {
                # -----------------------------------------------------
                # Policy and critics
                # -----------------------------------------------------

                "actor_state_dict":
                    agent.actor.state_dict(),

                "critic1_state_dict":
                    agent.critic1.state_dict(),

                "critic2_state_dict":
                    agent.critic2.state_dict(),

                "target_critic1_state_dict":
                    agent.target_critic1.state_dict(),

                "target_critic2_state_dict":
                    agent.target_critic2.state_dict(),

                # -----------------------------------------------------
                # Optimizers
                # -----------------------------------------------------

                "actor_optimizer_state_dict":
                    agent.actor_optimizer.state_dict(),

                "critic1_optimizer_state_dict":
                    agent.critic1_optimizer.state_dict(),

                "critic2_optimizer_state_dict":
                    agent.critic2_optimizer.state_dict(),

                "alpha_optimizer_state_dict":
                    agent.alpha_optimizer.state_dict(),

                # -----------------------------------------------------
                # Entropy coefficient
                # -----------------------------------------------------

                "log_alpha":
                    agent.log_alpha.detach().cpu(),

                # -----------------------------------------------------
                # Training progress
                # -----------------------------------------------------

                "step":
                    step,

                # -----------------------------------------------------
                # Replay buffer
                #
                # SAC is off-policy, so the replay buffer is part of
                # the practical training state.
                # -----------------------------------------------------

                "replay_buffer": {

                    "observations":
                        replay_buffer.observations,

                    "actions":
                        replay_buffer.actions,

                    "rewards":
                        replay_buffer.rewards,

                    "next_observations":
                        replay_buffer.next_observations,

                    "terminated":
                        replay_buffer.terminated,

                    "position":
                        replay_buffer.position,

                    "size":
                        replay_buffer.size,

                    "rng_state":
                        replay_buffer.rng.bit_generator.state,
                },

                # -----------------------------------------------------
                # Other RNG state
                # -----------------------------------------------------

                "exploration_rng_state":
                    rng.bit_generator.state,

                "torch_rng_state":
                    torch.get_rng_state(),

                # -----------------------------------------------------
                # Experiment configuration
                # -----------------------------------------------------

                "config": {

                    "observation_dim":
                        observation_dim,

                    "action_dim":
                        action_dim,

                    "hidden_dim":
                        HIDDEN_DIM,

                    "gamma":
                        GAMMA,

                    "tau":
                        TAU,

                    "actor_lr":
                        ACTOR_LR,

                    "critic_lr":
                        CRITIC_LR,

                    "alpha_lr":
                        ALPHA_LR,

                    "initial_alpha":
                        INITIAL_ALPHA,

                    "buffer_size":
                        BUFFER_SIZE,

                    "batch_size":
                        BATCH_SIZE,

                    "learning_starts":
                        LEARNING_STARTS,

                    "seed":
                        SEED,
                },
            },
            path,
        )

    # =================================================================
    # Initial environment reset
    # =================================================================

    observation, _ = env.reset(
        seed=SEED,
    )

    episode_return = 0.0
    episode_length = 0
    episode_count = 0

    # Rolling episode statistics.
    recent_returns = deque(
        maxlen=20
    )

    recent_lengths = deque(
        maxlen=20
    )

    recent_successes = deque(
        maxlen=20
    )

    # Latest 100 SAC updates for smoother logging.
    recent_update_metrics = deque(
        maxlen=100
    )

    # =================================================================
    # Training
    # =================================================================

    try:

        for step in range(
            1,
            TOTAL_TIMESTEPS + 1,
        ):

            # ---------------------------------------------------------
            # 1. Select action
            #
            # First 500 steps:
            #     uniform random exploration
            #
            # Afterwards:
            #     stochastic SAC policy
            # ---------------------------------------------------------

            if step <= LEARNING_STARTS:

                action = rng.uniform(
                    low=-1.0,
                    high=1.0,
                    size=action_dim,
                ).astype(np.float32)

            else:

                action = agent.select_action(
                    observation,
                    deterministic=False,
                )

            # ---------------------------------------------------------
            # 2. Interact with environment
            # ---------------------------------------------------------

            (
                next_observation,
                reward,
                terminated,
                truncated,
                _,
            ) = env.step(
                action
            )

            # ---------------------------------------------------------
            # 3. Store transition
            #
            # IMPORTANT:
            #
            # Store terminated only.
            #
            # Time-limit truncation is not a true terminal state, so
            # the SAC Bellman target should still bootstrap.
            # ---------------------------------------------------------

            replay_buffer.add(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                terminated=terminated,
            )

            # ---------------------------------------------------------
            # 4. Advance episode state
            # ---------------------------------------------------------

            observation = next_observation

            episode_return += reward
            episode_length += 1

            # ---------------------------------------------------------
            # 5. SAC gradient update
            #
            # One gradient update per environment step.
            # ---------------------------------------------------------

            if (
                step >= LEARNING_STARTS
                and len(replay_buffer) >= BATCH_SIZE
            ):

                batch = replay_buffer.sample(
                    batch_size=BATCH_SIZE,
                )

                metrics = agent.update(
                    batch
                )

                recent_update_metrics.append(
                    metrics
                )

            # ---------------------------------------------------------
            # 6. Episode ended
            # ---------------------------------------------------------

            if terminated or truncated:

                episode_count += 1

                recent_returns.append(
                    episode_return
                )

                recent_lengths.append(
                    episode_length
                )

                # If we reached the 10 s time limit without true
                # failure, the hover episode is considered successful.
                success = (
                    truncated
                    and not terminated
                )

                recent_successes.append(
                    float(success)
                )

                observation, _ = env.reset()

                episode_return = 0.0
                episode_length = 0

            # ---------------------------------------------------------
            # 7. Periodic logging
            # ---------------------------------------------------------

            if step % LOG_INTERVAL == 0:

                if recent_returns:

                    mean_return = float(
                        np.mean(
                            recent_returns
                        )
                    )

                    mean_length = float(
                        np.mean(
                            recent_lengths
                        )
                    )

                    success_rate = float(
                        np.mean(
                            recent_successes
                        )
                    )

                else:

                    mean_return = float("nan")
                    mean_length = float("nan")
                    success_rate = float("nan")

                print()

                print(
                    f"Step {step}/{TOTAL_TIMESTEPS}"
                )

                print(
                    f"Episodes: {episode_count}"
                )

                print(
                    f"Replay size: {len(replay_buffer)}"
                )

                print(
                    f"Mean return (last 20): "
                    f"{mean_return:.3f}"
                )

                print(
                    f"Mean episode length (last 20): "
                    f"{mean_length:.1f}"
                )

                print(
                    f"Success rate (last 20): "
                    f"{100.0 * success_rate:.1f}%"
                )

                if recent_update_metrics:

                    critic1_loss = float(
                        np.mean(
                            [
                                m["critic1_loss"]
                                for m
                                in recent_update_metrics
                            ]
                        )
                    )

                    critic2_loss = float(
                        np.mean(
                            [
                                m["critic2_loss"]
                                for m
                                in recent_update_metrics
                            ]
                        )
                    )

                    actor_loss = float(
                        np.mean(
                            [
                                m["actor_loss"]
                                for m
                                in recent_update_metrics
                            ]
                        )
                    )

                    actor_q = float(
                        np.mean(
                            [
                                m["actor_q_mean"]
                                for m
                                in recent_update_metrics
                            ]
                        )
                    )

                    alpha = float(
                        np.mean(
                            [
                                m["alpha"]
                                for m
                                in recent_update_metrics
                            ]
                        )
                    )

                    log_prob = float(
                        np.mean(
                            [
                                m["actor_log_prob_mean"]
                                for m
                                in recent_update_metrics
                            ]
                        )
                    )

                    print(
                        f"Critic 1 loss: "
                        f"{critic1_loss:.4f}"
                    )

                    print(
                        f"Critic 2 loss: "
                        f"{critic2_loss:.4f}"
                    )

                    print(
                        f"Actor loss: "
                        f"{actor_loss:.4f}"
                    )

                    print(
                        f"Actor Q: "
                        f"{actor_q:.4f}"
                    )

                    print(
                        f"Alpha: "
                        f"{alpha:.4f}"
                    )

                    print(
                        f"Mean log-prob: "
                        f"{log_prob:.4f}"
                    )

                else:

                    print(
                        "Learning has not started yet."
                    )

            # ---------------------------------------------------------
            # 8. Periodic checkpointing
            # ---------------------------------------------------------

            if step % CHECKPOINT_INTERVAL == 0:

                checkpoint_path = (
                    CHECKPOINT_DIR
                    / f"custom_sac_hover_{step}_steps.pt"
                )

                save_checkpoint(
                    path=checkpoint_path,
                    step=step,
                )

                print(
                    f"Checkpoint saved to: "
                    f"{checkpoint_path}"
                )

        # =================================================================
        # Final checkpoint
        # =================================================================

        save_checkpoint(
            path=MODEL_PATH,
            step=TOTAL_TIMESTEPS,
        )

        print()

        print(
            "Training finished successfully."
        )

        print(
            f"Final model saved to: "
            f"{MODEL_PATH}"
        )

    finally:

        env.close()


if __name__ == "__main__":
    main()