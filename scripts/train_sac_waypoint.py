"""
Stage 2A: Train SB3 SAC for randomized waypoint regulation.

Each episode:
    - Random initial position and orientation.
    - Random target waypoint.
    - Target remains fixed throughout the episode.

Actions:
    Direct normalized commands for the four motors.

Training:
    SAC with automatic entropy tuning.
"""

from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

from quadrotor_rl.envs.waypoint_tracking_env import WaypointTrackingEnv


# ============================================================
# Experiment configuration
# ============================================================

SEED = 42

TOTAL_TIMESTEPS = 20_000
CHECKPOINT_FREQ = 5_000

LEARNING_STARTS = 500
BUFFER_SIZE = 50_000
BATCH_SIZE = 256

LEARNING_RATE = 3e-4
GAMMA = 0.99
TAU = 0.005

DEVICE = "cpu"

CHECKPOINT_DIR = Path(
    "results/checkpoints/sac_waypoint_stage2"
)

MODEL_DIR = Path(
    "results/models"
)

FINAL_MODEL_PATH = (
    MODEL_DIR / "sac_waypoint_stage2_final.zip"
)


def main():

    # --------------------------------------------------------
    # 1. Create output directories.
    # --------------------------------------------------------

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 2. Create waypoint environment.
    #
    # Same physics and control frequencies as hover:
    #
    #     Physics: 240 Hz
    #     Control: 48 Hz
    #
    # Only the task/reference changes.
    # --------------------------------------------------------

    env = WaypointTrackingEnv(
        gui=False,
        physics_freq_hz=240,
        control_freq_hz=48,
    )

    try:

        print("=== SB3 SAC Waypoint Training ===")
        print(f"Seed: {SEED}")
        print(f"Timesteps: {TOTAL_TIMESTEPS}")
        print(f"Device: {DEVICE}")

        print(
            "Observation dimension:",
            env.observation_space.shape[0],
        )

        print(
            "Action dimension:",
            env.action_space.shape[0],
        )

        # ----------------------------------------------------
        # 3. Initialize SAC.
        #
        # Architecture:
        #     Actor: MLP [256, 256]
        #     Critics: two Q-networks
        #
        # Automatic entropy tuning:
        #     ent_coef="auto"
        #
        # Replay buffer:
        #     50,000 transitions
        #
        # Learning starts after 500 steps.
        # ----------------------------------------------------

        model = SAC(
            policy="MlpPolicy",
            env=env,

            learning_rate=LEARNING_RATE,
            buffer_size=BUFFER_SIZE,
            learning_starts=LEARNING_STARTS,
            batch_size=BATCH_SIZE,

            gamma=GAMMA,
            tau=TAU,

            train_freq=1,
            gradient_steps=1,

            ent_coef="auto",

            policy_kwargs={
                "net_arch": [256, 256],
            },

            seed=SEED,
            device=DEVICE,
            verbose=1,
        )

        # ----------------------------------------------------
        # 4. Save checkpoints every 5,000 steps.
        #
        # These are model snapshots for evaluation.
        #
        # We don't save replay buffers here because this is
        # a development run focused on model selection.
        # ----------------------------------------------------

        checkpoint_callback = CheckpointCallback(
            save_freq=CHECKPOINT_FREQ,
            save_path=str(CHECKPOINT_DIR),
            name_prefix="sac_waypoint",
            save_replay_buffer=False,
            save_vecnormalize=False,
            verbose=2,
        )

        # ----------------------------------------------------
        # 5. Train SAC.
        # ----------------------------------------------------

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_callback,
            log_interval=10,
        )

        # ----------------------------------------------------
        # 6. Save final policy.
        # ----------------------------------------------------

        model.save(
            str(FINAL_MODEL_PATH)
        )

        print()
        print("Training finished successfully.")

        print(
            f"Final model saved to: {FINAL_MODEL_PATH}"
        )

    finally:

        env.close()


if __name__ == "__main__":
    main()