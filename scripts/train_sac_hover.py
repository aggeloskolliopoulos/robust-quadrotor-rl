from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

from quadrotor_rl.envs.quadrotor_tracking_env import QuadrotorTrackingEnv


# ---------------------------------------------------------------------
# Stage-1 hover training configuration
# ---------------------------------------------------------------------
SEED = 42

TOTAL_TIMESTEPS = 100_000
CHECKPOINT_FREQ = 10_000

BUFFER_SIZE = 50_000
LEARNING_STARTS = 500


# ---------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------
RESULTS_DIR = Path("results")

CHECKPOINT_DIR = RESULTS_DIR / "checkpoints" / "sac_hover_stage1"
MODEL_DIR = RESULTS_DIR / "models"

FINAL_MODEL_PATH = MODEL_DIR / "sac_hover_stage1_final"


def main():
    # -----------------------------------------------------------------
    # Create output directories.
    # -----------------------------------------------------------------
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Training environment.
    #
    # Each reset samples the randomized initial position and attitude
    # that we defined in QuadrotorTrackingEnv.
    # -----------------------------------------------------------------
    env = QuadrotorTrackingEnv(gui=False)

    # -----------------------------------------------------------------
    # Checkpoint callback.
    #
    # With one environment, save_freq=10_000 means:
    #
    #   save model at approximately
    #   10k, 20k, ..., 100k environment timesteps.
    #
    # We save model weights only. The replay buffer is not needed for
    # evaluation of these checkpoints.
    # -----------------------------------------------------------------
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=str(CHECKPOINT_DIR),
        name_prefix="sac_hover",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=2,
    )

    # -----------------------------------------------------------------
    # SAC agent.
    #
    # We intentionally keep essentially the same setup as the smoke
    # test. At this stage we want to study the learning curve before
    # introducing hyperparameter tuning.
    #
    # Important SB3 defaults that remain unchanged include:
    #
    #   learning_rate = 3e-4
    #   batch_size     = 256
    #   gamma          = 0.99
    #   tau            = 0.005
    #   train_freq     = 1
    #   gradient_steps = 1
    #   ent_coef       = "auto"
    #
    # The explicit parameters below are the ones we have deliberately
    # chosen for this experiment.
    # -----------------------------------------------------------------
    model = SAC(
        policy="MlpPolicy",
        env=env,
        buffer_size=BUFFER_SIZE,
        learning_starts=LEARNING_STARTS,
        seed=SEED,
        device="cpu",
        verbose=1,
    )

    print("\nStarting Stage-1 SAC hover training...")
    print(f"Training seed:       {SEED}")
    print(f"Total timesteps:     {TOTAL_TIMESTEPS}")
    print(f"Checkpoint interval: {CHECKPOINT_FREQ}")
    print(f"Replay buffer size:  {BUFFER_SIZE}")
    print(f"Learning starts:     {LEARNING_STARTS}")
    print()

    try:
        # -------------------------------------------------------------
        # Main RL training loop.
        #
        # SB3 internally performs:
        #
        #   environment interaction
        #       -> replay buffer
        #       -> mini-batch sampling
        #       -> critic updates
        #       -> actor update
        #       -> entropy-temperature update
        #       -> target-network updates
        #
        # The callback periodically saves the policy parameters.
        # -------------------------------------------------------------
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_callback,
            progress_bar=False,
        )

        # -------------------------------------------------------------
        # Save one explicitly named final model as well.
        #
        # The 100k checkpoint should also exist, but this gives us a
        # convenient canonical path for the final Stage-1 model.
        # -------------------------------------------------------------
        model.save(str(FINAL_MODEL_PATH))

        print("\nStage-1 training completed.")
        print(f"Final model: {FINAL_MODEL_PATH}.zip")
        print(f"Checkpoints: {CHECKPOINT_DIR}")

    finally:
        env.close()


if __name__ == "__main__":
    main()