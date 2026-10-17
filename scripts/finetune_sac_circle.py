from pathlib import Path

import torch

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import get_schedule_fn

from quadrotor_rl.envs.randomized_circle_tracking_env import (
    RandomizedCircleTrackingEnv,
)


# ============================================================
# Configuration
# ============================================================

SEED = 42

PRECOLLECT_STEPS = 1_000
FINETUNE_STEPS = 20_000
CHECKPOINT_INTERVAL = 2_500

LEARNING_RATE = 3e-5

HOVER_MODEL = Path(
    "results/checkpoints/sac_hover_stage1/"
    "sac_hover_80000_steps.zip"
)

CHECKPOINT_DIR = Path(
    "results/checkpoints/sac_circle_conservative"
)

FINAL_MODEL = Path(
    "results/models/sac_circle_conservative_20k.zip"
)


def main():

    if not HOVER_MODEL.exists():
        raise FileNotFoundError(HOVER_MODEL)

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FINAL_MODEL.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    env = RandomizedCircleTrackingEnv(
        gui=False,
        radius_m=0.30,
        omega_min=0.20,
        omega_max=1.00,
    )

    try:

        # ----------------------------------------------------
        # 1. Load original pretrained hover policy.
        #
        # Start with a NEW 50k-capacity replay buffer.
        # ----------------------------------------------------

        model = SAC.load(
            str(HOVER_MODEL),
            env=env,
            device="cpu",
            custom_objects={
                "buffer_size": 50_000,
            },
        )

        model.set_random_seed(SEED)

        assert model.replay_buffer.size() == 0

        # ----------------------------------------------------
        # 2. Reduce learning rate.
        #
        # Updating learning_rate alone is insufficient:
        # SB3 also uses lr_schedule to update optimizers.
        # ----------------------------------------------------

        model.learning_rate = LEARNING_RATE

        model.lr_schedule = get_schedule_fn(
            LEARNING_RATE
        )

        # ----------------------------------------------------
        # 3. Prepare policy-based replay warm-up.
        #
        # learning_starts=0:
        #   SAC collects actions from its trained policy
        #   instead of sampling random actions.
        #
        # gradient_steps=0:
        #   NO neural network updates during warm-up.
        # ----------------------------------------------------

        model.learning_starts = 0
        model.gradient_steps = 0

        print()
        print("=== Conservative SAC Fine-Tuning ===")
        print(f"Starting model: {HOVER_MODEL}")
        print(f"Loaded steps: {model.num_timesteps}")
        print(f"Learning rate: {LEARNING_RATE}")
        print(f"Warm-up transitions: {PRECOLLECT_STEPS}")
        print(f"Fine-tuning steps: {FINETUNE_STEPS}")
        print(
            f"Replay capacity: "
            f"{model.replay_buffer.buffer_size}"
        )
        print()

        # Save actor parameters to verify that they
        # remain unchanged during replay collection.

        actor_before = [
            p.detach().clone()
            for p in model.actor.parameters()
        ]

        # ----------------------------------------------------
        # 4. Collect policy experiences WITHOUT learning.
        # ----------------------------------------------------

        print("Phase 1: Policy-based replay collection")

        model.learn(
            total_timesteps=PRECOLLECT_STEPS,
            reset_num_timesteps=True,
            log_interval=20,
        )

        actor_unchanged = all(
            torch.equal(before, after)
            for before, after in zip(
                actor_before,
                model.actor.parameters(),
            )
        )

        assert actor_unchanged, (
            "Actor changed during no-gradient warm-up!"
        )

        print(
            f"Collected transitions: "
            f"{model.replay_buffer.size()}"
        )

        print(
            f"Actor unchanged: {actor_unchanged}"
        )

        # ----------------------------------------------------
        # 5. Enable gradient updates.
        # ----------------------------------------------------

        model.gradient_steps = 1

        callback = CheckpointCallback(
            save_freq=CHECKPOINT_INTERVAL,
            save_path=str(CHECKPOINT_DIR),
            name_prefix="sac_circle_safe",
            save_replay_buffer=True,
            verbose=2,
        )

        print()
        print("Phase 2: Low-learning-rate fine-tuning")
        print()

        # reset_num_timesteps=False:
        #   - keeps the populated replay buffer
        #   - preserves training timestep continuity
        #
        # FINETUNE_STEPS additional interactions are run
        # after the 1k warm-up interactions.

        model.learn(
            total_timesteps=FINETUNE_STEPS,
            reset_num_timesteps=False,
            callback=callback,
            log_interval=20,
        )

        # ----------------------------------------------------
        # 6. Save final model.
        # ----------------------------------------------------

        model.save(str(FINAL_MODEL))

        model.save_replay_buffer(
            str(
                FINAL_MODEL.with_name(
                    "sac_circle_conservative_20k_replay.pkl"
                )
            )
        )

        print()
        print("Conservative fine-tuning completed.")
        print(f"Final model: {FINAL_MODEL}")

    finally:
        env.close()


if __name__ == "__main__":
    main()