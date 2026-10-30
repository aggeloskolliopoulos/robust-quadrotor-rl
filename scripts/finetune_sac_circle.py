"""
Experiment C: Position-Focused SAC Reward Ablation.

Hypothesis:
    Increasing the position-error penalty improves
    circular trajectory tracking.

Only reward change:
    position weight: 1.0 -> 5.0

Training configuration:
    Start: Hover SAC 80k
    Tasks: 25% hover / 75% randomized circle
    Replay warm-up: 2,000 steps without learning
    Fine-tuning: 100,000 steps
    Learning rate: 3e-5

All existing models and environments remain unchanged.
"""

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import get_schedule_fn

from quadrotor_rl.envs.quadrotor_tracking_env import (
    QuadrotorTrackingEnv,
)
from quadrotor_rl.envs.randomized_circle_tracking_env import (
    RandomizedCircleTrackingEnv,
)


# ============================================================
# 1. Configuration
# ============================================================

SEED = 42

HOVER_PROBABILITY = 0.25

POSITION_WEIGHT_OLD = 1.0
POSITION_WEIGHT_NEW = 5.0

PRECOLLECT_STEPS = 2_000
TRAINING_STEPS = 100_000

LEARNING_RATE = 3e-5
BUFFER_SIZE = 200_000

CHECKPOINT_INTERVAL = 25_000

HOVER_MODEL = Path(
    "results/checkpoints/sac_hover_stage1/"
    "sac_hover_80000_steps.zip"
)

CHECKPOINT_DIR = Path(
    "results/checkpoints/sac_reward_position5"
)

FINAL_MODEL = Path(
    "results/models/sac_reward_position5_100k.zip"
)

REPLAY_PATH = Path(
    "results/models/sac_reward_position5_100k_replay.pkl"
)


# ============================================================
# 2. Modified reward
# ============================================================

class PositionFocusedRewardMixin:
    """
    Preserve the original environment reward.

    Change ONLY the position-error coefficient
    from 1.0 to 5.0.
    """

    def _compute_reward(
        self,
        observation,
        action,
        terminated,
    ):

        # Original reward, including the failure penalty.
        original_reward = super()._compute_reward(
            observation=observation,
            action=action,
            terminated=terminated,
        )

        observation = np.asarray(
            observation,
            dtype=np.float32,
        )

        position_error = observation[0:3]

        squared_position_error = float(
            np.sum(position_error ** 2)
        )

        additional_penalty = (
            POSITION_WEIGHT_NEW - POSITION_WEIGHT_OLD
        ) * squared_position_error

        return float(
            original_reward - additional_penalty
        )


class PositionHoverEnv(
    PositionFocusedRewardMixin,
    QuadrotorTrackingEnv,
):
    pass


class PositionCircleEnv(
    PositionFocusedRewardMixin,
    RandomizedCircleTrackingEnv,
):
    pass


# ============================================================
# 3. Mixed-task environment
# ============================================================

class MixedTrackingEnv(gym.Env):

    def __init__(self):

        super().__init__()

        self.hover_env = PositionHoverEnv(
            gui=False,
            physics_freq_hz=240,
            control_freq_hz=48,
        )

        self.circle_env = PositionCircleEnv(
            gui=False,
            physics_freq_hz=240,
            control_freq_hz=48,
            radius_m=0.30,
            omega_min=0.20,
            omega_max=1.00,
        )

        self.observation_space = (
            self.hover_env.observation_space
        )

        self.action_space = (
            self.hover_env.action_space
        )

        assert (
            self.hover_env.observation_space.shape
            == self.circle_env.observation_space.shape
        )

        assert (
            self.hover_env.action_space.shape
            == self.circle_env.action_space.shape
        )

        self.active_env = None
        self.active_task = None

        self.episode_counts = {
            "hover": 0,
            "circle": 0,
        }

        self.step_counts = {
            "hover": 0,
            "circle": 0,
        }

    def reset(self, *, seed=None, options=None):

        super().reset(seed=seed)

        if self.np_random.random() < HOVER_PROBABILITY:

            self.active_task = "hover"
            self.active_env = self.hover_env

        else:

            self.active_task = "circle"
            self.active_env = self.circle_env

        self.episode_counts[self.active_task] += 1

        observation, info = self.active_env.reset(
            seed=seed,
            options=options,
        )

        info = dict(info)
        info["task"] = self.active_task

        return observation, info

    def step(self, action):

        obs, reward, terminated, truncated, info = (
            self.active_env.step(action)
        )

        self.step_counts[self.active_task] += 1

        info = dict(info)
        info["task"] = self.active_task

        return obs, reward, terminated, truncated, info

    def close(self):

        self.hover_env.close()
        self.circle_env.close()


# ============================================================
# 4. Verify that ONLY the position penalty changed
# ============================================================

def verify_reward(env):

    zero_action = np.zeros(
        4,
        dtype=np.float32,
    )

    for task_name, task_env in [
        ("hover", env.hover_env),
        ("circle", env.circle_env),
    ]:

        observation, _ = task_env.reset(
            seed=SEED
        )

        for terminated in [False, True]:

            original_reward = (
                QuadrotorTrackingEnv._compute_reward(
                    task_env,
                    observation,
                    zero_action,
                    terminated,
                )
            )

            new_reward = task_env._compute_reward(
                observation,
                zero_action,
                terminated,
            )

            position_error = observation[0:3]

            expected_difference = (
                -(POSITION_WEIGHT_NEW - POSITION_WEIGHT_OLD)
                * np.sum(position_error ** 2)
            )

            actual_difference = (
                new_reward - original_reward
            )

            assert np.isclose(
                actual_difference,
                expected_difference,
                atol=1e-7,
            ), (
                f"Reward mismatch for {task_name}, "
                f"terminated={terminated}"
            )

    print("Reward sanity checks: PASSED")
    print("Only the position-error weight changed.")


# ============================================================
# 5. Main
# ============================================================

def main():

    if not HOVER_MODEL.is_file():
        raise FileNotFoundError(HOVER_MODEL)

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FINAL_MODEL.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    env = MixedTrackingEnv()

    try:

        # ----------------------------------------------------
        # Phase 0: Reward verification
        # ----------------------------------------------------

        verify_reward(env)

        # ----------------------------------------------------
        # Phase 1: Load pretrained Hover SAC
        # ----------------------------------------------------

        model = SAC.load(
            str(HOVER_MODEL),
            env=env,
            device="cpu",
            custom_objects={
                "buffer_size": BUFFER_SIZE,
            },
        )

        model.set_random_seed(SEED)

        model.learning_rate = LEARNING_RATE
        model.lr_schedule = get_schedule_fn(
            LEARNING_RATE
        )

        # Policy-based replay collection.
        # No gradient updates during warm-up.
        model.learning_starts = 0
        model.gradient_steps = 0

        print()
        print("=" * 60)
        print("POSITION-FOCUSED SAC TRAINING")
        print("=" * 60)

        print(f"Starting model: {HOVER_MODEL}")
        print(f"Old position weight: {POSITION_WEIGHT_OLD}")
        print(f"New position weight: {POSITION_WEIGHT_NEW}")
        print("Angular weight: 0.01 (unchanged)")
        print("Velocity weight: 0.1 (unchanged)")
        print("Action weight: 0.01 (unchanged)")
        print(f"Learning rate: {LEARNING_RATE}")
        print(f"Replay capacity: {model.replay_buffer.buffer_size}")
        print(f"Warm-up steps: {PRECOLLECT_STEPS}")
        print(f"Training steps: {TRAINING_STEPS}")
        print()

        # ----------------------------------------------------
        # Phase 2: Policy-based replay warm-up
        # ----------------------------------------------------

        actor_before = [
            parameter.detach().clone()
            for parameter in model.actor.parameters()
        ]

        print("Phase 1: Replay warm-up")

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
            "Actor changed during warm-up!"
        )

        print(
            "Replay transitions:",
            model.replay_buffer.size(),
        )

        print(
            "Actor unchanged:",
            actor_unchanged,
        )

        # ----------------------------------------------------
        # Phase 3: Actual SAC fine-tuning
        # ----------------------------------------------------

        model.gradient_steps = 1

        callback = CheckpointCallback(
            save_freq=CHECKPOINT_INTERVAL,
            save_path=str(CHECKPOINT_DIR),
            name_prefix="sac_pos5",
            save_replay_buffer=True,
            verbose=2,
        )

        print()
        print("Phase 2: Position-focused fine-tuning")
        print()

        model.learn(
            total_timesteps=TRAINING_STEPS,
            reset_num_timesteps=False,
            callback=callback,
            log_interval=20,
        )

        # ----------------------------------------------------
        # Phase 4: Save final state
        # ----------------------------------------------------

        model.save(
            str(FINAL_MODEL)
        )

        model.save_replay_buffer(
            str(REPLAY_PATH)
        )

        print()
        print("=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)

        print(f"Final model: {FINAL_MODEL}")
        print(f"Replay buffer: {REPLAY_PATH}")

        print()
        print("Episodes by task:")
        print(env.episode_counts)

        print()
        print("Environment steps by task:")
        print(env.step_counts)

    finally:

        env.close()


if __name__ == "__main__":
    main()