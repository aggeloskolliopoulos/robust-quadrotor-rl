
"""
Generalized SAC tracking from the successful hover-200k checkpoint.

Single self-contained training script.

- No old tracking/curriculum weights loaded
- No old replay buffer loaded
- Pos5 reward unchanged
- Direct motor RPM control
- 25% hover, 75% randomized circles
- Random radius, phase, angular velocity
- 200k tracking steps after 2k replay warm-up

Usage:
    python -u scripts/train_generalized_from_hover200k.py --check-only
    python -u scripts/train_generalized_from_hover200k.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys

from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import get_schedule_fn


# ============================================================
# 1. Project imports
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from finetune_sac_circle import (
    PositionFocusedRewardMixin,
    PositionHoverEnv,
)

from quadrotor_rl.envs.circle_tracking_env import (
    CircleTrackingEnv,
)

from quadrotor_rl.envs.quadrotor_tracking_env import (
    QuadrotorTrackingEnv,
)


# ============================================================
# 2. Configuration
# ============================================================

DEFAULT_HOVER = ROOT / (
    "results/checkpoints/"
    "sac_pos5_scratch_hover_seed42_20261030_173955/"
    "sac_pos5_scratch_hover_200000_steps.zip"
)

LEARNING_RATE = 3e-5
REPLAY_CAPACITY = 200_000


# ============================================================
# 3. Randomized circle environment
# ============================================================

class GeneralizedCircleEnv(
    PositionFocusedRewardMixin,
    CircleTrackingEnv,
):
    """
    Circle tracking with randomized parameters.

    Every reset samples:
        radius: 0.15 - 0.45 m
        phase: 0 - 2*pi
        omega: 0.2 - 1.0 rad/s

    Reward: Pos5.
    Actions: 4 normalized direct motor RPM commands.
    """

    def __init__(self):

        super().__init__(
            gui=False,
            physics_freq_hz=240,
            control_freq_hz=48,
            radius_m=0.30,
            angular_speed_rad_s=0.60,
        )

        self.initial_phase_rad = 0.0

    def circle_reference(self, t):

        theta = (
            self.angular_speed_rad_s * float(t)
            + self.initial_phase_rad
        )

        r = self.radius_m
        w = self.angular_speed_rad_s

        target_pos = (
            np.asarray(
                self.circle_center,
                dtype=np.float32,
            )
            + np.array(
                [
                    r * math.cos(theta),
                    r * math.sin(theta),
                    0.0,
                ],
                dtype=np.float32,
            )
        )

        target_vel = np.array(
            [
                -r * w * math.sin(theta),
                r * w * math.cos(theta),
                0.0,
            ],
            dtype=np.float32,
        )

        return target_pos, target_vel

    def reset(self, *, seed=None, options=None):

        obs, info = super().reset(
            seed=seed,
            options=options,
        )

        old_pos = np.asarray(
            info["target_pos"],
            dtype=np.float32,
        ).copy()

        old_vel = np.asarray(
            info["target_vel"],
            dtype=np.float32,
        ).copy()

        # Randomize the new trajectory.

        self.radius_m = float(
            self.np_random.uniform(0.15, 0.45)
        )

        self.initial_phase_rad = float(
            self.np_random.uniform(0.0, 2 * math.pi)
        )

        self.angular_speed_rad_s = float(
            self.np_random.uniform(0.2, 1.0)
        )

        new_pos, new_vel = self.circle_reference(0.0)

        # Update observation for the new reference.
        # The actual physical drone state is unchanged.

        obs = np.asarray(
            obs,
            dtype=np.float32,
        ).copy()

        obs[0:3] += new_pos - old_pos
        obs[3:6] += new_vel - old_vel
        obs[6:9] = new_vel

        self.target_pos = new_pos.copy()
        self.target_vel = new_vel.copy()

        info = dict(info)

        info.update({
            "target_pos": new_pos.copy(),
            "target_vel": new_vel.copy(),
            "radius_m": self.radius_m,
            "phase_rad": self.initial_phase_rad,
            "omega_rad_s": self.angular_speed_rad_s,
        })

        return obs, info


# ============================================================
# 4. Mixed hover / circle environment
# ============================================================

class GeneralizedMixedEnv(gym.Env):
    """
    Sample the task at every episode reset:

        25% hover
        75% randomized circle

    The task probabilities refer to EPISODES,
    not necessarily to simulation steps.
    """

    def __init__(self):

        super().__init__()

        self.hover_env = PositionHoverEnv(
            gui=False,
            physics_freq_hz=240,
            control_freq_hz=48,
        )

        self.circle_env = GeneralizedCircleEnv()

        self.observation_space = (
            self.hover_env.observation_space
        )

        self.action_space = (
            self.hover_env.action_space
        )

        if (
            self.observation_space.shape != (16,)
            or self.action_space.shape != (4,)
        ):
            raise RuntimeError(
                "Expected 16D observations and 4D actions."
            )

        if (
            self.circle_env.observation_space.shape != (16,)
            or self.circle_env.action_space.shape != (4,)
        ):
            raise RuntimeError(
                "Hover and circle spaces are incompatible."
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

        self.success_counts = {
            "hover": 0,
            "circle": 0,
        }

        self.failure_counts = {
            "hover": 0,
            "circle": 0,
        }

    def reset(self, *, seed=None, options=None):

        super().reset(seed=seed)

        # Seed each subenvironment when a seed is supplied.
        # Subsequent resets continue its RNG sequence.

        if seed is not None:

            self.hover_env.reset(seed=seed + 101)
            self.circle_env.reset(seed=seed + 202)

        if self.np_random.random() < 0.25:

            self.active_task = "hover"
            self.active_env = self.hover_env

        else:

            self.active_task = "circle"
            self.active_env = self.circle_env

        self.episode_counts[self.active_task] += 1

        obs, info = self.active_env.reset(
            options=options,
        )

        info = dict(info)
        info["task"] = self.active_task

        return obs, info

    def step(self, action):

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = self.active_env.step(action)

        self.step_counts[self.active_task] += 1

        if terminated:

            self.failure_counts[self.active_task] += 1

        elif truncated:

            self.success_counts[self.active_task] += 1

        info = dict(info)
        info["task"] = self.active_task

        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )

    def close(self):

        self.hover_env.close()
        self.circle_env.close()


# ============================================================
# 5. Environment preflight
# ============================================================

def preflight(env):

    action = np.zeros(
        4,
        dtype=np.float32,
    )

    for name, subenv in (
        ("hover", env.hover_env),
        ("circle", env.circle_env),
    ):

        obs, _ = subenv.reset(seed=42)

        assert obs.shape == (16,)
        assert np.all(np.isfinite(obs))

        # Compare Pos5 with the original position weight 1.

        for terminated in (False, True):

            original_reward = float(
                QuadrotorTrackingEnv._compute_reward(
                    subenv,
                    obs,
                    action,
                    terminated,
                )
            )

            pos5_reward = float(
                subenv._compute_reward(
                    obs,
                    action,
                    terminated,
                )
            )

            expected_difference = (
                -4.0
                * float(
                    np.dot(
                        obs[:3],
                        obs[:3],
                    )
                )
            )

            if not np.isclose(
                pos5_reward - original_reward,
                expected_difference,
                atol=1e-6,
            ):

                raise RuntimeError(
                    f"Pos5 mismatch in {name}."
                )

    # Verify deterministic trajectory sampling.

    circle = env.circle_env

    obs1, info1 = circle.reset(seed=12345)

    parameters1 = (
        circle.radius_m,
        circle.initial_phase_rad,
        circle.angular_speed_rad_s,
    )

    obs2, info2 = circle.reset(seed=12345)

    parameters2 = (
        circle.radius_m,
        circle.initial_phase_rad,
        circle.angular_speed_rad_s,
    )

    np.testing.assert_allclose(obs1, obs2)
    np.testing.assert_allclose(
        info1["target_pos"],
        info2["target_pos"],
    )
    np.testing.assert_allclose(
        info1["target_vel"],
        info2["target_vel"],
    )

    if parameters1 != parameters2:

        raise RuntimeError(
            "Trajectory randomization is not reproducible."
        )

    reference_pos, reference_vel = (
        circle.circle_reference(0.0)
    )

    np.testing.assert_allclose(
        obs1[6:9],
        reference_vel,
        atol=1e-6,
    )

    np.testing.assert_allclose(
        info1["target_pos"],
        reference_pos,
        atol=1e-6,
    )

    if not (
        0.15 <= circle.radius_m <= 0.45
        and 0 <= circle.initial_phase_rad <= 2 * math.pi
        and 0.2 <= circle.angular_speed_rad_s <= 1.0
    ):
        raise RuntimeError(
            "Randomized trajectory parameters out of bounds."
        )

    print(
        "[PASS] Pos5 reward and randomized references.",
        flush=True,
    )


# ============================================================
# 6. Verify the starting hover checkpoint
# ============================================================

def check_hover_model(model):

    env = PositionHoverEnv(
        gui=False,
        physics_freq_hz=240,
        control_freq_hz=48,
    )

    successes = 0

    try:

        for seed in (1200, 1201, 1202):

            obs, _ = env.reset(seed=seed)

            success = False

            for _ in range(480):

                action, _ = model.predict(
                    obs,
                    deterministic=True,
                )

                (
                    obs,
                    _,
                    terminated,
                    truncated,
                    _,
                ) = env.step(action)

                if terminated or truncated:

                    success = bool(
                        truncated and not terminated
                    )

                    break

            successes += int(success)

            print(
                f"[HOVER {seed}] "
                f"{'PASS' if success else 'FAIL'}",
                flush=True,
            )

    finally:

        env.close()

    if successes < 2:

        raise RuntimeError(
            "Starting hover checkpoint failed "
            "the baseline test. Do not train."
        )


# ============================================================
# 7. Checkpoint callback
# ============================================================

class CheckpointCallback(BaseCallback):

    def __init__(
        self,
        out,
        env,
        warmup_steps,
        every,
    ):

        super().__init__()

        self.out = out
        self.train_env = env
        self.warmup_steps = warmup_steps
        self.every = every

        self.progress_file = (
            out / "checkpoint_progress.csv"
        )

        self.headers = [
            "tracking_steps",
            "model_timesteps",
            "hover_episodes",
            "circle_episodes",
            "hover_steps",
            "circle_steps",
            "hover_success",
            "circle_success",
            "hover_failures",
            "circle_failures",
            "checkpoint",
        ]

        with self.progress_file.open(
            "x",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=self.headers,
            )

            writer.writeheader()

    def _on_step(self):

        tracking_steps = (
            self.num_timesteps - self.warmup_steps
        )

        if (
            tracking_steps <= 0
            or tracking_steps % self.every != 0
        ):
            return True

        filename = (
            f"sac_gen_tracking_"
            f"{tracking_steps:07d}_steps.zip"
        )

        path = (
            self.out / "checkpoints" / filename
        )

        if path.exists():
            raise FileExistsError(path)

        self.model.save(str(path))

        env = self.train_env

        row = {
            "tracking_steps": tracking_steps,
            "model_timesteps": self.num_timesteps,

            "hover_episodes":
                env.episode_counts["hover"],

            "circle_episodes":
                env.episode_counts["circle"],

            "hover_steps":
                env.step_counts["hover"],

            "circle_steps":
                env.step_counts["circle"],

            "hover_success":
                env.success_counts["hover"],

            "circle_success":
                env.success_counts["circle"],

            "hover_failures":
                env.failure_counts["hover"],

            "circle_failures":
                env.failure_counts["circle"],

            "checkpoint": filename,
        }

        with self.progress_file.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=self.headers,
            )

            writer.writerow(row)

        print(
            f"\n[CHECKPOINT {tracking_steps:,}] "
            f"circle success={row['circle_success']} | "
            f"circle failures={row['circle_failures']}"
        )

        print(
            f"[SAVED] {filename}",
            flush=True,
        )

        return True


# ============================================================
# 8. Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--check-only",
        action="store_true",
    )

    parser.add_argument(
        "--hover-checkpoint",
        type=Path,
        default=DEFAULT_HOVER,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--tracking-steps",
        type=int,
        default=200_000,
    )

    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=2_000,
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25_000,
    )

    args = parser.parse_args()

    if min(
        args.tracking_steps,
        args.warmup_steps,
        args.checkpoint_every,
    ) <= 0:

        parser.error(
            "Steps and checkpoint interval must be positive."
        )

    if not args.hover_checkpoint.is_file():

        raise FileNotFoundError(
            f"Hover checkpoint not found: "
            f"{args.hover_checkpoint}"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = GeneralizedMixedEnv()

    try:

        preflight(env)

        # Load only the successful hover controller.

        model = SAC.load(
            str(args.hover_checkpoint),
            env=env,
            device="cpu",
            custom_objects={
                "buffer_size": REPLAY_CAPACITY,
            },
        )

        if (
            model.observation_space.shape != (16,)
            or model.action_space.shape != (4,)
        ):

            raise RuntimeError(
                "Loaded model architecture is incompatible."
            )

        if model.num_timesteps != 200_000:

            raise RuntimeError(
                f"Expected hover-200k model, "
                f"got {model.num_timesteps} timesteps."
            )

        # SAC.load() must start with an empty replay buffer.

        if (
            model.replay_buffer is None
            or model.replay_buffer.size() != 0
        ):

            raise RuntimeError(
                "Expected a new empty replay buffer."
            )

        model.set_random_seed(args.seed)

        print()
        print("=" * 60)
        print("GENERALIZED TRACKING FROM HOVER 200K")
        print("=" * 60)

        print(
            f"Starting checkpoint: "
            f"{args.hover_checkpoint}"
        )

        print(
            "Loaded actor and critics: HOVER 200K"
        )

        print(
            "Previous tracking model: NONE"
        )

        print(
            "Replay buffer: NEW and EMPTY",
            flush=True,
        )

        # Confirm that the starting controller works.

        check_hover_model(model)

        if args.check_only:

            print()
            print(
                "[PASS] Preflight complete."
            )

            print(
                "[PASS] No training started.",
                flush=True,
            )

            return

        # ----------------------------------------------------
        # Create a unique output directory.
        # ----------------------------------------------------

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        out = (
            ROOT
            / "results"
            / "checkpoints"
            / (
                "sac_generalized_from_hover200k_"
                f"seed{args.seed}_{timestamp}"
            )
        )

        out.mkdir(
            parents=True,
            exist_ok=False,
        )

        (
            out / "checkpoints"
        ).mkdir(exist_ok=False)

        # Save experiment configuration.

        config = {
            "starting_checkpoint": str(
                args.hover_checkpoint
            ),

            "loaded_prior_tracking_checkpoint": False,
            "loaded_replay_buffer": False,

            "hover_pretraining_timesteps": 200_000,
            "seed": args.seed,

            "warmup_steps": args.warmup_steps,
            "tracking_steps": args.tracking_steps,

            "checkpoint_every":
                args.checkpoint_every,

            "hover_episode_probability": 0.25,
            "circle_episode_probability": 0.75,

            "radius_m": [0.15, 0.45],
            "phase_rad": [0.0, 2 * math.pi],
            "omega_rad_s": [0.2, 1.0],

            "reward": "Pos5 unchanged",
            "control": "Direct motor RPM",

            "learning_rate": LEARNING_RATE,
            "replay_buffer_capacity": REPLAY_CAPACITY,
        }

        (
            out / "config.json"
        ).write_text(
            json.dumps(config, indent=2),
            encoding="utf-8",
        )

        print(
            f"\n[OUTPUT] {out}",
            flush=True,
        )

        # ----------------------------------------------------
        # Configure optimization.
        # ----------------------------------------------------

        model.learning_rate = LEARNING_RATE

        model.lr_schedule = get_schedule_fn(
            LEARNING_RATE
        )

        model.learning_starts = 0

        # ----------------------------------------------------
        # Phase 1: collect experience without updates.
        # ----------------------------------------------------

        model.gradient_steps = 0

        actor_before = [
            p.detach().clone()
            for p in model.actor.parameters()
        ]

        print()
        print("=" * 60)
        print("PHASE 1: REPLAY WARM-UP")
        print("=" * 60)

        print(
            f"Collecting {args.warmup_steps:,} transitions "
            "without gradient updates.",
            flush=True,
        )

        model.learn(
            total_timesteps=args.warmup_steps,
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

        if not actor_unchanged:

            raise RuntimeError(
                "Actor changed during warm-up."
            )

        replay_size = model.replay_buffer.size()

        if replay_size < args.warmup_steps:

            raise RuntimeError(
                "Insufficient warm-up replay transitions."
            )

        print(
            f"[PASS] Warm-up complete. "
            f"Replay size={replay_size}",
            flush=True,
        )

        # ----------------------------------------------------
        # Phase 2: train the SAC.
        # ----------------------------------------------------

        model.gradient_steps = 1

        callback = CheckpointCallback(
            out=out,
            env=env,
            warmup_steps=args.warmup_steps,
            every=args.checkpoint_every,
        )

        print()
        print("=" * 60)
        print("PHASE 2: GENERALIZED TRACKING")
        print("=" * 60)

        print(
            f"Tracking steps: "
            f"{args.tracking_steps:,}"
        )

        print(
            f"Learning rate: {LEARNING_RATE}"
        )

        print(
            f"Checkpoint interval: "
            f"{args.checkpoint_every:,}",
            flush=True,
        )

        model.learn(
            total_timesteps=args.tracking_steps,
            reset_num_timesteps=False,
            callback=callback,
            log_interval=20,
        )

        expected_timesteps = (
            args.warmup_steps + args.tracking_steps
        )

        if model.num_timesteps != expected_timesteps:

            raise RuntimeError(
                "Unexpected final timestep count."
            )

        # ----------------------------------------------------
        # Save final model, replay and counts.
        # ----------------------------------------------------

        final_model = (
            out / "sac_generalized_final.zip"
        )

        replay_path = (
            out / "replay_final.pkl"
        )

        model.save(str(final_model))
        model.save_replay_buffer(str(replay_path))

        counts = {
            "warmup_steps": args.warmup_steps,
            "tracking_steps": args.tracking_steps,

            "model_timesteps_since_reset":
                model.num_timesteps,

            "episodes": env.episode_counts,
            "steps": env.step_counts,

            "successes": env.success_counts,
            "failures": env.failure_counts,
        }

        (
            out / "training_counts.json"
        ).write_text(
            json.dumps(counts, indent=2),
            encoding="utf-8",
        )

        print()
        print("=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)

        print(f"Final model: {final_model}")
        print(f"Replay buffer: {replay_path}")

        print(
            "[NEXT] Evaluate all saved checkpoints.",
            flush=True,
        )

    finally:

        env.close()


if __name__ == "__main__":
    main()
