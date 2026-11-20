
"""
Controlled SAC fine-tuning from Curriculum 100k.

Two variants:
    baseline: original Pos5 reward
    z_focus:  Pos5 plus -3 * vertical_error^2

Both start from the SAME Curriculum checkpoint.
Both start with a NEW replay buffer.

No existing checkpoint is overwritten.
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


# ============================================================
# 1. Project imports
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from train_generalized_from_hover200k import (
    GeneralizedMixedEnv,
    preflight,
)


# ============================================================
# 2. Configuration
# ============================================================

SOURCE = ROOT / (
    "results/checkpoints/"
    "pos5_radius_phase_curriculum_seed42_20261106_173230/"
    "checkpoints/sac_curriculum_102000_steps.zip"
)

LR = 3e-5

# Pos5: x=5, y=5, z=5
# Z-focused: x=5, y=5, z=8
EXTRA_Z_WEIGHT = 3.0

BUFFER_CAPACITY = 200_000


# ============================================================
# 3. Reward wrapper
# ============================================================

class ExtraAltitudePenalty(gym.Wrapper):
    """
    Add a penalty for vertical tracking error.

    Does NOT modify:
        observations
        action space
        dynamics
        termination conditions
        original Pos5 reward terms
    """

    def __init__(self, env, additional_weight):

        super().__init__(env)

        self.additional_weight = float(
            additional_weight
        )

    def step(self, action):

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = self.env.step(action)

        # obs[0:3] = target_pos - actual_pos
        z_error = float(obs[2])

        if math.isfinite(z_error):

            reward = (
                float(reward)
                - self.additional_weight * z_error**2
            )

        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )


def make_env(variant):

    base = GeneralizedMixedEnv()

    if variant == "z_focus":

        return (
            ExtraAltitudePenalty(
                base,
                EXTRA_Z_WEIGHT,
            ),
            base,
        )

    return base, base


# ============================================================
# 4. Reward verification
# ============================================================

def sanity_checks():
    """
    Compare the same transitions in two environments.

    Expected difference:
        reward_z - reward_pos5 = -3 * e_z^2
    """

    plain = GeneralizedMixedEnv()

    focused = ExtraAltitudePenalty(
        GeneralizedMixedEnv(),
        EXTRA_Z_WEIGHT,
    )

    try:

        preflight(plain)

        for seed in (123, 456):

            obs_a, info_a = plain.reset(
                seed=seed
            )

            obs_b, info_b = focused.reset(
                seed=seed
            )

            np.testing.assert_allclose(
                obs_a,
                obs_b,
                atol=1e-6,
            )

            assert (
                info_a["task"] == info_b["task"]
            )

            action = np.zeros(
                4,
                dtype=np.float32,
            )

            (
                next_a,
                reward_a,
                term_a,
                trunc_a,
                _,
            ) = plain.step(action)

            (
                next_b,
                reward_b,
                term_b,
                trunc_b,
                _,
            ) = focused.step(action)

            np.testing.assert_allclose(
                next_a,
                next_b,
                atol=1e-5,
            )

            assert (
                term_a == term_b
                and trunc_a == trunc_b
            )

            expected = (
                -EXTRA_Z_WEIGHT
                * float(next_a[2]) ** 2
            )

            actual = float(
                reward_b - reward_a
            )

            if not np.isclose(
                actual,
                expected,
                atol=1e-5,
            ):

                raise RuntimeError(
                    "Unexpected reward difference: "
                    f"{actual} vs {expected}"
                )

        print(
            "[PASS] Reward difference verified: "
            "only z weight changes 5 -> 8.",
            flush=True,
        )

    finally:

        plain.close()
        focused.close()


# ============================================================
# 5. Checkpoint callback
# ============================================================

class SaveCheckpoints(BaseCallback):

    def __init__(
        self,
        output_dir,
        base_env,
        interval,
    ):

        super().__init__(verbose=0)

        self.output_dir = output_dir
        self.base_env = base_env
        self.interval = interval

        self.csv_path = (
            output_dir / "progress.csv"
        )

        self.fields = [
            "training_steps",
            "total_new_steps",
            "hover_episodes",
            "circle_episodes",
            "hover_success",
            "circle_success",
            "hover_failure",
            "circle_failure",
            "checkpoint",
        ]

        with self.csv_path.open(
            "x",
            newline="",
            encoding="utf-8",
        ) as file:

            csv.DictWriter(
                file,
                fieldnames=self.fields,
            ).writeheader()

    def _on_step(self):

        # This callback is used only during the
        # second model.learn() call, after warm-up.

        steps = self.n_calls

        if steps % self.interval != 0:
            return True

        name = (
            f"sac_curriculum_ablation_"
            f"{steps:06d}_steps.zip"
        )

        path = (
            self.output_dir
            / "checkpoints"
            / name
        )

        if path.exists():
            raise FileExistsError(path)

        self.model.save(str(path))

        env = self.base_env

        row = {
            "training_steps": steps,
            "total_new_steps": self.num_timesteps,

            "hover_episodes":
                env.episode_counts["hover"],

            "circle_episodes":
                env.episode_counts["circle"],

            "hover_success":
                env.success_counts["hover"],

            "circle_success":
                env.success_counts["circle"],

            "hover_failure":
                env.failure_counts["hover"],

            "circle_failure":
                env.failure_counts["circle"],

            "checkpoint": name,
        }

        with self.csv_path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=self.fields,
            )

            writer.writerow(row)

        print(
            f"[CHECKPOINT {steps:,}] "
            f"circle success={row['circle_success']} "
            f"failures={row['circle_failure']}",
            flush=True,
        )

        print(
            f"[SAVED] {path}",
            flush=True,
        )

        return True


# ============================================================
# 6. Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--variant",
        choices=("baseline", "z_focus"),
        default="baseline",
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=30_000,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=2_000,
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10_000,
    )

    args = parser.parse_args()

    if min(
        args.steps,
        args.warmup,
        args.checkpoint_every,
    ) <= 0:

        parser.error(
            "Training parameters must be positive."
        )

    if not SOURCE.is_file():

        raise FileNotFoundError(
            f"Curriculum checkpoint missing: {SOURCE}"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Verify the original reward and the extra penalty.

    sanity_checks()

    env, base_env = make_env(
        args.variant
    )

    try:

        # ----------------------------------------------------
        # Load the SAME curriculum checkpoint
        # for both variants.
        # ----------------------------------------------------

        model = SAC.load(
            str(SOURCE),
            env=env,
            device="cpu",
            custom_objects={
                "buffer_size": BUFFER_CAPACITY,
            },
        )

        if (
            model.observation_space.shape != (16,)
            or model.action_space.shape != (4,)
        ):

            raise RuntimeError(
                "Unexpected model observation/action space."
            )

        if (
            model.replay_buffer is None
            or model.replay_buffer.size() != 0
        ):

            raise RuntimeError(
                "Expected a fresh, empty replay buffer."
            )

        model.set_random_seed(
            args.seed
        )

        obs, _ = env.reset(
            seed=1234
        )

        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        if (
            np.shape(action) != (4,)
            or not np.all(np.isfinite(action))
        ):

            raise RuntimeError(
                "Loaded policy returned invalid actions."
            )

        print()
        print("=" * 60)
        print("CURRICULUM FINE-TUNING ABLATION")
        print("=" * 60)

        print(f"Variant: {args.variant}")
        print(f"Source: {SOURCE}")

        print(
            "Reward z weight:",
            8 if args.variant == "z_focus" else 5,
        )

        print(
            "[PASS] Loaded original Curriculum checkpoint."
        )

        print(
            "[PASS] New empty replay buffer."
        )

        if args.check_only:

            print(
                "[PASS] Preflight complete. "
                "No training started.",
                flush=True,
            )

            return

        # ----------------------------------------------------
        # Unique result directory
        # ----------------------------------------------------

        stamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        output = (
            ROOT
            / "results"
            / "checkpoints"
            / (
                f"curriculum_ablation_"
                f"{args.variant}_"
                f"seed{args.seed}_{stamp}"
            )
        )

        (
            output / "checkpoints"
        ).mkdir(
            parents=True,
            exist_ok=False,
        )

        config = {
            "variant": args.variant,
            "source_checkpoint": str(SOURCE),

            "training_seed": args.seed,

            "warmup_steps": args.warmup,
            "tracking_steps": args.steps,

            "checkpoint_interval":
                args.checkpoint_every,

            "learning_rate": LR,

            "position_weights_xyz": [
                5,
                5,
                8 if args.variant == "z_focus" else 5,
            ],

            "task_probabilities_per_episode": {
                "hover": 0.25,
                "circle": 0.75,
            },

            "radius_m": [0.15, 0.45],
            "phase_rad": [0.0, 2 * math.pi],
            "omega_rad_s": [0.2, 1.0],

            "replay_buffer_capacity":
                BUFFER_CAPACITY,

            "prior_replay_loaded": False,
            "control": "Direct motor RPM",
        }

        (
            output / "config.json"
        ).write_text(
            json.dumps(
                config,
                indent=2,
            ),
            encoding="utf-8",
        )

        # ----------------------------------------------------
        # Fine-tuning hyperparameters
        # ----------------------------------------------------

        model.learning_rate = LR

        model.lr_schedule = (
            lambda progress_remaining: LR
        )

        model.learning_starts = 0

        # ----------------------------------------------------
        # Warm-up: collect experience without updates
        # ----------------------------------------------------

        model.gradient_steps = 0

        actor_before = [
            p.detach().clone()
            for p in model.actor.parameters()
        ]

        print()
        print(
            f"[WARM-UP] {args.warmup:,} steps "
            "without gradient updates.",
            flush=True,
        )

        model.learn(
            total_timesteps=args.warmup,
            reset_num_timesteps=True,
        )

        actor_changed = any(
            not torch.equal(before, after)
            for before, after in zip(
                actor_before,
                model.actor.parameters(),
            )
        )

        if actor_changed:

            raise RuntimeError(
                "Actor changed during warm-up."
            )

        if (
            model.replay_buffer.size()
            < args.warmup
        ):

            raise RuntimeError(
                "Replay warm-up failed."
            )

        print(
            "[PASS] Warm-up complete.",
            flush=True,
        )

        # ----------------------------------------------------
        # Actual fine-tuning
        # ----------------------------------------------------

        model.gradient_steps = 1

        callback = SaveCheckpoints(
            output_dir=output,
            base_env=base_env,
            interval=args.checkpoint_every,
        )

        print()
        print(
            f"[TRAIN] {args.variant}: "
            f"{args.steps:,} steps",
            flush=True,
        )

        print(
            f"[OUTPUT] {output}",
            flush=True,
        )

        model.learn(
            total_timesteps=args.steps,
            reset_num_timesteps=False,
            callback=callback,
            log_interval=20,
        )

        if model.num_timesteps != (
            args.warmup + args.steps
        ):

            raise RuntimeError(
                "Unexpected final timestep count."
            )

        # ----------------------------------------------------
        # Save final model and statistics
        # ----------------------------------------------------

        final = (
            output
            / "sac_curriculum_ablation_final.zip"
        )

        model.save(str(final))

        counts = {
            "episodes": base_env.episode_counts,
            "steps": base_env.step_counts,
            "successes": base_env.success_counts,
            "failures": base_env.failure_counts,
            "new_model_timesteps":
                model.num_timesteps,
        }

        (
            output / "training_counts.json"
        ).write_text(
            json.dumps(
                counts,
                indent=2,
            ),
            encoding="utf-8",
        )

        print()
        print("=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)

        print(f"Variant: {args.variant}")
        print(f"Final model: {final}")

    finally:

        env.close()


if __name__ == "__main__":
    main()
