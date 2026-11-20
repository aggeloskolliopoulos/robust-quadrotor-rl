
"""Long nominal SAC fine-tuning from validated Baseline +30k.

Same Pos5 reward, mixed hover/randomized-circle distribution.
No prior checkpoints modified.

Check:
    python -u scripts/train_long_nominal.py --check-only

Train:
    python -u scripts/train_long_nominal.py
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys

from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback


# ============================================================
# 1. Project paths and imports
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from train_generalized_from_hover200k import GeneralizedMixedEnv


# ============================================================
# 2. Source checkpoint
# ============================================================

SOURCE = ROOT / (
    "results/checkpoints/"
    "curriculum_ablation_baseline_seed42_20261114_145522_495268/"
    "checkpoints/sac_curriculum_ablation_030000_steps.zip"
)


# ============================================================
# 3. Hyperparameters
# ============================================================

BUFFER_CAPACITY = 200_000

DEFAULT_STEPS = 200_000
DEFAULT_WARMUP = 2_000
DEFAULT_INTERVAL = 25_000

DEFAULT_LR = 1e-5
DEFAULT_SEED = 42


# ============================================================
# 4. Checkpoint callback
# ============================================================

class SaveCheckpoints(BaseCallback):
    """
    Save model checkpoints at fixed fine-tuning steps.

    Collect training episode statistics.

    IMPORTANT:
    Training success counts are diagnostic.
    The best model will be selected by evaluation,
    NOT by training episode counts.
    """

    def __init__(
        self,
        output_dir: Path,
        train_env,
        interval: int,
    ):
        super().__init__(verbose=0)

        self.output_dir = output_dir
        self.train_env = train_env
        self.interval = interval

        self.csv_path = (
            output_dir / "progress.csv"
        )

        self.fields = [
            "fine_tune_steps",
            "model_timesteps",
            "gradient_updates",
            "hover_episodes",
            "circle_episodes",
            "hover_steps",
            "circle_steps",
            "hover_success",
            "circle_success",
            "hover_failures",
            "circle_failures",
        ]

        with self.csv_path.open(
            "x",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=self.fields,
            )

            writer.writeheader()

    def counts(self):
        """
        Counts include the initial warm-up.

        They are cumulative within this run.
        """

        env = self.train_env

        return {
            "hover_episodes": int(
                env.episode_counts.get("hover", 0)
            ),

            "circle_episodes": int(
                env.episode_counts.get("circle", 0)
            ),

            "hover_steps": int(
                env.step_counts.get("hover", 0)
            ),

            "circle_steps": int(
                env.step_counts.get("circle", 0)
            ),

            "hover_success": int(
                env.success_counts.get("hover", 0)
            ),

            "circle_success": int(
                env.success_counts.get("circle", 0)
            ),

            "hover_failures": int(
                env.failure_counts.get("hover", 0)
            ),

            "circle_failures": int(
                env.failure_counts.get("circle", 0)
            ),
        }

    def _on_step(self) -> bool:

        if self.n_calls % self.interval != 0:
            return True

        steps = self.n_calls

        path = (
            self.output_dir
            / "checkpoints"
            / f"sac_long_nominal_{steps:06d}_steps.zip"
        )

        # Save through a temporary file to avoid leaving
        # a partially written checkpoint if interrupted.

        temp_path = path.with_name(
            path.stem + ".pending.zip"
        )

        self.model.save(str(temp_path))
        temp_path.replace(path)

        counts = self.counts()

        row = {
            "fine_tune_steps": steps,
            "model_timesteps": int(
                self.model.num_timesteps
            ),
            "gradient_updates": int(
                self.model._n_updates
            ),
            **counts,
        }

        with self.csv_path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=self.fields,
            )

            writer.writerow(row)

        print(
            f"[CHECKPOINT {steps:,}] "
            f"circle success={counts['circle_success']} "
            f"failures={counts['circle_failures']} "
            f"| hover success={counts['hover_success']} "
            f"failures={counts['hover_failures']}",
            flush=True,
        )

        print(
            f"[SAVED] {path}",
            flush=True,
        )

        return True


# ============================================================
# 5. Environment sanity checks
# ============================================================

def ensure_environment(env):

    if env.observation_space.shape != (16,):

        raise RuntimeError(
            "Unexpected observation shape: "
            f"{env.observation_space.shape}"
        )

    if env.action_space.shape != (4,):

        raise RuntimeError(
            "Unexpected action shape: "
            f"{env.action_space.shape}"
        )

    for attr in (
        "episode_counts",
        "step_counts",
        "success_counts",
        "failure_counts",
    ):

        if not hasattr(env, attr):

            raise RuntimeError(
                f"Training environment missing {attr}"
            )


# ============================================================
# 6. Set learning rate on loaded SAC model
# ============================================================

def set_learning_rate(
    model: SAC,
    learning_rate: float,
) -> None:

    model.learning_rate = learning_rate

    model.lr_schedule = (
        lambda _: learning_rate
    )

    optimizers = (
        model.actor.optimizer,
        model.critic.optimizer,
        getattr(
            model,
            "ent_coef_optimizer",
            None,
        ),
    )

    for optimizer in optimizers:

        if optimizer is not None:

            for group in optimizer.param_groups:

                group["lr"] = learning_rate


# ============================================================
# 7. Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=DEFAULT_INTERVAL,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
    )

    args = parser.parse_args()

    if (
        args.steps <= 0
        or args.warmup <= 0
        or args.checkpoint_every <= 0
    ):

        parser.error(
            "--steps, --warmup and "
            "--checkpoint-every must be positive"
        )

    if not 0 < args.lr < 1:

        parser.error(
            "--lr must be between 0 and 1"
        )

    if not SOURCE.is_file():

        raise FileNotFoundError(
            "Missing original Baseline checkpoint: "
            f"{SOURCE}"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Same environment used in our previous
    # Curriculum Baseline fine-tuning experiment.

    env = GeneralizedMixedEnv()

    try:

        ensure_environment(env)

        # Loading the model initializes a new,
        # empty replay buffer.

        model = SAC.load(
            str(SOURCE),
            env=env,
            device="cpu",
            custom_objects={
                "buffer_size": BUFFER_CAPACITY
            },
        )

        model.set_random_seed(args.seed)

        set_learning_rate(
            model,
            args.lr,
        )

        if model.replay_buffer.size() != 0:

            raise RuntimeError(
                "Expected an empty replay buffer."
            )

        print()
        print("=" * 60)
        print("LONG NOMINAL SAC FINE-TUNING")
        print("=" * 60)

        print(f"Source: {SOURCE}")
        print(f"New training steps: {args.steps:,}")
        print(
            f"Policy-collection warm-up: "
            f"{args.warmup:,}"
        )
        print(f"Learning rate: {args.lr:g}")
        print(
            f"Checkpoints every: "
            f"{args.checkpoint_every:,}"
        )

        print("Reward: unchanged Pos5")
        print(
            "Training: 25% hover, "
            "75% randomized circles"
        )
        print(
            "R=[0.15,0.45], "
            "phase=[0,2pi], "
            "omega=[0.2,1.0]"
        )

        print(
            "[PASS] Source checkpoint and "
            "new empty buffer loaded.",
            flush=True,
        )

        # ----------------------------------------------------
        # Preflight: NO training
        # ----------------------------------------------------

        if args.check_only:

            obs, _ = env.reset(
                seed=args.seed
            )

            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            (
                next_obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            if (
                not np.all(np.isfinite(next_obs))
                or not np.isfinite(reward)
            ):

                raise RuntimeError(
                    "Nonfinite preflight transition."
                )

            print(
                "[PASS] Environment reset, "
                "SAC action, and transition."
            )

            print(
                "[PASS] Preflight complete. "
                "No training started."
            )

            return

        # ----------------------------------------------------
        # New output directory
        # ----------------------------------------------------

        stamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        output = (
            ROOT
            / "results"
            / "checkpoints"
            / (
                "sac_long_nominal_from_baseline30k_"
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
            "source": str(SOURCE),

            "training_steps": args.steps,

            "warmup_steps_no_gradients":
                args.warmup,

            "checkpoint_interval":
                args.checkpoint_every,

            "learning_rate": args.lr,

            "buffer_capacity":
                BUFFER_CAPACITY,

            "seed": args.seed,

            "device": "cpu",

            "reward": "Pos5 unchanged",

            "distribution": (
                "25% hover / 75% circle; "
                "R U(.15,.45), "
                "phi U(0,2pi), "
                "omega U(.2,1.0)"
            ),

            "note": (
                "Actor frozen during warm-up; "
                "new replay buffer; "
                "checkpoint steps exclude warm-up"
            ),
        }

        (
            output / "config.json"
        ).write_text(
            json.dumps(
                config,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )

        # ----------------------------------------------------
        # Warm-up without gradient updates
        # ----------------------------------------------------

        frozen_actor = {
            key: value.detach().cpu().clone()

            for key, value
            in model.actor.state_dict().items()
        }

        model.gradient_steps = 0

        # Collect policy actions rather than
        # random actions during warm-up.

        model.learning_starts = 0

        print()
        print(
            f"[WARM-UP] {args.warmup:,} steps, "
            "NO gradient updates.",
            flush=True,
        )

        model.learn(
            total_timesteps=args.warmup,
            reset_num_timesteps=True,
            log_interval=20,
        )

        # Verify actor weights stayed unchanged.

        for key, before in frozen_actor.items():

            after = (
                model.actor.state_dict()[key]
                .detach()
                .cpu()
            )

            if not torch.equal(
                before,
                after,
            ):

                raise RuntimeError(
                    "Actor changed during "
                    "no-gradient warm-up: "
                    f"{key}"
                )

        if model.replay_buffer.size() < min(
            args.warmup,
            BUFFER_CAPACITY,
        ):

            raise RuntimeError(
                "Replay buffer did not collect "
                "enough transitions."
            )

        print(
            "[PASS] Warm-up complete, "
            "actor unchanged, "
            f"buffer={model.replay_buffer.size()}.",
            flush=True,
        )

        # ----------------------------------------------------
        # Real fine-tuning
        # ----------------------------------------------------

        model.gradient_steps = 1

        callback = SaveCheckpoints(
            output_dir=output,
            train_env=env,
            interval=args.checkpoint_every,
        )

        print()
        print(
            f"[TRAIN] {args.steps:,} additional steps.",
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

        # ----------------------------------------------------
        # Final model and counters
        # ----------------------------------------------------

        final_path = (
            output / "sac_long_nominal_final.zip"
        )

        model.save(
            str(final_path)
        )

        (
            output / "training_counts.json"
        ).write_text(
            json.dumps(
                callback.counts(),
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )

        print()
        print("=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)

        print(
            f"Final model: {final_path}"
        )

        print(
            f"Progress CSV: "
            f"{output / 'progress.csv'}"
        )

        print(
            "Do not assume the final checkpoint "
            "is best; evaluate all checkpoints."
        )

    finally:

        env.close()


if __name__ == "__main__":
    main()
