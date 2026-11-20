
"""SAC nominal fine-tuning from the validated Baseline +30k.

Same GeneralizedMixedEnv, Pos5 reward and 25% hover / 75% randomized circles.
This script never overwrites the source Baseline checkpoint.

Fresh run, recommended 100k additional steps::

    python -u scripts/train_long_nominal.py --check-only
    python -u scripts/train_long_nominal.py \
        --steps 100000 --warmup 20000 --checkpoint-every 10000 \
        --save-buffer-every 25000 --lr 0.000003 --seed 42

Resume from a *paired* model + replay-buffer checkpoint::

    python -u scripts/train_long_nominal.py \
        --resume-from /absolute/path/to/checkpoints/sac_long_nominal_050000_steps.zip \
        --steps 100000 --warmup 20000 --checkpoint-every 10000 \
        --save-buffer-every 25000 --lr 0.000003 --seed 42

On resume, --steps means the TOTAL additional steps since Baseline 30k;
it does not mean extra steps after the saved checkpoint.

Checkpoints are written at the start of the next rollout, which (for the
usual SAC train_freq=1) is *after* the previous transition was stored AND
its gradient update completed. The terminal checkpoint is saved after learn().

Resuming with SB3 is not bitwise identical: a new physics episode, fresh
random-number-generator streams and a fresh SB3 training schedule are used.
The actor/critics, optimizers, entropy coefficient and replay data are kept.
Only load replay-buffer .pkl files that you generated and trust.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from train_generalized_from_hover200k import GeneralizedMixedEnv

SOURCE = ROOT / (
    "results/checkpoints/"
    "curriculum_ablation_baseline_seed42_20261114_145522_495268/"
    "checkpoints/sac_curriculum_ablation_030000_steps.zip"
)

BUFFER_CAPACITY = 200_000
DEFAULT_STEPS = 100_000
DEFAULT_WARMUP = 20_000
DEFAULT_MODEL_INTERVAL = 10_000
DEFAULT_BUFFER_INTERVAL = 25_000
DEFAULT_LR = 3e-6
DEFAULT_SEED = 42

COUNT_KEYS = (
    "hover_episodes",
    "circle_episodes",
    "hover_steps",
    "circle_steps",
    "hover_success",
    "circle_success",
    "hover_failures",
    "circle_failures",
)

CSV_FIELDS = (
    "fine_tune_steps",
    "model_timesteps",
    "gradient_updates",
    *COUNT_KEYS,
)


def ensure_environment(env: GeneralizedMixedEnv) -> None:
    if env.observation_space.shape != (16,):
        raise RuntimeError(
            f"Unexpected observation shape: {env.observation_space.shape}"
        )
    if env.action_space.shape != (4,):
        raise RuntimeError(
            f"Unexpected action shape: {env.action_space.shape}"
        )
    for key in ("episode_counts", "step_counts", "success_counts", "failure_counts"):
        if not hasattr(env, key):
            raise RuntimeError(f"Training environment missing {key}")


def set_learning_rate(model: SAC, learning_rate: float) -> None:
    """Match the original fine-tuning logic for actor/critic/entropy LR."""
    model.learning_rate = learning_rate
    model.lr_schedule = lambda _: learning_rate
    for optimizer in (
        model.actor.optimizer,
        model.critic.optimizer,
        getattr(model, "ent_coef_optimizer", None),
    ):
        if optimizer is not None:
            for group in optimizer.param_groups:
                group["lr"] = learning_rate


def current_counts(env: GeneralizedMixedEnv) -> dict[str, int]:
    return {
        "hover_episodes": int(env.episode_counts.get("hover", 0)),
        "circle_episodes": int(env.episode_counts.get("circle", 0)),
        "hover_steps": int(env.step_counts.get("hover", 0)),
        "circle_steps": int(env.step_counts.get("circle", 0)),
        "hover_success": int(env.success_counts.get("hover", 0)),
        "circle_success": int(env.success_counts.get("circle", 0)),
        "hover_failures": int(env.failure_counts.get("hover", 0)),
        "circle_failures": int(env.failure_counts.get("circle", 0)),
    }


def write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_name(path.stem + ".pending.json")
    try:
        with temp.open("x", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def save_snapshot(
    model: SAC,
    output_dir: Path,
    basename: str,
    fine_tune_steps: int,
    warmup_steps: int,
    training_counts: dict[str, int],
    learning_rate: float,
    seed: int,
    with_buffer: bool,
) -> tuple[Path, Path | None, Path]:
    """Safely write a model and optionally replay buffer, then commit metadata.

    The JSON manifest is written last. A manifest with has_replay_buffer=True
    and valid file sizes identifies a complete resumable pair. If the process
    is interrupted while saving, .pending files or files without a manifest
    are not considered resumable by this script.
    """
    model_path = output_dir / f"{basename}.zip"
    buffer_path = output_dir / f"{basename}.pkl" if with_buffer else None
    manifest_path = output_dir / f"{basename}.json"

    if manifest_path.exists():
        raise FileExistsError(f"Checkpoint already committed: {manifest_path}")

    model_temp = output_dir / f"{basename}.pending.zip"
    buffer_temp = output_dir / f"{basename}.pending.pkl"

    if any(p.exists() for p in (model_path, model_temp)):
        raise FileExistsError(f"Model checkpoint already exists: {model_path}")
    if with_buffer and any(p.exists() for p in (buffer_path, buffer_temp)):
        raise FileExistsError(f"Replay buffer checkpoint already exists: {buffer_path}")

    try:
        model.save(str(model_temp))
        if with_buffer:
            model.save_replay_buffer(str(buffer_temp))

        # Publish the artifacts only after both temporary saves succeeded.
        model_temp.replace(model_path)
        if with_buffer:
            buffer_temp.replace(buffer_path)

        metadata = {
            "schema_version": 1,
            "fine_tune_steps": int(fine_tune_steps),
            "warmup_steps": int(warmup_steps),
            "model_timesteps": int(model.num_timesteps),
            "gradient_updates": int(model._n_updates),
            "model_file": model_path.name,
            "model_size_bytes": model_path.stat().st_size,
            "has_replay_buffer": with_buffer,
            "buffer_file": buffer_path.name if with_buffer else None,
            "buffer_size_bytes": buffer_path.stat().st_size if with_buffer else None,
            "buffer_capacity": int(model.replay_buffer.buffer_size),
            "buffer_entries": int(model.replay_buffer.size()),
            "learning_rate": float(learning_rate),
            "seed": int(seed),
            "training_counts": training_counts,
        }
        write_json_atomically(manifest_path, metadata)
    finally:
        model_temp.unlink(missing_ok=True)
        buffer_temp.unlink(missing_ok=True)

    return model_path, buffer_path, manifest_path


def load_resume_metadata(model_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    """Require a committed *pair* rather than silently loading an empty buffer."""
    model_path = model_path.expanduser().resolve()
    if model_path.suffix != ".zip":
        raise ValueError("--resume-from must point to a .zip model checkpoint")

    manifest_path = model_path.with_suffix(".json")
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No checkpoint manifest: {manifest_path}\n"
            "Only checkpoints created with this updated script can be resumed."
        )

    with manifest_path.open(encoding="utf-8") as f:
        metadata = json.load(f)

    if metadata.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint manifest version")
    if not metadata.get("has_replay_buffer"):
        raise ValueError(
            f"{model_path.name} is model-only; choose a checkpoint with a .pkl"
        )
    if metadata.get("model_file") != model_path.name:
        raise ValueError("Model filename does not match the checkpoint manifest")

    buffer_name = metadata.get("buffer_file")
    if not buffer_name or Path(buffer_name).name != buffer_name:
        raise ValueError("Invalid replay-buffer filename in manifest")
    buffer_path = model_path.parent / buffer_name

    for path, size_key in (
        (model_path, "model_size_bytes"),
        (buffer_path, "buffer_size_bytes"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing resume artifact: {path}")
        if path.stat().st_size != metadata[size_key]:
            raise RuntimeError(f"Resume artifact size mismatch: {path}")

    return model_path, buffer_path, metadata


class SaveCheckpoints(BaseCallback):
    """Save model-only checkpoints and periodic model + buffer pairs.

    on_rollout_start is deliberately used instead of on_step: for the usual
    SAC train_freq=1, this fires AFTER the prior transition was added to the
    replay buffer AND the corresponding SAC gradient update completed.
    """

    def __init__(
        self,
        output_dir: Path,
        train_env: GeneralizedMixedEnv,
        model_interval: int,
        buffer_interval: int,
        warmup_steps: int,
        initial_step: int,
        previous_counts: dict[str, int],
        learning_rate: float,
        seed: int,
    ):
        super().__init__(verbose=0)
        self.output_dir = output_dir
        self.train_env = train_env
        self.model_interval = model_interval
        self.buffer_interval = buffer_interval
        self.warmup_steps = warmup_steps
        self.initial_step = initial_step
        self.previous_counts = previous_counts
        self.learning_rate = learning_rate
        self.seed = seed
        self.last_saved_step = initial_step
        self.csv_path = output_dir / "progress.csv"

        with self.csv_path.open("x", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()

    def counts(self) -> dict[str, int]:
        now = current_counts(self.train_env)
        return {
            key: int(self.previous_counts.get(key, 0)) + now[key]
            for key in COUNT_KEYS
        }

    def _on_rollout_start(self) -> None:
        # After warm-up, model.num_timesteps == warmup + fine-tune steps.
        step = int(self.model.num_timesteps) - self.warmup_steps
        if step <= self.last_saved_step:
            return
        needs_model = step % self.model_interval == 0
        needs_buffer = step % self.buffer_interval == 0
        if not (needs_model or needs_buffer):
            return

        counts = self.counts()
        name = f"sac_long_nominal_{step:06d}_steps"
        checkpoint_dir = self.output_dir / "checkpoints"
        model_path, buffer_path, _ = save_snapshot(
            model=self.model,
            output_dir=checkpoint_dir,
            basename=name,
            fine_tune_steps=step,
            warmup_steps=self.warmup_steps,
            training_counts=counts,
            learning_rate=self.learning_rate,
            seed=self.seed,
            with_buffer=needs_buffer,
        )

        row = {
            "fine_tune_steps": step,
            "model_timesteps": int(self.model.num_timesteps),
            "gradient_updates": int(self.model._n_updates),
            **counts,
        }
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writerow(row)

        self.last_saved_step = step
        print(
            f"[CHECKPOINT {step:,}] "
            f"circle success={counts['circle_success']} "
            f"failures={counts['circle_failures']} | "
            f"hover success={counts['hover_success']} "
            f"failures={counts['hover_failures']}",
            flush=True,
        )
        print(f"[SAVED MODEL] {model_path}", flush=True)
        if buffer_path is not None:
            print(
                f"[SAVED REPLAY BUFFER] {buffer_path} "
                f"({self.model.replay_buffer.size():,} transitions)",
                flush=True,
            )

    def _on_step(self) -> bool:
        return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_MODEL_INTERVAL)
    parser.add_argument("--save-buffer-every", type=int, default=DEFAULT_BUFFER_INTERVAL)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--resume-from", type=Path, default=None,
        help="Existing paired model .zip checkpoint; warm-up is NOT repeated",
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    if any(value <= 0 for value in (
        args.steps, args.warmup, args.checkpoint_every, args.save_buffer_every
    )):
        parser.error("steps, warmup, and checkpoint intervals must be positive")
    if not 0 < args.lr < 1:
        parser.error("--lr must be between 0 and 1")
    if not SOURCE.is_file() and args.resume_from is None:
        raise FileNotFoundError(f"Missing Baseline checkpoint: {SOURCE}")

    resume_meta = None
    resume_buffer_path = None
    resume_model_path = None
    start_step = 0
    previous_counts = {key: 0 for key in COUNT_KEYS}

    if args.resume_from is not None:
        resume_model_path, resume_buffer_path, resume_meta = load_resume_metadata(
            args.resume_from
        )
        start_step = int(resume_meta["fine_tune_steps"])
        previous_counts = {
            key: int(resume_meta["training_counts"][key])
            for key in COUNT_KEYS
        }
        if args.steps <= start_step:
            parser.error(
                f"--steps={args.steps} must exceed resumed step={start_step}; "
                "--steps is the total fine-tune target"
            )
        for key, actual, expected in (
            ("warmup_steps", args.warmup, resume_meta["warmup_steps"]),
            ("learning_rate", args.lr, resume_meta["learning_rate"]),
            ("seed", args.seed, resume_meta["seed"]),
            ("buffer_capacity", BUFFER_CAPACITY, resume_meta["buffer_capacity"]),
        ):
            if actual != expected:
                parser.error(
                    f"Resume mismatch for {key}: requested {actual}, "
                    f"checkpoint has {expected}"
                )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = GeneralizedMixedEnv()
    try:
        ensure_environment(env)
        source_path = resume_model_path or SOURCE
        model = SAC.load(
            str(source_path),
            env=env,
            device="cpu",
            custom_objects={"buffer_size": BUFFER_CAPACITY},
        )
        model.set_random_seed(args.seed)
        set_learning_rate(model, args.lr)

        if resume_meta is None:
            if model.replay_buffer.size() != 0:
                raise RuntimeError("Expected a new, empty replay buffer")
        else:
            # The serialized buffer is only safe to load from a trusted source.
            model.load_replay_buffer(str(resume_buffer_path))
            model.learning_starts = 0
            model.gradient_steps = 1
            if model.replay_buffer.buffer_size != BUFFER_CAPACITY:
                raise RuntimeError("Loaded replay buffer capacity mismatch")
            if model.replay_buffer.size() != resume_meta["buffer_entries"]:
                raise RuntimeError("Loaded replay buffer entry-count mismatch")
            if int(model.num_timesteps) != int(resume_meta["model_timesteps"]):
                raise RuntimeError("Loaded model timestep mismatch")
            if int(model._n_updates) != int(resume_meta["gradient_updates"]):
                raise RuntimeError("Loaded model optimizer update-count mismatch")
            if int(model.num_timesteps) - args.warmup != start_step:
                raise RuntimeError("Resume step does not match warm-up + training")

        print("\n" + "=" * 60)
        print("LONG NOMINAL SAC FINE-TUNING (WITH REPLAY CHECKPOINTS)")
        print("=" * 60)
        print(f"Source model: {source_path}")
        print(f"Target fine-tuning steps: {args.steps:,}")
        print(f"Completed fine-tuning steps: {start_step:,}")
        print(f"Warm-up (no gradients): {args.warmup:,}")
        print(f"Learning rate: {args.lr:g}")
        print(f"Model checkpoints every: {args.checkpoint_every:,}")
        print(f"Replay-buffer checkpoints every: {args.save_buffer_every:,}")
        print("Reward: unchanged Pos5")
        print("Training: 25% hover, 75% randomized circles")
        print("R=[0.15,0.45], phase=[0,2pi], omega=[0.2,1.0]")
        if resume_meta is None:
            print("[PASS] Original Baseline loaded with EMPTY replay buffer.")
        else:
            print(
                f"[PASS] Model + buffer restored: "
                f"{model.replay_buffer.size():,} transitions."
            )

        if args.check_only:
            obs, _ = env.reset(seed=args.seed)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if not np.all(np.isfinite(obs)) or not np.isfinite(reward):
                raise RuntimeError("Nonfinite preflight transition")
            print("[PASS] Environment, policy, and transition")
            print("[PASS] Preflight complete; NO training started")
            return

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output = ROOT / "results" / "checkpoints" / (
            f"sac_long_nominal_from_baseline30k_seed{args.seed}_{stamp}"
        )
        (output / "checkpoints").mkdir(parents=True, exist_ok=False)

        config = {
            "source_baseline": str(SOURCE),
            "loaded_model": str(source_path),
            "resumed_from": str(resume_model_path) if resume_model_path else None,
            "starting_fine_tune_step": start_step,
            "target_fine_tune_steps": args.steps,
            "new_training_steps_this_process": args.steps - start_step,
            "warmup_steps_no_gradients": args.warmup,
            "checkpoint_interval": args.checkpoint_every,
            "replay_buffer_checkpoint_interval": args.save_buffer_every,
            "learning_rate": args.lr,
            "buffer_capacity": BUFFER_CAPACITY,
            "seed": args.seed,
            "device": "cpu",
            "reward": "Pos5 unchanged",
            "distribution": (
                "25% hover / 75% circle; R U(.15,.45), "
                "phi U(0,2pi), omega U(.2,1.0)"
            ),
            "resume_is_bitwise_exact": False,
            "note": (
                "Model-only checkpoints cannot be resumed with replay data. "
                "Replay snapshots are paired with a model and JSON manifest. "
                "Warm-up is performed only for new runs."
            ),
        }
        write_json_atomically(output / "config.json", config)

        if resume_meta is None:
            frozen_actor = {
                key: value.detach().cpu().clone()
                for key, value in model.actor.state_dict().items()
            }
            model.gradient_steps = 0
            model.learning_starts = 0  # policy actions, not random actions
            print(
                f"\n[WARM-UP] {args.warmup:,} steps, no gradients.",
                flush=True,
            )
            model.learn(
                total_timesteps=args.warmup,
                reset_num_timesteps=True,
                log_interval=20,
            )
            for key, before in frozen_actor.items():
                after = model.actor.state_dict()[key].detach().cpu()
                if not torch.equal(before, after):
                    raise RuntimeError(f"Actor changed in warm-up: {key}")
            if model.replay_buffer.size() < min(args.warmup, BUFFER_CAPACITY):
                raise RuntimeError("Replay buffer collected too few transitions")
            if int(model.num_timesteps) != args.warmup:
                raise RuntimeError("Unexpected number of warm-up timesteps")
            print(
                f"[PASS] Actor unchanged; replay buffer has "
                f"{model.replay_buffer.size():,} entries.",
                flush=True,
            )
        else:
            print("\n[RESUME] Warm-up skipped; replay buffer retained.")

        model.gradient_steps = 1
        model.learning_starts = 0

        callback = SaveCheckpoints(
            output_dir=output,
            train_env=env,
            model_interval=args.checkpoint_every,
            buffer_interval=args.save_buffer_every,
            warmup_steps=args.warmup,
            initial_step=start_step,
            previous_counts=previous_counts,
            learning_rate=args.lr,
            seed=args.seed,
        )
        print(
            f"\n[TRAIN] {args.steps - start_step:,} more fine-tuning steps.\n"
            f"[OUTPUT] {output}",
            flush=True,
        )
        model.learn(
            total_timesteps=args.steps - start_step,
            reset_num_timesteps=False,
            callback=callback,
            log_interval=20,
        )

        final_step = int(model.num_timesteps) - args.warmup
        if final_step != args.steps:
            raise RuntimeError(
                f"Expected {args.steps:,} completed steps; got {final_step:,}"
            )

        final_counts = callback.counts()
        # Save the terminal step under the standard numbered filename as well,
        # so existing checkpoint-evaluation scripts can find the +100k model.
        numbered_model, numbered_buffer, numbered_manifest = save_snapshot(
            model=model,
            output_dir=output / "checkpoints",
            basename=f"sac_long_nominal_{final_step:06d}_steps",
            fine_tune_steps=final_step,
            warmup_steps=args.warmup,
            training_counts=final_counts,
            learning_rate=args.lr,
            seed=args.seed,
            with_buffer=True,
        )

        # Convenient root-level 'final' aliases. Use filesystem hard links to
        # avoid storing a second ~200k-transition buffer; fall back to copies.
        final_model = output / "sac_long_nominal_final.zip"
        final_buffer = output / "sac_long_nominal_final.pkl"
        final_meta = output / "sac_long_nominal_final.json"
        for original, alias in (
            (numbered_model, final_model),
            (numbered_buffer, final_buffer),
        ):
            try:
                os.link(original, alias)
            except OSError:
                shutil.copy2(original, alias)

        with numbered_manifest.open(encoding="utf-8") as f:
            final_metadata = json.load(f)
        final_metadata["model_file"] = final_model.name
        final_metadata["buffer_file"] = final_buffer.name
        write_json_atomically(final_meta, final_metadata)
        write_json_atomically(output / "training_counts.json", final_counts)

        # The callback fires before the NEXT rollout, so the very last step
        # would otherwise be absent from progress.csv.
        if callback.last_saved_step != final_step:
            with callback.csv_path.open("a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow({
                    "fine_tune_steps": final_step,
                    "model_timesteps": int(model.num_timesteps),
                    "gradient_updates": int(model._n_updates),
                    **final_counts,
                })

        print("\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)
        print(f"Final numbered checkpoint: {numbered_model}")
        print(f"Final model: {final_model}")
        print(f"Final replay buffer: {final_buffer}")
        print(f"Final resume manifest: {final_meta}")
        print(f"Progress CSV: {output / 'progress.csv'}")
        print("Select the best checkpoint using held-out evaluation, not train reward.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
