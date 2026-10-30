
"""Train a fresh SB3 SAC on the existing Position-Focused hover reward.

Run from the project root:
    python scripts/train_pos5_hover_scratch.py --check-only
    python scripts/train_pos5_hover_scratch.py
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from finetune_sac_circle import PositionHoverEnv
from quadrotor_rl.envs.quadrotor_tracking_env import QuadrotorTrackingEnv

REFERENCE_MODEL = (
    ROOT
    / "results/checkpoints/sac_hover_stage1"
    / "sac_hover_80000_steps.zip"
)

N_TRAIN = 200_000
CHECKPOINT_EVERY = 20_000

# Development seeds, NOT final held-out test seeds
EVALUATION_SEEDS = tuple(range(1200, 1220))

# 10 seconds at 48 Hz
MAX_EVALUATION_STEPS = 480


# ============================================================
# 2. REWARD SANITY CHECK
# ============================================================

def reward_sanity_check():
    """Verify that Pos5 changes only the position weight."""

    old_env = QuadrotorTrackingEnv()
    new_env = PositionHoverEnv()

    try:
        old_obs, _ = old_env.reset(seed=12345)
        new_obs, _ = new_env.reset(seed=12345)

        assert np.shape(old_obs) == (16,)
        assert new_env.action_space.shape == (4,)

        np.testing.assert_allclose(
            old_obs, new_obs, atol=1e-5, rtol=0
        )

        action = np.zeros(4, dtype=np.float32)

        old_next, old_reward, old_term, old_trunc, _ = (
            old_env.step(action)
        )

        new_next, new_reward, new_term, new_trunc, _ = (
            new_env.step(action)
        )

        np.testing.assert_allclose(
            old_next, new_next, atol=1e-5, rtol=0
        )

        assert (old_term, old_trunc) == (new_term, new_trunc)

        # New reward = old reward - 4 * ||position_error||^2
        expected_difference = -4.0 * float(
            np.dot(new_next[:3], new_next[:3])
        )

        actual_difference = float(new_reward - old_reward)

        if not np.isclose(
            actual_difference,
            expected_difference,
            atol=1e-4,
            rtol=1e-4,
        ):
            raise AssertionError(
                "Pos5 reward check failed: "
                f"actual={actual_difference}, "
                f"expected={expected_difference}"
            )

        print("[PASS] Observation/action dimensions verified.")
        print("[PASS] Position-Focused reward verified.")

    finally:
        old_env.close()
        new_env.close()


# ============================================================
# 3. COPY HYPERPARAMETERS (NOT PRETRAINED WEIGHTS)
# ============================================================

def baseline_sac_config(reference: SAC) -> dict:
    """Extract SAC settings from the original Hover 80k model."""

    keys = (
        "learning_rate",
        "buffer_size",
        "learning_starts",
        "batch_size",
        "tau",
        "gamma",
        "train_freq",
        "gradient_steps",
        "action_noise",
        "replay_buffer_class",
        "replay_buffer_kwargs",
        "optimize_memory_usage",
        "ent_coef",
        "target_update_interval",
        "target_entropy",
        "use_sde",
        "sde_sample_freq",
        "use_sde_at_warmup",
        "policy_kwargs",
    )

    accepted = set(inspect.signature(SAC.__init__).parameters)

    return {
        name: getattr(reference, name)
        for name in keys
        if name in accepted and hasattr(reference, name)
    }


# ============================================================
# 4. DETERMINISTIC HOVER EVALUATION
# ============================================================

def evaluate_hover(model: SAC, env: PositionHoverEnv) -> dict:
    """Evaluate a checkpoint using fixed development seeds."""

    full_rmse = []
    late_rmse = []

    for seed in EVALUATION_SEEDS:

        obs, _ = env.reset(seed=seed)

        squared_errors = []
        succeeded = False

        for step in range(MAX_EVALUATION_STEPS):

            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            obs, _, terminated, truncated, _ = env.step(action)

            position_error = obs[:3]

            squared_errors.append(
                float(np.dot(position_error, position_error))
            )

            if terminated or truncated:
                succeeded = bool(truncated and not terminated)
                break

        else:
            raise RuntimeError(
                "Evaluation episode exceeded 480 steps."
            )

        if succeeded:

            episode_full = math.sqrt(
                float(np.mean(squared_errors))
            )

            full_rmse.append(episode_full)

            # Last 5 seconds
            if len(squared_errors) > 5 * 48:

                episode_late = math.sqrt(
                    float(np.mean(squared_errors[5 * 48:]))
                )

                late_rmse.append(episode_late)

    n_success = len(full_rmse)
    n_total = len(EVALUATION_SEEDS)

    return {
        "n_seeds": n_total,
        "n_success": n_success,
        "survival_rate": n_success / n_total,
        "full_rmse_m": (
            float(np.mean(full_rmse))
            if full_rmse else math.nan
        ),
        "late_rmse_m": (
            float(np.mean(late_rmse))
            if late_rmse else math.nan
        ),
    }


# ============================================================
# 5. CHECKPOINT AND EVALUATION CALLBACK
# ============================================================

class SaveAndEvaluate(BaseCallback):

    def __init__(self, run_dir: Path, eval_env: PositionHoverEnv):

        super().__init__(verbose=0)

        self.run_dir = run_dir
        self.eval_env = eval_env

        self.metrics_file = (
            run_dir / "hover_development_metrics.csv"
        )

        self.best_key = None

        with self.metrics_file.open("x", newline="") as f:

            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "steps",
                    "n_seeds",
                    "n_success",
                    "survival_rate",
                    "full_rmse_m",
                    "late_rmse_m",
                    "checkpoint",
                ],
            )

            writer.writeheader()

    def _on_step(self) -> bool:

        steps = self.num_timesteps

        if steps % CHECKPOINT_EVERY != 0:
            return True

        # Save checkpoint
        checkpoint = (
            self.run_dir
            / f"sac_pos5_scratch_hover_{steps:06d}_steps.zip"
        )

        if checkpoint.exists():
            raise FileExistsError(
                f"Refusing to overwrite: {checkpoint}"
            )

        self.model.save(str(checkpoint))

        # Evaluate checkpoint
        metrics = evaluate_hover(
            self.model,
            self.eval_env,
        )

        row = {
            "steps": steps,
            **metrics,
            "checkpoint": checkpoint.name,
        }

        with self.metrics_file.open("a", newline="") as f:

            writer = csv.DictWriter(
                f,
                fieldnames=row.keys(),
            )

            writer.writerow(row)

        print(
            f"\n[DEV EVAL {steps:6d}] "
            f"survival={metrics['n_success']}/"
            f"{metrics['n_seeds']} "
            f"full={metrics['full_rmse_m']:.5f} m "
            f"late={metrics['late_rmse_m']:.5f} m "
            f"| {checkpoint.name}\n",
            flush=True,
        )

        # Model selection:
        # 1. Highest survival
        # 2. Lowest full position RMSE
        key = (
            -metrics["survival_rate"],
            metrics["full_rmse_m"],
        )

        if self.best_key is None or key < self.best_key:

            self.best_key = key

            best_file = (
                self.run_dir
                / "best_development_checkpoint.txt"
            )

            best_file.write_text(
                checkpoint.name + "\n",
                encoding="utf-8",
            )

            print(
                f"[BEST DEV] {checkpoint.name}",
                flush=True,
            )

        return True


# ============================================================
# 6. MAIN TRAINING FUNCTION
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Fresh SAC training with Pos5 hover reward"
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run sanity checks without training",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    # Verify original checkpoint exists
    if not REFERENCE_MODEL.is_file():

        raise FileNotFoundError(
            f"Reference checkpoint not found: {REFERENCE_MODEL}"
        )

    # Check reward implementation
    reward_sanity_check()

    # Load original SAC ONLY to recover hyperparameters
    reference = SAC.load(
        str(REFERENCE_MODEL),
        device="cpu",
    )

    sac_settings = baseline_sac_config(reference)

    print(
        f"[INFO] Original model: "
        f"{reference.num_timesteps} steps"
    )

    print("[INFO] SAC configuration:")

    for key, value in sac_settings.items():
        print(f"    {key} = {value}")

    # --------------------------------------------------------
    # PREFLIGHT CHECK
    # --------------------------------------------------------

    if args.check_only:

        test_env = PositionHoverEnv()

        try:
            fresh = SAC(
                reference.policy_class,
                test_env,
                device="cpu",
                seed=args.seed,
                verbose=0,
                **sac_settings,
            )

            assert fresh.num_timesteps == 0

            print(
                "[PASS] Fresh SAC initialized successfully."
            )

            print(
                "[PASS] Actor and critics are randomly initialized."
            )

        finally:
            test_env.close()

        print("[PASS] All preflight checks completed.")
        print("[INFO] No training started.")

        return

    # --------------------------------------------------------
    # SET RANDOM SEEDS
    # --------------------------------------------------------

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    set_random_seed(args.seed)

    # --------------------------------------------------------
    # CREATE UNIQUE OUTPUT DIRECTORY
    # --------------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    run_dir = (
        ROOT
        / "results"
        / "checkpoints"
        / f"sac_pos5_scratch_hover_seed{args.seed}_{timestamp}"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    # --------------------------------------------------------
    # CREATE ENVIRONMENTS
    # --------------------------------------------------------

    train_env = PositionHoverEnv()
    eval_env = PositionHoverEnv()

    try:

        assert (
            reference.observation_space.shape
            == train_env.observation_space.shape
            == (16,)
        )

        assert (
            reference.action_space.shape
            == train_env.action_space.shape
            == (4,)
        )

        train_env.reset(seed=args.seed)

        # ----------------------------------------------------
        # FRESH SAC MODEL
        # ----------------------------------------------------

        # IMPORTANT:
        # SAC(...) creates a new actor and new critics.
        # We do NOT use SAC.load(...) for the new model.

        model = SAC(
            reference.policy_class,
            train_env,
            seed=args.seed,
            device="cpu",
            verbose=1,
            **sac_settings,
        )

        assert model.num_timesteps == 0

        # ----------------------------------------------------
        # SAVE EXPERIMENT CONFIGURATION
        # ----------------------------------------------------

        metadata = {
            "experiment": "Pos5 from-scratch hover",
            "train_steps": N_TRAIN,
            "checkpoint_interval": CHECKPOINT_EVERY,
            "validation_seeds": list(EVALUATION_SEEDS),
            "training_seed": args.seed,
            "initialization": "Fresh actor and critics",
            "hyperparameter_reference": str(REFERENCE_MODEL),
            "environment": "PositionHoverEnv",
            "reward_weights": {
                "position": 5.0,
                "velocity": 0.1,
                "angular": 0.01,
                "action": 0.01,
            },
            "failure_penalty": -10,
            "rmse_aggregation": (
                "Mean of per-episode RMSE, "
                "successful episodes only"
            ),
            "sac_settings": {
                key: repr(value)
                for key, value in sac_settings.items()
            },
        }

        config_path = run_dir / "experiment.json"

        config_path.write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )

        print("\n====================================")
        print("POSITION-FOCUSED SAC FROM SCRATCH")
        print("====================================")
        print(f"Training steps: {N_TRAIN}")
        print(f"Position weight: 5.0")
        print(f"Checkpoint interval: {CHECKPOINT_EVERY}")
        print(f"Seed: {args.seed}")
        print(f"Output: {run_dir}")
        print("====================================\n")

        # ----------------------------------------------------
        # CALLBACK
        # ----------------------------------------------------

        callback = SaveAndEvaluate(
            run_dir,
            eval_env,
        )

        # ----------------------------------------------------
        # START TRAINING
        # ----------------------------------------------------

        model.learn(
            total_timesteps=N_TRAIN,
            callback=callback,
            log_interval=10,
        )

        print("\n[FINISHED] Training completed.")
        print(f"Results saved in: {run_dir}")
        print(
            "Check best_development_checkpoint.txt "
            "for the selected model."
        )

    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
