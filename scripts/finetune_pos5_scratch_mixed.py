
"""
Stage 2: Fine-tune the Pos5 Scratch Hover 200k SAC.

Starting checkpoint:
    Scratch Pos5 Hover 200k

Training:
    2k replay warm-up (without gradient updates)
    100k additional learning steps
    25% Hover / 75% Randomized Circle

Reward:
    Position weight = 5 (unchanged)

Uses the existing MixedTrackingEnv and training pipeline
from finetune_sac_circle.py.

Run:
    python scripts/finetune_pos5_scratch_mixed.py --check-only
    python -u scripts/finetune_pos5_scratch_mixed.py
"""

import argparse
import contextlib
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

from stable_baselines3 import SAC


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "scripts"))

import finetune_sac_circle as training


START_MODEL = (
    ROOT
    / "results/checkpoints"
    / "sac_pos5_scratch_hover_seed42_20261030_173955"
    / "sac_pos5_scratch_hover_200000_steps.zip"
)

ORIGINAL_SCRIPT = ROOT / "scripts/finetune_sac_circle.py"


# ============================================================
# 2. EXPERIMENT CONFIGURATION
# ============================================================

SEED = 42

WARMUP_STEPS = 2_000
TRAINING_STEPS = 100_000

LEARNING_RATE = 3e-5
BUFFER_SIZE = 200_000

CHECKPOINT_INTERVAL = 25_000

HOVER_PROBABILITY = 0.25

POSITION_WEIGHT = 5.0


# ============================================================
# 3. CONSOLE + FILE LOGGING
# ============================================================

class Tee:
    """Write output to both terminal and log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            stream.write(message)
        self.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


# ============================================================
# 4. APPLY CONFIGURATION
# ============================================================

def configure_training(run_dir: Path):

    training.SEED = SEED

    training.HOVER_PROBABILITY = HOVER_PROBABILITY

    training.POSITION_WEIGHT_OLD = 1.0
    training.POSITION_WEIGHT_NEW = POSITION_WEIGHT

    training.PRECOLLECT_STEPS = WARMUP_STEPS
    training.TRAINING_STEPS = TRAINING_STEPS

    training.LEARNING_RATE = LEARNING_RATE
    training.BUFFER_SIZE = BUFFER_SIZE

    training.CHECKPOINT_INTERVAL = CHECKPOINT_INTERVAL

    # IMPORTANT:
    # Replace only the starting model and output paths.
    # The existing training logic stays unchanged.

    training.HOVER_MODEL = START_MODEL

    training.CHECKPOINT_DIR = (
        run_dir / "checkpoints"
    )

    training.FINAL_MODEL = (
        run_dir / "sac_pos5_scratch_mixed_100k.zip"
    )

    training.REPLAY_PATH = (
        run_dir / "sac_pos5_scratch_mixed_replay.pkl"
    )


# ============================================================
# 5. PREFLIGHT CHECK
# ============================================================

def preflight():

    if not START_MODEL.is_file():
        raise FileNotFoundError(
            f"Starting checkpoint not found: {START_MODEL}"
        )

    if not ORIGINAL_SCRIPT.is_file():
        raise FileNotFoundError(
            f"Original training script missing: {ORIGINAL_SCRIPT}"
        )

    print("[INFO] Starting checkpoint:", START_MODEL)

    env = training.MixedTrackingEnv()

    try:
        # Check the existing Pos5 reward implementation.
        training.verify_reward(env)

        assert env.observation_space.shape == (16,)
        assert env.action_space.shape == (4,)

        model = SAC.load(
            str(START_MODEL),
            env=env,
            device="cpu",
            custom_objects={
                "buffer_size": BUFFER_SIZE,
            },
        )

        assert model.observation_space.shape == (16,)
        assert model.action_space.shape == (4,)

        print("[PASS] Starting SAC checkpoint loaded.")
        print("[INFO] Model timesteps:", model.num_timesteps)

        print("[PASS] Observation/action spaces verified.")
        print("[PASS] Pos5 mixed-task environment verified.")
        print("[PASS] Reward sanity check completed.")

    finally:
        env.close()


# ============================================================
# 6. MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--check-only",
        action="store_true",
    )

    args = parser.parse_args()

    preflight()

    if args.check_only:
        print("[PASS] All preflight checks completed.")
        print("[INFO] No training started.")
        return

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    run_dir = (
        ROOT
        / "results/checkpoints"
        / f"sac_pos5_scratch_mixed_seed{SEED}_{timestamp}"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    configure_training(run_dir)

    # Save experiment configuration for reproducibility.
    script_hash = hashlib.sha256(
        ORIGINAL_SCRIPT.read_bytes()
    ).hexdigest()

    config = {
        "experiment": "Pos5 Scratch Mixed Fine-Tuning",
        "starting_checkpoint": str(START_MODEL),
        "seed": SEED,
        "warmup_steps": WARMUP_STEPS,
        "training_steps": TRAINING_STEPS,
        "learning_rate": LEARNING_RATE,
        "buffer_size": BUFFER_SIZE,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "hover_probability": HOVER_PROBABILITY,
        "circle_probability": 1.0 - HOVER_PROBABILITY,
        "position_weight": POSITION_WEIGHT,
        "original_training_script_sha256": script_hash,
    }

    (run_dir / "experiment.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )

    log_file = run_dir / "training.log"

    print("\n" + "=" * 60)
    print("POS5 SCRATCH -> MIXED FINE-TUNING")
    print("=" * 60)
    print("Starting model:", START_MODEL)
    print("Warm-up steps:", WARMUP_STEPS)
    print("Training steps:", TRAINING_STEPS)
    print("Learning rate:", LEARNING_RATE)
    print("Position weight:", POSITION_WEIGHT)
    print("Output directory:", run_dir)
    print("=" * 60 + "\n")

    # Save all Python terminal output while training.
    with log_file.open(
        "w",
        encoding="utf-8",
        buffering=1,
    ) as log:

        with contextlib.redirect_stdout(
            Tee(sys.stdout, log)
        ), contextlib.redirect_stderr(
            Tee(sys.stderr, log)
        ):

            # Run the previously validated training pipeline.
            training.main()

    print("\n[FINISHED] Mixed training completed.")
    print("[INFO] Results:", run_dir)


if __name__ == "__main__":
    main()
