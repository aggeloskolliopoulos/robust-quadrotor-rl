
"""
Targeted radius/phase curriculum for the Scratch Pos5 SAC.

Stage 0: 2k policy-based replay warm-up (no gradient updates)
Stage 1: 50k learning steps, R ~ U(0.15, 0.35)
Stage 2: 50k learning steps, R ~ U(0.15, 0.45)

Phi ~ U(0, 2*pi), omega ~ U(0.2, 1.0)
Hover probability: 25%
Reward: existing Pos5, unchanged

Run:
    python scripts/train_pos5_radius_phase_curriculum.py --check-only
    python -u scripts/train_pos5_radius_phase_curriculum.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys

from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.utils import get_schedule_fn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import finetune_sac_circle as previous_training

from quadrotor_rl.envs.circle_tracking_env import (
    CircleTrackingEnv,
)


# ============================================================
# 1. Configuration
# ============================================================

START_MODEL = ROOT / (
    "results/checkpoints/"
    "sac_pos5_scratch_mixed_seed42_20261030_190357/"
    "checkpoints/sac_pos5_77000_steps.zip"
)

SEED = 42

WARMUP_STEPS = 2_000
TRAINING_STEPS = 100_000
STAGE1_STEPS = 50_000

CHECKPOINT_EVERY = 25_000

LEARNING_RATE = 3e-5
BUFFER_SIZE = 200_000


# ============================================================
# 2. Position-focused circle with randomized radius/phase
# ============================================================

class CurriculumCircleEnv(
    previous_training.PositionFocusedRewardMixin,
    CircleTrackingEnv,
):
    """
    Keep physics, observations, action space, termination,
    and Pos5 reward unchanged.

    Randomize the reference at each episode reset.
    """

    def __init__(self, **kwargs):

        self.initial_phase_rad = 0.0

        self.radius_min = 0.15
        self.radius_max = 0.35

        super().__init__(
            radius_m=0.30,
            angular_speed_rad_s=0.20,
            **kwargs,
        )

    def circle_reference(self, t):

        theta = (
            self.angular_speed_rad_s * t
            + self.initial_phase_rad
        )

        r = self.radius_m
        omega = self.angular_speed_rad_s

        position = self.circle_center + np.array(
            [
                r * np.cos(theta),
                r * np.sin(theta),
                0.0,
            ],
            dtype=np.float32,
        )

        velocity = np.array(
            [
                -r * omega * np.sin(theta),
                 r * omega * np.cos(theta),
                 0.0,
            ],
            dtype=np.float32,
        )

        return position.astype(np.float32), velocity

    def set_radius_range(self, low, high):

        if not (0 < low <= high):
            raise ValueError("Invalid radius range")

        self.radius_min = float(low)
        self.radius_max = float(high)

    def reset(self, *, seed=None, options=None):

        # Parent reset samples the initial physical state.
        # Its observation initially uses the previous
        # reference, which we correct below.

        obs, info = super().reset(
            seed=seed,
            options=options,
        )

        old_pos = self.target_pos.copy()
        old_vel = self.target_vel.copy()

        # Gymnasium's seeded generator.
        self.radius_m = float(
            self.np_random.uniform(
                self.radius_min,
                self.radius_max,
            )
        )

        self.initial_phase_rad = float(
            self.np_random.uniform(
                0.0,
                2.0 * np.pi,
            )
        )

        self.angular_speed_rad_s = float(
            self.np_random.uniform(
                0.20,
                1.00,
            )
        )

        # New reference at t=0.
        new_pos, new_vel = self.circle_reference(0.0)

        self.target_pos = new_pos
        self.target_vel = new_vel

        self.elapsed_steps = 0

        # Correct the 16D observation.
        obs = obs.copy()

        obs[0:3] += new_pos - old_pos
        obs[3:6] += new_vel - old_vel
        obs[6:9] = new_vel

        info = dict(info)

        info.update(
            radius_m=self.radius_m,
            initial_phase_rad=self.initial_phase_rad,
            angular_speed_rad_s=self.angular_speed_rad_s,
            target_pos=new_pos.copy(),
            target_vel=new_vel.copy(),
            reference_time_s=0.0,
        )

        return obs, info


# ============================================================
# 3. Reuse the existing mixed-task curriculum
# ============================================================

class CurriculumMixedEnv(
    previous_training.MixedTrackingEnv
):

    def __init__(self):

        super().__init__()

        # Replace only the circle task.
        # Keep the existing hover task and task sampler.

        old_circle = self.circle_env

        self.circle_env = CurriculumCircleEnv(
            gui=False,
            physics_freq_hz=240,
            control_freq_hz=48,
        )

        old_circle.close()


# ============================================================
# 4. Automatic curriculum stage transition
# ============================================================

class SwitchRadiusStage(BaseCallback):

    def __init__(self, env):

        super().__init__()

        self.env = env
        self.switched = False

    def _on_step(self):

        # Model timesteps include the 2k warm-up.
        if (
            not self.switched
            and self.num_timesteps
            >= WARMUP_STEPS + STAGE1_STEPS
        ):

            self.env.circle_env.set_radius_range(
                0.15,
                0.45,
            )

            self.switched = True

            print(
                "\n[CURRICULUM] Stage 2 started: "
                "R ~ U(0.15, 0.45) m\n",
                flush=True,
            )

        return True


# ============================================================
# 5. Preflight
# ============================================================

def preflight(env):

    if not START_MODEL.is_file():
        raise FileNotFoundError(START_MODEL)

    # The existing reward verification also checks
    # that ONLY the position coefficient is changed.
    previous_training.verify_reward(env)

    assert env.observation_space.shape == (16,)
    assert env.action_space.shape == (4,)

    # Verify seeded reproducibility.
    obs1, info1 = env.circle_env.reset(seed=8877)
    obs2, info2 = env.circle_env.reset(seed=8877)

    np.testing.assert_allclose(
        obs1,
        obs2,
        atol=1e-6,
        rtol=0,
    )

    for key in (
        "radius_m",
        "initial_phase_rad",
        "angular_speed_rad_s",
    ):
        assert np.isclose(info1[key], info2[key])

    # Verify correct reference and observation
    # for several randomized episodes.
    for seed in (8877, 8878, 8879):

        obs, info = env.circle_env.reset(seed=seed)

        p, v = env.circle_env.circle_reference(0.0)

        np.testing.assert_allclose(
            p,
            info["target_pos"],
            atol=1e-6,
        )

        np.testing.assert_allclose(
            v,
            obs[6:9],
            atol=1e-6,
        )

        np.testing.assert_allclose(
            obs[:3],
            p - info["initial_pos"],
            atol=1e-5,
        )

        assert 0.15 <= info["radius_m"] <= 0.35
        assert 0 <= info["initial_phase_rad"] <= 2 * math.pi
        assert 0.2 <= info["angular_speed_rad_s"] <= 1.0

    model = SAC.load(
        str(START_MODEL),
        env=env,
        device="cpu",
        custom_objects={
            "buffer_size": BUFFER_SIZE,
        },
    )

    print(
        "[PASS] Pos5 reward and randomized "
        "reference verified."
    )

    print("[PASS] Observation/action spaces verified.")
    print("[PASS] SAC checkpoint loaded successfully.")

    print(
        "[INFO] Starting checkpoint timesteps:",
        model.num_timesteps,
    )

    return model


# ============================================================
# 6. Training
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--check-only",
        action="store_true",
    )

    args = parser.parse_args()

    env = CurriculumMixedEnv()

    try:

        model = preflight(env)

        if args.check_only:
            print("[PASS] All preflight checks completed.")
            print("[INFO] No training started.")
            return

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        out = (
            ROOT
            / "results/checkpoints"
            / (
                f"pos5_radius_phase_curriculum_"
                f"seed{SEED}_{timestamp}"
            )
        )

        out.mkdir(
            parents=True,
            exist_ok=False,
        )

        config = {
            "starting_checkpoint": str(START_MODEL),
            "training_seed": SEED,
            "warmup_steps": WARMUP_STEPS,
            "training_steps": TRAINING_STEPS,
            "stage1_learning_steps": STAGE1_STEPS,
            "stage1_radius_range_m": [0.15, 0.35],
            "stage2_radius_range_m": [0.15, 0.45],
            "phase_rad": [0, 2 * math.pi],
            "omega_rad_s": [0.2, 1.0],
            "hover_probability": (
                previous_training.HOVER_PROBABILITY
            ),
            "learning_rate": LEARNING_RATE,
            "replay_capacity": BUFFER_SIZE,
            "reward": "Existing Pos5, unchanged",
        }

        (out / "config.json").write_text(
            json.dumps(config, indent=2),
            encoding="utf-8",
        )

        # Same fine-tuning hyperparameters
        # as the previous successful experiments.
        model.set_random_seed(SEED)

        model.learning_rate = LEARNING_RATE
        model.lr_schedule = get_schedule_fn(
            LEARNING_RATE
        )

        # ----------------------------------------------------
        # Replay warm-up, without gradient updates
        # ----------------------------------------------------

        model.learning_starts = 0
        model.gradient_steps = 0

        actor_before = [
            p.detach().clone()
            for p in model.actor.parameters()
        ]

        print(
            "\n[WARMUP] Collecting 2000 transitions...",
            flush=True,
        )

        model.learn(
            total_timesteps=WARMUP_STEPS,
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

        assert actor_unchanged

        print(
            "[PASS] Actor unchanged during warm-up."
        )

        print(
            "[INFO] Replay transitions:",
            model.replay_buffer.size(),
        )

        # ----------------------------------------------------
        # Actual fine-tuning
        # ----------------------------------------------------

        model.gradient_steps = 1

        callbacks = CallbackList([
            CheckpointCallback(
                save_freq=CHECKPOINT_EVERY,
                save_path=str(out / "checkpoints"),
                name_prefix="sac_curriculum",
                save_replay_buffer=True,
                verbose=1,
            ),
            SwitchRadiusStage(env),
        ])

        print("\n" + "=" * 60)
        print("POS5 RADIUS / PHASE CURRICULUM")
        print("=" * 60)

        print("Starting model:", START_MODEL)
        print("Stage 1: 50k steps, radius 0.15–0.35 m")
        print("Stage 2: 50k steps, radius 0.15–0.45 m")
        print("Phase: U(0, 2pi)")
        print("Omega: U(0.2, 1.0)")
        print("Position weight: 5")
        print("Output:", out)
        print("=" * 60 + "\n")

        model.learn(
            total_timesteps=TRAINING_STEPS,
            reset_num_timesteps=False,
            callback=callbacks,
            log_interval=20,
        )

        # ----------------------------------------------------
        # Save final state
        # ----------------------------------------------------

        final_path = (
            out / "sac_curriculum_final_100k.zip"
        )

        model.save(str(final_path))

        model.save_replay_buffer(
            str(out / "replay_final.pkl")
        )

        print("\n[FINISHED] Training completed.")
        print("Final model:", final_path)
        print("Episodes by task:", env.episode_counts)
        print("Steps by task:", env.step_counts)

    finally:
        env.close()


if __name__ == "__main__":
    main()
