
"""
Zero-shot circle radius/phase generalization.

Compare:
    - Scratch Pos5 SAC +75k
    - Previous Pos5 SAC
    - PID

No training. No changes to existing models.

Run:
    python scripts/eval_radius_phase.py --check-only
    python -u scripts/eval_radius_phase.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys

from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC


# ============================================================
# 1. Paths and experiment configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import verify_nominal_pos5 as benchmark

from quadrotor_rl.envs.circle_tracking_env import (
    CircleTrackingEnv,
)


SCRATCH_PATH = ROOT / (
    "results/checkpoints/"
    "sac_pos5_scratch_mixed_seed42_20261030_190357/"
    "checkpoints/sac_pos5_77000_steps.zip"
)

PREVIOUS_PATH = ROOT / (
    "results/checkpoints/sac_reward_position5/"
    "sac_pos5_102000_steps.zip"
)

RADII = (0.15, 0.30, 0.45)

PHASES = (
    0.0,
    math.pi / 2,
    math.pi,
    3 * math.pi / 2,
)

OMEGAS = (0.2, 0.6, 1.0)

SEEDS = tuple(range(4000, 4020))


# ============================================================
# 2. Circle environment with initial phase
# ============================================================

class PhaseCircleTrackingEnv(CircleTrackingEnv):
    """
    Same quadrotor environment as CircleTrackingEnv.

    Only the reference trajectory changes:

        theta = omega*t + phi

        x_d = cx + R*cos(theta)
        y_d = cy + R*sin(theta)
        z_d = cz

        vx_d = -R*omega*sin(theta)
        vy_d =  R*omega*cos(theta)
        vz_d = 0

    Physics, observation, actions, reward, and termination
    are unchanged.
    """

    def __init__(
        self,
        *,
        initial_phase_rad=0.0,
        **kwargs,
    ):

        if not np.isfinite(initial_phase_rad):
            raise ValueError(
                "initial_phase_rad must be finite"
            )

        self.initial_phase_rad = float(
            initial_phase_rad
        )

        super().__init__(**kwargs)

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

        return (
            position.astype(np.float32),
            velocity,
        )

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):

        observation, info = super().reset(
            seed=seed,
            options=options,
        )

        info = dict(info)

        info["initial_phase_rad"] = (
            self.initial_phase_rad
        )

        return observation, info


# ============================================================
# 3. Sanity checks
# ============================================================

def preflight():

    for path in (SCRATCH_PATH, PREVIOUS_PATH):

        if not path.is_file():
            raise FileNotFoundError(path)

    # Verify phi=0 reproduces the original environment.

    base = CircleTrackingEnv(
        radius_m=0.30,
        angular_speed_rad_s=0.6,
    )

    phase0 = PhaseCircleTrackingEnv(
        radius_m=0.30,
        angular_speed_rad_s=0.6,
        initial_phase_rad=0.0,
    )

    try:

        obs_a, _ = base.reset(seed=3999)
        obs_b, _ = phase0.reset(seed=3999)

        np.testing.assert_allclose(
            obs_a,
            obs_b,
            atol=1e-6,
            rtol=0,
        )

        for t in (0.0, 0.5, 3.7):

            pa, va = base.circle_reference(t)
            pb, vb = phase0.circle_reference(t)

            np.testing.assert_allclose(
                pa, pb, atol=1e-6, rtol=0
            )

            np.testing.assert_allclose(
                va, vb, atol=1e-6, rtol=0
            )

        print(
            "[PASS] Phase 0 matches original "
            "CircleTrackingEnv."
        )

    finally:

        base.close()
        phase0.close()

    # Verify phi=pi/2 gives the correct initial target.

    env = PhaseCircleTrackingEnv(
        radius_m=0.30,
        angular_speed_rad_s=0.6,
        initial_phase_rad=math.pi / 2,
    )

    try:

        obs, info = env.reset(seed=3999)

        expected_pos = (
            env.circle_center
            + np.array(
                [0, 0.30, 0],
                dtype=np.float32,
            )
        )

        expected_vel = np.array(
            [-0.18, 0, 0],
            dtype=np.float32,
        )

        np.testing.assert_allclose(
            info["target_pos"],
            expected_pos,
            atol=1e-6,
        )

        np.testing.assert_allclose(
            info["target_vel"],
            expected_vel,
            atol=1e-6,
        )

        np.testing.assert_allclose(
            obs[6:9],
            expected_vel,
            atol=1e-6,
        )

        print(
            "[PASS] Phase pi/2 reference and "
            "observation are consistent."
        )

        # Verify compatibility with the existing
        # PID evaluation pipeline.

        result, _ = benchmark.evaluate_episode(
            env,
            None,
            seed=3999,
        )

        assert "survived" in result
        assert "full_rmse_m" in result
        assert "late_rmse_m" in result

        print(
            "[PASS] Existing PID evaluator works "
            "with PhaseCircleTrackingEnv."
        )

        print(
            "[INFO] Episode metric keys:",
            sorted(result),
        )

    finally:

        env.close()


# ============================================================
# 4. Summary helpers
# ============================================================

def report(rows, group_keys):

    groups = defaultdict(list)

    for row in rows:

        key = tuple(
            row[name] for name in group_keys
        )

        groups[key].append(row)

    for group, sample in sorted(groups.items()):

        successful = [
            row
            for row in sample
            if int(row["survived"]) == 1
        ]

        def mean_success(metric):

            values = [
                float(row[metric])
                for row in successful
                if metric in row
                and row[metric] is not None
            ]

            if not values:
                return float("nan")

            return float(np.mean(values))

        descriptor = " ".join(
            f"{name}={value}"
            for name, value in zip(group_keys, group)
        )

        full = mean_success("full_rmse_m")
        late = mean_success("late_rmse_m")
        velocity = mean_success("velocity_rmse_m_s")

        print(
            f"{descriptor:70s} | "
            f"success={len(successful):3d}/"
            f"{len(sample):3d} | "
            f"full={100*full:6.2f}cm | "
            f"late={100*late:6.2f}cm | "
            f"vel={velocity:.4f}m/s"
        )


# ============================================================
# 5. Main
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

        print(
            "[PASS] All preflight checks passed."
        )

        print(
            "[INFO] No full benchmark started."
        )

        return

    # --------------------------------------------------------
    # Load controllers
    # --------------------------------------------------------

    policies = {
        "Scratch_Pos5_75k": SAC.load(
            str(SCRATCH_PATH),
            device="cpu",
        ),

        "Previous_Pos5": SAC.load(
            str(PREVIOUS_PATH),
            device="cpu",
        ),

        # Existing evaluator interprets None as PID.
        "PID": None,
    }

    # --------------------------------------------------------
    # Create unique output folder
    # --------------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output = (
        ROOT
        / "results/evaluation"
        / f"circle_radius_phase_generalization_{timestamp}"
    )

    output.mkdir(
        parents=True,
        exist_ok=False,
    )

    config = {
        "purpose": "Zero-shot radius and phase generalization",
        "radii_m": RADII,
        "phases_rad": PHASES,
        "omegas_rad_s": OMEGAS,
        "seeds": SEEDS,
        "models": {
            "Scratch_Pos5_75k": str(SCRATCH_PATH),
            "Previous_Pos5": str(PREVIOUS_PATH),
            "PID": "DSLPIDControl",
        },
        "note": (
            "RMSE is the mean of per-episode RMSE "
            "among survivors. Failures reported separately."
        ),
    }

    (output / "experiment.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    total = (
        len(RADII)
        * len(PHASES)
        * len(OMEGAS)
        * len(SEEDS)
        * len(policies)
    )

    print()
    print("=" * 65)
    print("ZERO-SHOT RADIUS / PHASE GENERALIZATION")
    print("=" * 65)
    print("Models:", list(policies))
    print("Radii:", RADII)
    print("Phases:", PHASES)
    print("Angular speeds:", OMEGAS)
    print("Seeds:", len(SEEDS))
    print("Total episodes:", total)
    print("Output:", output)
    print("=" * 65)

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    rows = []

    csv_path = output / "episodes.csv"

    with csv_path.open(
        "x",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = None

        for radius in RADII:

            for phase in PHASES:

                for omega in OMEGAS:

                    env = PhaseCircleTrackingEnv(
                        gui=False,
                        physics_freq_hz=240,
                        control_freq_hz=48,
                        radius_m=radius,
                        angular_speed_rad_s=omega,
                        initial_phase_rad=phase,
                    )

                    try:

                        for name, policy in policies.items():

                            for seed in SEEDS:

                                result, _ = (
                                    benchmark.evaluate_episode(
                                        env,
                                        policy,
                                        seed,
                                    )
                                )

                                row = {
                                    **result,
                                    "controller": name,
                                    "radius_m": radius,
                                    "phase_rad": phase,
                                    "omega_rad_s": omega,
                                    "seed": seed,
                                }

                                if writer is None:

                                    writer = csv.DictWriter(
                                        f,
                                        fieldnames=list(row.keys()),
                                    )

                                    writer.writeheader()

                                writer.writerow(row)

                                # Save each episode immediately.
                                f.flush()

                                rows.append(row)

                    finally:

                        env.close()

                    print(
                        f"[DONE] R={radius:.2f} "
                        f"phi={phase/math.pi:.2f}pi "
                        f"omega={omega:.1f} "
                        f"| rows={len(rows)}",
                        flush=True,
                    )

    # --------------------------------------------------------
    # Summaries
    # --------------------------------------------------------

    print("\n=== OVERALL ===")

    report(
        rows,
        ("controller",),
    )

    print("\n=== BY RADIUS ===")

    report(
        rows,
        ("radius_m", "controller"),
    )

    print("\n=== BY PHASE ===")

    report(
        rows,
        ("phase_rad", "controller"),
    )

    print("\n=== BY SPEED ===")

    report(
        rows,
        ("omega_rad_s", "controller"),
    )

    print()
    print("[FINISHED] Evaluation completed.")
    print("Episode CSV:", csv_path)


if __name__ == "__main__":
    main()
