
"""
Development benchmark for radius/phase curriculum checkpoints.

6 controllers x 36 trajectories x 10 seeds = 2160 episodes.
No training. All SAC policies evaluated deterministically.

Run:
    python scripts/eval_curriculum_checkpoints.py --check-only
    python -u scripts/eval_curriculum_checkpoints.py
"""

import argparse
import csv
import json
import math
from itertools import product
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

import verify_nominal_pos5 as benchmark
from eval_radius_phase import PhaseCircleTrackingEnv


ROOT = Path(__file__).resolve().parents[1]

CURRICULUM_DIR = (
    ROOT
    / "results/checkpoints"
    / "pos5_radius_phase_curriculum_seed42_20261106_173230"
)

ORIGINAL_MODEL = (
    ROOT
    / "results/checkpoints"
    / "sac_pos5_scratch_mixed_seed42_20261030_190357"
    / "checkpoints/sac_pos5_77000_steps.zip"
)

MODEL_PATHS = {
    "Scratch_original": ORIGINAL_MODEL,
    "Curriculum_25k": (
        CURRICULUM_DIR / "checkpoints/sac_curriculum_27000_steps.zip"
    ),
    "Curriculum_50k": (
        CURRICULUM_DIR / "checkpoints/sac_curriculum_52000_steps.zip"
    ),
    "Curriculum_75k": (
        CURRICULUM_DIR / "checkpoints/sac_curriculum_77000_steps.zip"
    ),
    "Curriculum_100k": (
        CURRICULUM_DIR / "checkpoints/sac_curriculum_102000_steps.zip"
    ),
}

RADII = (0.15, 0.30, 0.45)
PHASES = (0.0, math.pi / 2, math.pi, 3 * math.pi / 2)
OMEGAS = (0.2, 0.6, 1.0)
SEEDS = tuple(range(5000, 5010))

NAMES = list(MODEL_PATHS) + ["PID"]


def matched(a, b):
    return math.isclose(float(a), float(b), abs_tol=1e-8)


def compute_summary(rows):
    good = [r for r in rows if bool(r["survived"])]

    def mean(key):
        if not good:
            return float("nan")
        return float(np.mean([float(r[key]) for r in good]))

    return {
        "episodes": len(rows),
        "successes": len(good),
        "survival_pct": (
            100 * len(good) / len(rows)
            if rows else float("nan")
        ),
        "full_cm": 100 * mean("full_rmse_m"),
        "late_cm": 100 * mean("late_rmse_m"),
        "velocity_m_s": mean("velocity_rmse_m_s"),
    }


def print_section(title, rows, summary_writer, group):
    print(f"\n=== {title} ===", flush=True)

    for name in NAMES:
        subset = [
            r for r in rows
            if r["controller"] == name
        ]

        s = compute_summary(subset)

        print(
            f"{name:20s} | "
            f"success={s['successes']:3d}/{s['episodes']:3d} | "
            f"survival={s['survival_pct']:6.1f}% | "
            f"full={s['full_cm']:7.3f}cm | "
            f"late={s['late_cm']:7.3f}cm | "
            f"velocity={s['velocity_m_s']:.4f}m/s",
            flush=True,
        )

        summary_writer.writerow({
            "group": group,
            "controller": name,
            **s,
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    print("=== CHECKPOINT CHECK ===")

    for name, path in MODEL_PATHS.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing checkpoint for {name}: {path}"
            )
        print(f"[OK] {name}: {path.name}")

    if args.check_only:
        print("[PASS] All checkpoints found. No evaluation started.")
        return

    policies = {
        name: SAC.load(str(path), device="cpu")
        for name, path in MODEL_PATHS.items()
    }
    policies["PID"] = None

    out = (
        ROOT
        / "results/evaluation"
        / (
            "curriculum_checkpoint_development_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    )
    out.mkdir(parents=True, exist_ok=False)

    config = {
        "purpose": "Development checkpoint selection, not held-out confirmation",
        "radii_m": list(RADII),
        "phases_rad": list(PHASES),
        "omegas_rad_s": list(OMEGAS),
        "seeds": list(SEEDS),
        "models": {
            name: str(path)
            for name, path in MODEL_PATHS.items()
        },
        "pid": "Existing verify_nominal_pos5 PID evaluator",
    }

    (out / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    rows = []
    total = (
        len(RADII) * len(PHASES)
        * len(OMEGAS) * len(SEEDS)
        * len(policies)
    )

    print(f"\nTotal episodes: {total}")
    print(f"Results: {out}")
    print("Starting evaluation...\n", flush=True)

    csv_path = out / "episodes.csv"

    # Write every episode immediately.
    with csv_path.open("x", newline="", encoding="utf-8") as f:
        writer = None

        for radius, phase, omega in product(
            RADII, PHASES, OMEGAS
        ):
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
                        metrics, _ = benchmark.evaluate_episode(
                            env, policy, seed
                        )

                        row = {
                            **metrics,
                            "controller": name,
                            "radius_m": radius,
                            "phase_rad": phase,
                            "omega_rad_s": omega,
                            "seed": seed,
                        }

                        if writer is None:
                            writer = csv.DictWriter(
                                f, fieldnames=list(row)
                            )
                            writer.writeheader()

                        writer.writerow(row)
                        f.flush()
                        rows.append(row)

            finally:
                env.close()

            print(
                f"[DONE] R={radius:.2f} "
                f"phi={phase/math.pi:.2f}pi "
                f"omega={omega:.1f} "
                f"| {len(rows)}/{total}",
                flush=True,
            )

    summary_path = out / "summary.csv"

    with summary_path.open(
        "x", newline="", encoding="utf-8"
    ) as f:
        summary_writer = csv.DictWriter(
            f,
            fieldnames=[
                "group",
                "controller",
                "episodes",
                "successes",
                "survival_pct",
                "full_cm",
                "late_cm",
                "velocity_m_s",
            ],
        )
        summary_writer.writeheader()

        print_section(
            "OVERALL",
            rows,
            summary_writer,
            "overall",
        )

        for radius in RADII:
            subset = [
                r for r in rows
                if matched(r["radius_m"], radius)
            ]
            print_section(
                f"RADIUS {radius:.2f} m",
                subset,
                summary_writer,
                f"radius_{radius:.2f}",
            )

        nominal = [
            r for r in rows
            if matched(r["radius_m"], 0.30)
            and matched(r["phase_rad"], 0.0)
        ]

        print_section(
            "ORIGINAL NOMINAL TASK (R=0.30, phi=0)",
            nominal,
            summary_writer,
            "original_nominal",
        )

        for phase in PHASES:
            hard = [
                r for r in rows
                if matched(r["radius_m"], 0.45)
                and matched(r["phase_rad"], phase)
            ]

            print_section(
                f"HARD R=0.45, phi={phase/math.pi:.2f}pi",
                hard,
                summary_writer,
                f"hard_phase_{phase/math.pi:.2f}pi",
            )

    print("\n[FINISHED] Evaluation completed.")
    print("Episodes:", csv_path)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
