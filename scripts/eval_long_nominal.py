
"""Screen long nominal SAC checkpoints on new evaluation seeds.

36 circular scenarios x 10 seeds = 360 episodes/controller.
Baseline + 8 long-training checkpoints = 3240 episodes.

All models evaluated with deterministic actions.
No training. No checkpoints are modified.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys

from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC


# ============================================================
# 1. Paths and imports
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_radius_phase import PhaseCircleTrackingEnv


BASELINE = ROOT / (
    "results/checkpoints/"
    "curriculum_ablation_baseline_seed42_20261114_145522_495268/"
    "checkpoints/sac_curriculum_ablation_030000_steps.zip"
)

RUN_DIR = ROOT / (
    "results/checkpoints/"
    "sac_long_nominal_from_baseline30k_seed42_20261115_130123_642163"
)

STEPS = list(
    range(25_000, 200_001, 25_000)
)

MODELS = {
    "Baseline_30k": BASELINE,
}

MODELS.update({
    f"Long_{steps // 1000}k":
        RUN_DIR
        / "checkpoints"
        / f"sac_long_nominal_{steps:06d}_steps.zip"

    for steps in STEPS
})


# ============================================================
# 2. Evaluation settings
# ============================================================

RADII = (
    0.15,
    0.30,
    0.45,
)

PHASES = (
    0.0,
    math.pi / 2,
    math.pi,
    3 * math.pi / 2,
)

OMEGAS = (
    0.2,
    0.6,
    1.0,
)

MAX_STEPS = 480
LATE_START = 240

METRICS = (
    "full_xyz_m",
    "late_xyz_m",
    "late_xy_m",
    "late_z_m",
)


# ============================================================
# 3. Evaluate one episode
# ============================================================

def evaluate_episode(env, model, seed):

    obs, _ = env.reset(seed=seed)

    pos_errors = []
    survived = False

    for _ in range(MAX_STEPS):

        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        pos_errors.append(
            np.asarray(
                obs[:3],
                dtype=np.float64,
            ).copy()
        )

        if terminated or truncated:

            survived = bool(
                truncated and not terminated
            )

            break

    result = {
        "survived": int(survived),
        "duration_s": len(pos_errors) / 48,
    }

    result.update({
        m: float("nan")
        for m in METRICS
    })

    # Only completed episodes contribute to RMSE.
    if survived:

        e = np.asarray(
            pos_errors,
            dtype=np.float64,
        )

        late = e[LATE_START:]

        def euclidean_rmse(x):

            return float(
                np.sqrt(
                    np.mean(
                        np.sum(x**2, axis=1)
                    )
                )
            )

        result.update({
            "full_xyz_m":
                euclidean_rmse(e),

            "late_xyz_m":
                euclidean_rmse(late),

            "late_xy_m":
                euclidean_rmse(late[:, :2]),

            "late_z_m":
                euclidean_rmse(late[:, 2:3]),
        })

    return result


# ============================================================
# 4. Summary and matched-success statistics
# ============================================================

def episode_key(row):

    return (
        row["radius"],
        row["phase"],
        row["omega"],
        row["seed"],
    )


def stats(rows):

    successful = [
        r
        for r in rows
        if r["survived"]
    ]

    summary = {
        "survival": len(successful),
    }

    for metric in METRICS:

        summary[metric] = (
            100 * float(
                np.mean([
                    r[metric]
                    for r in successful
                ])
            )
            if successful
            else float("nan")
        )

    return summary


def compare_with_baseline(
    rows,
    baseline_rows,
    name,
):

    candidate = {
        episode_key(r): r
        for r in rows
        if r["survived"]
    }

    baseline = {
        episode_key(r): r
        for r in baseline_rows
        if r["survived"]
    }

    shared = sorted(
        set(candidate) & set(baseline)
    )

    candidate_only = len(
        set(candidate) - set(baseline)
    )

    baseline_only = len(
        set(baseline) - set(candidate)
    )

    print()
    print(
        f"{name}: matched={len(shared)}, "
        f"candidate-only successes={candidate_only}, "
        f"baseline-only successes={baseline_only}"
    )

    if not shared:
        return

    for metric in METRICS:

        candidate_mean = (
            100 * float(
                np.mean([
                    candidate[k][metric]
                    for k in shared
                ])
            )
        )

        baseline_mean = (
            100 * float(
                np.mean([
                    baseline[k][metric]
                    for k in shared
                ])
            )
        )

        print(
            f"  {metric:<12}: "
            f"{candidate_mean:.3f} vs "
            f"{baseline_mean:.3f} cm "
            f"| delta="
            f"{candidate_mean - baseline_mean:+.3f} cm"
        )


# ============================================================
# 5. Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed-start",
        type=int,
        default=13000,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
    )

    args = parser.parse_args()

    if args.seeds <= 0:

        parser.error(
            "--seeds must be positive"
        )

    # Check all checkpoint paths before evaluation.

    for name, path in MODELS.items():

        if not path.is_file():

            raise FileNotFoundError(
                f"{name}: {path}"
            )

    print(
        f"[PASS] Found all "
        f"{len(MODELS)} model checkpoints.",
        flush=True,
    )

    # --------------------------------------------------------
    # Quick preflight
    # --------------------------------------------------------

    if args.check_only:

        model = SAC.load(
            str(BASELINE),
            device="cpu",
        )

        env = PhaseCircleTrackingEnv(
            gui=False,
            physics_freq_hz=240,
            control_freq_hz=48,
            radius_m=0.30,
            initial_phase_rad=0.0,
            angular_speed_rad_s=0.6,
        )

        try:

            result = evaluate_episode(
                env,
                model,
                args.seed_start,
            )

            print(
                "[CHECK] One baseline episode:",
                result,
            )

        finally:

            env.close()

        print(
            "[PASS] Preflight complete. "
            "No full benchmark."
        )

        return

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output_dir = (
        ROOT
        / "results"
        / "evaluation"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    output_csv = output_dir / (
        f"long_nominal_screen_{timestamp}.csv"
    )

    fields = [
        "controller",
        "radius",
        "phase",
        "omega",
        "seed",
        "survived",
        "duration_s",
        *METRICS,
    ]

    seeds = range(
        args.seed_start,
        args.seed_start + args.seeds,
    )

    episodes_per_model = (
        len(RADII)
        * len(PHASES)
        * len(OMEGAS)
        * args.seeds
    )

    results = {}

    print()
    print("=" * 64)
    print("LONG NOMINAL CHECKPOINT SCREENING")
    print("=" * 64)

    print(
        f"Controllers: {len(MODELS)}"
    )

    print(
        f"Episodes/controller: "
        f"{episodes_per_model}"
    )

    print(
        f"Total episodes: "
        f"{len(MODELS) * episodes_per_model}"
    )

    print(
        f"CSV: {output_csv}",
        flush=True,
    )

    # --------------------------------------------------------
    # Evaluate all controllers
    # --------------------------------------------------------

    with output_csv.open(
        "x",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for name, path in MODELS.items():

            print()
            print(
                f"[EVALUATING] {name}",
                flush=True,
            )

            model = SAC.load(
                str(path),
                device="cpu",
            )

            rows = []

            for radius in RADII:

                for phase in PHASES:

                    for omega in OMEGAS:

                        env = PhaseCircleTrackingEnv(
                            gui=False,
                            physics_freq_hz=240,
                            control_freq_hz=48,
                            radius_m=radius,
                            initial_phase_rad=phase,
                            angular_speed_rad_s=omega,
                        )

                        try:

                            for seed in seeds:

                                metrics = evaluate_episode(
                                    env,
                                    model,
                                    seed,
                                )

                                row = {
                                    "controller": name,
                                    "radius": radius,
                                    "phase": phase,
                                    "omega": omega,
                                    "seed": seed,
                                    **metrics,
                                }

                                rows.append(row)
                                writer.writerow(row)

                        finally:

                            env.close()

                        f.flush()

            results[name] = rows
            summary = stats(rows)

            print(
                f"[DONE] {name}: "
                f"survival="
                f"{summary['survival']}/"
                f"{episodes_per_model}, "
                f"full="
                f"{summary['full_xyz_m']:.3f}cm, "
                f"late="
                f"{summary['late_xyz_m']:.3f}cm, "
                f"lateZ="
                f"{summary['late_z_m']:.3f}cm",
                flush=True,
            )

            del model

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 64)
    print("ALL CONTROLLERS")
    print("=" * 64)

    print(
        f"{'Controller':<16} "
        f"{'Surv.':>8} "
        f"{'Full':>9} "
        f"{'Late':>9} "
        f"{'Late XY':>9} "
        f"{'Late Z':>9}"
    )

    for name, rows in results.items():

        s = stats(rows)

        print(
            f"{name:<16} "
            f"{s['survival']:>3}/"
            f"{episodes_per_model:<4} "
            f"{s['full_xyz_m']:>9.3f} "
            f"{s['late_xyz_m']:>9.3f} "
            f"{s['late_xy_m']:>9.3f} "
            f"{s['late_z_m']:>9.3f}"
        )

    print(
        "RMSE in cm, successful episodes only."
    )

    # --------------------------------------------------------
    # Fair pairwise comparisons
    # --------------------------------------------------------

    print()
    print("=" * 64)
    print("MATCHED SUCCESS VS BASELINE")
    print("=" * 64)

    for name, rows in results.items():

        if name != "Baseline_30k":

            compare_with_baseline(
                rows,
                results["Baseline_30k"],
                name,
            )

    # --------------------------------------------------------
    # Hard scenario
    # --------------------------------------------------------

    print()
    print("=" * 64)
    print("HARD SCENARIO: R=.45 phi=0 omega=1")
    print("=" * 64)

    for name, rows in results.items():

        hard = [
            r
            for r in rows
            if (
                r["radius"] == 0.45
                and r["phase"] == 0.0
                and r["omega"] == 1.0
            )
        ]

        successes = sum(
            r["survived"]
            for r in hard
        )

        print(
            f"{name:<16} "
            f"{successes}/{len(hard)}"
        )

    print()
    print(
        f"[FINISHED] {output_csv}",
        flush=True,
    )


if __name__ == "__main__":
    main()
