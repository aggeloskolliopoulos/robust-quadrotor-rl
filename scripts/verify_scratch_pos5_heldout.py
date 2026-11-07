
"""
Held-out nominal benchmark:
    PID vs Previous Pos5 vs Scratch Pos5 +75k.

Reuses verify_nominal_pos5.py.
Does not modify existing evaluation files or checkpoints.
"""

import sys
import csv
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import verify_nominal_pos5 as ev


# ------------------------------------------------------------
# 1. New held-out seeds
# ------------------------------------------------------------

ev.CIRCLE_SEEDS = range(3000, 3050)
ev.HOVER_SEEDS = range(3100, 3150)

ev.OMEGAS = (0.2, 0.4, 0.6, 0.8, 1.0)
ev.PLOT_OMEGAS = (0.4, 1.0)


# ------------------------------------------------------------
# 2. SAC checkpoints
# PID is handled by the original evaluator.
# ------------------------------------------------------------

ev.MODEL_PATHS = {
    "Previous_Pos5": (
        ROOT
        / "results/checkpoints/sac_reward_position5"
        / "sac_pos5_102000_steps.zip"
    ),

    "Scratch_Pos5_75k": (
        ROOT
        / "results/checkpoints"
        / "sac_pos5_scratch_mixed_seed42_20261030_190357"
        / "checkpoints/sac_pos5_77000_steps.zip"
    ),
}


# ------------------------------------------------------------
# 3. New output directory
# ------------------------------------------------------------

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

ev.OUT = (
    ROOT
    / "results/evaluation"
    / f"scratch_pos5_heldout_{timestamp}"
)

if ev.OUT.exists():
    raise FileExistsError(ev.OUT)


# ------------------------------------------------------------
# 4. Preflight checks
# ------------------------------------------------------------

assert callable(ev.main)

for name, path in ev.MODEL_PATHS.items():
    if not path.is_file():
        raise FileNotFoundError(f"{name}: {path}")

    print(f"[OK] {name}: {path}")

print("\nHELD-OUT EXPERIMENT")
print("Circle seeds:", list(ev.CIRCLE_SEEDS))
print("Hover seeds:", list(ev.HOVER_SEEDS))
print("Angular speeds:", ev.OMEGAS)
print("Output:", ev.OUT)
print()



# ============================================================
# Correct paired bootstrap for the new controller names
# ============================================================

import numpy as np


def paired_bootstrap_scratch(circle_rows, key, n_boot=4000):
    """
    Compare Scratch_Pos5_75k against:
      1. PID
      2. Previous_Pos5

    Paired bootstrap over seed clusters.
    Each seed contributes its mean difference across 5 speeds.

    Negative difference means Scratch SAC is better.

    Save circle results BEFORE statistical analysis.
    """

    # Preserve the 750 completed circle evaluations even if
    # the statistical analysis encounters an error.
    ev.OUT.mkdir(parents=True, exist_ok=True)

    csv_path = ev.OUT / "circle_episodes.csv"

    if not csv_path.exists():
        ev.save_csv(csv_path, circle_rows)
        print(f"[SAVED] {csv_path}", flush=True)

    indexed = {
        (
            row["controller"],
            row["omega_rad_s"],
            row["seed"],
        ): row
        for row in circle_rows
    }

    for baseline_name in ("PID", "Previous_Pos5"):

        seed_differences = []
        incomplete_seeds = []

        for seed in ev.CIRCLE_SEEDS:

            differences = []

            for omega in ev.OMEGAS:

                scratch = indexed.get(
                    ("Scratch_Pos5_75k", omega, seed)
                )

                baseline = indexed.get(
                    (baseline_name, omega, seed)
                )

                if scratch is None or baseline is None:
                    break

                if not (
                    scratch["survived"]
                    and baseline["survived"]
                ):
                    break

                differences.append(
                    float(scratch[key])
                    - float(baseline[key])
                )

            if len(differences) == len(ev.OMEGAS):
                seed_differences.append(
                    float(np.mean(differences))
                )
            else:
                incomplete_seeds.append(seed)

        if not seed_differences:
            print(
                f"[CI UNAVAILABLE] Scratch vs {baseline_name}: "
                "no complete paired successful seeds."
            )
            continue

        diffs = np.asarray(
            seed_differences, dtype=np.float64
        )

        rng = np.random.default_rng(2026)

        indices = rng.integers(
            low=0,
            high=len(diffs),
            size=(n_boot, len(diffs)),
        )

        boot_means = np.mean(
            diffs[indices],
            axis=1,
        )

        ci_low, ci_high = np.percentile(
            boot_means, [2.5, 97.5]
        )

        mean_diff = float(np.mean(diffs))

        print()
        print("=" * 65)
        print(f"PAIRED BOOTSTRAP: {key}")
        print(f"Scratch_Pos5_75k vs {baseline_name}")
        print("=" * 65)
        print(f"Paired seed clusters: {len(diffs)}")
        print(f"Incomplete seed clusters: {len(incomplete_seeds)}")
        print(f"Mean difference: {mean_diff * 100:.4f} cm")
        print(
            f"95% bootstrap CI: "
            f"[{ci_low * 100:.4f}, {ci_high * 100:.4f}] cm"
        )

        if ci_high < 0:
            print(
                "Result: Scratch has lower RMSE; "
                "95% CI excludes zero."
            )
        elif ci_low > 0:
            print(
                "Result: Scratch has higher RMSE; "
                "95% CI excludes zero."
            )
        else:
            print(
                "Result: CI includes zero; "
                "no clear difference at this level."
            )

        if incomplete_seeds:
            print(
                "WARNING: CI is conditional on complete "
                "successful pairs. Report failures separately."
            )


# Replace only the statistical-analysis function.
# The environment, PID, SAC loading, and metrics remain unchanged.
ev.paired_bootstrap = paired_bootstrap_scratch

# Run the original evaluation pipeline.
ev.main()



# ------------------------------------------------------------
# 6. Verify output files and controller coverage
# ------------------------------------------------------------

for filename, expected_rows in (
    ("circle_episodes.csv", 750),
    ("hover_episodes.csv", 150),
):
    path = ev.OUT / filename

    if not path.is_file():
        raise FileNotFoundError(
            f"Expected evaluation output missing: {path}"
        )

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"\n{filename}: {len(rows)} rows")

    if len(rows) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} rows, "
            f"found {len(rows)}. "
            "Check that PID and both SAC models were evaluated."
        )

    if not any(
        str(value) == "PID"
        for row in rows
        for value in row.values()
    ):
        raise RuntimeError(
            f"PID results not found in {filename}."
        )

print("\n[PASS] Held-out evaluation completed.")
print("Results:", ev.OUT)
