
"""Extended research analysis: mass, inertia, motor and paired gust.

Run from the repository root:
    python -u scripts/research_analysis_extended.py

Reads existing CSV files only. No training or simulations.
Separate inertia/motor sweeps are not merged.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONTROLLERS = ("PID", "Baseline_30k", "Final_20k")

EPISODE_KEY = (
    "radius_m",
    "phase_rad",
    "omega_rad_s",
    "seed",
)

COLORS = {
    "PID": "#2271B2",
    "Baseline_30k": "#E1812C",
    "Final_20k": "#339967",
}

DISPLAY = {
    "PID": "PID",
    "Baseline_30k": "SAC Baseline 30k",
    "Final_20k": "SAC Final 20k",
}


# ============================================================
# 1. CSV utilities
# ============================================================

def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def find_sweeps(
    folder, pattern, level_col, min_rows, keep_all
):
    candidates = []

    for path in folder.glob(pattern):
        rows = read_csv(path)

        if len(rows) < min_rows:
            continue

        required = set(EPISODE_KEY) | {
            "controller", "survived", level_col
        }

        if not rows or not required.issubset(rows[0]):
            continue

        controllers = {
            r["controller"] for r in rows
        }

        if not {
            "PID", "Baseline_30k"
        }.issubset(controllers):
            continue

        candidates.append((
            len(rows),
            path.stat().st_mtime,
            path,
            rows,
        ))

    candidates.sort(
        key=lambda item: (item[0], item[1]),
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(
            f"No substantial benchmark: {pattern}"
        )

    return (
        candidates if keep_all
        else candidates[:1]
    )


# ============================================================
# 2. Matching episodes
# ============================================================

def episode_index(rows):
    indexed = {}

    for row in rows:
        key = tuple(
            row[c] for c in EPISODE_KEY
        )

        if key in indexed:
            raise ValueError(
                f"Duplicate episode: {key}"
            )

        indexed[key] = row

    return indexed


def matched_pairs(a_rows, b_rows, metric):
    a = episode_index(a_rows)
    b = episode_index(b_rows)

    pairs = []

    for key in sorted(a.keys() & b.keys()):
        ra = a[key]
        rb = b[key]

        # Tracking only when both flights completed.
        if metric != "survived":
            if not (
                int(ra["survived"])
                and int(rb["survived"])
            ):
                continue

        va = float(ra[metric])
        vb = float(rb[metric])

        if np.isfinite(va) and np.isfinite(vb):
            pairs.append((
                str(ra["seed"]),
                va - vb,
            ))

    return pairs


# ============================================================
# 3. Seed-block bootstrap
# ============================================================

def seed_bootstrap(
    pairs,
    factor=1.0,
    n_boot=3000,
    random_seed=42,
):
    if not pairs:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            0,
            0,
        )

    grouped = defaultdict(list)

    for seed, value in pairs:
        grouped[str(seed)].append(
            float(value) * factor
        )

    blocks = list(grouped.values())

    sums = np.asarray([
        sum(block) for block in blocks
    ])

    counts = np.asarray([
        len(block) for block in blocks
    ])

    estimate = (
        sums.sum() / counts.sum()
    )

    rng = np.random.default_rng(
        random_seed
    )

    picks = rng.integers(
        0,
        len(blocks),
        size=(n_boot, len(blocks)),
    )

    samples = (
        sums[picks].sum(axis=1)
        / counts[picks].sum(axis=1)
    )

    low, high = np.percentile(
        samples, [2.5, 97.5]
    )

    return (
        float(estimate),
        float(low),
        float(high),
        int(counts.sum()),
        len(blocks),
    )


def compare(
    output,
    scenario,
    dataset,
    a_rows,
    b_rows,
    a_label,
    b_label,
    n_boot,
):
    for metric, units, factor in (
        ("survived", "survival_pp", 100.0),
        (
            "late_xyz_m",
            "late_xyz_cm_both_survived",
            100.0,
        ),
    ):
        pairs = matched_pairs(
            a_rows, b_rows, metric
        )

        mean, low, high, n, seeds = (
            seed_bootstrap(
                pairs,
                factor=factor,
                n_boot=n_boot,
            )
        )

        output.append({
            "dataset": dataset,
            "scenario": str(scenario),
            "comparison":
                f"{a_label}_minus_{b_label}",
            "metric": units,
            "estimate": mean,
            "ci95_low": low,
            "ci95_high": high,
            "n_matched": n,
            "n_seed_blocks": seeds,
        })


# ============================================================
# 4. Difference in tracking degradation
# ============================================================

def compare_degradation(
    output,
    scenario,
    dataset,
    rows,
    level_col,
    nominal,
    a_label,
    b_label,
    n_boot,
):
    """
    (A_dist - A_nominal) - (B_dist - B_nominal)

    Include an episode only if all four
    controller/scenario flights completed.
    """

    indexes = {}

    for label in (a_label, b_label):
        for condition, level in (
            ("dist", scenario),
            ("nom", nominal),
        ):
            subset = [
                r for r in rows
                if (
                    r["controller"] == label
                    and r[level_col] == level
                )
            ]

            indexes[(label, condition)] = (
                episode_index(subset)
            )

    common = set.intersection(
        *(
            set(index.keys())
            for index in indexes.values()
        )
    )

    pairs = []

    for key in sorted(common):
        all_rows = [
            index[key]
            for index in indexes.values()
        ]

        if not all(
            int(r["survived"]) == 1
            for r in all_rows
        ):
            continue

        def value(label, condition):
            return float(
                indexes[
                    (label, condition)
                ][key]["late_xyz_m"]
            )

        delta_a = (
            value(a_label, "dist")
            - value(a_label, "nom")
        )

        delta_b = (
            value(b_label, "dist")
            - value(b_label, "nom")
        )

        difference = delta_a - delta_b

        if np.isfinite(difference):
            pairs.append((
                indexes[
                    (a_label, "dist")
                ][key]["seed"],
                difference,
            ))

    mean, low, high, n, seeds = (
        seed_bootstrap(
            pairs,
            factor=100.0,
            n_boot=n_boot,
        )
    )

    output.append({
        "dataset": dataset,
        "scenario": str(scenario),
        "comparison":
            f"{a_label}_minus_{b_label}",
        "metric":
            "difference_in_degradation_cm_all4_survived",
        "estimate": mean,
        "ci95_low": low,
        "ci95_high": high,
        "n_matched": n,
        "n_seed_blocks": seeds,
    })


# ============================================================
# 5. Summary
# ============================================================

def summarize(
    rows, dataset, scenario, controller, source
):
    sample = [
        r for r in rows
        if r["controller"] == controller
    ]

    successful = [
        r for r in sample
        if int(r["survived"]) == 1
    ]

    late = (
        100 * float(
            np.mean([
                float(r["late_xyz_m"])
                for r in successful
            ])
        )
        if successful
        else float("nan")
    )

    return {
        "dataset": dataset,
        "scenario": str(scenario),
        "controller": controller,
        "n": len(sample),
        "survived": len(successful),
        "survival_pct": (
            100 * len(successful) / len(sample)
            if sample else float("nan")
        ),
        "late_xyz_cm_survivors": late,
        "n_seeds": len({
            r["seed"] for r in sample
        }),
        "source": source,
    }


def scenario_order(name, values):
    if name in ("mass", "inertia"):
        return sorted(values, key=float)

    def motor_key(s):
        if s == "nominal":
            return (0, 0, 0)

        if s.startswith("uniform_"):
            return (
                1,
                -int(s.split("_")[-1]),
                0,
            )

        if s.startswith("motor"):
            motor, eta = s.split("_")
            return (
                2 + int(motor[5:]),
                -int(eta),
                0,
            )

        return (99, 0, s)

    return sorted(values, key=motor_key)


def nominal_level(name, levels):
    if name == "motor":
        return (
            "nominal"
            if "nominal" in levels
            else None
        )

    for value in (0, 100):
        for level in levels:
            if float(level) == value:
                return level

    return None


# ============================================================
# 6. Mass / inertia / motor analysis
# ============================================================

def analyze_sweep(
    name,
    sweep_number,
    path,
    rows,
    level_col,
    summaries,
    comparisons,
    n_boot,
    out_dir,
):
    dataset = (
        "mass"
        if name == "mass"
        else f"{name}_sweep{sweep_number}"
    )

    levels = scenario_order(
        name,
        {r[level_col] for r in rows},
    )

    controllers = [
        c for c in CONTROLLERS
        if any(
            r["controller"] == c
            for r in rows
        )
    ]

    nominal = nominal_level(
        name, levels
    )

    print(
        f"\n[SOURCE] {dataset}: {path.name}"
    )
    print(
        f"  rows={len(rows)}, levels={levels}"
    )
    print(
        f"  nominal={nominal}"
    )

    for level in levels:
        group = [
            r for r in rows
            if r[level_col] == level
        ]

        for controller in controllers:
            summaries.append(
                summarize(
                    group,
                    dataset,
                    level,
                    controller,
                    path.name,
                )
            )

        # Controller-vs-controller comparisons.
        for a, b in (
            ("Baseline_30k", "PID"),
            ("Final_20k", "Baseline_30k"),
        ):
            if (
                a not in controllers
                or b not in controllers
            ):
                continue

            compare(
                comparisons,
                level,
                dataset,
                [
                    r for r in group
                    if r["controller"] == a
                ],
                [
                    r for r in group
                    if r["controller"] == b
                ],
                a,
                b,
                n_boot,
            )

        if nominal is None or level == nominal:
            continue

        # Compare each controller with its nominal.
        for controller in controllers:
            compare(
                comparisons,
                level,
                dataset,
                [
                    r for r in rows
                    if (
                        r[level_col] == level
                        and r["controller"] == controller
                    )
                ],
                [
                    r for r in rows
                    if (
                        r[level_col] == nominal
                        and r["controller"] == controller
                    )
                ],
                f"{controller}_perturbed",
                f"{controller}_nominal",
                n_boot,
            )

        # Difference-in-degradation, same episodes.
        for a, b in (
            ("Baseline_30k", "PID"),
            ("Final_20k", "Baseline_30k"),
        ):
            if a in controllers and b in controllers:
                compare_degradation(
                    comparisons,
                    level,
                    dataset,
                    rows,
                    level_col,
                    nominal,
                    a,
                    b,
                    n_boot,
                )

    # --------------------------------------------------------
    # Survival and tracking figures
    # --------------------------------------------------------

    x = (
        np.arange(len(levels))
        if name == "motor"
        else np.asarray(
            [float(v) for v in levels]
        )
    )

    plots = (
        (
            "survival_pct",
            "survival",
            "Completed episodes (%)",
        ),
        (
            "late_xyz_cm_survivors",
            "late_rmse_survivors",
            "Late XYZ RMSE (cm) — completed only",
        ),
    )

    for metric, suffix, ylabel in plots:
        fig, ax = plt.subplots(
            figsize=(
                max(8.3, 1.15 * len(levels)),
                4.7,
            ),
            constrained_layout=True,
        )

        for controller in controllers:
            available = {
                r["scenario"]: r
                for r in summaries
                if (
                    r["dataset"] == dataset
                    and r["controller"] == controller
                )
            }

            y = [
                available[level][metric]
                for level in levels
            ]

            ax.plot(
                x,
                y,
                "-o",
                linewidth=1.8,
                markersize=5,
                color=COLORS[controller],
                label=DISPLAY[controller],
            )

        if name == "motor":
            ax.set_xticks(
                x,
                levels,
                rotation=35,
                ha="right",
            )
            ax.set_xlabel(
                "Motor effectiveness scenario"
            )

        else:
            ax.set_xticks(x)
            ax.set_xlabel(
                f"{name.capitalize()} percentage "
                "(as recorded in CSV)"
            )

        if metric == "survival_pct":
            ax.set_ylim(-4, 104)

        ax.set_ylabel(ylabel)

        ax.set_title(
            f"{name.capitalize()} robustness "
            f"— sweep {sweep_number}"
        )

        ax.grid(alpha=0.22)
        ax.legend(frameon=False)

        output = (
            out_dir
            / f"{dataset}_{suffix}.png"
        )

        fig.savefig(output, dpi=220)
        plt.close(fig)

        print(f"[FIGURE] {output}")

    # --------------------------------------------------------
    # Terminal summary
    # --------------------------------------------------------

    for level in levels:
        current = [
            r for r in summaries
            if (
                r["dataset"] == dataset
                and r["scenario"] == str(level)
            )
        ]

        entries = [
            (
                f"{DISPLAY[r['controller']]}: "
                f"{r['survived']}/{r['n']}, "
                f"late={r['late_xyz_cm_survivors']:.2f}cm"
            )
            for r in current
        ]

        print(
            f"  {str(level):>10}: "
            + " | ".join(entries)
        )


# ============================================================
# 7. Paired gust recovery
# ============================================================

def analyze_gust(
    path,
    rows,
    summaries,
    comparisons,
    n_boot,
    out_dir,
):
    print(
        f"\n[SOURCE] gust: {path.name}"
        f" ({len(rows)} rows)"
    )

    pid = episode_index([
        r for r in rows
        if r["controller"] == "PID"
    ])

    sac = episode_index([
        r for r in rows
        if r["controller"] == "Baseline_30k"
    ])

    common = sorted(
        pid.keys() & sac.keys()
    )

    if not common:
        raise ValueError(
            "No matched gust trials"
        )

    # Validate that the wind input was identical.
    for key in common:
        if (
            pid[key]["wind_hash"]
            != sac[key]["wind_hash"]
        ):
            raise ValueError(
                f"Wind mismatch: {key}"
            )

    # Both controllers must have completed
    # their own nominal and gust episodes.
    valid = [
        key for key in common
        if (
            int(pid[key]["paired_success"])
            and int(sac[key]["paired_success"])
        )
    ]

    print(
        f"[GUST] Matched trials: {len(common)}"
    )
    print(
        f"[GUST] Jointly successful: {len(valid)}"
    )

    for controller, index in (
        ("PID", pid),
        ("Baseline_30k", sac),
    ):
        own = list(index.values())

        count = sum(
            int(r["paired_success"])
            for r in own
        )

        summaries.append({
            "dataset": "gust_paired",
            "scenario": "nominal_and_gust",
            "controller": controller,
            "n": len(own),
            "survived": count,
            "survival_pct":
                100 * count / len(own)
                if own else float("nan"),
            "late_xyz_cm_survivors":
                float("nan"),
            "n_seeds": len({
                r["seed"] for r in own
            }),
            "source": path.name,
        })

    measures = (
        (
            "peak_delta_3_6_cm",
            "Peak deviation (cm)",
            1.0,
        ),
        (
            "post_gust_rms_delta_4_6_cm",
            "Post-gust RMS deviation (cm)",
            1.0,
        ),
        (
            "post_gust_time_above_1cm_s",
            "Time outside 1 cm (s)",
            1.0,
        ),
        (
            "settled",
            "Settled by episode end (%)",
            100.0,
        ),
    )

    fig, axes = plt.subplots(
        2, 2,
        figsize=(10.5, 7.1),
        constrained_layout=True,
    )

    for ax, (field, label, factor) in zip(
        axes.flat, measures
    ):
        pid_values = []
        sac_values = []
        differences = []

        for key in valid:
            rp = pid[key]
            rs = sac[key]

            if field == "settled":
                pval = float(
                    rp["recovery_status"]
                    == "settled"
                )
                sval = float(
                    rs["recovery_status"]
                    == "settled"
                )

            else:
                pval = float(rp[field])
                sval = float(rs[field])

            if not (
                np.isfinite(pval)
                and np.isfinite(sval)
            ):
                continue

            pid_values.append((
                rp["seed"], pval
            ))

            sac_values.append((
                rs["seed"], sval
            ))

            differences.append((
                rp["seed"],
                sval - pval,
            ))

        pid_stats = seed_bootstrap(
            pid_values,
            factor=factor,
            n_boot=n_boot,
        )

        sac_stats = seed_bootstrap(
            sac_values,
            factor=factor,
            n_boot=n_boot,
        )

        gap_stats = seed_bootstrap(
            differences,
            factor=factor,
            n_boot=n_boot,
        )

        comparisons.append({
            "dataset": "gust_paired",
            "scenario": "jointly_successful",
            "comparison":
                "Baseline_30k_minus_PID",
            "metric": field,
            "estimate": gap_stats[0],
            "ci95_low": gap_stats[1],
            "ci95_high": gap_stats[2],
            "n_matched": gap_stats[3],
            "n_seed_blocks": gap_stats[4],
        })

        centers = [
            pid_stats[0],
            sac_stats[0],
        ]

        lower = [
            max(0, stat[0] - stat[1])
            for stat in (
                pid_stats, sac_stats
            )
        ]

        upper = [
            max(0, stat[2] - stat[0])
            for stat in (
                pid_stats, sac_stats
            )
        ]

        ax.bar(
            [0, 1],
            centers,
            width=0.56,
            color=[
                COLORS["PID"],
                COLORS["Baseline_30k"],
            ],
        )

        ax.errorbar(
            [0, 1],
            centers,
            yerr=[lower, upper],
            fmt="none",
            color="black",
            capsize=3,
        )

        ax.set_xticks(
            [0, 1],
            ["PID", "SAC Baseline"],
        )

        ax.set_title(label)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)

        print(
            f"[GUST] {field}: "
            f"SAC-PID={gap_stats[0]:+.3f}, "
            f"CI=[{gap_stats[1]:+.3f}, "
            f"{gap_stats[2]:+.3f}], "
            f"n={gap_stats[3]}, "
            f"seed_blocks={gap_stats[4]}"
        )

    output = (
        out_dir / "gust_paired_recovery.png"
    )

    fig.savefig(output, dpi=220)
    plt.close(fig)

    print(f"[FIGURE] {output}")


# ============================================================
# 8. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("results/evaluation"),
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/research_analysis"),
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=3000,
    )

    args = parser.parse_args()

    if args.bootstrap < 100:
        parser.error(
            "--bootstrap must be >=100"
        )

    if not args.data_dir.is_dir():
        parser.error(
            f"Missing folder: {args.data_dir}"
        )

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries = []
    comparisons = []

    print("=" * 70)
    print("EXTENDED ROBUSTNESS RESEARCH ANALYSIS")
    print("=" * 70)

    specs = (
        (
            "mass",
            "*mass*robustness*.csv",
            "mass_percent",
            5000,
            False,
        ),
        (
            "inertia",
            "*inertia*robustness*.csv",
            "inertia_percent",
            1000,
            True,
        ),
        (
            "motor",
            "*motor*robustness*.csv",
            "scenario",
            1000,
            True,
        ),
    )

    for (
        name,
        pattern,
        level_col,
        min_rows,
        keep_all,
    ) in specs:

        sweeps = find_sweeps(
            args.data_dir,
            pattern,
            level_col,
            min_rows,
            keep_all,
        )

        for i, (_, _, path, rows) in enumerate(
            sweeps, 1
        ):
            analyze_sweep(
                name,
                i,
                path,
                rows,
                level_col,
                summaries,
                comparisons,
                args.bootstrap,
                args.out_dir,
            )

    # --------------------------------------------------------
    # Gust has a different schema: paired_success,
    # not the ordinary survived column.
    # --------------------------------------------------------

    gust_candidates = []

    for path in args.data_dir.glob(
        "gust_paired_diagnostic_*.csv"
    ):
        rows = read_csv(path)

        required = set(EPISODE_KEY) | {
            "controller",
            "paired_success",
            "recovery_status",
            "wind_hash",
        }

        if (
            len(rows) >= 100
            and required.issubset(rows[0])
        ):
            gust_candidates.append((
                len(rows),
                path.stat().st_mtime,
                path,
                rows,
            ))

    if not gust_candidates:
        raise FileNotFoundError(
            "Missing full gust paired CSV"
        )

    _, _, gust_path, gust_rows = max(
        gust_candidates
    )

    analyze_gust(
        gust_path,
        gust_rows,
        summaries,
        comparisons,
        args.bootstrap,
        args.out_dir,
    )

    # --------------------------------------------------------
    # Save CSV outputs
    # --------------------------------------------------------

    summary_path = (
        args.out_dir / "extra_summary.csv"
    )

    compare_path = (
        args.out_dir
        / "extra_paired_bootstrap.csv"
    )

    write_csv(
        summary_path,
        summaries,
        [
            "dataset",
            "scenario",
            "controller",
            "n",
            "survived",
            "survival_pct",
            "late_xyz_cm_survivors",
            "n_seeds",
            "source",
        ],
    )

    write_csv(
        compare_path,
        comparisons,
        [
            "dataset",
            "scenario",
            "comparison",
            "metric",
            "estimate",
            "ci95_low",
            "ci95_high",
            "n_matched",
            "n_seed_blocks",
        ],
    )

    print(f"\n[TABLE] {summary_path}")
    print(f"[TABLE] {compare_path}")

    # --------------------------------------------------------
    # Print paired comparison results
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("SAC BASELINE MINUS PID")
    print("=" * 90)

    for item in comparisons:
        if (
            item["comparison"]
            != "Baseline_30k_minus_PID"
        ):
            continue

        print(
            f"{item['dataset']:<17} "
            f"{item['scenario']:<14} "
            f"{item['metric']:<40} "
            f"{item['estimate']:+.3f} "
            f"[{item['ci95_low']:+.3f}, "
            f"{item['ci95_high']:+.3f}] "
            f"n={item['n_matched']}"
        )

    print(
        "\n[FINISHED] Research analysis complete."
    )

    print(
        "Bootstrap resamples entire seed blocks."
    )

    print(
        "Tracking comparisons use jointly "
        "completed episodes only."
    )

    print(
        "Inertia and motor sweep files "
        "were analyzed separately."
    )


if __name__ == "__main__":
    main()
