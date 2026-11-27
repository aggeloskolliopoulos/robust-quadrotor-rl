
"""Research analysis for PID vs SAC quadrotor robustness.

Reads existing evaluation CSVs. Does not run simulations.

Outputs:
    results/research_analysis/
        latency_survival.png
        latency_late_rmse_survivors.png
        wind_late_rmse_degradation.png
        sensor_noise_late_rmse_degradation.png
        summary.csv
        paired_bootstrap.csv
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


CONTROLLERS = ("PID", "Baseline_30k")

EPISODE_KEY = (
    "radius_m",
    "phase_rad",
    "omega_rad_s",
    "seed",
)


# ============================================================
# 1. Read and write CSV files
# ============================================================

def read_csv(path):

    with path.open(
        newline="",
        encoding="utf-8",
    ) as file:

        return list(
            csv.DictReader(file)
        )


def write_csv(path, rows, columns):

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns,
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 2. Select complete evaluation files
# ============================================================

def best_dataset(folder, pattern, required):

    choices = []

    for path in folder.glob(pattern):

        rows = read_csv(path)

        if not rows or "scenario" not in rows[0]:
            continue

        scenarios = {
            row["scenario"]
            for row in rows
        }

        controllers = {
            row["controller"]
            for row in rows
        }

        if not required.issubset(scenarios):
            continue

        if not set(CONTROLLERS).issubset(
            controllers
        ):
            continue

        choices.append((
            len(rows),
            path.stat().st_mtime,
            path,
            rows,
        ))

    if not choices:

        raise FileNotFoundError(
            f"No suitable CSV found for {pattern}"
        )

    _, _, path, rows = max(
        choices,
        key=lambda x: (x[0], x[1]),
    )

    return path, rows


def get_latency(folder):
    """
    Select:
      - Original complete command benchmark
      - Original complete observation benchmark
      - Combined independent confirmation

    Never double-count identical episodes.
    """

    datasets = [
        (path, read_csv(path))
        for path in folder.glob(
            "latency_threshold_*.csv"
        )
    ]

    if not datasets:
        raise FileNotFoundError(
            "No latency_threshold_*.csv files"
        )

    full = {}

    for kind in ("command", "observation"):

        choices = []

        for path, rows in datasets:

            subset = [
                r for r in rows
                if r.get("kind") == kind
            ]

            levels = {
                int(r["substeps"])
                for r in subset
            }

            if set(range(6)).issubset(levels):

                choices.append((
                    len(subset),
                    path.stat().st_mtime,
                    path,
                    subset,
                ))

        if not choices:

            raise FileNotFoundError(
                f"Missing complete {kind} benchmark"
            )

        _, _, path, subset = max(
            choices,
            key=lambda x: (x[0], x[1]),
        )

        full[kind] = (path, subset)

    # Find the independent 3/4/5-substep confirmation.
    confirmations = []

    for path, rows in datasets:

        if len(rows) < 1000:
            continue

        kinds = {
            r.get("kind")
            for r in rows
        }

        levels = {
            int(r["substeps"])
            for r in rows
        }

        if (
            kinds == {"command", "observation"}
            and levels == {3, 4, 5}
        ):

            confirmations.append((
                path.stat().st_mtime,
                path,
                rows,
            ))

    if confirmations:

        _, confirmation_path, confirmation = max(
            confirmations
        )

    else:

        confirmation_path = None
        confirmation = []

        print(
            "[NOTE] No combined confirmation found"
        )

    combined = []

    for kind, (path, rows) in full.items():

        print(
            f"[SOURCE] {kind}: {path.name} "
            f"({len(rows)} rows)"
        )

        combined.extend(rows)

    if confirmation:

        print(
            "[SOURCE] confirmation: "
            f"{confirmation_path.name} "
            f"({len(confirmation)} rows)"
        )

        combined.extend(confirmation)

    # Reject duplicated experiments.
    seen = set()

    for row in combined:

        key = (
            row["kind"],
            row["substeps"],
            row["controller"],
            *(
                row[column]
                for column in EPISODE_KEY
            ),
        )

        if key in seen:

            raise ValueError(
                f"Duplicate latency episode: {key}"
            )

        seen.add(key)

    return combined


# ============================================================
# 3. Summary statistics
# ============================================================

def summary(
    rows,
    *,
    dataset,
    scenario,
    controller,
    source,
):

    group = [
        r for r in rows
        if r["controller"] == controller
    ]

    successful = [
        r for r in group
        if int(r["survived"]) == 1
    ]

    if successful:

        late_rmse_cm = 100 * float(
            np.mean([
                float(r["late_xyz_m"])
                for r in successful
            ])
        )

    else:

        late_rmse_cm = float("nan")

    return {
        "dataset": dataset,
        "scenario": str(scenario),
        "controller": controller,

        "n": len(group),
        "survived": len(successful),

        "survival_pct": (
            100 * len(successful) / len(group)
            if group else float("nan")
        ),

        "late_xyz_cm_success_only":
            late_rmse_cm,

        "n_seeds": len({
            r["seed"]
            for r in group
        }),

        "source": source,
    }


# ============================================================
# 4. Matched episode comparison
# ============================================================

def index_by_episode(rows):

    index = {}

    for row in rows:

        key = tuple(
            row[column]
            for column in EPISODE_KEY
        )

        if key in index:

            raise ValueError(
                f"Duplicate episode key: {key}"
            )

        index[key] = row

    return index


def bootstrap_seed_mean(
    seed_values,
    *,
    n_boot=3000,
):
    """
    Bootstrap entire seed blocks.

    Each seed block contains all matched trajectories
    available for that seed.

    Resampling seeds preserves within-seed dependence.
    """

    if not seed_values:

        return (
            float("nan"),
            float("nan"),
            float("nan"),
            0,
            0,
        )

    blocks = defaultdict(list)

    for seed, value in seed_values:

        blocks[seed].append(
            float(value)
        )

    groups = list(
        blocks.values()
    )

    sums = np.asarray(
        [sum(group) for group in groups],
        dtype=float,
    )

    counts = np.asarray(
        [len(group) for group in groups],
        dtype=int,
    )

    point = float(
        sums.sum() / counts.sum()
    )

    rng = np.random.default_rng(43)

    picks = rng.integers(
        0,
        len(groups),
        size=(n_boot, len(groups)),
    )

    bootstrap = (
        sums[picks].sum(axis=1)
        / counts[picks].sum(axis=1)
    )

    low, high = np.percentile(
        bootstrap,
        [2.5, 97.5],
    )

    return (
        point,
        float(low),
        float(high),
        int(counts.sum()),
        len(groups),
    )


def paired_difference(
    rows_a,
    rows_b,
    field,
    *,
    multiplier=1.0,
    n_boot=3000,
):
    """
    Return A-B and seed-block bootstrap 95% CI.

    RMSE is compared only when BOTH episodes survived.
    Survival differences use ALL matched episodes.
    """

    a = index_by_episode(rows_a)
    b = index_by_episode(rows_b)

    matched = sorted(
        a.keys() & b.keys()
    )

    pairs = []

    for key in matched:

        ra = a[key]
        rb = b[key]

        if field == "late_xyz_m":

            if (
                int(ra["survived"]) != 1
                or int(rb["survived"]) != 1
            ):
                continue

        va = float(ra[field])
        vb = float(rb[field])

        if (
            np.isfinite(va)
            and np.isfinite(vb)
        ):

            pairs.append((
                ra["seed"],
                multiplier * (va - vb),
            ))

    return bootstrap_seed_mean(
        pairs,
        n_boot=n_boot,
    )


def add_comparison(
    output,
    *,
    dataset,
    scenario,
    comparison,
    a,
    b,
    n_boot,
):

    metrics = (
        (
            "survived",
            "survival_difference_pp",
            100.0,
        ),
        (
            "late_xyz_m",
            "late_xyz_difference_cm",
            100.0,
        ),
    )

    for field, metric, factor in metrics:

        mean, low, high, n, n_seeds = (
            paired_difference(
                a,
                b,
                field,
                multiplier=factor,
                n_boot=n_boot,
            )
        )

        output.append({
            "dataset": dataset,
            "scenario": str(scenario),
            "comparison": comparison,
            "metric": metric,

            "mean": mean,
            "ci95_low": low,
            "ci95_high": high,

            "n_matched": n,
            "n_seed_blocks": n_seeds,
        })


# ============================================================
# 5. Analyze wind or sensor experiments
# ============================================================

def scenario_analysis(
    rows,
    *,
    dataset,
    scenarios,
    source,
    summaries,
    comparisons,
    n_boot,
):

    for scenario in scenarios:

        group = [
            r for r in rows
            if r["scenario"] == scenario
        ]

        for controller in CONTROLLERS:

            summaries.append(
                summary(
                    group,
                    dataset=dataset,
                    scenario=scenario,
                    controller=controller,
                    source=source,
                )
            )

        # SAC - PID, same scenario and matched episodes.
        add_comparison(
            comparisons,
            dataset=dataset,
            scenario=scenario,
            comparison="SAC_minus_PID",

            a=[
                r for r in group
                if r["controller"] == "Baseline_30k"
            ],

            b=[
                r for r in group
                if r["controller"] == "PID"
            ],

            n_boot=n_boot,
        )

        if scenario == "nominal":
            continue

        # Each controller's degradation vs its own nominal.
        for controller in CONTROLLERS:

            add_comparison(
                comparisons,
                dataset=dataset,
                scenario=scenario,

                comparison=(
                    f"{controller}:scenario_minus_nominal"
                ),

                a=[
                    r for r in rows
                    if (
                        r["scenario"] == scenario
                        and r["controller"] == controller
                    )
                ],

                b=[
                    r for r in rows
                    if (
                        r["scenario"] == "nominal"
                        and r["controller"] == controller
                    )
                ],

                n_boot=n_boot,
            )


# ============================================================
# 6. Plot tracking degradation with paired CIs
# ============================================================

def plot_degradation(
    comparisons,
    dataset,
    scenarios,
    labels,
    output_file,
):

    fig, ax = plt.subplots(
        figsize=(8, 4.7),
        constrained_layout=True,
    )

    x = np.arange(
        len(scenarios),
        dtype=float,
    )

    width = 0.35

    for offset, controller in (
        (-0.5, "PID"),
        (0.5, "Baseline_30k"),
    ):

        values = []
        lower = []
        upper = []

        for scenario in scenarios:

            matches = [
                r for r in comparisons
                if (
                    r["dataset"] == dataset
                    and r["scenario"] == scenario
                    and r["comparison"] ==
                    f"{controller}:scenario_minus_nominal"
                    and r["metric"] ==
                    "late_xyz_difference_cm"
                )
            ]

            if len(matches) != 1:

                raise ValueError(
                    f"Missing analysis: {scenario}, "
                    f"{controller}"
                )

            result = matches[0]

            values.append(
                result["mean"]
            )

            lower.append(
                max(
                    0,
                    result["mean"]
                    - result["ci95_low"],
                )
            )

            upper.append(
                max(
                    0,
                    result["ci95_high"]
                    - result["mean"],
                )
            )

        ax.bar(
            x + offset * width,
            values,
            width,

            yerr=np.asarray([
                lower,
                upper,
            ]),

            capsize=3,

            label=(
                "PID"
                if controller == "PID"
                else "SAC Baseline 30k"
            ),
        )

    ax.axhline(
        0,
        linewidth=0.8,
        color="black",
    )

    ax.set_xticks(x, labels)

    ax.set_ylabel(
        "Change in late XYZ RMSE vs nominal (cm)"
    )

    ax.set_title(
        "Tracking degradation: "
        + (
            "wind disturbance"
            if dataset == "wind"
            else "sensor noise"
        )
    )

    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    fig.savefig(
        output_file,
        dpi=220,
    )

    plt.close(fig)

    print(
        f"[FIGURE] {output_file}"
    )


# ============================================================
# 7. Latency analysis and figures
# ============================================================

def latency_analysis(
    rows,
    summaries,
    comparisons,
    n_boot,
    output_dir,
):

    levels = tuple(range(6))

    for kind in ("command", "observation"):

        for level in levels:

            scenario = f"{kind}_{level}"

            group = [
                r for r in rows
                if (
                    r["kind"] == kind
                    and int(r["substeps"]) == level
                )
            ]

            for controller in CONTROLLERS:

                summaries.append(
                    summary(
                        group,
                        dataset="latency",
                        scenario=scenario,
                        controller=controller,
                        source="original + confirmation",
                    )
                )

            add_comparison(
                comparisons,
                dataset="latency",
                scenario=scenario,
                comparison="SAC_minus_PID",

                a=[
                    r for r in group
                    if r["controller"] == "Baseline_30k"
                ],

                b=[
                    r for r in group
                    if r["controller"] == "PID"
                ],

                n_boot=n_boot,
            )

    delays = [
        1000 * level / 240
        for level in levels
    ]

    # --------------------------------------------------------
    # Figure A: Survival vs latency
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 5),
        constrained_layout=True,
    )

    for kind, linestyle in (
        ("command", "-"),
        ("observation", "--"),
    ):

        for controller, marker in (
            ("PID", "o"),
            ("Baseline_30k", "s"),
        ):

            values = []

            for level in levels:

                item = next(
                    r for r in summaries
                    if (
                        r["dataset"] == "latency"
                        and r["scenario"] ==
                        f"{kind}_{level}"
                        and r["controller"] == controller
                    )
                )

                values.append(
                    item["survival_pct"]
                )

            ax.plot(
                delays,
                values,
                marker=marker,
                linestyle=linestyle,

                label=(
                    f"{'PID' if controller == 'PID' else 'SAC'} "
                    f"- {kind}"
                ),
            )

    ax.set_title(
        "Survival under command and observation latency"
    )

    ax.set_xlabel(
        "Added pure delay (ms)"
    )

    ax.set_ylabel(
        "Episode survival (%)"
    )

    ax.set_ylim(-4, 104)
    ax.set_xticks(delays)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    output = output_dir / "latency_survival.png"

    fig.savefig(
        output,
        dpi=220,
    )

    plt.close(fig)

    print(f"[FIGURE] {output}")

    # --------------------------------------------------------
    # Figure B: RMSE among survivors
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 5),
        constrained_layout=True,
    )

    for kind, linestyle in (
        ("command", "-"),
        ("observation", "--"),
    ):

        for controller, marker in (
            ("PID", "o"),
            ("Baseline_30k", "s"),
        ):

            values = []

            for level in levels:

                item = next(
                    r for r in summaries
                    if (
                        r["dataset"] == "latency"
                        and r["scenario"] ==
                        f"{kind}_{level}"
                        and r["controller"] == controller
                    )
                )

                values.append(
                    item["late_xyz_cm_success_only"]
                )

            ax.plot(
                delays,
                values,
                marker=marker,
                linestyle=linestyle,

                label=(
                    f"{'PID' if controller == 'PID' else 'SAC'} "
                    f"- {kind}"
                ),
            )

    ax.set_title(
        "Tracking error on completed episodes only"
    )

    ax.set_xlabel(
        "Added pure delay (ms)"
    )

    ax.set_ylabel(
        "Late XYZ RMSE (cm)"
    )

    ax.set_xticks(delays)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    output = (
        output_dir
        / "latency_late_rmse_survivors.png"
    )

    fig.savefig(
        output,
        dpi=220,
    )

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
            "--bootstrap must be at least 100"
        )

    if not args.data_dir.is_dir():

        parser.error(
            f"Missing results folder: {args.data_dir}"
        )

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries = []
    comparisons = []

    print("\n" + "=" * 70)
    print("QUADROTOR ROBUSTNESS RESEARCH ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # A. Latency
    # --------------------------------------------------------

    latency_rows = get_latency(
        args.data_dir
    )

    latency_analysis(
        latency_rows,
        summaries,
        comparisons,
        args.bootstrap,
        args.out_dir,
    )

    # --------------------------------------------------------
    # B. Wind (independent confirmation dataset)
    # --------------------------------------------------------

    wind_scenarios = (
        "nominal",
        "constant_x10",
        "random_constant",
        "turbulent",
    )

    wind_file, wind_rows = best_dataset(
        args.data_dir,
        "wind_robustness_*.csv",
        set(wind_scenarios),
    )

    print(
        f"[SOURCE] wind: {wind_file.name} "
        f"({len(wind_rows)} rows)"
    )

    scenario_analysis(
        wind_rows,
        dataset="wind",
        scenarios=wind_scenarios,
        source=wind_file.name,
        summaries=summaries,
        comparisons=comparisons,
        n_boot=args.bootstrap,
    )

    plot_degradation(
        comparisons,
        "wind",
        wind_scenarios[1:],
        [
            "Constant\n10 mN",
            "Random\nconstant",
            "Turbulent",
        ],
        args.out_dir
        / "wind_late_rmse_degradation.png",
    )

    # --------------------------------------------------------
    # C. Sensor noise
    # --------------------------------------------------------

    sensor_scenarios = (
        "nominal",
        "noise_low",
        "noise_moderate",
    )

    sensor_file, sensor_rows = best_dataset(
        args.data_dir,
        "sensor_latency_robustness_*.csv",
        set(sensor_scenarios),
    )

    print(
        f"[SOURCE] sensor: {sensor_file.name} "
        f"({len(sensor_rows)} rows)"
    )

    scenario_analysis(
        sensor_rows,
        dataset="sensor",
        scenarios=sensor_scenarios,
        source=sensor_file.name,
        summaries=summaries,
        comparisons=comparisons,
        n_boot=args.bootstrap,
    )

    plot_degradation(
        comparisons,
        "sensor",
        sensor_scenarios[1:],
        [
            "Low noise",
            "Moderate noise",
        ],
        args.out_dir
        / "sensor_noise_late_rmse_degradation.png",
    )

    # --------------------------------------------------------
    # D. Save tables
    # --------------------------------------------------------

    summary_file = (
        args.out_dir / "summary.csv"
    )

    comparison_file = (
        args.out_dir / "paired_bootstrap.csv"
    )

    write_csv(
        summary_file,
        summaries,
        [
            "dataset",
            "scenario",
            "controller",
            "n",
            "survived",
            "survival_pct",
            "late_xyz_cm_success_only",
            "n_seeds",
            "source",
        ],
    )

    write_csv(
        comparison_file,
        comparisons,
        [
            "dataset",
            "scenario",
            "comparison",
            "metric",
            "mean",
            "ci95_low",
            "ci95_high",
            "n_matched",
            "n_seed_blocks",
        ],
    )

    print(f"\n[TABLE] {summary_file}")
    print(f"[TABLE] {comparison_file}")

    # --------------------------------------------------------
    # E. Terminal summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LATENCY: SAC - PID SURVIVAL (PERCENTAGE POINTS)")
    print("=" * 70)

    for kind in ("command", "observation"):

        for level in (3, 4, 5):

            item = next(
                r for r in comparisons
                if (
                    r["dataset"] == "latency"
                    and r["scenario"] ==
                    f"{kind}_{level}"
                    and r["comparison"] == "SAC_minus_PID"
                    and r["metric"] ==
                    "survival_difference_pp"
                )
            )

            print(
                f"{kind:<12} "
                f"{level*1000/240:5.2f} ms: "
                f"{item['mean']:+6.1f} pp "
                f"[{item['ci95_low']:+.1f}, "
                f"{item['ci95_high']:+.1f}] "
                f"n={item['n_matched']}, "
                f"seeds={item['n_seed_blocks']}"
            )

    for dataset in ("wind", "sensor"):

        print("\n" + "=" * 70)
        print(
            f"{dataset.upper()}: "
            "LATE XYZ CHANGE VS NOMINAL (CM)"
        )
        print("=" * 70)

        for item in comparisons:

            if (
                item["dataset"] == dataset
                and item["metric"] ==
                "late_xyz_difference_cm"
                and ":scenario_minus_nominal"
                in item["comparison"]
            ):

                controller = (
                    item["comparison"].split(":")[0]
                )

                print(
                    f"{item['scenario']:<18} "
                    f"{controller:<13} "
                    f"{item['mean']:+.3f} cm "
                    f"[{item['ci95_low']:+.3f}, "
                    f"{item['ci95_high']:+.3f}] "
                    f"n={item['n_matched']}"
                )

    print("\n[FINISHED] Research analysis complete.")

    print(
        "95% seed-block bootstrap intervals."
    )

    print(
        "RMSE comparisons use paired completed episodes only."
    )

    print(
        "No simulations or training were performed."
    )


if __name__ == "__main__":
    main()
