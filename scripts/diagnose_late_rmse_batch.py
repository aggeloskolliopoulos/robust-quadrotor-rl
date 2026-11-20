
"""
Multi-seed diagnostic for late tracking errors of two frozen SAC policies.

Place this file in scripts/ next to diagnose_late_rmse.py.
It reuses run_episode() from that file, but NEVER calls its plotting code.
No training and no changes to checkpoints.

Usage:
    python scripts/diagnose_late_rmse_batch.py --check-only
    python -u scripts/diagnose_late_rmse_batch.py
    python -u scripts/diagnose_late_rmse_batch.py --seed-start 7000 --seeds 20
"""

import argparse
import csv
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC

from diagnose_late_rmse import ROOT, MODEL_PATHS, SCENARIOS, run_episode

LATE_START = 5.0
LATE_MID = 7.5


def calculate_metrics(trace):
    """Per-episode metrics. Errors are target minus actual position."""
    t = np.asarray(trace["time"], dtype=float)
    e = np.asarray(trace["position_error"], dtype=float)
    v = np.asarray(trace["velocity_error"], dtype=float)
    mask = t >= LATE_START
    m1 = (t >= LATE_START) & (t < LATE_MID)
    m2 = t >= LATE_MID

    def rms(a):
        return float(np.sqrt(np.mean(np.asarray(a, dtype=float) ** 2)))

    result = {
        "survived": bool(trace["survived"]),
        "duration_s": float(trace["duration_s"]),
        "full_rmse_m": rms(np.linalg.norm(e, axis=1)),
    }

    late_fields = (
        "late_rmse_m", "late_xy_rmse_m", "late_z_rmse_m",
        "late_v_rmse_m_s", "late_z_bias_m", "late_z_std_m",
        "late_z_first_rmse_m", "late_z_last_rmse_m",
        "late_radial_bias_m", "late_rpm_effort",
    )

    if not (mask.any() and m1.any() and m2.any()):
        result.update({field: float("nan") for field in late_fields})
        return result

    ez = e[mask, 2]
    rpm = np.asarray(trace["rpm"], dtype=float)[mask]
    hover = float(trace["hover_rpm"])

    result.update({
        "late_rmse_m": rms(np.linalg.norm(e[mask], axis=1)),
        "late_xy_rmse_m": rms(np.linalg.norm(e[mask, :2], axis=1)),
        "late_z_rmse_m": rms(ez),
        "late_v_rmse_m_s": rms(np.linalg.norm(v[mask], axis=1)),
        "late_z_bias_m": float(np.mean(ez)),
        "late_z_std_m": float(np.std(ez)),
        "late_z_first_rmse_m": rms(e[m1, 2]),
        "late_z_last_rmse_m": rms(e[m2, 2]),
        "late_radial_bias_m": float(
            np.mean(np.asarray(trace["radial_error"])[mask])
        ),
        "late_rpm_effort": float(
            np.mean(np.sum(((rpm-hover)/hover)**2, axis=1))
        ),
    })

    return result


def mean_field(records, field):
    return float(np.mean([r[field] for r in records]))


def plot_scenario(out, name, paired_traces):
    """Plot only trajectories in which BOTH controllers completed 10 s."""
    if not paired_traces:
        return

    fig, panel_grid = plt.subplots(
        1, 3, figsize=(16, 4.4), squeeze=False
    )
    panels = list(panel_grid[0, :])

    for controller in MODEL_PATHS:
        traces = [pair[controller] for pair in paired_traces]
        t = np.asarray(traces[0]["time"], dtype=float)

        # All successful episodes share the same control timestep count.
        if any(len(tr["time"]) != len(t) for tr in traces):
            print(
                f"[WARN] {name}: inconsistent trace lengths, "
                "skipping plots"
            )
            plt.close(fig)
            return

        ep = np.stack(
            [tr["position_error"] for tr in traces], axis=0
        )

        z_err = ep[:, :, 2] * 100  # cm, signed target - actual
        xy = np.linalg.norm(ep[:, :, :2], axis=2) * 100
        xyz = np.linalg.norm(ep, axis=2) * 100

        panels[0].plot(
            t, np.mean(z_err, axis=0), label=controller
        )
        panels[0].fill_between(
            t,
            np.percentile(z_err, 10, axis=0),
            np.percentile(z_err, 90, axis=0),
            alpha=0.13,
        )

        panels[1].plot(
            t,
            np.sqrt(np.mean(xy**2, axis=0)),
            label=controller,
        )

        panels[2].plot(
            t,
            np.sqrt(np.mean(xyz**2, axis=0)),
            label=controller,
        )

    titles = (
        "Mean signed altitude error (cm) / 10-90% band",
        "Across-seed XY RMSE at each time (cm)",
        "Across-seed position RMSE at each time (cm)",
    )

    for ax, title in zip(panels, titles):
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.axvspan(LATE_START, 10.0, alpha=0.07)
        ax.grid(alpha=0.3)
        ax.legend()

    panels[0].axhline(
        0, linestyle="--", linewidth=1, color="gray"
    )

    fig.suptitle(
        f"{name} | {len(paired_traces)} matched successes"
    )
    fig.tight_layout()
    fig.savefig(out / f"{name}_multiseed.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=7000)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    if args.seeds < 1:
        parser.error("--seeds must be positive")

    for controller, path in MODEL_PATHS.items():
        if not path.is_file():
            raise FileNotFoundError(f"{controller}: {path}")

    print("[OK] Both model checkpoints exist.", flush=True)
    print(
        "[INFO] Reusing run_episode() from diagnose_late_rmse.py",
        flush=True,
    )

    if args.check_only:
        print(
            "[OK] Preflight. Planned episodes: "
            f"{len(SCENARIOS)*args.seeds*len(MODEL_PATHS)}"
        )
        return

    models = {
        name: SAC.load(str(path), device="cpu")
        for name, path in MODEL_PATHS.items()
    }

    out = ROOT / "results" / "evaluation" / (
        "late_rmse_multiseed_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out.mkdir(parents=True, exist_ok=False)

    print(f"[OUTPUT] {out}", flush=True)

    episode_rows = []
    summary_rows = []
    pair_rows = []

    seed_list = range(
        args.seed_start,
        args.seed_start + args.seeds,
    )

    for scenario_name, radius, phase, omega in SCENARIOS:
        print(
            f"\n=== {scenario_name}: "
            f"R={radius}, phase={phase:.4f}, omega={omega} ===",
            flush=True,
        )

        successful = {
            name: [] for name in MODEL_PATHS
        }
        paired = []

        for seed in seed_list:
            outputs = {}

            for name, model in models.items():
                trace = run_episode(
                    model, radius, phase, omega, seed
                )
                metrics = calculate_metrics(trace)

                outputs[name] = (metrics, trace)

                episode_rows.append({
                    "scenario": scenario_name,
                    "seed": seed,
                    "controller": name,
                    **metrics,
                })

                if metrics["survived"]:
                    successful[name].append(metrics)

            if all(
                outputs[name][0]["survived"]
                for name in MODEL_PATHS
            ):
                paired.append({
                    name: outputs[name][1]
                    for name in MODEL_PATHS
                })

                a, b = (
                    outputs[name][0]
                    for name in ("Original", "Curriculum")
                )

                pair_rows.append({
                    "scenario": scenario_name,
                    "seed": seed,
                    "delta_late_rmse_cm": 100 * (
                        b["late_rmse_m"] - a["late_rmse_m"]
                    ),
                    "delta_late_xy_rmse_cm": 100 * (
                        b["late_xy_rmse_m"] - a["late_xy_rmse_m"]
                    ),
                    "delta_late_z_rmse_cm": 100 * (
                        b["late_z_rmse_m"] - a["late_z_rmse_m"]
                    ),
                    "delta_late_velocity_rmse_m_s": (
                        b["late_v_rmse_m_s"]
                        - a["late_v_rmse_m_s"]
                    ),
                })

            print(
                f"  seed {seed}: "
                + ", ".join(
                    f"{name}="
                    f"{'OK' if outputs[name][0]['survived'] else 'FAIL'}"
                    for name in MODEL_PATHS
                ),
                flush=True,
            )

        paired_count = len(paired)

        print(
            f"  Matched surviving seeds: "
            f"{paired_count}/{args.seeds}"
        )

        if not paired:
            continue

        for name in MODEL_PATHS:
            # Comparisons use exactly the same surviving seeds.
            matched_metrics = [
                calculate_metrics(pair[name])
                for pair in paired
            ]

            row = {
                "scenario": scenario_name,
                "controller": name,
                "survivals_total": len(successful[name]),
                "episodes_total": args.seeds,
                "matched_successes": paired_count,
            }

            for field in (
                "full_rmse_m",
                "late_rmse_m",
                "late_xy_rmse_m",
                "late_z_rmse_m",
                "late_v_rmse_m_s",
                "late_z_bias_m",
                "late_z_std_m",
                "late_z_first_rmse_m",
                "late_z_last_rmse_m",
                "late_radial_bias_m",
                "late_rpm_effort",
            ):
                row[field] = mean_field(
                    matched_metrics, field
                )

            # Use squared errors for the z contribution.
            row["z_fraction_of_squared_position_error_pct"] = (
                100
                * np.mean([
                    x["late_z_rmse_m"] ** 2
                    for x in matched_metrics
                ])
                / np.mean([
                    x["late_rmse_m"] ** 2
                    for x in matched_metrics
                ])
            )

            summary_rows.append(row)

            print(
                f"  {name:10s}: "
                f"late={100*row['late_rmse_m']:.3f}cm, "
                f"XY={100*row['late_xy_rmse_m']:.3f}cm, "
                f"Z={100*row['late_z_rmse_m']:.3f}cm, "
                f"Z bias={100*row['late_z_bias_m']:+.3f}cm, "
                f"Z std={100*row['late_z_std_m']:.3f}cm, "
                f"Z share="
                f"{row['z_fraction_of_squared_position_error_pct']:.1f}%, "
                f"V={row['late_v_rmse_m_s']:.4f}m/s"
            )

            print(
                f"             Z-RMSE "
                f"5-7.5s="
                f"{100*row['late_z_first_rmse_m']:.3f}cm, "
                f"7.5-10s="
                f"{100*row['late_z_last_rmse_m']:.3f}cm"
            )

        differences = [
            r for r in pair_rows
            if r["scenario"] == scenario_name
        ]

        print(
            "  Curriculum - Original (paired): "
            f"late="
            f"{mean_field(differences, 'delta_late_rmse_cm'):+.3f}cm, "
            f"XY="
            f"{mean_field(differences, 'delta_late_xy_rmse_cm'):+.3f}cm, "
            f"Z="
            f"{mean_field(differences, 'delta_late_z_rmse_cm'):+.3f}cm"
        )

        plot_scenario(
            out,
            scenario_name,
            paired,
        )

    def save_csv(filename, rows):
        if not rows:
            return

        with (
            out / filename
        ).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(rows)

    save_csv("episodes.csv", episode_rows)
    save_csv("matched_summary.csv", summary_rows)
    save_csv("paired_differences.csv", pair_rows)

    print(f"\n[FINISHED] Results saved in: {out}")
    print(
        "CSV: episodes.csv, matched_summary.csv, "
        "paired_differences.csv"
    )
    print(
        "PNG: A_nominal_multiseed.png, "
        "B_phase_multiseed.png, "
        "C_radius_multiseed.png"
    )


if __name__ == "__main__":
    main()
