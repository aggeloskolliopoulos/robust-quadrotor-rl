
"""Paired gust diagnostic: real position with vs without a gust.

Requires scripts/eval_wind_robustness.py (stochastic version).
Controllers: PID and frozen SAC Baseline 30k.
No training or modifications to existing environments.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime

import numpy as np
from stable_baselines3 import SAC
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel

import eval_wind_robustness as wind


THRESHOLD_M = 0.01
HOLD_S = 0.50


# ============================================================
# 1. Run one episode and record true physical positions
# ============================================================

def rollout(env, controller, policy, seed):

    obs, info = env.reset(seed=seed)

    pid = (
        DSLPIDControl(drone_model=DroneModel.CF2X)
        if controller == "PID"
        else None
    )

    positions = []
    terminated = truncated = False

    for _ in range(env.max_steps):

        if pid is not None:
            action, _ = wind.pid_action(pid, env)
        else:
            action, _ = policy.predict(
                obs, deterministic=True
            )

        obs, _, terminated, truncated, _ = env.step(action)

        positions.append(
            np.asarray(
                env.sim._getDroneStateVector(0)[0:3],
                dtype=np.float64,
            ).copy()
        )

        if terminated or truncated:
            break

    steps = len(positions)

    if env.sim._wind_substeps != (
        steps * env.sim.PYB_STEPS_PER_CTRL
    ):
        raise RuntimeError(
            "Wind missed physics substeps"
        )

    return {
        "positions": np.asarray(
            positions, dtype=np.float64
        ),
        "survived": bool(
            truncated
            and not terminated
            and steps == env.max_steps
        ),
        "duration_s": steps / env.control_freq_hz,
        "info": info,
    }


# ============================================================
# 2. Compare the paired trajectories
# ============================================================

def analyze_pair(nominal, gust, control_hz):

    # Same initial position and attitude.
    for key in ("initial_pos", "initial_rpy"):
        np.testing.assert_array_equal(
            nominal["info"][key],
            gust["info"][key],
        )

    n = min(
        len(nominal["positions"]),
        len(gust["positions"]),
    )

    time_s = (
        np.arange(1, n + 1, dtype=np.float64)
        / control_hz
    )

    # Difference between real physical trajectories.
    delta = np.linalg.norm(
        gust["positions"][:n]
        - nominal["positions"][:n],
        axis=1,
    )

    # Before the gust, both trajectories must coincide.
    pre = delta[time_s <= wind.GUST_START]

    if len(pre) and np.max(pre) > 1e-6:
        raise RuntimeError(
            "Paired trajectories differ BEFORE the gust: "
            f"{np.max(pre):.6g} m"
        )

    paired_success = (
        nominal["survived"] and gust["survived"]
    )

    result = {
        "nominal_survived": int(nominal["survived"]),
        "gust_survived": int(gust["survived"]),
        "nominal_duration_s": nominal["duration_s"],
        "gust_duration_s": gust["duration_s"],
        "paired_success": int(paired_success),

        "peak_delta_3_6_cm": float("nan"),
        "peak_delta_time_s": float("nan"),
        "post_gust_rms_delta_4_6_cm": float("nan"),
        "post_gust_time_above_1cm_s": float("nan"),
        "settling_time_s": float("nan"),
        "recovery_status": "excluded_failure",
    }

    # Failures are recorded but excluded from averages
    # of full-length transient trajectories.
    if not paired_success:
        return result, time_s, delta

    # --------------------------------------------------------
    # Peak deviation during and shortly after the gust
    # --------------------------------------------------------

    peak_mask = (
        (time_s >= 3.0)
        & (time_s < 6.0)
    )

    peak_indices = np.flatnonzero(peak_mask)

    peak_idx = peak_indices[
        np.argmax(delta[peak_mask])
    ]

    result["peak_delta_3_6_cm"] = (
        100.0 * float(delta[peak_idx])
    )

    result["peak_delta_time_s"] = float(
        time_s[peak_idx]
    )

    # --------------------------------------------------------
    # Post-gust deviation, t = 4..6 s
    # --------------------------------------------------------

    post_mask = (
        (time_s >= 4.0)
        & (time_s < 6.0)
    )

    result["post_gust_rms_delta_4_6_cm"] = (
        100.0
        * float(
            np.sqrt(
                np.mean(delta[post_mask] ** 2)
            )
        )
    )

    result["post_gust_time_above_1cm_s"] = (
        float(
            np.count_nonzero(
                delta[time_s >= 4.0] > THRESHOLD_M
            )
        )
        / control_hz
    )

    # --------------------------------------------------------
    # Settling time
    #
    # The position must return within 1 cm of the
    # nominal trajectory and remain there until the
    # episode ends, with at least 0.5 s of observations.
    #
    # This prevents false recovery from a brief dip below
    # the threshold followed by another excursion.
    # --------------------------------------------------------

    excursions = np.flatnonzero(
        (time_s >= 3.0)
        & (delta > THRESHOLD_M)
    )

    hold_steps = int(
        round(HOLD_S * control_hz)
    )

    if len(excursions) == 0:

        result["recovery_status"] = "no_excursion"
        result["settling_time_s"] = 0.0

    else:

        last_exceedance = int(excursions[-1])

        for index in np.flatnonzero(time_s >= 4.0):

            if (
                index > last_exceedance
                and index + hold_steps <= n
                and np.all(
                    delta[index:] <= THRESHOLD_M
                )
            ):

                result["settling_time_s"] = max(
                    0.0,
                    float(time_s[index] - 4.0),
                )

                result["recovery_status"] = "settled"
                break

        else:

            result["recovery_status"] = "not_settled"

    return result, time_s, delta


# ============================================================
# 3. Main evaluation
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--check-only",
        action="store_true",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
    )

    parser.add_argument(
        "--seed-start",
        type=int,
        default=26000,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    if args.seeds < 1 or args.seed_start < 0:
        parser.error("Invalid seeds")

    if not wind.mass.BASELINE.is_file():
        raise FileNotFoundError(wind.mass.BASELINE)

    # Reuse the wind physics preflight.
    wind.preflight()

    # --------------------------------------------------------
    # Paired physics preflight
    # --------------------------------------------------------

    if args.check_only:

        calm = wind.make_env(
            "nominal", 0.30, 0.0, 0.60
        )

        gust = wind.make_env(
            "random_gust", 0.30, 0.0, 0.60
        )

        try:

            a = rollout(
                calm, "PID", None, 26000
            )

            b = rollout(
                gust, "PID", None, 26000
            )

            metrics, _, _ = analyze_pair(
                a, b, calm.control_freq_hz
            )

            if not metrics["paired_success"]:
                raise RuntimeError(
                    "Preflight pair failed"
                )

            if metrics["peak_delta_3_6_cm"] <= 0.001:
                raise RuntimeError(
                    "Gust had no measurable effect"
                )

            print(
                "[PASS] Identical trajectories "
                "before gust"
            )

            print(
                "[PASS] Gust peak deviation: "
                f"{metrics['peak_delta_3_6_cm']:.4f} cm"
            )

        finally:

            calm.close()
            gust.close()

        print("[PASS] Diagnostic preflight completed")
        return

    # --------------------------------------------------------
    # Load the two controllers
    # --------------------------------------------------------

    policies = {
        "PID": None,

        "Baseline_30k": SAC.load(
            str(wind.mass.BASELINE),
            device="cpu",
        ),
    }

    radii = (
        (0.30,)
        if args.quick
        else wind.mass.RADII
    )

    phases = (
        (0.0,)
        if args.quick
        else wind.mass.PHASES
    )

    omegas = (
        (0.60,)
        if args.quick
        else wind.mass.OMEGAS
    )

    seeds = range(
        args.seed_start,
        args.seed_start + (
            2 if args.quick else args.seeds
        ),
    )

    # --------------------------------------------------------
    # Outputs
    # --------------------------------------------------------

    output_dir = (
        wind.mass.ROOT
        / "results/evaluation"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    outfile = output_dir / (
        f"gust_paired_diagnostic_{stamp}.csv"
    )

    tracefile = output_dir / (
        f"gust_paired_example_trace_{stamp}.csv"
    )

    columns = [
        "controller",
        "radius_m",
        "phase_rad",
        "omega_rad_s",
        "seed",
        "wind_hash",
        "wind_amplitude_mn",
        "wind_direction_rad",
        "nominal_survived",
        "gust_survived",
        "nominal_duration_s",
        "gust_duration_s",
        "paired_success",
        "peak_delta_3_6_cm",
        "peak_delta_time_s",
        "post_gust_rms_delta_4_6_cm",
        "post_gust_time_above_1cm_s",
        "settling_time_s",
        "recovery_status",
    ]

    all_rows = []
    hashes = {}
    example_trace = []

    per_controller = (
        len(radii)
        * len(phases)
        * len(omegas)
        * len(seeds)
    )

    print()
    print("=" * 65)
    print("PAIRED RANDOM GUST DIAGNOSTIC")
    print("PID vs SAC BASELINE 30k")
    print("=" * 65)

    print(
        f"Matched pairs/controller: {per_controller}"
    )

    print(
        f"Total simulated episodes: {4 * per_controller}"
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    with outfile.open(
        "x",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns,
        )

        writer.writeheader()

        for name, model in policies.items():

            print(
                f"\n[RUN] {name}",
                flush=True,
            )

            for radius in radii:

                for phase in phases:

                    for omega in omegas:

                        calm_env = wind.make_env(
                            "nominal",
                            radius,
                            phase,
                            omega,
                        )

                        gust_env = wind.make_env(
                            "random_gust",
                            radius,
                            phase,
                            omega,
                        )

                        try:

                            for seed in seeds:

                                calm = rollout(
                                    calm_env,
                                    name,
                                    model,
                                    seed,
                                )

                                gust = rollout(
                                    gust_env,
                                    name,
                                    model,
                                    seed,
                                )

                                metrics, t, delta = (
                                    analyze_pair(
                                        calm,
                                        gust,
                                        calm_env.control_freq_hz,
                                    )
                                )

                                key = (
                                    radius,
                                    phase,
                                    omega,
                                    seed,
                                )

                                wind_hash = (
                                    gust["info"]["wind_hash"]
                                )

                                old_hash = hashes.setdefault(
                                    key,
                                    wind_hash,
                                )

                                if old_hash != wind_hash:
                                    raise RuntimeError(
                                        "Different gusts "
                                        "between PID and SAC"
                                    )

                                row = {
                                    "controller": name,
                                    "radius_m": radius,
                                    "phase_rad": phase,
                                    "omega_rad_s": omega,
                                    "seed": seed,
                                    "wind_hash": wind_hash,
                                    "wind_amplitude_mn":
                                        gust["info"][
                                            "wind_amplitude_mn"
                                        ],
                                    "wind_direction_rad":
                                        gust["info"][
                                            "wind_direction_rad"
                                        ],
                                    **metrics,
                                }

                                writer.writerow(row)
                                all_rows.append(row)

                                # Save one example trajectory
                                # for later visualization.

                                if (
                                    radius == 0.30
                                    and phase == 0.0
                                    and omega == 0.60
                                    and seed == args.seed_start
                                ):

                                    for idx, ts in enumerate(t):

                                        example_trace.append({
                                            "controller": name,
                                            "time_s": ts,
                                            "delta_m": delta[idx],
                                            "nominal_x":
                                                calm["positions"][idx, 0],
                                            "nominal_y":
                                                calm["positions"][idx, 1],
                                            "nominal_z":
                                                calm["positions"][idx, 2],
                                            "gust_x":
                                                gust["positions"][idx, 0],
                                            "gust_y":
                                                gust["positions"][idx, 1],
                                            "gust_z":
                                                gust["positions"][idx, 2],
                                        })

                        finally:

                            calm_env.close()
                            gust_env.close()

                        file.flush()

            # ----------------------------------------------------
            # Per-controller summary
            # ----------------------------------------------------

            group = [
                r for r in all_rows
                if r["controller"] == name
            ]

            complete = [
                r for r in group
                if r["paired_success"]
            ]

            recovered = [
                r for r in complete
                if r["recovery_status"]
                in ("settled", "no_excursion")
            ]

            def mean(metric):

                if not complete:
                    return float("nan")

                return float(
                    np.mean([
                        r[metric]
                        for r in complete
                    ])
                )

            mean_settling = (
                float(
                    np.mean([
                        r["settling_time_s"]
                        for r in recovered
                    ])
                )
                if recovered
                else float("nan")
            )

            no_excursion = sum(
                r["recovery_status"] == "no_excursion"
                for r in complete
            )

            nominal_ok = sum(
                r["nominal_survived"]
                for r in group
            )

            gust_ok = sum(
                r["gust_survived"]
                for r in group
            )

            print(
                f"[DONE] {name}: "
                f"nominal={nominal_ok}/{len(group)}, "
                f"gust={gust_ok}/{len(group)}, "
                f"paired={len(complete)}",
                flush=True,
            )

            print(
                f"  peak(3-6s) = "
                f"{mean('peak_delta_3_6_cm'):.3f} cm"
            )

            print(
                f"  post RMS(4-6s) = "
                f"{mean('post_gust_rms_delta_4_6_cm'):.3f} cm"
            )

            print(
                f"  settled={len(recovered)}/{len(complete)}, "
                f"no_excursion={no_excursion}, "
                f"mean_settling={mean_settling:.3f}s"
            )

    # ========================================================
    # Matched PID vs SAC comparison
    # ========================================================

    by_controller = {
        name: {
            (
                r["radius_m"],
                r["phase_rad"],
                r["omega_rad_s"],
                r["seed"],
            ): r

            for r in all_rows

            if (
                r["controller"] == name
                and r["paired_success"]
            )
        }

        for name in policies
    }

    common = sorted(
        set(by_controller["PID"])
        & set(by_controller["Baseline_30k"])
    )

    print()
    print("=" * 65)
    print(
        f"MATCHED SAC VS PID: {len(common)} "
        "paired-success cases"
    )
    print("=" * 65)

    for metric in (
        "peak_delta_3_6_cm",
        "post_gust_rms_delta_4_6_cm",
    ):

        pid_mean = (
            np.mean([
                by_controller["PID"][k][metric]
                for k in common
            ])
            if common else float("nan")
        )

        sac_mean = (
            np.mean([
                by_controller["Baseline_30k"][k][metric]
                for k in common
            ])
            if common else float("nan")
        )

        print(
            f"{metric}: "
            f"SAC={sac_mean:.3f} cm, "
            f"PID={pid_mean:.3f} cm, "
            f"delta={sac_mean-pid_mean:+.3f} cm"
        )

    # ========================================================
    # Example time-series CSV for future plots
    # ========================================================

    if example_trace:

        with tracefile.open(
            "x",
            newline="",
            encoding="utf-8",
        ) as f:

            twriter = csv.DictWriter(
                f,
                fieldnames=list(
                    example_trace[0]
                ),
            )

            twriter.writeheader()
            twriter.writerows(example_trace)

    print()
    print(f"[FINISHED] {outfile}")

    if example_trace:
        print(f"[EXAMPLE TRACE] {tracefile}")

    print(
        "Recovery: within 1 cm of own nominal "
        "path until episode end."
    )

    print(
        "Failed pairs are counted in survival, "
        "but excluded from transient averages."
    )


if __name__ == "__main__":
    main()
