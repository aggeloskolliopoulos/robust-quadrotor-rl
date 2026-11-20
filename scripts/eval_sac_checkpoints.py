
"""
Nominal benchmark: SAC Baseline_30k vs DSLPIDControl.

36 circle scenarios x 20 new seeds = 720 episodes/controller.

Both controllers use the same PhaseCircleTrackingEnv,
same initial seeds, motor limits, termination conditions
and tracking error metrics.

No training. No checkpoints are modified.

Run:
    python -u scripts/eval_sac_checkpoints.py --check-only
    python -u scripts/eval_sac_checkpoints.py
"""

from __future__ import annotations

import argparse
import csv
import math
import sys

from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from stable_baselines3 import SAC

from gym_pybullet_drones.control.DSLPIDControl import (
    DSLPIDControl,
)

from gym_pybullet_drones.utils.enums import DroneModel


# ============================================================
# 1. Paths and imports
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_radius_phase import PhaseCircleTrackingEnv


SAC_PATH = ROOT / (
    "results/checkpoints/"
    "curriculum_ablation_baseline_seed42_20261114_145522_495268/"
    "checkpoints/sac_curriculum_ablation_030000_steps.zip"
)


# ============================================================
# 2. Benchmark settings
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

PHYSICS_HZ = 240
CONTROL_HZ = 48

STEPS_PER_EPISODE = 480
LATE_START = 240

METRICS = (
    "full_rmse_m",
    "late_rmse_m",
    "late_xy_rmse_m",
    "late_z_rmse_m",
    "velocity_rmse_m_s",
)


# ============================================================
# 3. Environment
# ============================================================

def make_env(radius, phase, omega):

    return PhaseCircleTrackingEnv(
        gui=False,
        physics_freq_hz=PHYSICS_HZ,
        control_freq_hz=CONTROL_HZ,
        radius_m=radius,
        initial_phase_rad=phase,
        angular_speed_rad_s=omega,
    )


# ============================================================
# 4. Convert PID motor RPM to normalized action
# ============================================================

def rpm_to_action(rpm, sim):
    """
    Inverse of the piecewise action-to-RPM mapping.

    a = -1 -> RPM = 0
    a =  0 -> RPM = hover RPM
    a = +1 -> RPM = maximum RPM

    Both controllers use the same actuator limits.
    """

    rpm = np.asarray(
        rpm,
        dtype=np.float64,
    ).reshape(4)

    if not np.all(np.isfinite(rpm)):
        raise RuntimeError(
            "PID returned non-finite RPM values."
        )

    hover = float(sim.HOVER_RPM)
    maximum = float(sim.MAX_RPM)

    if not 0 < hover < maximum:
        raise RuntimeError(
            "Invalid simulator motor limits."
        )

    rpm = np.clip(
        rpm,
        0.0,
        maximum,
    )

    action = np.where(
        rpm <= hover,
        rpm / hover - 1.0,
        (rpm - hover) / (maximum - hover),
    )

    action = np.asarray(
        np.clip(action, -1.0, 1.0),
        dtype=np.float32,
    )

    # Check that normalized actions reconstruct
    # the original physical RPM commands.

    reconstructed = (
        sim._normalizedActionToRPM(action)
    )

    if not np.allclose(
        reconstructed,
        rpm,
        atol=0.01,
        rtol=0,
    ):
        raise RuntimeError(
            "PID RPM / normalized action mismatch."
        )

    return action


# ============================================================
# 5. PID controller action
# ============================================================

def pid_action(env, observation, pid):
    """
    Obtain the true simulator state.

    BaseAviary state vector:
      position         [0:3]
      quaternion       [3:7]
      velocity         [10:13]
      angular velocity [13:16]

    Our 16D observation:
      target_pos - position [0:3]
      target_vel - velocity [3:6]

    Reconstruct the reference from the same
    observation supplied to the SAC.
    """

    state = np.asarray(
        env.sim._getDroneStateVector(0),
        dtype=np.float64,
    )

    if state.shape != (20,):
        raise RuntimeError(
            f"Unexpected physical state: {state.shape}"
        )

    pos = state[0:3]
    quat = state[3:7]
    vel = state[10:13]
    angular_vel = state[13:16]

    obs = np.asarray(
        observation,
        dtype=np.float64,
    )

    target_pos = pos + obs[0:3]
    target_vel = vel + obs[3:6]

    rpm, _, _ = pid.computeControl(
        control_timestep=1.0 / CONTROL_HZ,

        cur_pos=pos.copy(),
        cur_quat=quat.copy(),
        cur_vel=vel.copy(),
        cur_ang_vel=angular_vel.copy(),

        target_pos=target_pos,
        target_vel=target_vel,

        target_rpy=np.zeros(3),
    )

    return rpm_to_action(
        rpm,
        env.sim,
    )


# ============================================================
# 6. RMSE metric
# ============================================================

def rmse_values(errors):
    """
    Euclidean vector RMSE.

    sqrt(mean(||error_t||^2))

    Compatible with the RMSE definition
    used in our previous SAC evaluations.
    """

    errors = np.asarray(
        errors,
        dtype=np.float64,
    )

    return float(
        np.sqrt(
            np.mean(
                np.sum(errors**2, axis=1)
            )
        )
    )


# ============================================================
# 7. Evaluate one episode
# ============================================================

def evaluate_episode(
    env,
    controller,
    seed,
    model=None,
):

    observation, _ = env.reset(
        seed=seed
    )

    # Fresh PID for EVERY episode.
    # No integral-state carry-over.

    pid = (
        DSLPIDControl(
            drone_model=DroneModel.CF2X
        )
        if controller == "PID"
        else None
    )

    position_errors = []
    velocity_errors = []

    survived = False

    for _ in range(STEPS_PER_EPISODE):

        if controller == "PID":

            action = pid_action(
                env,
                observation,
                pid,
            )

        else:

            action, _ = model.predict(
                observation,
                deterministic=True,
            )

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        observation = np.asarray(
            observation,
            dtype=np.float64,
        )

        position_errors.append(
            observation[:3].copy()
        )

        velocity_errors.append(
            observation[3:6].copy()
        )

        if terminated or truncated:

            survived = bool(
                truncated and not terminated
            )

            break

    result = {
        "survived": int(survived),

        "duration_s": (
            len(position_errors) / CONTROL_HZ
        ),

        **{
            field: float("nan")
            for field in METRICS
        },
    }

    # RMSE only for complete episodes.
    # Failures are accounted for in survival.

    if survived:

        e = np.asarray(
            position_errors,
            dtype=np.float64,
        )

        v = np.asarray(
            velocity_errors,
            dtype=np.float64,
        )

        late = e[LATE_START:]

        result.update({
            "full_rmse_m":
                rmse_values(e),

            "late_rmse_m":
                rmse_values(late),

            "late_xy_rmse_m":
                rmse_values(late[:, :2]),

            "late_z_rmse_m":
                float(
                    np.sqrt(
                        np.mean(late[:, 2] ** 2)
                    )
                ),

            "velocity_rmse_m_s":
                rmse_values(v),
        })

    return result


# ============================================================
# 8. Summary metrics
# ============================================================

def episode_key(row):

    return (
        row["radius"],
        row["phase"],
        row["omega"],
        row["seed"],
    )


def mean_success(rows, metric):

    values = [
        row[metric]
        for row in rows
        if row["survived"]
    ]

    if not values:
        return float("nan")

    return float(
        np.mean(values)
    )


def show_controller_summary(name, rows):

    n = len(rows)

    successes = sum(
        row["survived"]
        for row in rows
    )

    print()
    print(
        f"{name}: survival "
        f"{successes}/{n} "
        f"({100 * successes/n:.2f}%)"
    )

    for metric in METRICS:

        value = mean_success(
            rows,
            metric,
        )

        if metric == "velocity_rmse_m_s":

            unit = "m/s"

        else:

            value *= 100
            unit = "cm"

        print(
            f"  {metric:<22} "
            f"{value:.4f} {unit}"
        )

    hard = [
        row
        for row in rows
        if (
            row["radius"] == 0.45
            and row["phase"] == 0.0
            and row["omega"] == 1.0
        )
    ]

    hard_success = sum(
        row["survived"]
        for row in hard
    )

    print(
        f"  HARD R=.45 phi=0 omega=1: "
        f"{hard_success}/{len(hard)}"
    )


# ============================================================
# 9. Paired bootstrap
# ============================================================

def seed_cluster_bootstrap(
    groups,
    seed_list,
    rng,
    repeats=2000,
):
    """
    Resample entire initial-condition seed clusters.

    This accounts for correlation between scenarios
    evaluated with the same initial-condition seed.
    """

    samples = []

    for _ in range(repeats):

        drawn = rng.choice(
            seed_list,
            len(seed_list),
            replace=True,
        )

        values = [
            value
            for seed in drawn
            for value in groups[seed]
        ]

        if values:

            samples.append(
                float(np.mean(values))
            )

    if not samples:

        return (
            float("nan"),
            float("nan"),
        )

    return tuple(
        np.quantile(
            samples,
            [0.025, 0.975],
        )
    )


# ============================================================
# 10. Matched-success comparison
# ============================================================

def compare(
    rows_sac,
    rows_pid,
    seed_list,
):

    sac = {
        episode_key(row): row
        for row in rows_sac
    }

    pid = {
        episode_key(row): row
        for row in rows_pid
    }

    if sac.keys() != pid.keys():
        raise RuntimeError(
            "SAC and PID test scenarios differ."
        )

    common = [
        k
        for k in sac
        if (
            sac[k]["survived"]
            and pid[k]["survived"]
        )
    ]

    only_sac = sum(
        sac[k]["survived"]
        and not pid[k]["survived"]
        for k in sac
    )

    only_pid = sum(
        pid[k]["survived"]
        and not sac[k]["survived"]
        for k in sac
    )

    print()
    print("=" * 60)
    print("PAIRED SAC VS PID")
    print("=" * 60)

    print(
        f"Common successful episodes: "
        f"{len(common)}/{len(sac)}"
    )

    print(
        f"SAC success, PID failure: {only_sac}"
    )

    print(
        f"PID success, SAC failure: {only_pid}"
    )

    rng = np.random.default_rng(
        20261114
    )

    # --------------------------------------------------------
    # Survival difference
    # --------------------------------------------------------

    survival_by_seed = defaultdict(list)

    for k in sac:

        survival_by_seed[k[-1]].append(
            int(sac[k]["survived"])
            - int(pid[k]["survived"])
        )

    survival_differences = [
        x
        for group in survival_by_seed.values()
        for x in group
    ]

    delta_survival = (
        100 * np.mean(survival_differences)
    )

    ci_survival = seed_cluster_bootstrap(
        survival_by_seed,
        seed_list,
        rng,
    )

    print()
    print(
        f"Survival difference SAC-PID: "
        f"{delta_survival:+.3f} percentage points"
    )

    print(
        "  Seed-cluster bootstrap 95% CI: "
        f"[{100*ci_survival[0]:+.3f}, "
        f"{100*ci_survival[1]:+.3f}] pp"
    )

    # --------------------------------------------------------
    # RMSE on the SAME successful episodes
    # --------------------------------------------------------

    print()
    print(
        "MATCHED-SUCCESS MEAN PER-EPISODE RMSE"
    )

    for metric in METRICS:

        if not common:

            print(
                f"  {metric}: unavailable"
            )

            continue

        sac_mean = float(
            np.mean([
                sac[k][metric]
                for k in common
            ])
        )

        pid_mean = float(
            np.mean([
                pid[k][metric]
                for k in common
            ])
        )

        delta = sac_mean - pid_mean

        groups = defaultdict(list)

        for k in common:

            groups[k[-1]].append(
                sac[k][metric]
                - pid[k][metric]
            )

        ci = seed_cluster_bootstrap(
            groups,
            seed_list,
            rng,
        )

        if metric == "velocity_rmse_m_s":

            scale = 1
            unit = "m/s"

        else:

            scale = 100
            unit = "cm"

        print(
            f"  {metric:<22} "
            f"SAC={scale*sac_mean:.4f}, "
            f"PID={scale*pid_mean:.4f}, "
            f"delta={scale*delta:+.4f} {unit}"
        )

        print(
            "    Seed-cluster bootstrap 95% CI: "
            f"[{scale*ci[0]:+.4f}, "
            f"{scale*ci[1]:+.4f}] {unit}"
        )

    print()
    print(
        "Note: CIs describe variation across "
        "evaluation seeds, not across independently "
        "trained SAC policies."
    )


# ============================================================
# 11. Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed-start",
        type=int,
        default=12000,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
    )

    args = parser.parse_args()

    if args.seeds <= 0:

        parser.error(
            "--seeds must be positive."
        )

    if not SAC_PATH.is_file():

        raise FileNotFoundError(
            SAC_PATH
        )

    model = SAC.load(
        str(SAC_PATH),
        device="cpu",
    )

    seed_list = list(
        range(
            args.seed_start,
            args.seed_start + args.seeds,
        )
    )

    # --------------------------------------------------------
    # Preflight: test SAC and PID on one shared scenario.
    # --------------------------------------------------------

    if args.check_only:

        env = make_env(
            0.30,
            0.0,
            0.6,
        )

        try:

            print(
                "[CHECK] One scenario/seed, "
                "both controllers:",
                flush=True,
            )

            for controller in ("SAC", "PID"):

                result = evaluate_episode(
                    env,
                    controller,
                    seed_list[0],
                    model=(
                        model
                        if controller == "SAC"
                        else None
                    ),
                )

                print(
                    controller,
                    result,
                    flush=True,
                )

        finally:

            env.close()

        print(
            "[PASS] Preflight completed. "
            "No full benchmark run.",
            flush=True,
        )

        return

    # --------------------------------------------------------
    # Output CSV
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
        f"sac_vs_pid_{timestamp}.csv"
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

    results = {
        "SAC": [],
        "PID": [],
    }

    scenario_count = (
        len(RADII)
        * len(PHASES)
        * len(OMEGAS)
    )

    print()
    print("=" * 60)
    print("NOMINAL SAC VS PID BENCHMARK")
    print("=" * 60)

    print(
        f"Scenarios: {scenario_count}"
    )

    print(
        f"Seeds: {len(seed_list)}"
    )

    print(
        f"Episodes/controller: "
        f"{scenario_count * len(seed_list)}"
    )

    print(
        f"Total episodes: "
        f"{2 * scenario_count * len(seed_list)}"
    )

    print(
        f"Output CSV: {output_csv}",
        flush=True,
    )

    # --------------------------------------------------------
    # Evaluate same scenarios and seeds.
    # --------------------------------------------------------

    with output_csv.open(
        "x",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        for radius in RADII:

            for phase in PHASES:

                for omega in OMEGAS:

                    env = make_env(
                        radius,
                        phase,
                        omega,
                    )

                    try:

                        for seed in seed_list:

                            for controller in (
                                "SAC",
                                "PID",
                            ):

                                metrics = evaluate_episode(
                                    env,
                                    controller,
                                    seed,
                                    model=(
                                        model
                                        if controller == "SAC"
                                        else None
                                    ),
                                )

                                row = {
                                    "controller": controller,
                                    "radius": radius,
                                    "phase": phase,
                                    "omega": omega,
                                    "seed": seed,
                                    **metrics,
                                }

                                results[controller].append(
                                    row
                                )

                                writer.writerow(row)

                            file.flush()

                    finally:

                        env.close()

                    n = len(seed_list)

                    sac_success = sum(
                        row["survived"]
                        for row in results["SAC"][-n:]
                    )

                    pid_success = sum(
                        row["survived"]
                        for row in results["PID"][-n:]
                    )

                    print(
                        f"R={radius:.2f}, "
                        f"phi={phase:.3f}, "
                        f"omega={omega:.1f}: "
                        f"SAC={sac_success}/{n}, "
                        f"PID={pid_success}/{n}",
                        flush=True,
                    )

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("ALL CONTROLLERS")
    print("=" * 60)

    for controller in ("SAC", "PID"):

        show_controller_summary(
            controller,
            results[controller],
        )

    compare(
        results["SAC"],
        results["PID"],
        seed_list,
    )

    print()
    print("=" * 60)
    print("BENCHMARK COMPLETED")
    print("=" * 60)

    print(
        f"[FINISHED] Results saved to:\n"
        f"{output_csv}",
        flush=True,
    )


if __name__ == "__main__":
    main()
