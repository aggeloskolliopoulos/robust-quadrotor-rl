
"""Zero-shot mass robustness benchmark.

Compare:
    - Nominal DSLPIDControl
    - SAC Baseline 30k
    - SAC Final 20k

Mass perturbations:
    -20%, -10%, 0%, +10%, +20%

Same reference trajectories, initial seeds, physical actuator
constants, and termination conditions across all controllers.

No training. No modification to existing SAC models or URDF files.
"""

from __future__ import annotations

import argparse
import csv
import math

from datetime import datetime
from pathlib import Path

import numpy as np
import pybullet as p

from stable_baselines3 import SAC

from gym_pybullet_drones.control.DSLPIDControl import (
    DSLPIDControl,
)
from gym_pybullet_drones.utils.enums import DroneModel

from eval_controllers import pid_action
from eval_radius_phase import PhaseCircleTrackingEnv


# ============================================================
# 1. Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

BASELINE = ROOT / (
    "results/checkpoints/"
    "curriculum_ablation_baseline_seed42_"
    "20261114_145522_495268/"
    "checkpoints/"
    "sac_curriculum_ablation_030000_steps.zip"
)

FINAL20 = ROOT / (
    "results/checkpoints/"
    "sac_long_nominal_from_baseline30k_"
    "seed42_20261120_150049_642310/"
    "checkpoints/"
    "sac_long_nominal_020000_steps.zip"
)

CONTROLLERS = (
    "PID",
    "Baseline_30k",
    "Final_20k",
)

RADII = (0.15, 0.30, 0.45)

PHASES = (
    0.0,
    math.pi / 2,
    math.pi,
    3 * math.pi / 2,
)

OMEGAS = (0.2, 0.6, 1.0)

MASS_LEVELS = (
    -20.0,
    -10.0,
    0.0,
    10.0,
    20.0,
)

METRICS = (
    "full_xyz_m",
    "late_xyz_m",
    "late_xy_m",
    "late_z_m",
)


# ============================================================
# 2. Environment with physical mass uncertainty
# ============================================================

class MassPerturbedCircleEnv(PhaseCircleTrackingEnv):
    """
    Apply the mass perturbation to the real PyBullet body.

    Important:
        QuadrotorTrackingEnv.reset() resets PyBullet and
        reloads the drone URDF.

        Therefore, changeDynamics() must be called AFTER
        every parent reset.

    The controller and motor parameters remain nominal.
    """

    def __init__(
        self,
        *,
        mass_percent=0.0,
        **kwargs,
    ):

        self.mass_percent = float(mass_percent)

        if self.mass_percent <= -100.0:
            raise ValueError(
                "Mass variation must be greater than -100%"
            )

        super().__init__(**kwargs)

        self.nominal_mass = float(self.sim.M)

        self.nominal_hover_rpm = float(
            self.sim.HOVER_RPM
        )

        self.nominal_max_rpm = float(
            self.sim.MAX_RPM
        )

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):

        # First perform the regular simulator reset.
        # This reloads the original URDF.

        observation, info = super().reset(
            seed=seed,
            options=options,
        )

        sim = self.sim

        body_id = int(sim.DRONE_IDS[0])
        client_id = int(sim.CLIENT)

        # -1 identifies the base link in PyBullet.

        before = p.getDynamicsInfo(
            body_id,
            -1,
            physicsClientId=client_id,
        )

        base_mass = float(before[0])

        # Preserve the original inertia tensor.
        inertia = np.asarray(
            before[2],
            dtype=np.float64,
        )

        # Check that the physical base mass corresponds
        # to the nominal drone mass from the URDF.

        if not np.isclose(
            base_mass,
            self.nominal_mass,
            rtol=1e-5,
            atol=1e-8,
        ):

            raise RuntimeError(
                "Unexpected URDF base mass: "
                f"{base_mass} kg; simulator expects "
                f"{self.nominal_mass} kg"
            )

        desired_mass = (
            self.nominal_mass
            * (1.0 + self.mass_percent / 100.0)
        )

        # Leave nominal physics entirely unchanged.
        # For nonzero perturbations, change ONLY the mass.
        # Explicitly preserve the inertia diagonal.

        if self.mass_percent != 0.0:

            p.changeDynamics(
                body_id,
                -1,
                mass=desired_mass,
                localInertiaDiagonal=inertia.tolist(),
                physicsClientId=client_id,
            )

        # Read the actual dynamics back from PyBullet.

        after = p.getDynamicsInfo(
            body_id,
            -1,
            physicsClientId=client_id,
        )

        actual_mass = float(after[0])

        actual_inertia = np.asarray(
            after[2],
            dtype=np.float64,
        )

        if not np.isclose(
            actual_mass,
            desired_mass,
            rtol=1e-6,
            atol=1e-9,
        ):

            raise RuntimeError(
                "Physical mass update failed: "
                f"expected {desired_mass}, "
                f"got {actual_mass}"
            )

        np.testing.assert_allclose(
            actual_inertia,
            inertia,
            rtol=1e-5,
            atol=1e-10,
        )

        # Ensure no nominal simulator/control parameters
        # were changed accidentally.

        if not (
            np.isclose(sim.M, self.nominal_mass)
            and np.isclose(
                sim.HOVER_RPM,
                self.nominal_hover_rpm,
            )
            and np.isclose(
                sim.MAX_RPM,
                self.nominal_max_rpm,
            )
            and np.isclose(
                self.hover_rpm,
                self.nominal_hover_rpm,
            )
        ):

            raise RuntimeError(
                "Nominal actuator/controller "
                "parameters changed"
            )

        info = dict(info)

        info.update({
            "physical_mass_kg": actual_mass,
            "nominal_mass_kg": self.nominal_mass,
            "mass_percent": self.mass_percent,
        })

        return observation, info


# ============================================================
# 3. Evaluate one episode
# ============================================================

def run_episode(
    env,
    controller,
    model,
    seed,
):

    observation, reset_info = env.reset(
        seed=seed
    )

    # A new PID instance is created for every episode
    # to reset its integrator and derivative memory.

    pid = (
        DSLPIDControl(
            drone_model=DroneModel.CF2X
        )
        if controller == "PID"
        else None
    )

    position_errors = []

    terminated = False
    truncated = False

    for _ in range(env.max_steps):

        # ---------------------------------------------
        # Controller
        # ---------------------------------------------

        if pid is not None:

            action, _ = pid_action(
                pid,
                env,
            )

        else:

            action, _ = model.predict(
                observation,
                deterministic=True,
            )

        # ---------------------------------------------
        # Apply the action to the SAME environment
        # ---------------------------------------------

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        # Observation[:3] is the position error:
        #
        # target_position - current_position

        position_errors.append(
            np.asarray(
                observation[:3],
                dtype=np.float64,
            ).copy()
        )

        if terminated or truncated:
            break

    survived = bool(
        truncated
        and not terminated
        and len(position_errors) == env.max_steps
    )

    result = {
        "survived": int(survived),
        "duration_s": (
            len(position_errors)
            / env.control_freq_hz
        ),
        "physical_mass_kg": float(
            reset_info["physical_mass_kg"]
        ),
    }

    # Only completed episodes contribute to RMSE.
    # Failures are tracked separately via survival.

    result.update({
        metric: float("nan")
        for metric in METRICS
    })

    if survived:

        errors = np.asarray(
            position_errors,
            dtype=np.float64,
        )

        late = errors[
            env.max_steps // 2:
        ]

        def rms(array):

            return float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            array ** 2,
                            axis=1,
                        )
                    )
                )
            )

        result.update({
            "full_xyz_m":
                rms(errors),

            "late_xyz_m":
                rms(late),

            "late_xy_m":
                rms(late[:, :2]),

            "late_z_m":
                rms(late[:, 2:3]),
        })

    return result


# ============================================================
# 4. Statistical helpers
# ============================================================

def episode_key(row):

    return (
        row["radius_m"],
        row["phase_rad"],
        row["omega_rad_s"],
        row["seed"],
    )


def summary(rows):

    successful = [
        row
        for row in rows
        if row["survived"]
    ]

    means = {}

    for metric in METRICS:

        means[metric] = (
            100.0 * float(
                np.mean([
                    row[metric]
                    for row in successful
                ])
            )
            if successful
            else float("nan")
        )

    return (
        len(successful),
        len(rows),
        means,
    )


def matched_report(
    candidate,
    reference,
    name,
):

    candidate_success = {
        episode_key(row): row
        for row in candidate
        if row["survived"]
    }

    reference_success = {
        episode_key(row): row
        for row in reference
        if row["survived"]
    }

    common = sorted(
        candidate_success.keys()
        & reference_success.keys()
    )

    candidate_only = len(
        candidate_success.keys()
        - reference_success.keys()
    )

    reference_only = len(
        reference_success.keys()
        - candidate_success.keys()
    )

    print(
        f"  {name}: matched={len(common)}, "
        f"candidate-only={candidate_only}, "
        f"reference-only={reference_only}"
    )

    if not common:
        return

    for metric in (
        "late_xyz_m",
        "late_xy_m",
        "late_z_m",
        "full_xyz_m",
    ):

        candidate_mean = (
            100.0 * np.mean([
                candidate_success[k][metric]
                for k in common
            ])
        )

        reference_mean = (
            100.0 * np.mean([
                reference_success[k][metric]
                for k in common
            ])
        )

        print(
            f"    {metric}: "
            f"{candidate_mean:.3f} vs "
            f"{reference_mean:.3f} cm "
            f"| delta="
            f"{candidate_mean-reference_mean:+.3f} cm"
        )


# ============================================================
# 5. Environment constructor
# ============================================================

def make_env(
    mass_percent,
    radius,
    phase,
    omega,
):

    return MassPerturbedCircleEnv(
        gui=False,
        physics_freq_hz=240,
        control_freq_hz=48,
        mass_percent=mass_percent,
        radius_m=radius,
        initial_phase_rad=phase,
        angular_speed_rad_s=omega,
    )


# ============================================================
# 6. Physical and nominal preflight
# ============================================================

def preflight():

    # Compare original environment with our mass=0 version.

    original = PhaseCircleTrackingEnv(
        radius_m=0.30,
        initial_phase_rad=math.pi / 2,
        angular_speed_rad_s=0.60,
    )

    nominal = make_env(
        0.0,
        0.30,
        math.pi / 2,
        0.60,
    )

    try:

        obs_original, _ = original.reset(
            seed=17000
        )

        obs_nominal, info = nominal.reset(
            seed=17000
        )

        np.testing.assert_allclose(
            obs_original,
            obs_nominal,
            rtol=0,
            atol=1e-7,
        )

        # Apply exactly the same action.

        action = np.zeros(
            4,
            dtype=np.float32,
        )

        next_original, *_ = original.step(
            action
        )

        next_nominal, *_ = nominal.step(
            action
        )

        np.testing.assert_allclose(
            next_original,
            next_nominal,
            rtol=0,
            atol=1e-6,
        )

        print(
            "[PASS] Mass 0% matches original "
            "PhaseCircleTrackingEnv"
        )

        print(
            "[INFO] Nominal mass = "
            f"{info['physical_mass_kg']*1000:.3f} g"
        )

    finally:

        original.close()
        nominal.close()

    # Verify that the new physical mass survives resets.

    env = make_env(
        20.0,
        0.30,
        0.0,
        0.60,
    )

    try:

        for seed in (17000, 17001):

            _, info = env.reset(
                seed=seed
            )

            expected_mass = 0.027 * 1.20

            if not np.isclose(
                info["physical_mass_kg"],
                expected_mass,
                atol=1e-7,
            ):

                raise RuntimeError(
                    "Unexpected +20% mass after reset"
                )

            print(
                f"[PASS] seed={seed}: "
                f"+20% mass = "
                f"{info['physical_mass_kg']*1000:.3f} g"
            )

        observation, _ = env.reset(
            seed=17002
        )

        action = np.zeros(
            4,
            dtype=np.float32,
        )

        next_obs, reward, *_ = env.step(
            action
        )

        if (
            not np.all(np.isfinite(next_obs))
            or not np.isfinite(reward)
        ):

            raise RuntimeError(
                "Nonfinite perturbed physics step"
            )

        print(
            "[PASS] Perturbed physics step; "
            "nominal RPM mapping unchanged"
        )

    finally:

        env.close()

    print(
        "[PASS] Preflight complete. "
        "No evaluation grid started."
    )


# ============================================================
# 7. Main benchmark
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
        default=17000,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--mass-levels",
        type=float,
        nargs="+",
        default=MASS_LEVELS,
    )

    args = parser.parse_args()

    if args.seeds < 1:

        parser.error(
            "--seeds must be positive"
        )

    if any(
        mass <= -100
        for mass in args.mass_levels
    ):

        parser.error(
            "Mass changes must be above -100%"
        )

    # Check the two frozen SAC models.

    for path in (BASELINE, FINAL20):

        if not path.is_file():

            raise FileNotFoundError(path)

    # Verify the physical simulation before benchmarking.

    preflight()

    if args.check_only:
        return

    models = {
        "PID": None,
        "Baseline_30k": SAC.load(
            str(BASELINE),
            device="cpu",
        ),
        "Final_20k": SAC.load(
            str(FINAL20),
            device="cpu",
        ),
    }

    radii = RADII
    phases = PHASES
    omegas = OMEGAS

    seeds = range(
        args.seed_start,
        args.seed_start + args.seeds,
    )

    # Small, fast test before the full benchmark.

    if args.quick:

        radii = (0.30,)
        phases = (0.0,)
        omegas = (0.60,)

        seeds = range(
            args.seed_start,
            args.seed_start + 2,
        )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    output = (
        ROOT
        / "results/evaluation"
        / f"mass_robustness_{timestamp}.csv"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "controller",
        "mass_percent",
        "physical_mass_kg",
        "radius_m",
        "phase_rad",
        "omega_rad_s",
        "seed",
        "survived",
        "duration_s",
        *METRICS,
    ]

    all_rows = []

    scenarios = (
        len(radii)
        * len(phases)
        * len(omegas)
    )

    total_episodes = (
        len(CONTROLLERS)
        * len(args.mass_levels)
        * scenarios
        * len(seeds)
    )

    print()
    print("=" * 70)
    print("ZERO-SHOT MASS ROBUSTNESS")
    print("PID vs TWO FROZEN SAC CONTROLLERS")
    print("=" * 70)

    print(
        f"Mass levels: {args.mass_levels}"
    )

    print(
        f"Scenarios: {scenarios}"
    )

    print(
        f"Episodes/controller/mass: "
        f"{scenarios*len(seeds)}"
    )

    print(
        f"Total episodes: {total_episodes}"
    )

    print(
        f"CSV: {output}",
        flush=True,
    )

    # --------------------------------------------------------
    # Run all controllers in all mass conditions
    # --------------------------------------------------------

    with output.open(
        "x",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for mass in args.mass_levels:

            for controller in CONTROLLERS:

                group = []

                print(
                    f"\n[RUN] mass={mass:+g}% "
                    f"controller={controller}",
                    flush=True,
                )

                for radius in radii:

                    for phase in phases:

                        for omega in omegas:

                            env = make_env(
                                mass,
                                radius,
                                phase,
                                omega,
                            )

                            try:

                                for seed in seeds:

                                    result = run_episode(
                                        env,
                                        controller,
                                        models[controller],
                                        seed,
                                    )

                                    row = {
                                        "controller": controller,
                                        "mass_percent": float(mass),
                                        "radius_m": float(radius),
                                        "phase_rad": float(phase),
                                        "omega_rad_s": float(omega),
                                        "seed": int(seed),
                                        **result,
                                    }

                                    group.append(row)
                                    all_rows.append(row)

                                    writer.writerow(row)

                            finally:

                                env.close()

                            # Keep partial results available
                            # even if execution is interrupted.

                            f.flush()

                successes, n, means = summary(
                    group
                )

                print(
                    f"[DONE] {controller:<13} "
                    f"mass={mass:+g}%: "
                    f"survival={successes}/{n}, "
                    f"full={means['full_xyz_m']:.3f} cm, "
                    f"late={means['late_xyz_m']:.3f} cm, "
                    f"lateXY={means['late_xy_m']:.3f} cm, "
                    f"lateZ={means['late_z_m']:.3f} cm",
                    flush=True,
                )

    # ========================================================
    # Overall summary
    # ========================================================

    print()
    print("=" * 70)
    print("SUMMARY - RMSE IN CM, SUCCESSFUL EPISODES ONLY")
    print("=" * 70)

    print(
        f"{'Mass':>7} "
        f"{'Controller':<14} "
        f"{'Survival':>10} "
        f"{'Full':>8} "
        f"{'Late':>8} "
        f"{'LateXY':>8} "
        f"{'LateZ':>8}"
    )

    for mass in args.mass_levels:

        for name in CONTROLLERS:

            rows = [
                r
                for r in all_rows
                if (
                    r["mass_percent"] == mass
                    and r["controller"] == name
                )
            ]

            successes, n, means = summary(rows)

            print(
                f"{mass:+6g}% "
                f"{name:<14} "
                f"{successes:4}/{n:<5} "
                f"{means['full_xyz_m']:8.3f} "
                f"{means['late_xyz_m']:8.3f} "
                f"{means['late_xy_m']:8.3f} "
                f"{means['late_z_m']:8.3f}"
            )

    # ========================================================
    # Paired controller comparisons at the same mass
    # ========================================================

    print()
    print("=" * 70)
    print("PAIRED CONTROLLER COMPARISONS AT THE SAME MASS")
    print("=" * 70)

    for mass in args.mass_levels:

        subset = {
            name: [
                r
                for r in all_rows
                if (
                    r["mass_percent"] == mass
                    and r["controller"] == name
                )
            ]

            for name in CONTROLLERS
        }

        print(
            f"\nMass = {mass:+g}%"
        )

        matched_report(
            subset["Baseline_30k"],
            subset["PID"],
            "Baseline vs PID",
        )

        matched_report(
            subset["Final_20k"],
            subset["PID"],
            "Final20 vs PID",
        )

        matched_report(
            subset["Final_20k"],
            subset["Baseline_30k"],
            "Final20 vs Baseline",
        )

    # ========================================================
    # Compare mass-perturbed and nominal performance
    # ========================================================

    if 0.0 in args.mass_levels:

        print()
        print("=" * 70)
        print("PAIRED MASS DEGRADATION VS NOMINAL")
        print("=" * 70)

        for name in CONTROLLERS:

            reference = [
                r
                for r in all_rows
                if (
                    r["controller"] == name
                    and r["mass_percent"] == 0.0
                )
            ]

            for mass in args.mass_levels:

                if mass == 0.0:
                    continue

                candidate = [
                    r
                    for r in all_rows
                    if (
                        r["controller"] == name
                        and r["mass_percent"] == mass
                    )
                ]

                matched_report(
                    candidate,
                    reference,
                    f"{name} {mass:+g}% vs nominal",
                )

    # ========================================================
    # Identify the most difficult scenario groups
    # ========================================================

    print()
    print("=" * 70)
    print("WORST FAILURE GROUPS")
    print("=" * 70)

    groups = {}

    for row in all_rows:

        grouping = (
            row["mass_percent"],
            row["controller"],
            row["radius_m"],
            row["phase_rad"],
            row["omega_rad_s"],
        )

        groups.setdefault(
            grouping, []
        ).append(row)

    failures = []

    for grouping, rows in groups.items():

        nfail = sum(
            1 - r["survived"]
            for r in rows
        )

        if nfail:

            failures.append(
                (
                    nfail,
                    grouping,
                    len(rows),
                )
            )

    for nfail, grouping, n in sorted(
        failures,
        reverse=True,
    )[:12]:

        (
            mass,
            name,
            radius,
            phase,
            omega,
        ) = grouping

        print(
            f"mass={mass:+g}% "
            f"{name:<13} "
            f"R={radius:.2f} "
            f"phase={phase:.3f} "
            f"omega={omega:.1f}: "
            f"failures={nfail}/{n}"
        )

    print()
    print(
        f"[FINISHED] {output}"
    )

    print(
        "Interpret survival first. "
        "RMSE is conditional on completing the episode."
    )


if __name__ == "__main__":
    main()
