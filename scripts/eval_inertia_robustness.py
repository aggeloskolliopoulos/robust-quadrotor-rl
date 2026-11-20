
"""Zero-shot inertia robustness: nominal PID vs two frozen SAC policies.

Only PyBullet's base-link inertia diagonal changes. Mass, actuator mapping,
references, PID gains, reward and SAC networks remain nominal.

Requires scripts/eval_mass_robustness.py from the prior experiment.
"""

from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime

import numpy as np
import pybullet as p
from stable_baselines3 import SAC

import eval_mass_robustness as mass


LEVELS = (-20.0, -10.0, 0.0, 10.0, 20.0)


# ============================================================
# 1. Environment with inertia uncertainty
# ============================================================

class InertiaPerturbedCircleEnv(mass.PhaseCircleTrackingEnv):
    """Modify physical inertia after every PyBullet reset, not sim.J/PID."""

    def __init__(
        self,
        *,
        inertia_percent: float = 0.0,
        **kwargs,
    ):
        self.inertia_percent = float(inertia_percent)

        if (
            not np.isfinite(self.inertia_percent)
            or self.inertia_percent <= -100
        ):
            raise ValueError(
                "Inertia perturbation must be finite and > -100%"
            )

        super().__init__(**kwargs)

        self.nominal_mass = float(self.sim.M)

        self.nominal_model_inertia = np.asarray(
            self.sim.J, dtype=np.float64
        ).copy()

        self.nominal_hover_rpm = float(self.sim.HOVER_RPM)
        self.nominal_max_rpm = float(self.sim.MAX_RPM)

        self.actual_base_inertia = None

    def reset(self, *, seed=None, options=None):

        # The parent reset reloads the drone URDF.
        obs, info = super().reset(
            seed=seed,
            options=options,
        )

        body = int(self.sim.DRONE_IDS[0])
        client = int(self.sim.CLIENT)

        before = p.getDynamicsInfo(
            body, -1, physicsClientId=client
        )

        initial_mass = float(before[0])

        initial_inertia = np.asarray(
            before[2], dtype=np.float64
        )

        if not np.isclose(
            initial_mass,
            self.nominal_mass,
            rtol=1e-6,
            atol=1e-9,
        ):
            raise RuntimeError(
                "Reset did not restore nominal mass"
            )

        if (
            not np.all(np.isfinite(initial_inertia))
            or np.any(initial_inertia <= 0)
        ):
            raise RuntimeError(
                "Invalid initial PyBullet base inertia"
            )

        # Scale all three physical inertia components.
        target = initial_inertia * (
            1.0 + self.inertia_percent / 100.0
        )

        # At 0%, leave the original physics unchanged.
        if self.inertia_percent != 0.0:

            p.changeDynamics(
                body,
                -1,
                localInertiaDiagonal=target.tolist(),
                physicsClientId=client,
            )

        # Verify what PyBullet actually uses.
        after = p.getDynamicsInfo(
            body, -1, physicsClientId=client
        )

        actual_mass = float(after[0])

        actual_inertia = np.asarray(
            after[2], dtype=np.float64
        )

        np.testing.assert_allclose(
            actual_inertia,
            target,
            rtol=1e-5,
            atol=1e-12,
        )

        # Mass must remain nominal.
        if not np.isclose(
            actual_mass,
            self.nominal_mass,
            rtol=1e-6,
            atol=1e-9,
        ):
            raise RuntimeError(
                "Physical mass changed while varying inertia"
            )

        # The simulator's nominal model parameters are
        # deliberately not modified.
        np.testing.assert_array_equal(
            self.sim.J,
            self.nominal_model_inertia,
        )

        if not np.isclose(
            float(self.sim.HOVER_RPM),
            self.nominal_hover_rpm,
        ):
            raise RuntimeError(
                "Nominal HOVER_RPM was changed"
            )

        if not np.isclose(
            float(self.sim.MAX_RPM),
            self.nominal_max_rpm,
        ):
            raise RuntimeError(
                "Nominal MAX_RPM was changed"
            )

        if not np.isclose(
            float(self.hover_rpm),
            self.nominal_hover_rpm,
        ):
            raise RuntimeError(
                "RL action-to-RPM mapping was changed"
            )

        self.actual_base_inertia = (
            actual_inertia.copy()
        )

        info = dict(info)

        info.update({
            "physical_mass_kg": actual_mass,
            "inertia_percent": self.inertia_percent,
            "base_inertia_kg_m2": actual_inertia.copy(),
            "nominal_base_inertia_kg_m2":
                initial_inertia.copy(),
        })

        return obs, info


# ============================================================
# 2. Environment constructor
# ============================================================

def make_env(
    percent: float,
    radius: float,
    phase: float,
    omega: float,
):

    return InertiaPerturbedCircleEnv(
        gui=False,
        physics_freq_hz=240,
        control_freq_hz=48,
        inertia_percent=percent,
        radius_m=radius,
        initial_phase_rad=phase,
        angular_speed_rad_s=omega,
    )


# ============================================================
# 3. Physics preflight
# ============================================================

def preflight():
    """Verify nominal equality, reset persistence and dynamics."""

    # 0% inertia must reproduce the existing environment.
    original = mass.PhaseCircleTrackingEnv(
        radius_m=0.30,
        initial_phase_rad=math.pi / 2,
        angular_speed_rad_s=0.60,
    )

    zero = make_env(
        0.0, 0.30, math.pi / 2, 0.60
    )

    try:

        a, _ = original.reset(seed=19000)
        b, _ = zero.reset(seed=19000)

        np.testing.assert_allclose(
            a, b, rtol=0, atol=1e-7
        )

        action = np.array(
            [0.10, -0.05, 0.03, -0.02],
            dtype=np.float32,
        )

        for _ in range(5):

            a, *_ = original.step(action)
            b, *_ = zero.step(action)

            np.testing.assert_allclose(
                a, b, rtol=0, atol=1e-6
            )

        print(
            "[PASS] Inertia 0% reproduces original environment"
        )

    finally:
        original.close()
        zero.close()

    # Verify perturbations after repeated resets.
    for percent in (-20.0, 20.0):

        env = make_env(
            percent, 0.30, 0.0, 0.60
        )

        try:

            for seed in (19000, 19001):

                _, info = env.reset(seed=seed)

                ratio = (
                    info["base_inertia_kg_m2"]
                    / info["nominal_base_inertia_kg_m2"]
                )

                np.testing.assert_allclose(
                    ratio,
                    np.ones(3) * (1 + percent / 100),
                    rtol=1e-5,
                    atol=1e-6,
                )

                print(
                    f"[PASS] reset seed={seed}, "
                    f"J={percent:+g}%: "
                    f"diagonal={info['base_inertia_kg_m2']}, "
                    f"mass={info['physical_mass_kg'] * 1000:.3f} g"
                )

        finally:
            env.close()

    # Crucial verification: changing the inertia must
    # actually change the rotational physical response.
    nominal = make_env(
        0.0, 0.30, 0.0, 0.60
    )

    perturbed = make_env(
        20.0, 0.30, 0.0, 0.60
    )

    try:

        nominal_obs, _ = nominal.reset(seed=19002)

        perturbed_obs, _ = perturbed.reset(
            seed=19002
        )

        np.testing.assert_allclose(
            nominal_obs,
            perturbed_obs,
            rtol=0,
            atol=1e-7,
        )

        action = np.array(
            [0.10, -0.05, 0.03, -0.02],
            dtype=np.float32,
        )

        for _ in range(8):

            nominal_obs, *_ = nominal.step(action)

            perturbed_obs, *_ = perturbed.step(action)

        # Last three observation components are angular velocity.
        delta = float(
            np.linalg.norm(
                nominal_obs[-3:]
                - perturbed_obs[-3:]
            )
        )

        if (
            not np.isfinite(delta)
            or delta < 1e-6
        ):
            raise RuntimeError(
                "Inertia perturbation had no measurable "
                f"rotational effect (delta={delta})"
            )

        print(
            "[PASS] Physical angular response changes: "
            f"delta={delta:.6g} rad/s"
        )

    finally:
        nominal.close()
        perturbed.close()

    print(
        "[PASS] Inertia preflight complete"
    )


# ============================================================
# 4. Reporting
# ============================================================

def report_group(
    rows: list[dict],
    percent: float,
    name: str,
):

    survived, count, means = mass.summary(rows)

    print(
        f"[DONE] J={percent:+g}% {name:<13} "
        f"survival={survived}/{count} "
        f"full={means['full_xyz_m']:.3f} cm "
        f"late={means['late_xyz_m']:.3f} cm "
        f"lateXY={means['late_xy_m']:.3f} cm "
        f"lateZ={means['late_z_m']:.3f} cm",
        flush=True,
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

    parser.add_argument(
        "--quick",
        action="store_true",
    )

    parser.add_argument(
        "--seed-start",
        type=int,
        default=19000,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--inertia-levels",
        type=float,
        nargs="+",
        default=list(LEVELS),
    )

    args = parser.parse_args()

    if args.seeds < 1:
        parser.error("--seeds must be positive")

    if any(
        not np.isfinite(x) or x <= -100
        for x in args.inertia_levels
    ):
        parser.error(
            "All inertia levels must be finite and > -100%"
        )

    if len(set(args.inertia_levels)) != len(args.inertia_levels):
        parser.error(
            "Inertia levels must be unique"
        )

    # Reuse model paths from the successful mass benchmark.
    for path in (mass.BASELINE, mass.FINAL20):

        if not path.is_file():
            raise FileNotFoundError(path)

    preflight()

    if args.check_only:
        return

    policies = {
        "PID": None,
        "Baseline_30k": SAC.load(
            str(mass.BASELINE), device="cpu"
        ),
        "Final_20k": SAC.load(
            str(mass.FINAL20), device="cpu"
        ),
    }

    radii = (0.30,) if args.quick else mass.RADII
    phases = (0.0,) if args.quick else mass.PHASES
    omegas = (0.6,) if args.quick else mass.OMEGAS

    seeds = range(
        args.seed_start,
        args.seed_start + (
            2 if args.quick else args.seeds
        ),
    )

    n_per_cell = (
        len(radii)
        * len(phases)
        * len(omegas)
        * len(seeds)
    )

    output_dir = (
        mass.ROOT
        / "results"
        / "evaluation"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = output_dir / (
        "inertia_robustness_"
        + datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )
        + ".csv"
    )

    fields = [
        "controller",
        "inertia_percent",
        "physical_mass_kg",
        "actual_ixx",
        "actual_iyy",
        "actual_izz",
        "radius_m",
        "phase_rad",
        "omega_rad_s",
        "seed",
        "survived",
        "duration_s",
        *mass.METRICS,
    ]

    data = {}

    total = (
        n_per_cell
        * len(args.inertia_levels)
        * len(policies)
    )

    print("\n" + "=" * 65)
    print("ZERO-SHOT INERTIA ROBUSTNESS")
    print("PID vs TWO FROZEN SAC CONTROLLERS")
    print("=" * 65)

    print(
        f"J levels: {args.inertia_levels}"
    )

    print(
        f"Episodes per controller/J: {n_per_cell}"
    )

    print(
        f"Total episodes: {total}"
    )

    print(
        f"Output: {output_file}",
        flush=True,
    )

    # --------------------------------------------------------
    # Evaluate each inertia level and controller
    # --------------------------------------------------------

    with output_file.open(
        "x",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f, fieldnames=fields
        )

        writer.writeheader()

        for percent in args.inertia_levels:

            for name, policy in policies.items():

                group = []

                print(
                    f"\n[RUN] J={percent:+g}% "
                    f"controller={name}",
                    flush=True,
                )

                for radius in radii:

                    for phase in phases:

                        for omega in omegas:

                            env = make_env(
                                percent,
                                radius,
                                phase,
                                omega,
                            )

                            try:

                                for seed in seeds:

                                    # Identical evaluator as mass benchmark.
                                    result = mass.run_episode(
                                        env,
                                        name,
                                        policy,
                                        seed,
                                    )

                                    if env.actual_base_inertia is None:
                                        raise RuntimeError(
                                            "Missing inertia after reset"
                                        )

                                    j = env.actual_base_inertia

                                    row = {
                                        "controller": name,
                                        "inertia_percent": percent,
                                        "radius_m": radius,
                                        "phase_rad": phase,
                                        "omega_rad_s": omega,
                                        "seed": seed,
                                        "actual_ixx": float(j[0]),
                                        "actual_iyy": float(j[1]),
                                        "actual_izz": float(j[2]),
                                        **result,
                                    }

                                    group.append(row)
                                    writer.writerow(row)

                            finally:
                                env.close()

                            # Partial results survive interruptions.
                            f.flush()

                data[(percent, name)] = group

                report_group(
                    group,
                    percent,
                    name,
                )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("SUMMARY - RMSE IN CM, SUCCESSFUL EPISODES ONLY")
    print("=" * 65)

    print(
        f"{'Inertia':>8} "
        f"{'Controller':<14} "
        f"{'Survival':>10} "
        f"{'Full':>8} "
        f"{'Late':>8} "
        f"{'LateXY':>8} "
        f"{'LateZ':>8}"
    )

    for percent in args.inertia_levels:

        for name in policies:

            ok, n, metrics = mass.summary(
                data[(percent, name)]
            )

            print(
                f"{percent:+7g}% "
                f"{name:<14} "
                f"{ok:4}/{n:<5} "
                f"{metrics['full_xyz_m']:8.3f} "
                f"{metrics['late_xyz_m']:8.3f} "
                f"{metrics['late_xy_m']:8.3f} "
                f"{metrics['late_z_m']:8.3f}"
            )

    # --------------------------------------------------------
    # Matched controller comparisons
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("PAIRED CONTROLLER COMPARISONS AT THE SAME INERTIA")
    print("=" * 65)

    for percent in args.inertia_levels:

        print(f"\nJ={percent:+g}%")

        mass.matched_report(
            data[(percent, "Baseline_30k")],
            data[(percent, "PID")],
            "Baseline vs PID",
        )

        mass.matched_report(
            data[(percent, "Final_20k")],
            data[(percent, "PID")],
            "Final20 vs PID",
        )

        mass.matched_report(
            data[(percent, "Final_20k")],
            data[(percent, "Baseline_30k")],
            "Final20 vs Baseline",
        )

    # --------------------------------------------------------
    # Degradation relative to nominal inertia
    # --------------------------------------------------------

    if 0.0 in args.inertia_levels:

        print("\n" + "=" * 65)
        print("PAIRED INERTIA DEGRADATION VS NOMINAL")
        print("=" * 65)

        for name in policies:

            for percent in args.inertia_levels:

                if percent != 0:

                    mass.matched_report(
                        data[(percent, name)],
                        data[(0.0, name)],
                        f"{name}: {percent:+g}% vs 0%",
                    )

    # --------------------------------------------------------
    # Identify difficult scenarios
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("WORST FAILURE GROUPS")
    print("=" * 65)

    groups = {}

    for rows in data.values():

        for row in rows:

            key = (
                row["inertia_percent"],
                row["controller"],
                row["radius_m"],
                row["phase_rad"],
                row["omega_rad_s"],
            )

            groups.setdefault(key, []).append(row)

    worst = []

    for key, rows in groups.items():

        fails = [
            r for r in rows
            if not r["survived"]
        ]

        if fails:

            worst.append((
                len(fails),
                key,
                len(rows),
                np.mean([
                    r["duration_s"]
                    for r in fails
                ]),
            ))

    for count, key, total_group, t_fail in sorted(
        worst,
        key=lambda x: x[0],
        reverse=True,
    )[:12]:

        percent, name, radius, phase, omega = key

        print(
            f"J={percent:+g}% "
            f"{name:<13} "
            f"R={radius:.2f} "
            f"phi={phase:.3f} "
            f"omega={omega:.1f}: "
            f"failures={count}/{total_group}, "
            f"mean_fail_t={t_fail:.3f}s"
        )

    print(
        f"\n[FINISHED] {output_file}"
    )

    print(
        "Survival first; RMSE conditional on "
        "completing the episode."
    )


if __name__ == "__main__":
    main()
