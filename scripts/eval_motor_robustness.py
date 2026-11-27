
"""Zero-shot motor/propeller effectiveness robustness benchmark.

Compare nominal PID, SAC Baseline 30k and SAC Final 20k on the same
circular trajectories and initial seeds. No changes to model files.

Uniform losses and one degraded motor are evaluated separately.
Requires scripts/eval_mass_robustness.py and eval_radius_phase.py.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from types import MethodType

import numpy as np
import pybullet as p
from stable_baselines3 import SAC
from gym_pybullet_drones.utils.enums import DroneModel, Physics

import eval_mass_robustness as mass


# Motor indices correspond to the rotor links 0,1,2,3 in BaseAviary.
SCENARIOS = {
    "nominal": (1.00, 1.00, 1.00, 1.00),

    "uniform_95": (0.95, 0.95, 0.95, 0.95),
    "uniform_90": (0.90, 0.90, 0.90, 0.90),

    "motor0_95": (0.95, 1.00, 1.00, 1.00),
    "motor0_90": (0.90, 1.00, 1.00, 1.00),
    "motor0_85": (0.85, 1.00, 1.00, 1.00),
    "motor0_80": (0.80, 1.00, 1.00, 1.00),

    "motor1_85": (1.00, 0.85, 1.00, 1.00),
    "motor2_85": (1.00, 1.00, 0.85, 1.00),
    "motor3_85": (1.00, 1.00, 1.00, 0.85),
}


# ============================================================
# 1. Replacement ONLY for the low-level PYB motor forces
# ============================================================

def motor_effectiveness_physics(sim, rpm, nth_drone):
    """Physics.PYB motor forces with actual thrust/torque multipliers.

    The nominal _physics implementation is called unchanged
    when eta=1.

    No change to commanded RPM, nominal KF/KM, PID gains,
    or RL action mapping.
    """

    original_rpm = rpm
    rpm = np.asarray(rpm, dtype=np.float64).reshape(4)
    eta = sim._motor_eta

    # Record the commands and physical forces.
    forces = (rpm ** 2) * sim.KF * eta
    reaction_torques = (rpm ** 2) * sim.KM * eta

    sim._motor_last_rpm = rpm.copy()
    sim._motor_last_forces_n = forces.copy()
    sim._motor_last_reaction_torques_nm = (
        reaction_torques.copy()
    )
    sim._motor_physics_calls += 1

    if np.all(eta == 1.0):

        # Identical nominal physics implementation.
        return sim._motor_original_physics(
            original_rpm, nth_drone
        )

    if sim.DRONE_MODEL != DroneModel.CF2X:
        raise RuntimeError(
            "This benchmark expects DroneModel.CF2X"
        )

    # Match the yaw-torque signs in BaseAviary._physics.
    yaw_torque = (
        -reaction_torques[0]
        + reaction_torques[1]
        - reaction_torques[2]
        + reaction_torques[3]
    )

    # The four propeller forces act on links 0..3.
    for motor in range(4):

        p.applyExternalForce(
            sim.DRONE_IDS[nth_drone],
            motor,
            forceObj=[
                0.0,
                0.0,
                float(forces[motor]),
            ],
            posObj=[0.0, 0.0, 0.0],
            flags=p.LINK_FRAME,
            physicsClientId=sim.CLIENT,
        )

    # Apply reaction yaw torque exactly as BaseAviary does.
    p.applyExternalTorque(
        sim.DRONE_IDS[nth_drone],
        4,
        torqueObj=[0.0, 0.0, float(yaw_torque)],
        flags=p.LINK_FRAME,
        physicsClientId=sim.CLIENT,
    )


# ============================================================
# 2. Same circle environment, modified physical motors
# ============================================================

class MotorEffectivenessCircleEnv(
    mass.PhaseCircleTrackingEnv
):
    def __init__(self, *, effectiveness, **kwargs):

        eta = np.asarray(
            effectiveness, dtype=np.float64
        )

        if (
            eta.shape != (4,)
            or not np.all(np.isfinite(eta))
            or np.any(eta <= 0.0)
            or np.any(eta > 1.0)
        ):
            raise ValueError(
                "effectiveness must contain four "
                "finite numbers in (0,1]"
            )

        self.effectiveness = eta.copy()

        super().__init__(**kwargs)

        sim = self.sim

        if (
            sim.PHYSICS != Physics.PYB
            or sim.NUM_DRONES != 1
        ):
            raise RuntimeError(
                "Expected one drone using Physics.PYB"
            )

        if sim.DRONE_MODEL != DroneModel.CF2X:
            raise RuntimeError(
                "Expected DroneModel.CF2X"
            )

        # Save the nominal physical and control constants.
        self._nominal_parameters = (
            float(sim.M),
            float(sim.KF),
            float(sim.KM),
            float(sim.HOVER_RPM),
            float(sim.MAX_RPM),
            float(self.hover_rpm),
            float(self.max_rpm),
        )

        # Override the physics method only for this instance.
        # Do not modify the installed gym-pybullet-drones code.
        sim._motor_original_physics = sim._physics
        sim._motor_eta = self.effectiveness.copy()
        sim._motor_physics_calls = 0
        sim._motor_last_rpm = None
        sim._motor_last_forces_n = None
        sim._motor_last_reaction_torques_nm = None

        sim._physics = MethodType(
            motor_effectiveness_physics,
            sim,
        )

    def reset(self, *, seed=None, options=None):

        observation, info = super().reset(
            seed=seed,
            options=options,
        )

        sim = self.sim

        # Ensure no nominal model or controller constants
        # have been modified by the physics perturbation.
        params = (
            float(sim.M),
            float(sim.KF),
            float(sim.KM),
            float(sim.HOVER_RPM),
            float(sim.MAX_RPM),
            float(self.hover_rpm),
            float(self.max_rpm),
        )

        np.testing.assert_allclose(
            params,
            self._nominal_parameters,
            rtol=0,
            atol=0,
        )

        np.testing.assert_array_equal(
            sim._motor_eta,
            self.effectiveness,
        )

        # Physical drone mass must stay nominal.
        body = int(sim.DRONE_IDS[0])

        actual_mass = float(
            p.getDynamicsInfo(
                body,
                -1,
                physicsClientId=sim.CLIENT,
            )[0]
        )

        if not np.isclose(
            actual_mass,
            sim.M,
            rtol=1e-6,
            atol=1e-9,
        ):
            raise RuntimeError(
                "Unexpected physical mass change"
            )

        info = dict(info)

        # mass.run_episode() uses this field.
        info["physical_mass_kg"] = actual_mass
        info["motor_effectiveness"] = (
            self.effectiveness.copy()
        )

        return observation, info


def make_env(
    effectiveness,
    radius=0.30,
    phase=0.0,
    omega=0.60,
):

    return MotorEffectivenessCircleEnv(
        effectiveness=effectiveness,
        gui=False,
        physics_freq_hz=240,
        control_freq_hz=48,
        radius_m=radius,
        initial_phase_rad=phase,
        angular_speed_rad_s=omega,
    )


# ============================================================
# 3. Physics preflight
# ============================================================

def preflight():

    # First verify that 100% effectiveness is equivalent
    # to the original unmodified simulation.
    original = mass.PhaseCircleTrackingEnv(
        radius_m=0.30,
        initial_phase_rad=0.0,
        angular_speed_rad_s=0.60,
    )

    nominal = make_env(SCENARIOS["nominal"])

    try:

        obs_a, _ = original.reset(seed=22000)
        obs_b, _ = nominal.reset(seed=22000)

        np.testing.assert_allclose(
            obs_a, obs_b, rtol=0, atol=1e-7
        )

        action = np.array(
            [0.10, -0.05, 0.03, -0.02],
            dtype=np.float32,
        )

        for _ in range(8):

            obs_a, *_ = original.step(action)
            obs_b, *_ = nominal.step(action)

            np.testing.assert_allclose(
                obs_a, obs_b, rtol=0, atol=1e-6
            )

        if nominal.sim._motor_physics_calls != 8 * 5:
            raise RuntimeError(
                "Unexpected number of physics substeps"
            )

        print(
            "[PASS] eta=100% reproduces original environment"
        )

    finally:
        original.close()
        nominal.close()

    # Compare actual physical responses at identical RPM.
    nominal = make_env(SCENARIOS["nominal"])
    uniform = make_env(SCENARIOS["uniform_90"])
    single = make_env(SCENARIOS["motor0_90"])

    try:

        observations = []

        for env in (nominal, uniform, single):

            obs, _ = env.reset(seed=22001)
            observations.append(obs)

        for obs in observations[1:]:

            np.testing.assert_allclose(
                obs,
                observations[0],
                rtol=0,
                atol=1e-7,
            )

        # Action=0 means nominal hover RPM.
        hover_action = np.zeros(
            4, dtype=np.float32
        )

        for env in (nominal, uniform, single):

            env.step(hover_action)

            if env.sim._motor_physics_calls != 5:
                raise RuntimeError(
                    "Expected five 240 Hz physics substeps"
                )

            rpm = env.sim._motor_last_rpm
            actual = env.sim._motor_last_forces_n

            expected = (
                (rpm ** 2)
                * env.sim.KF
                * env.effectiveness
            )

            np.testing.assert_allclose(
                actual,
                expected,
                rtol=1e-12,
                atol=0,
            )

        # Uniform 90% should reduce vertical acceleration.
        vz_nom = float(
            nominal.sim._getDroneStateVector(0)[12]
        )

        vz_uniform = float(
            uniform.sim._getDroneStateVector(0)[12]
        )

        if not vz_uniform < vz_nom - 1e-4:
            raise RuntimeError(
                "Uniform thrust loss had no vertical response"
            )

        # Losing 10% thrust at only motor 0 must change
        # rotational motion.
        omega_nom = (
            nominal.sim._getDroneStateVector(0)[13:16]
        )

        omega_single = (
            single.sim._getDroneStateVector(0)[13:16]
        )

        angular_delta = float(
            np.linalg.norm(omega_single - omega_nom)
        )

        if not angular_delta > 1e-4:
            raise RuntimeError(
                "Single-motor loss had no angular response"
            )

        print(
            "[PASS] Uniform 90% changes vertical velocity: "
            f"delta_vz={vz_uniform-vz_nom:+.6f} m/s"
        )

        print(
            "[PASS] Motor 0 at 90% changes angular velocity: "
            f"delta_omega={angular_delta:.6f} rad/s"
        )

        print(
            "[PASS] Physical motor forces match "
            "eta * KF * RPM**2"
        )

        print(
            "[PASS] Nominal RPM mapping, mass, KF and KM "
            "remain unchanged"
        )

    finally:
        nominal.close()
        uniform.close()
        single.close()

    print(
        "[PASS] Motor effectiveness preflight complete"
    )


# ============================================================
# 4. Main benchmark
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
        default=22000,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=tuple(SCENARIOS),
        default=list(SCENARIOS),
    )

    args = parser.parse_args()

    if args.seeds < 1:
        parser.error("--seeds must be positive")

    if len(set(args.scenarios)) != len(args.scenarios):
        parser.error("--scenarios must be unique")

    # First verify physical implementation.
    preflight()

    if args.check_only:
        return

    # Frozen policies from previous experiments.
    for model_path in (mass.BASELINE, mass.FINAL20):
        if not model_path.is_file():
            raise FileNotFoundError(model_path)

    policies = {
        "PID": None,

        "Baseline_30k": SAC.load(
            str(mass.BASELINE),
            device="cpu",
        ),

        "Final_20k": SAC.load(
            str(mass.FINAL20),
            device="cpu",
        ),
    }

    radii = (0.30,) if args.quick else mass.RADII
    phases = (0.0,) if args.quick else mass.PHASES
    omegas = (0.60,) if args.quick else mass.OMEGAS

    seeds = range(
        args.seed_start,
        args.seed_start
        + (2 if args.quick else args.seeds),
    )

    trajectories = (
        len(radii)
        * len(phases)
        * len(omegas)
    )

    count_per_cell = trajectories * len(seeds)

    total = (
        count_per_cell
        * len(policies)
        * len(args.scenarios)
    )

    output_dir = (
        mass.ROOT / "results" / "evaluation"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = output_dir / (
        "motor_robustness_"
        + datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )
        + ".csv"
    )

    fields = [
        "controller",
        "scenario",
        "eta0",
        "eta1",
        "eta2",
        "eta3",
        "physical_mass_kg",
        "radius_m",
        "phase_rad",
        "omega_rad_s",
        "seed",
        "survived",
        "duration_s",
        *mass.METRICS,
    ]

    data = {}

    print("\n" + "=" * 72)
    print("ZERO-SHOT MOTOR / PROPELLER EFFECTIVENESS")
    print("=" * 72)

    print(f"Scenarios: {args.scenarios}")
    print(f"Trajectories: {trajectories}")

    print(
        "Episodes per controller/scenario: "
        f"{count_per_cell}"
    )

    print(f"Total episodes: {total}")
    print(f"Output CSV: {output_file}", flush=True)

    # --------------------------------------------------------
    # Run all scenarios and controllers
    # --------------------------------------------------------

    with output_file.open(
        "x",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for scenario in args.scenarios:

            eta = SCENARIOS[scenario]

            for name, model in policies.items():

                group = []

                print(
                    f"\n[RUN] scenario={scenario}, "
                    f"controller={name}",
                    flush=True,
                )

                for radius in radii:

                    for phase in phases:

                        for omega in omegas:

                            env = make_env(
                                eta,
                                radius,
                                phase,
                                omega,
                            )

                            try:

                                for seed in seeds:

                                    outcome = mass.run_episode(
                                        env,
                                        name,
                                        model,
                                        seed,
                                    )

                                    row = {
                                        "controller": name,
                                        "scenario": scenario,
                                        "eta0": eta[0],
                                        "eta1": eta[1],
                                        "eta2": eta[2],
                                        "eta3": eta[3],
                                        "radius_m": radius,
                                        "phase_rad": phase,
                                        "omega_rad_s": omega,
                                        "seed": int(seed),
                                        **outcome,
                                    }

                                    writer.writerow(row)
                                    group.append(row)

                            finally:
                                env.close()

                            # Preserve partial CSV results.
                            f.flush()

                data[(scenario, name)] = group

                successes, count, metrics = (
                    mass.summary(group)
                )

                print(
                    f"[DONE] {name:<13} "
                    f"{scenario:<12} "
                    f"survival={successes}/{count} "
                    f"full={metrics['full_xyz_m']:.3f}cm "
                    f"late={metrics['late_xyz_m']:.3f}cm "
                    f"XY={metrics['late_xy_m']:.3f}cm "
                    f"Z={metrics['late_z_m']:.3f}cm",
                    flush=True,
                )

    # ========================================================
    # Overall summary
    # ========================================================

    print("\n" + "=" * 72)
    print("SUMMARY - RMSE CM (SUCCESSFUL EPISODES ONLY)")
    print("=" * 72)

    print(
        f"{'Scenario':<14} "
        f"{'Controller':<14} "
        f"{'Survival':>10} "
        f"{'Full':>8} "
        f"{'Late':>8} "
        f"{'LateXY':>8} "
        f"{'LateZ':>8}"
    )

    for scenario in args.scenarios:

        for name in policies:

            successes, count, metrics = mass.summary(
                data[(scenario, name)]
            )

            print(
                f"{scenario:<14} "
                f"{name:<14} "
                f"{successes:4}/{count:<5} "
                f"{metrics['full_xyz_m']:8.3f} "
                f"{metrics['late_xyz_m']:8.3f} "
                f"{metrics['late_xy_m']:8.3f} "
                f"{metrics['late_z_m']:8.3f}"
            )

    # ========================================================
    # Matched PID / SAC comparisons
    # ========================================================

    print("\n" + "=" * 72)
    print(
        "PAIRED CONTROLLER COMPARISONS "
        "AT SAME EFFECTIVENESS"
    )
    print("=" * 72)

    for scenario in args.scenarios:

        print(f"\nScenario: {scenario}")

        mass.matched_report(
            data[(scenario, "Baseline_30k")],
            data[(scenario, "PID")],
            "Baseline vs PID",
        )

        mass.matched_report(
            data[(scenario, "Final_20k")],
            data[(scenario, "PID")],
            "Final20 vs PID",
        )

        mass.matched_report(
            data[(scenario, "Final_20k")],
            data[(scenario, "Baseline_30k")],
            "Final20 vs Baseline",
        )

    # ========================================================
    # Performance degradation versus nominal
    # ========================================================

    if "nominal" in args.scenarios:

        print("\n" + "=" * 72)
        print(
            "PAIRED EFFECTIVENESS DEGRADATION VS NOMINAL"
        )
        print("=" * 72)

        for name in policies:

            for scenario in args.scenarios:

                if scenario == "nominal":
                    continue

                mass.matched_report(
                    data[(scenario, name)],
                    data[("nominal", name)],
                    f"{name}: {scenario} vs nominal",
                )

    # ========================================================
    # Worst failure groups
    # ========================================================

    print("\n" + "=" * 72)
    print("WORST FAILURE GROUPS")
    print("=" * 72)

    groups = {}

    for rows in data.values():

        for row in rows:

            key = (
                row["scenario"],
                row["controller"],
                row["radius_m"],
                row["phase_rad"],
                row["omega_rad_s"],
            )

            groups.setdefault(
                key, []
            ).append(row)

    worst = []

    for key, rows in groups.items():

        failed = [
            row
            for row in rows
            if not row["survived"]
        ]

        if failed:

            mean_time = float(
                np.mean([
                    row["duration_s"]
                    for row in failed
                ])
            )

            worst.append((
                len(failed),
                key,
                len(rows),
                mean_time,
            ))

    for failures, key, n, mean_time in sorted(
        worst,
        key=lambda x: x[0],
        reverse=True,
    )[:12]:

        scenario, name, radius, phase, omega = key

        print(
            f"{scenario:<13} "
            f"{name:<13} "
            f"R={radius:.2f} "
            f"phi={phase:.3f} "
            f"omega={omega:.1f}: "
            f"failures={failures}/{n}, "
            f"mean_fail_t={mean_time:.3f}s"
        )

    print(f"\n[FINISHED] {output_file}")

    print(
        "Interpret survival first; "
        "RMSE only for completed episodes."
    )

    print(
        "Motor index 0 is the first rotor only, "
        "not an all-rotors test."
    )


if __name__ == "__main__":
    main()
