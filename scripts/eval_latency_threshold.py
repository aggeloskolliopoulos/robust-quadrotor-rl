
"""Physics-substep latency threshold: PID vs frozen SAC Baseline 30k.

Command latency delays ACTUAL motor RPM at physics frequency.
Observation latency uses physical states sampled at 240 Hz.

Uses the existing eval_sensor_latency_robustness.py.

Run:
    python -u scripts/eval_latency_threshold.py --check-only
    python -u scripts/eval_latency_threshold.py --quick
    python -u scripts/eval_latency_threshold.py --kind command --seeds 3
    python -u scripts/eval_latency_threshold.py --kind observation --seeds 3
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, deque
from datetime import datetime
from types import MethodType

import numpy as np
import pybullet as p
from stable_baselines3 import SAC

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics

import eval_sensor_latency_robustness as sensor


LEVELS = (0, 1, 2, 3, 4, 5)
CONTROLLERS = ("PID", "Baseline_30k")


# ============================================================
# 1. Physics-level monitoring and command delay
# ============================================================

def physics_with_latency(sim, rpm, nth_drone):
    """Called BEFORE each PyBullet physics step."""

    if nth_drone != 0:
        raise RuntimeError("Expected one drone")

    body = int(sim.DRONE_IDS[0])

    # Get REAL physical state at the beginning
    # of the current 1/240-second substep.
    pos, quat = p.getBasePositionAndOrientation(
        body,
        physicsClientId=sim.CLIENT,
    )

    vel, angular_vel = p.getBaseVelocity(
        body,
        physicsClientId=sim.CLIENT,
    )

    state = np.zeros(20, dtype=np.float64)

    state[0:3] = pos
    state[3:7] = quat
    state[7:10] = p.getEulerFromQuaternion(quat)
    state[10:13] = vel
    state[13:16] = angular_vel

    sim._latency_states.append(state)

    # Current requested motor command.
    commanded = np.asarray(
        rpm, dtype=np.float64
    ).copy()

    # Delay the physical RPM command, if requested.
    if sim._latency_command_steps:

        sim._latency_pending.append(commanded)
        applied = sim._latency_pending.popleft()

    else:
        applied = commanded

    sim._latency_last_commanded_rpm = commanded
    sim._latency_last_applied_rpm = applied.copy()

    sim._latency_applied_rpm_log.append(
        applied.copy()
    )

    # Apply delayed or current RPM to original physics.
    sim._latency_original_physics(
        applied,
        nth_drone,
    )


# ============================================================
# 2. Environment
# ============================================================

def make_env(
    kind,
    delay_substeps,
    radius=0.30,
    phase=0.0,
    omega=0.60,
):

    if kind not in ("command", "observation"):
        raise ValueError(kind)

    if delay_substeps not in LEVELS:
        raise ValueError(delay_substeps)

    env = sensor.make_env(
        radius, phase, omega
    )

    sim = env.sim

    if (
        sim.PHYSICS != Physics.PYB
        or sim.NUM_DRONES != 1
        or sim.PYB_FREQ != 240
        or sim.CTRL_FREQ != 48
    ):
        env.close()
        raise RuntimeError(
            "Expected single-drone PYB at 240/48 Hz"
        )

    # Modify physics of this instance only.
    sim._latency_original_physics = sim._physics

    sim._physics = MethodType(
        physics_with_latency,
        sim,
    )

    sim._latency_command_steps = (
        delay_substeps
        if kind == "command"
        else 0
    )

    sim._latency_states = []
    sim._latency_pending = deque()
    sim._latency_applied_rpm_log = []

    sim._latency_last_commanded_rpm = None
    sim._latency_last_applied_rpm = None

    return env


def reset_pipeline(env, seed):

    obs, info = env.reset(seed=seed)
    sim = env.sim

    sim._latency_states.clear()
    sim._latency_applied_rpm_log.clear()

    sim._latency_last_commanded_rpm = None
    sim._latency_last_applied_rpm = None

    # Start with hover RPM for commands in transit.
    sim._latency_pending = deque([
        np.full(
            4,
            sim.HOVER_RPM,
            dtype=np.float64,
        )
        for _ in range(
            sim._latency_command_steps
        )
    ])

    return obs, info


# ============================================================
# 3. Observation delayed by physics substeps
# ============================================================

def measured_state(
    env,
    kind,
    delay_substeps,
):

    true_state = np.asarray(
        env.sim._getDroneStateVector(0),
        dtype=np.float64,
    )

    if (
        kind == "observation"
        and delay_substeps > 0
    ):

        history = env.sim._latency_states

        if not history:
            # Initial t=0: no past samples.
            return true_state.copy(), true_state

        # At time t, select the physical state at:
        # t - delay_substeps / 240.
        index = max(
            0,
            len(history) - delay_substeps,
        )

        measurement = history[index].copy()

    else:
        measurement = true_state.copy()

    return measurement, true_state


# ============================================================
# 4. Failure diagnostics
# ============================================================

def failure_reason(env):

    state = np.asarray(
        env.sim._getDroneStateVector(0),
        dtype=np.float64,
    )

    if not np.all(np.isfinite(state)):
        return "numerical"

    if state[2] < 0.10:
        return "low_altitude"

    if np.linalg.norm(
        env.target_pos - state[:3]
    ) > 2.0:
        return "position_bound"

    if np.any(
        np.abs(state[7:9]) > np.deg2rad(80)
    ):
        return "tilt_bound"

    return "other"


# ============================================================
# 5. Run one episode
# ============================================================

def run_episode(
    env,
    kind,
    substeps,
    controller,
    policy,
    seed,
):

    obs, _ = reset_pipeline(env, seed)

    pid = (
        DSLPIDControl(
            drone_model=DroneModel.CF2X
        )
        if controller == "PID"
        else None
    )

    errors = []

    max_tilt_deg = 0.0
    max_angular_speed = 0.0

    terminated = False
    truncated = False

    for _ in range(env.max_steps):

        measurement, true_state = measured_state(
            env,
            kind,
            substeps,
        )

        # -----------------------------------------------
        # Original controllers, delayed measurements.
        # -----------------------------------------------

        if pid is not None:

            action = sensor.measured_pid_action(
                pid,
                env,
                measurement,
            )

        else:

            sac_obs = sensor.sac_observation(
                obs,
                true_state,
                measurement,
            )

            action, _ = policy.predict(
                sac_obs,
                deterministic=True,
            )

        # -----------------------------------------------
        # Advance the simulator.
        #
        # RPM delay happens INSIDE the physics callback.
        # -----------------------------------------------

        (
            obs,
            _,
            terminated,
            truncated,
            _,
        ) = env.step(action)

        errors.append(
            np.asarray(
                obs[:3],
                dtype=np.float64,
            ).copy()
        )

        state = np.asarray(
            env.sim._getDroneStateVector(0)
        )

        max_tilt_deg = max(
            max_tilt_deg,
            float(
                np.rad2deg(
                    np.max(
                        np.abs(state[7:9])
                    )
                )
            ),
        )

        max_angular_speed = max(
            max_angular_speed,
            float(
                np.linalg.norm(
                    state[13:16]
                )
            ),
        )

        if terminated or truncated:
            break

    # ================================================
    # Results
    # ================================================

    n = len(errors)

    expected_physics_steps = (
        n * env.sim.PYB_STEPS_PER_CTRL
    )

    if (
        len(env.sim._latency_states)
        != expected_physics_steps
    ):
        raise RuntimeError(
            "Missing physics-level states"
        )

    survived = bool(
        truncated
        and not terminated
        and n == env.max_steps
    )

    result = {
        "survived": int(survived),

        "duration_s":
            n / env.control_freq_hz,

        "reason":
            "completed"
            if survived
            else failure_reason(env),

        "max_tilt_deg": max_tilt_deg,

        "max_angular_speed_rad_s":
            max_angular_speed,

        **{
            metric: float("nan")
            for metric in sensor.mass.METRICS
        },
    }

    if survived:

        errors = np.asarray(errors)

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
            "full_xyz_m": rms(errors),
            "late_xyz_m": rms(late),
            "late_xy_m": rms(late[:, :2]),
            "late_z_m": rms(late[:, 2:3]),
        })

    return result


# ============================================================
# 6. Preflight
# ============================================================

def preflight():

    # --------------------------------------------------------
    # A. Zero-delay physics must match original physics.
    # --------------------------------------------------------

    original = sensor.make_env()
    patched = make_env("command", 0)

    try:

        obs_a, _ = original.reset(seed=30100)
        obs_b, _ = reset_pipeline(
            patched, seed=30100
        )

        np.testing.assert_allclose(
            obs_a, obs_b,
            rtol=0, atol=1e-7,
        )

        action = np.array(
            [0.10, -0.05, 0.03, -0.02],
            dtype=np.float32,
        )

        for _ in range(10):

            obs_a, *_ = original.step(action)
            obs_b, *_ = patched.step(action)

            np.testing.assert_allclose(
                obs_a, obs_b,
                rtol=0, atol=1e-6,
            )

        assert (
            len(patched.sim._latency_states)
            == 50
        )

        print(
            "[PASS] Zero-delay physics equals original"
        )

    finally:
        original.close()
        patched.close()

    # --------------------------------------------------------
    # B. 5 physics steps command delay must equal
    #    the old 1-control-step command delay.
    # --------------------------------------------------------

    old = sensor.make_env()
    new = make_env("command", 5)

    try:

        obs_a, _ = old.reset(seed=30101)
        obs_b, _ = reset_pipeline(
            new, seed=30101
        )

        fifo = sensor.CommandDelay(1)

        actions = [
            np.array(
                [0.10, -0.05, 0.03, -0.02],
                dtype=np.float32,
            ),

            np.array(
                [-0.06, 0.08, -0.03, 0.07],
                dtype=np.float32,
            ),

            np.zeros(
                4, dtype=np.float32
            ),
        ]

        for i in range(10):

            action = actions[
                i % len(actions)
            ]

            # Old control-step delay.
            delayed = fifo.apply(action)

            obs_a, *_ = old.step(delayed)

            # New physics-step delay.
            obs_b, *_ = new.step(action)

            np.testing.assert_allclose(
                obs_a, obs_b,
                rtol=0, atol=1e-6,
            )

        print(
            "[PASS] 5 physics steps equal "
            "1 control-step command delay"
        )

    finally:
        old.close()
        new.close()

    # --------------------------------------------------------
    # C. 5 physics steps observation delay must equal
    #    the previous control-step state.
    # --------------------------------------------------------

    env = make_env(
        "observation", 5
    )

    try:

        reset_pipeline(
            env, seed=30102
        )

        previous = np.asarray(
            env.sim._getDroneStateVector(0)
        ).copy()

        action = np.array(
            [0.03, -0.02, 0.01, -0.01],
            dtype=np.float32,
        )

        for _ in range(10):

            env.step(action)

            measurement, _ = measured_state(
                env,
                "observation",
                5,
            )

            np.testing.assert_allclose(
                measurement[:16],
                previous[:16],
                rtol=0,
                atol=1e-7,
            )

            previous = np.asarray(
                env.sim._getDroneStateVector(0)
            ).copy()

        print(
            "[PASS] 5 physics steps equal "
            "1 control-step observation delay"
        )

    finally:
        env.close()

    # --------------------------------------------------------
    # D. Verify actual 1-substep RPM delay.
    # --------------------------------------------------------

    one = make_env("command", 1)

    try:

        reset_pipeline(
            one, seed=30103
        )

        action = np.array(
            [0.10, -0.05, 0.03, -0.02],
            dtype=np.float32,
        )

        expected = np.asarray(
            one._action_to_rpm(action),
            dtype=np.float64,
        )

        one.step(action)

        applied = (
            one.sim._latency_applied_rpm_log
        )

        assert len(applied) == 5

        # First physics step uses old hover RPM.
        np.testing.assert_allclose(
            applied[0],
            np.full(
                4, one.sim.HOVER_RPM
            ),
            rtol=0,
            atol=1e-8,
        )

        # Following four physics steps use new RPM.
        for item in applied[1:]:

            np.testing.assert_allclose(
                item,
                expected,
                rtol=0,
                atol=1e-8,
            )

        print(
            "[PASS] 1-substep RPM delay "
            "switches after 4.167 ms"
        )

    finally:
        one.close()

    print(
        "[PASS] Latency preflight complete"
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
        "--kind",
        choices=(
            "command",
            "observation",
            "both",
        ),
        default="command",
    )

    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=list(LEVELS),
    )

    parser.add_argument(
        "--seed-start",
        type=int,
        default=30000,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    if (
        args.seeds < 1
        or args.seed_start < 0
    ):
        parser.error(
            "Invalid seed arguments"
        )

    if (
        len(set(args.levels))
        != len(args.levels)
        or any(
            value not in LEVELS
            for value in args.levels
        )
    ):
        parser.error(
            "Levels must be distinct integers 0..5"
        )

    preflight()

    if args.check_only:
        return

    if not sensor.mass.BASELINE.is_file():
        raise FileNotFoundError(
            sensor.mass.BASELINE
        )

    policies = {
        "PID": None,

        "Baseline_30k": SAC.load(
            str(sensor.mass.BASELINE),
            device="cpu",
        ),
    }

    kinds = (
        ("command", "observation")
        if args.kind == "both"
        else (args.kind,)
    )

    radii = (
        (0.30,)
        if args.quick
        else sensor.mass.RADII
    )

    phases = (
        (0.0,)
        if args.quick
        else sensor.mass.PHASES
    )

    omegas = (
        (0.60,)
        if args.quick
        else sensor.mass.OMEGAS
    )

    seeds = range(
        args.seed_start,
        args.seed_start
        + (2 if args.quick else args.seeds),
    )

    output_dir = (
        sensor.mass.ROOT
        / "results"
        / "evaluation"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = output_dir / (
        "latency_threshold_"
        + datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )
        + ".csv"
    )

    fields = [
        "kind",
        "substeps",
        "delay_ms",
        "controller",
        "radius_m",
        "phase_rad",
        "omega_rad_s",
        "seed",
        "survived",
        "duration_s",
        "reason",
        "max_tilt_deg",
        "max_angular_speed_rad_s",
        *sensor.mass.METRICS,
    ]

    total = (
        len(kinds)
        * len(args.levels)
        * len(policies)
        * len(radii)
        * len(phases)
        * len(omegas)
        * len(seeds)
    )

    print(
        f"\nLatency threshold: {total} episodes"
    )

    print(
        f"CSV: {output}",
        flush=True,
    )

    # ========================================================
    # Evaluate
    # ========================================================

    with output.open(
        "x",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        for kind in kinds:

            for substeps in args.levels:

                for controller, policy in policies.items():

                    rows = []

                    for radius in radii:

                        for phase in phases:

                            for omega in omegas:

                                env = make_env(
                                    kind,
                                    substeps,
                                    radius,
                                    phase,
                                    omega,
                                )

                                try:

                                    for seed in seeds:

                                        result = run_episode(
                                            env,
                                            kind,
                                            substeps,
                                            controller,
                                            policy,
                                            seed,
                                        )

                                        row = {
                                            "kind": kind,
                                            "substeps": substeps,

                                            "delay_ms":
                                                1000 * substeps / 240,

                                            "controller": controller,
                                            "radius_m": radius,
                                            "phase_rad": phase,
                                            "omega_rad_s": omega,
                                            "seed": seed,

                                            **result,
                                        }

                                        writer.writerow(row)
                                        rows.append(row)

                                finally:
                                    env.close()

                                file.flush()

                    successful = [
                        row for row in rows
                        if row["survived"]
                    ]

                    failed = [
                        row for row in rows
                        if not row["survived"]
                    ]

                    if failed:

                        median_failure = float(
                            np.median([
                                row["duration_s"]
                                for row in failed
                            ])
                        )

                    else:
                        median_failure = float("nan")

                    reason_counts = dict(
                        Counter(
                            row["reason"]
                            for row in failed
                        )
                    )

                    if successful:

                        late_cm = 100 * float(
                            np.mean([
                                row["late_xyz_m"]
                                for row in successful
                            ])
                        )

                    else:
                        late_cm = float("nan")

                    print(
                        f"{kind:<11} "
                        f"{substeps}/240s | "
                        f"{controller:<13} "
                        f"survival="
                        f"{len(successful)}/{len(rows)} "
                        f"late={late_cm:.3f}cm "
                        f"median_fail="
                        f"{median_failure:.3f}s "
                        f"reasons={reason_counts}",
                        flush=True,
                    )

    print(
        f"\n[FINISHED] {output}"
    )


if __name__ == "__main__":
    main()
