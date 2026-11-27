
"""Sensor noise, observation latency, command latency: PID vs frozen SAC.

Run from repository root:
    python -u scripts/eval_sensor_latency_robustness.py --check-only

Requires existing eval_mass_robustness.py and eval_controllers.py.
No changes to plant dynamics, trained weights, or PID gains.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math

from collections import deque
from datetime import datetime

import numpy as np
from scipy.spatial.transform import Rotation
from stable_baselines3 import SAC

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel

from eval_controllers import pid_action, pid_rpm_to_action
import eval_mass_robustness as mass


# ============================================================
# 1. Scenarios and sensor uncertainty
# ============================================================

# (noise level, observation delay steps, command delay steps)
SCENARIOS = {
    "nominal":        ("none",     0, 0),
    "noise_low":      ("low",      0, 0),
    "noise_moderate": ("moderate", 0, 0),

    "obs_delay_1":    ("none",     1, 0),
    "obs_delay_2":    ("none",     2, 0),

    "cmd_delay_1":    ("none",     0, 1),
    "cmd_delay_2":    ("none",     0, 2),

    "combined_low":   ("low",      1, 1),
}

# Standard deviations PER AXIS:
# position (m), velocity (m/s),
# attitude (rad), angular velocity (rad/s).
NOISE = {
    "none": (
        0.0, 0.0, 0.0, 0.0
    ),
    "low": (
        0.002,
        0.010,
        math.radians(0.2),
        0.010,
    ),
    "moderate": (
        0.005,
        0.030,
        math.radians(0.5),
        0.030,
    ),
}

CONTROLLERS = ("PID", "Baseline_30k")


# ============================================================
# 2. Precompute paired sensor noise
# ============================================================

def noise_samples(seed, radius, phase, omega, n):
    """
    Common random errors for both controllers.

    Noise RNG is independent of environment randomization.
    Same trajectory and seed produce identical random errors.
    """

    key = [
        int(seed),
        round(radius * 1_000_000),
        round(
            (phase % (2 * math.pi))
            * 1_000_000
        ),
        round(omega * 1_000_000),
        50627,
    ]

    rng = np.random.default_rng(
        np.random.SeedSequence(key)
    )

    # 3 position + 3 velocity + 3 attitude
    # + 3 angular velocity random variables.
    z = rng.standard_normal((n, 12))

    digest = hashlib.sha256(
        z.tobytes()
    ).hexdigest()[:16]

    return z, digest


# ============================================================
# 3. Sensor channel: noise and observation delay
# ============================================================

class SensorChannel:
    """
    Generate one noisy state estimate every control step.

    A measurement with delay_steps=1 becomes available
    one control interval after it was sampled.

    The true simulator state is never modified.
    """

    def __init__(self, level, delay_steps, z):

        self.std = NOISE[level]
        self.delay = int(delay_steps)

        self.z = np.asarray(
            z, dtype=np.float64
        )

        self.history = []

        if (
            self.delay < 0
            or self.z.ndim != 2
            or self.z.shape[1] != 12
        ):
            raise ValueError(
                "Bad sensor delay or Gaussian sample shape"
            )

    def sample(self, true_state):

        k = len(self.history)

        if k >= len(self.z):
            raise RuntimeError(
                "Insufficient precomputed sensor noise"
            )

        state = np.asarray(
            true_state, dtype=np.float64
        )

        if state.shape != (20,):
            raise ValueError(
                f"Expected 20D state, got {state.shape}"
            )

        # Copy the state; never modify the true state.
        measured = state.copy()

        (
            s_pos,
            s_vel,
            s_att,
            s_rate,
        ) = self.std

        z = self.z[k]

        # Position measurement error.
        measured[0:3] += s_pos * z[0:3]

        # Linear velocity measurement error.
        measured[10:13] += s_vel * z[3:6]

        # Angular velocity measurement error.
        measured[13:16] += s_rate * z[9:12]

        # Attitude perturbation represented as a
        # small random 3D rotation.
        #
        # This preserves a valid unit quaternion.
        if s_att:

            q_noise = Rotation.from_rotvec(
                s_att * z[6:9]
            )

            q_true = Rotation.from_quat(
                measured[3:7]
            )

            measured[3:7] = (
                q_noise * q_true
            ).as_quat()

        # Save newly generated measurement.
        self.history.append(measured)

        # Return an earlier measurement if delayed.
        selected = max(0, k - self.delay)

        return self.history[selected].copy()


# ============================================================
# 4. Command delay
# ============================================================

class CommandDelay:
    """
    Delay normalized motor commands using a FIFO queue.

    Startup commands are nominal hover commands (action=0).

    This is a pure command delay, not a motor inertia model.
    """

    def __init__(self, steps):

        self.steps = int(steps)

        if self.steps < 0:
            raise ValueError(
                "Negative command delay"
            )

        self.queue = deque(
            np.zeros(4, dtype=np.float32)
            for _ in range(self.steps)
        )

    def apply(self, action):

        cmd = np.asarray(
            action, dtype=np.float32
        ).reshape(4).copy()

        cmd = np.clip(
            cmd, -1.0, 1.0
        )

        if not self.steps:
            return cmd

        self.queue.append(cmd)

        return self.queue.popleft()


# ============================================================
# 5. Environment
# ============================================================

def make_env(
    radius=0.30,
    phase=0.0,
    omega=0.60,
):

    return mass.PhaseCircleTrackingEnv(
        gui=False,
        physics_freq_hz=240,
        control_freq_hz=48,
        radius_m=radius,
        initial_phase_rad=phase,
        angular_speed_rad_s=omega,
    )


# ============================================================
# 6. PID action using measured state
# ============================================================

def measured_pid_action(
    pid,
    env,
    measurement,
):
    """
    Original DSLPIDControl with measured, not true, state.

    Current target position and velocity are known
    trajectory commands, not noisy sensor measurements.

    PID gains and RPM/action conversion remain unchanged.
    """

    rpm, _, _ = pid.computeControl(
        control_timestep=1.0 / env.control_freq_hz,

        cur_pos=measurement[0:3].copy(),
        cur_quat=measurement[3:7].copy(),
        cur_vel=measurement[10:13].copy(),
        cur_ang_vel=measurement[13:16].copy(),

        target_pos=env.target_pos.copy(),
        target_vel=env.target_vel.copy(),
        target_rpy=np.zeros(3),
    )

    return pid_rpm_to_action(rpm, env)


# ============================================================
# 7. SAC observation using the SAME measurement
# ============================================================

def sac_observation(
    true_obs,
    true_state,
    measurement,
):
    """
    Build the SAC 16D observation from measured quantities.

    SAC observation:
        [position_error (3),
         velocity_error (3),
         target_velocity (3),
         quaternion (4),
         angular_velocity (3)]

    Start from the environment's true observation to
    preserve its reference trajectory convention.

    Replace only physical quantities with measurements.
    """

    obs = np.asarray(
        true_obs, dtype=np.float32
    ).copy()

    # position_error = target_pos - measured_position
    obs[0:3] += (
        true_state[0:3]
        - measurement[0:3]
    ).astype(np.float32)

    # velocity_error = target_vel - measured_velocity
    obs[3:6] += (
        true_state[10:13]
        - measurement[10:13]
    ).astype(np.float32)

    # Target velocity obs[6:9] stays unchanged.

    # Normalize the quaternion sign convention.
    q = measurement[3:7].copy()

    if q[3] < 0.0:
        q = -q

    obs[9:13] = q.astype(np.float32)

    obs[13:16] = measurement[13:16].astype(
        np.float32
    )

    return obs


# ============================================================
# 8. Metrics
# ============================================================

def xyz_rms(errors):

    errors = np.asarray(
        errors, dtype=np.float64
    )

    return float(
        np.sqrt(
            np.mean(
                np.sum(errors ** 2, axis=1)
            )
        )
    )


# ============================================================
# 9. Evaluate one episode
# ============================================================

def run_episode(
    env,
    controller,
    policy,
    scenario,
    seed,
):

    (
        noise_level,
        obs_delay,
        cmd_delay,
    ) = SCENARIOS[scenario]

    true_obs, _ = env.reset(seed=seed)

    # Precompute the same noise sequence for PID/SAC.
    samples, noise_hash = noise_samples(
        seed,
        env.radius_m,
        env.initial_phase_rad,
        env.angular_speed_rad_s,
        env.max_steps + 1,
    )

    sensor = SensorChannel(
        noise_level,
        obs_delay,
        samples,
    )

    command_channel = CommandDelay(
        cmd_delay
    )

    pid = (
        DSLPIDControl(
            drone_model=DroneModel.CF2X
        )
        if controller == "PID"
        else None
    )

    # Initial physical state.
    true_state = env.sim._getDroneStateVector(0)

    # State estimate available to controller.
    measured = sensor.sample(true_state)

    sac_obs = sac_observation(
        true_obs,
        true_state,
        measured,
    )

    errors = []

    terminated = False
    truncated = False

    for _ in range(env.max_steps):

        # ----------------------------------------------------
        # Controller command based on available measurement.
        # ----------------------------------------------------

        if pid is not None:

            desired_action = measured_pid_action(
                pid,
                env,
                measured,
            )

        else:

            desired_action, _ = policy.predict(
                sac_obs,
                deterministic=True,
            )

        # ----------------------------------------------------
        # Apply command latency, if configured.
        # ----------------------------------------------------

        applied_action = command_channel.apply(
            desired_action
        )

        # ----------------------------------------------------
        # Advance TRUE physical environment.
        #
        # Reward, termination and errors use TRUE state.
        # ----------------------------------------------------

        (
            true_obs,
            _,
            terminated,
            truncated,
            _,
        ) = env.step(applied_action)

        errors.append(
            np.asarray(
                true_obs[:3],
                dtype=np.float64,
            ).copy()
        )

        if terminated or truncated:
            break

        # ----------------------------------------------------
        # Generate the measurement for the next decision.
        # ----------------------------------------------------

        true_state = env.sim._getDroneStateVector(0)

        measured = sensor.sample(
            true_state
        )

        sac_obs = sac_observation(
            true_obs,
            true_state,
            measured,
        )

    # ========================================================
    # Episode results
    # ========================================================

    steps = len(errors)

    survived = bool(
        truncated
        and not terminated
        and steps == env.max_steps
    )

    result = {
        "survived": int(survived),

        "duration_s":
            steps / env.control_freq_hz,

        "noise_hash": noise_hash,
        "noise_level": noise_level,
        "obs_delay_steps": obs_delay,
        "cmd_delay_steps": cmd_delay,

        **{
            metric: float("nan")
            for metric in mass.METRICS
        },
    }

    # Tracking metrics only for completed episodes.
    if survived:

        errors = np.asarray(errors)

        late = errors[
            env.max_steps // 2:
        ]

        result.update({
            "full_xyz_m": xyz_rms(errors),
            "late_xyz_m": xyz_rms(late),
            "late_xy_m": xyz_rms(late[:, :2]),
            "late_z_m": xyz_rms(late[:, 2:3]),
        })

    return result


# ============================================================
# 10. Preflight validation
# ============================================================

def preflight():

    # --------------------------------------------------------
    # A. Reproducible, independent noise.
    # --------------------------------------------------------

    a, ha = noise_samples(
        29000, 0.30, 0.0, 0.60, 481
    )

    b, hb = noise_samples(
        29000, 0.30, 0.0, 0.60, 481
    )

    c, hc = noise_samples(
        29001, 0.30, 0.0, 0.60, 481
    )

    np.testing.assert_array_equal(a, b)

    assert ha == hb
    assert ha != hc
    assert not np.array_equal(a, c)

    # --------------------------------------------------------
    # B. Observation delay.
    # --------------------------------------------------------

    state0 = np.zeros(20)
    state0[6] = 1.0

    state1 = state0.copy()
    state1[0] = 0.12

    state2 = state0.copy()
    state2[0] = 0.21

    channel = SensorChannel(
        "none", 1, a
    )

    np.testing.assert_array_equal(
        channel.sample(state0), state0
    )

    np.testing.assert_array_equal(
        channel.sample(state1), state0
    )

    np.testing.assert_array_equal(
        channel.sample(state2), state1
    )

    # --------------------------------------------------------
    # C. Noise and valid quaternion.
    # --------------------------------------------------------

    noisy = SensorChannel(
        "low", 0, a
    ).sample(state0)

    assert np.linalg.norm(noisy[0:3]) > 0

    assert abs(
        np.linalg.norm(noisy[3:7]) - 1.0
    ) < 1e-12

    # --------------------------------------------------------
    # D. Motor command delay.
    # --------------------------------------------------------

    cmd = np.array(
        [0.1, -0.1, 0.2, -0.2],
        dtype=np.float32,
    )

    delay = CommandDelay(2)

    np.testing.assert_array_equal(
        delay.apply(cmd),
        np.zeros(4),
    )

    np.testing.assert_array_equal(
        delay.apply(cmd),
        np.zeros(4),
    )

    np.testing.assert_array_equal(
        delay.apply(np.zeros(4)),
        cmd,
    )

    # --------------------------------------------------------
    # E. Compare nominal SAC observation and PID action
    # with the ORIGINAL environment implementation.
    # --------------------------------------------------------

    env = make_env()

    try:

        true_obs, _ = env.reset(seed=29002)

        state = np.asarray(
            env.sim._getDroneStateVector(0)
        )

        original_state = state.copy()

        measured = SensorChannel(
            "none", 0, a
        ).sample(state)

        np.testing.assert_array_equal(
            state,
            original_state,
        )

        np.testing.assert_array_equal(
            measured,
            state,
        )

        rebuilt_obs = sac_observation(
            true_obs,
            state,
            measured,
        )

        np.testing.assert_array_equal(
            rebuilt_obs,
            true_obs,
        )

        old_pid = DSLPIDControl(
            drone_model=DroneModel.CF2X
        )

        new_pid = DSLPIDControl(
            drone_model=DroneModel.CF2X
        )

        old_action, _ = pid_action(
            old_pid,
            env,
        )

        new_action = measured_pid_action(
            new_pid,
            env,
            measured,
        )

        np.testing.assert_allclose(
            old_action,
            new_action,
            rtol=0,
            atol=1e-6,
        )

        # Verify equivalence after one real physics step.
        obs_next, *_ = env.step(
            np.zeros(4, dtype=np.float32)
        )

        state_next = env.sim._getDroneStateVector(0)

        np.testing.assert_array_equal(
            sac_observation(
                obs_next,
                state_next,
                state_next,
            ),
            obs_next,
        )

    finally:
        env.close()

    print(
        "[PASS] Sensor noise reproducible and paired"
    )

    print(
        "[PASS] Observation and motor-command delays"
    )

    print(
        "[PASS] Valid noisy quaternions"
    )

    print(
        "[PASS] Nominal SAC observation unchanged"
    )

    print(
        "[PASS] Nominal PID action equals original pid_action"
    )

    print(
        "[PASS] True simulator state untouched"
    )


# ============================================================
# 11. Main benchmark
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
        default=29000,
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

    if (
        args.seeds < 1
        or args.seed_start < 0
    ):
        parser.error(
            "--seeds must be positive and "
            "--seed-start nonnegative"
        )

    if len(args.scenarios) != len(
        set(args.scenarios)
    ):
        parser.error(
            "Scenarios must be unique"
        )

    # Always validate before running benchmarks.
    preflight()

    if args.check_only:
        return

    if not mass.BASELINE.is_file():
        raise FileNotFoundError(
            mass.BASELINE
        )

    policies = {
        "PID": None,

        "Baseline_30k": SAC.load(
            str(mass.BASELINE),
            device="cpu",
        ),
    }

    radii = (
        (0.30,)
        if args.quick
        else mass.RADII
    )

    phases = (
        (0.0,)
        if args.quick
        else mass.PHASES
    )

    omegas = (
        (0.60,)
        if args.quick
        else mass.OMEGAS
    )

    seeds = range(
        args.seed_start,
        args.seed_start
        + (2 if args.quick else args.seeds),
    )

    count_per_cell = (
        len(radii)
        * len(phases)
        * len(omegas)
        * len(seeds)
    )

    total = (
        count_per_cell
        * len(policies)
        * len(args.scenarios)
    )

    # --------------------------------------------------------
    # Output CSV
    # --------------------------------------------------------

    output_dir = (
        mass.ROOT
        / "results"
        / "evaluation"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outfile = output_dir / (
        "sensor_latency_robustness_"
        + datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )
        + ".csv"
    )

    fields = [
        "controller",
        "scenario",
        "radius_m",
        "phase_rad",
        "omega_rad_s",
        "seed",
        "noise_level",
        "obs_delay_steps",
        "cmd_delay_steps",
        "noise_hash",
        "survived",
        "duration_s",
        *mass.METRICS,
    ]

    data = {}

    paired_hashes = {}

    print("\n" + "=" * 72)
    print(
        "SENSOR NOISE / LATENCY: PID VS SAC BASELINE 30k"
    )
    print("=" * 72)

    print(
        f"Scenarios: {args.scenarios}"
    )

    print(
        f"Episodes per controller/scenario: "
        f"{count_per_cell}"
    )

    print(
        f"Total episodes: {total}"
    )

    print(
        f"CSV: {outfile}",
        flush=True,
    )

    # ========================================================
    # Evaluation
    # ========================================================

    with outfile.open(
        "x",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        for scenario in args.scenarios:

            for controller, policy in policies.items():

                group = []

                print(
                    f"\n[RUN] {scenario} | {controller}",
                    flush=True,
                )

                for radius in radii:

                    for phase in phases:

                        for omega in omegas:

                            env = make_env(
                                radius,
                                phase,
                                omega,
                            )

                            try:

                                for seed in seeds:

                                    result = run_episode(
                                        env,
                                        controller,
                                        policy,
                                        scenario,
                                        seed,
                                    )

                                    key = (
                                        scenario,
                                        radius,
                                        phase,
                                        omega,
                                        seed,
                                    )

                                    old_hash = (
                                        paired_hashes.setdefault(
                                            key,
                                            result["noise_hash"],
                                        )
                                    )

                                    if (
                                        old_hash
                                        != result["noise_hash"]
                                    ):
                                        raise RuntimeError(
                                            "Unmatched sensor noise streams"
                                        )

                                    row = {
                                        "controller": controller,
                                        "scenario": scenario,
                                        "radius_m": radius,
                                        "phase_rad": phase,
                                        "omega_rad_s": omega,
                                        "seed": int(seed),
                                        **result,
                                    }

                                    writer.writerow(row)
                                    group.append(row)

                            finally:
                                env.close()

                            file.flush()

                data[(scenario, controller)] = group

                successes, count, m = mass.summary(
                    group
                )

                print(
                    f"[DONE] {scenario:<17} "
                    f"{controller:<13} "
                    f"survival={successes}/{count} "
                    f"full={m['full_xyz_m']:.3f}cm "
                    f"late={m['late_xyz_m']:.3f}cm "
                    f"XY={m['late_xy_m']:.3f}cm "
                    f"Z={m['late_z_m']:.3f}cm",
                    flush=True,
                )

    # ========================================================
    # Summary
    # ========================================================

    print("\n" + "=" * 72)
    print(
        "SUMMARY - RMSE IN CM, SUCCESSFUL EPISODES ONLY"
    )
    print("=" * 72)

    print(
        f"{'Scenario':<18} "
        f"{'Controller':<14} "
        f"{'Survival':>10} "
        f"{'Full':>8} "
        f"{'Late':>8} "
        f"{'LateXY':>8} "
        f"{'LateZ':>8}"
    )

    for scenario in args.scenarios:

        for controller in CONTROLLERS:

            successes, count, m = mass.summary(
                data[(scenario, controller)]
            )

            print(
                f"{scenario:<18} "
                f"{controller:<14} "
                f"{successes:4}/{count:<5} "
                f"{m['full_xyz_m']:8.3f} "
                f"{m['late_xyz_m']:8.3f} "
                f"{m['late_xy_m']:8.3f} "
                f"{m['late_z_m']:8.3f}"
            )

    # ========================================================
    # Matched SAC/PID comparison
    # ========================================================

    print("\n" + "=" * 72)
    print("MATCHED SAC VS PID")
    print("=" * 72)

    for scenario in args.scenarios:

        print(f"\n{scenario}")

        mass.matched_report(
            data[(scenario, "Baseline_30k")],
            data[(scenario, "PID")],
            "SAC vs PID",
        )

    # ========================================================
    # Paired robustness degradation vs nominal
    # ========================================================

    if "nominal" in args.scenarios:

        print("\n" + "=" * 72)
        print("PAIRED DEGRADATION VS NOMINAL")
        print("=" * 72)

        for controller in CONTROLLERS:

            for scenario in args.scenarios:

                if scenario == "nominal":
                    continue

                mass.matched_report(
                    data[(scenario, controller)],
                    data[("nominal", controller)],
                    f"{controller}: {scenario} vs nominal",
                )

    print(f"\n[FINISHED] {outfile}")

    print(
        "True state used for termination and tracking."
    )

    print(
        "1 control step = 20.833 ms."
    )


if __name__ == "__main__":
    main()
