
"""Zero-shot wind-like disturbance benchmark.

Controllers:
    PID
    SAC Baseline 30k

Scenarios:
    nominal
    constant_x10
    random_constant
    random_gust
    turbulent

Each (scenario, trajectory, seed) generates a deterministic
240-Hz force trace, identical for PID and SAC.

Requires the existing eval_mass_robustness.py.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math

from datetime import datetime
from types import MethodType

import numpy as np
import pybullet as p

from stable_baselines3 import SAC

from gym_pybullet_drones.control.DSLPIDControl import (
    DSLPIDControl,
)
from gym_pybullet_drones.utils.enums import (
    DroneModel,
    Physics,
)

from eval_controllers import pid_action
import eval_mass_robustness as mass


SCENARIOS = (
    "nominal",
    "constant_x10",
    "random_constant",
    "random_gust",
    "turbulent",
)

CONTROLLERS = (
    "PID",
    "Baseline_30k",
)

GUST_START = 3.0
GUST_END = 4.0
GUST_RAMP_S = 0.20

RECOVERY_THRESHOLD_M = 0.05
RECOVERY_HOLD_S = 0.50


# ============================================================
# 1. Create a reproducible 240-Hz wind profile
# ============================================================

def make_wind_profile(
    scenario,
    seed,
    radius,
    phase,
    omega,
    n,
    physics_hz,
):

    if scenario not in SCENARIOS:
        raise ValueError(scenario)

    # Stable seed, independent of the global NumPy RNG.
    # Includes both initial-state seed and trajectory.
    key = [
        int(seed),
        int(round(radius * 1_000_000)),
        int(round(
            (phase % (2 * math.pi)) * 1_000_000
        )),
        int(round(omega * 1_000_000)),
        SCENARIOS.index(scenario) + 100,
    ]

    rng = np.random.default_rng(
        np.random.SeedSequence(key)
    )

    dt = 1.0 / physics_hz

    t = np.arange(
        n,
        dtype=np.float64,
    ) * dt

    profile = np.zeros(
        (n, 3),
        dtype=np.float64,
    )

    amplitude = 0.0
    theta = float("nan")

    # --------------------------------------------------------
    # A. Fixed deterministic 10 mN along world X
    # --------------------------------------------------------

    if scenario == "constant_x10":

        profile[:, 0] = 0.010

        amplitude = 0.010
        theta = 0.0

    # --------------------------------------------------------
    # B. Randomized constant or gust
    # --------------------------------------------------------

    elif scenario in (
        "random_constant",
        "random_gust",
    ):

        if scenario == "random_constant":
            lo, hi = 0.005, 0.015
        else:
            lo, hi = 0.010, 0.020

        amplitude = float(
            rng.uniform(lo, hi)
        )

        theta = float(
            rng.uniform(0, 2 * math.pi)
        )

        direction = np.array(
            [
                math.cos(theta),
                math.sin(theta),
            ],
            dtype=np.float64,
        )

        if scenario == "random_constant":

            profile[:, :2] = (
                amplitude * direction
            )

        else:

            # Smooth gust, active only between 3 and 4 s.
            # 0.2 s raised-cosine ramp at both ends.

            s = t - GUST_START

            envelope = np.zeros(
                n,
                dtype=np.float64,
            )

            attack = (
                (s >= 0)
                & (s < GUST_RAMP_S)
            )

            plateau = (
                (s >= GUST_RAMP_S)
                & (
                    s <= (
                        GUST_END
                        - GUST_START
                        - GUST_RAMP_S
                    )
                )
            )

            release = (
                s > (
                    GUST_END
                    - GUST_START
                    - GUST_RAMP_S
                )
            ) & (
                s < (
                    GUST_END
                    - GUST_START
                )
            )

            envelope[attack] = (
                0.5
                * (
                    1 - np.cos(
                        math.pi
                        * s[attack]
                        / GUST_RAMP_S
                    )
                )
            )

            envelope[plateau] = 1.0

            envelope[release] = (
                0.5
                * (
                    1 - np.cos(
                        math.pi
                        * (
                            GUST_END
                            - GUST_START
                            - s[release]
                        )
                        / GUST_RAMP_S
                    )
                )
            )

            profile[:, :2] = (
                amplitude
                * envelope[:, None]
                * direction[None, :]
            )

    # --------------------------------------------------------
    # C. Temporally correlated turbulent disturbance
    # --------------------------------------------------------

    elif scenario == "turbulent":

        # Random mean horizontal force: 0..8 mN.
        amplitude = float(
            rng.uniform(0.0, 0.008)
        )

        theta = float(
            rng.uniform(0, 2 * math.pi)
        )

        mean = amplitude * np.array(
            [
                math.cos(theta),
                math.sin(theta),
            ],
            dtype=np.float64,
        )

        # Exact discrete-time OU / AR(1) process.
        tau = 0.50
        sigma = 0.003

        rho = math.exp(-dt / tau)

        noise_scale = (
            sigma
            * math.sqrt(1 - rho * rho)
        )

        deviation = rng.normal(
            0,
            sigma,
            size=2,
        )

        for k in range(n):

            if k > 0:

                deviation = (
                    rho * deviation
                    + noise_scale
                    * rng.normal(size=2)
                )

            profile[k, :2] = (
                mean + deviation
            )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    force_norm = np.linalg.norm(
        profile[:, :2],
        axis=1,
    )

    metadata = {
        "wind_amplitude_mn":
            1000 * amplitude,

        "wind_direction_rad":
            theta,

        "wind_peak_mn":
            1000 * float(np.max(force_norm)),

        "wind_rms_mn":
            1000 * float(
                np.sqrt(
                    np.mean(force_norm ** 2)
                )
            ),

        "wind_hash":
            hashlib.sha256(
                profile.tobytes()
            ).hexdigest()[:16],
    }

    return profile, metadata


# ============================================================
# 2. Physical force injection
# ============================================================

def wind_physics(sim, rpm, nth_drone):

    # Preserve the original motor physics.
    sim._wind_original_physics(
        rpm,
        nth_drone,
    )

    k = sim._wind_substeps

    if k >= len(sim._wind_profile):

        raise RuntimeError(
            "Wind trace shorter than simulation"
        )

    force = sim._wind_profile[k]

    if np.any(force):

        body = int(
            sim.DRONE_IDS[nth_drone]
        )

        com_pos, _ = (
            p.getBasePositionAndOrientation(
                body,
                physicsClientId=sim.CLIENT,
            )
        )

        p.applyExternalForce(
            body,
            -1,
            forceObj=force.tolist(),
            posObj=com_pos,
            flags=p.WORLD_FRAME,
            physicsClientId=sim.CLIENT,
        )

        sim._wind_active_substeps += 1

    sim._wind_substeps += 1


# ============================================================
# 3. Wind environment
# ============================================================

class StochasticWindEnv(
    mass.PhaseCircleTrackingEnv
):

    def __init__(
        self,
        *,
        scenario,
        **kwargs,
    ):

        self.wind_scenario = scenario

        super().__init__(**kwargs)

        sim = self.sim

        if (
            sim.PHYSICS != Physics.PYB
            or sim.NUM_DRONES != 1
            or sim.DRONE_MODEL != DroneModel.CF2X
            or sim.PYB_FREQ != 240
            or sim.CTRL_FREQ != 48
        ):

            raise RuntimeError(
                "Expected CF2X, PYB, 240/48 Hz"
            )

        self._nominal = np.array(
            [
                sim.M,
                sim.KF,
                sim.KM,
                sim.HOVER_RPM,
                sim.MAX_RPM,
                self.hover_rpm,
                self.max_rpm,
            ],
            dtype=np.float64,
        )

        sim._wind_original_physics = sim._physics

        sim._physics = MethodType(
            wind_physics,
            sim,
        )

        sim._wind_substeps = 0
        sim._wind_active_substeps = 0

        sim._wind_profile = np.zeros(
            (
                self.max_steps
                * sim.PYB_STEPS_PER_CTRL,
                3,
            ),
            dtype=np.float64,
        )

        self.wind_metadata = {}

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):

        obs, info = super().reset(
            seed=seed,
            options=options,
        )

        sim = self.sim

        # Confirm nominal actuator/control parameters.
        current = np.array(
            [
                sim.M,
                sim.KF,
                sim.KM,
                sim.HOVER_RPM,
                sim.MAX_RPM,
                self.hover_rpm,
                self.max_rpm,
            ],
            dtype=np.float64,
        )

        np.testing.assert_array_equal(
            current,
            self._nominal,
        )

        # Physical mass remains nominal.
        physical_mass = float(
            p.getDynamicsInfo(
                int(sim.DRONE_IDS[0]),
                -1,
                physicsClientId=sim.CLIENT,
            )[0]
        )

        if not np.isclose(
            physical_mass,
            sim.M,
            rtol=1e-6,
            atol=1e-9,
        ):

            raise RuntimeError(
                "Physical mass changed unexpectedly"
            )

        # Rebuild the SAME profile for the same
        # scenario, trajectory, and seed.
        (
            sim._wind_profile,
            self.wind_metadata,
        ) = make_wind_profile(
            self.wind_scenario,
            0 if seed is None else int(seed),
            float(self.radius_m),
            float(self.initial_phase_rad),
            float(self.angular_speed_rad_s),
            self.max_steps
            * sim.PYB_STEPS_PER_CTRL,
            sim.PYB_FREQ,
        )

        sim._wind_substeps = 0
        sim._wind_active_substeps = 0

        info = dict(info)
        info.update(self.wind_metadata)

        return obs, info


def make_env(
    scenario,
    radius=0.30,
    phase=0.0,
    omega=0.60,
):

    return StochasticWindEnv(
        scenario=scenario,
        gui=False,
        physics_freq_hz=240,
        control_freq_hz=48,
        radius_m=radius,
        initial_phase_rad=phase,
        angular_speed_rad_s=omega,
    )


# ============================================================
# 4. Evaluate an episode
# ============================================================

def xyz_rmse(errors):

    return float(
        np.sqrt(
            np.mean(
                np.sum(
                    errors ** 2,
                    axis=1,
                )
            )
        )
    )


def run_episode(
    env,
    controller,
    model,
    seed,
):

    obs, info = env.reset(seed=seed)

    pid = (
        DSLPIDControl(
            drone_model=DroneModel.CF2X
        )
        if controller == "PID"
        else None
    )

    errors = []

    terminated = False
    truncated = False

    for _ in range(env.max_steps):

        if pid is not None:

            action, _ = pid_action(
                pid,
                env,
            )

        else:

            action, _ = model.predict(
                obs,
                deterministic=True,
            )

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

        if terminated or truncated:
            break

    errors = np.asarray(
        errors,
        dtype=np.float64,
    )

    n = len(errors)

    survived = bool(
        truncated
        and not terminated
        and n == env.max_steps
    )

    sim = env.sim

    expected = (
        n * sim.PYB_STEPS_PER_CTRL
    )

    if sim._wind_substeps != expected:

        raise RuntimeError(
            "Wind callback missed physics substeps"
        )

    active_expected = np.count_nonzero(
        np.any(
            sim._wind_profile[:expected],
            axis=1,
        )
    )

    if (
        sim._wind_active_substeps
        != active_expected
    ):

        raise RuntimeError(
            "Applied wind substeps mismatch"
        )

    result = {
        "survived": int(survived),

        "duration_s":
            n / env.control_freq_hz,

        **{
            metric: float("nan")
            for metric in mass.METRICS
        },

        "pre_gust_mean_error_m": float("nan"),
        "gust_peak_3_to_5_m": float("nan"),
        "gust_excess_peak_m": float("nan"),
        "recovery_time_s": float("nan"),

        **{
            key: info[key]
            for key in (
                "wind_amplitude_mn",
                "wind_direction_rad",
                "wind_peak_mn",
                "wind_rms_mn",
                "wind_hash",
            )
        },
    }

    # RMSE for completed episodes only.
    if survived:

        late = errors[
            env.max_steps // 2:
        ]

        result.update({
            "full_xyz_m":
                xyz_rmse(errors),

            "late_xyz_m":
                xyz_rmse(late),

            "late_xy_m":
                xyz_rmse(late[:, :2]),

            "late_z_m":
                xyz_rmse(late[:, 2:3]),
        })

    # Extra metrics for gust response.
    if env.wind_scenario == "random_gust":

        times = (
            np.arange(1, n + 1)
            / env.control_freq_hz
        )

        norm = np.linalg.norm(
            errors,
            axis=1,
        )

        pre = norm[
            (times >= 2.0)
            & (times < GUST_START)
        ]

        during = norm[
            (times >= GUST_START)
            & (times < 5.0)
        ]

        if len(pre) and len(during):

            pre_mean = float(
                np.mean(pre)
            )

            peak = float(
                np.max(during)
            )

            result["pre_gust_mean_error_m"] = (
                pre_mean
            )

            result["gust_peak_3_to_5_m"] = peak

            result["gust_excess_peak_m"] = (
                peak - pre_mean
            )

        # Recovery criterion: XYZ error <= 5 cm
        # for 0.5 continuous seconds after gust.

        hold = round(
            RECOVERY_HOLD_S
            * env.control_freq_hz
        )

        for start in np.flatnonzero(
            times >= GUST_END
        ):

            if (
                start + hold <= n
                and np.all(
                    norm[start:start + hold]
                    <= RECOVERY_THRESHOLD_M
                )
            ):

                result["recovery_time_s"] = float(
                    times[start] - GUST_END
                )

                break

    return result


# ============================================================
# 5. Preflight
# ============================================================

def preflight():

    # A. Wind trace must be reproducible.
    for scenario in SCENARIOS:

        a, meta_a = make_wind_profile(
            scenario,
            26000,
            0.30,
            0.0,
            0.60,
            2400,
            240,
        )

        b, meta_b = make_wind_profile(
            scenario,
            26000,
            0.30,
            0.0,
            0.60,
            2400,
            240,
        )

        np.testing.assert_array_equal(a, b)

        if (
            meta_a["wind_hash"]
            != meta_b["wind_hash"]
        ):
            raise RuntimeError(
                "Nondeterministic wind profiles"
            )

        if (
            not np.all(np.isfinite(a))
            or not np.all(a[:, 2] == 0)
        ):
            raise RuntimeError(
                "Invalid wind profile"
            )

    print(
        "[PASS] Reproducible wind profiles"
    )

    # B. Gust timing and amplitude.
    gust, metadata = make_wind_profile(
        "random_gust",
        26000,
        0.30,
        0.0,
        0.60,
        2400,
        240,
    )

    if (
        np.any(gust[:3 * 240])
        or np.any(gust[4 * 240:])
    ):

        raise RuntimeError(
            "Gust outside 3-4 second interval"
        )

    if not (
        10 <= metadata["wind_amplitude_mn"] <= 20
    ):

        raise RuntimeError(
            "Random gust amplitude out of range"
        )

    if not np.any(
        gust[3 * 240:4 * 240]
    ):

        raise RuntimeError(
            "Gust never activated"
        )

    print(
        "[PASS] Gust timing 3-4 seconds"
    )

    # C. Nominal simulation matches original.
    original = mass.PhaseCircleTrackingEnv(
        radius_m=0.30,
        initial_phase_rad=0.0,
        angular_speed_rad_s=0.60,
    )

    nominal = make_env("nominal")

    try:

        a, _ = original.reset(seed=26000)
        b, _ = nominal.reset(seed=26000)

        np.testing.assert_allclose(
            a, b, atol=1e-7, rtol=0
        )

        action = np.array(
            [0.10, -0.05, 0.03, -0.02],
            dtype=np.float32,
        )

        for _ in range(8):

            a, *_ = original.step(action)
            b, *_ = nominal.step(action)

            np.testing.assert_allclose(
                a, b, atol=1e-6, rtol=0
            )

        if (
            nominal.sim._wind_substeps != 40
            or nominal.sim._wind_active_substeps != 0
        ):

            raise RuntimeError(
                "Nominal physics substep mismatch"
            )

        print(
            "[PASS] Nominal equals original simulation"
        )

    finally:

        original.close()
        nominal.close()

    # D. Verify real physical displacement.
    calm = make_env("nominal")
    forced = make_env("constant_x10")

    try:

        calm.reset(seed=26001)
        forced.reset(seed=26001)

        action = np.zeros(
            4,
            dtype=np.float32,
        )

        for _ in range(8):

            calm.step(action)
            forced.step(action)

        x0 = float(
            calm.sim._getDroneStateVector(0)[0]
        )

        x1 = float(
            forced.sim._getDroneStateVector(0)[0]
        )

        if (
            x1 - x0 <= 0.001
            or forced.sim._wind_active_substeps != 40
        ):

            raise RuntimeError(
                "Constant wind had no physical effect"
            )

        print(
            "[PASS] 10 mN changes X position: "
            f"{x1-x0:.5f} m"
        )

    finally:

        calm.close()
        forced.close()

    # E. Verify same trace for repeated seed.
    first = make_env("turbulent")
    second = make_env("turbulent")

    try:

        first.reset(seed=26002)
        second.reset(seed=26002)

        np.testing.assert_array_equal(
            first.sim._wind_profile,
            second.sim._wind_profile,
        )

        second.reset(seed=26003)

        if np.array_equal(
            first.sim._wind_profile,
            second.sim._wind_profile,
        ):

            raise RuntimeError(
                "Different seeds gave identical wind"
            )

        print(
            "[PASS] Paired wind traces reproducible"
        )

    finally:

        first.close()
        second.close()

    print(
        "[PASS] Wind preflight completed"
    )


# ============================================================
# 6. Main benchmark
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

    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=SCENARIOS,
        default=list(SCENARIOS),
    )

    args = parser.parse_args()

    if (
        args.seeds < 1
        or args.seed_start < 0
        or len(args.scenarios) != len(set(args.scenarios))
    ):

        parser.error(
            "Use nonnegative seeds and unique scenarios"
        )

    preflight()

    if args.check_only:
        return

    if not mass.BASELINE.is_file():
        raise FileNotFoundError(mass.BASELINE)

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

    seed_range = range(
        args.seed_start,
        args.seed_start + (
            2 if args.quick else args.seeds
        ),
    )

    n_per = (
        len(radii)
        * len(phases)
        * len(omegas)
        * len(seed_range)
    )

    total = (
        n_per
        * len(args.scenarios)
        * len(policies)
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

    outfile = output_dir / (
        "wind_robustness_"
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
        "wind_amplitude_mn",
        "wind_direction_rad",
        "wind_peak_mn",
        "wind_rms_mn",
        "wind_hash",
        "survived",
        "duration_s",
        *mass.METRICS,
        "pre_gust_mean_error_m",
        "gust_peak_3_to_5_m",
        "gust_excess_peak_m",
        "recovery_time_s",
    ]

    data = {}

    # Confirms that PID and SAC get identical wind.
    wind_checksums = {}

    print()
    print("=" * 68)
    print("STOCHASTIC WIND ROBUSTNESS")
    print("PID vs SAC BASELINE 30k")
    print("=" * 68)

    print(
        f"Scenarios: {args.scenarios}"
    )

    print(
        f"Episodes/controller/scenario: {n_per}"
    )

    print(
        f"Total episodes: {total}"
    )

    print(
        f"Output: {outfile}",
        flush=True,
    )

    with outfile.open(
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

            for name, policy in policies.items():

                group = []

                print(
                    f"\n[RUN] {scenario} / {name}",
                    flush=True,
                )

                for radius in radii:

                    for phase in phases:

                        for omega in omegas:

                            env = make_env(
                                scenario,
                                radius,
                                phase,
                                omega,
                            )

                            try:

                                for seed in seed_range:

                                    result = run_episode(
                                        env,
                                        name,
                                        policy,
                                        seed,
                                    )

                                    key = (
                                        scenario,
                                        radius,
                                        phase,
                                        omega,
                                        seed,
                                    )

                                    previous = (
                                        wind_checksums.setdefault(
                                            key,
                                            result["wind_hash"],
                                        )
                                    )

                                    if (
                                        previous
                                        != result["wind_hash"]
                                    ):

                                        raise RuntimeError(
                                            "Different wind traces "
                                            "between controllers"
                                        )

                                    row = {
                                        "controller": name,
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

                            f.flush()

                data[(scenario, name)] = group

                successes, n, metrics = (
                    mass.summary(group)
                )

                print(
                    f"[DONE] {scenario:<16} "
                    f"{name:<13} "
                    f"{successes}/{n} "
                    f"full={metrics['full_xyz_m']:.3f}cm "
                    f"late={metrics['late_xyz_m']:.3f}cm "
                    f"XY={metrics['late_xy_m']:.3f}cm "
                    f"Z={metrics['late_z_m']:.3f}cm",
                    flush=True,
                )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 68)
    print("SUMMARY - SUCCESSFUL EPISODES ONLY")
    print("=" * 68)

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

        for name in CONTROLLERS:

            ok, n, m = mass.summary(
                data[(scenario, name)]
            )

            print(
                f"{scenario:<18} "
                f"{name:<14} "
                f"{ok:4}/{n:<5} "
                f"{m['full_xyz_m']:8.3f} "
                f"{m['late_xyz_m']:8.3f} "
                f"{m['late_xy_m']:8.3f} "
                f"{m['late_z_m']:8.3f}"
            )

    # ========================================================
    # Matched controller comparisons
    # ========================================================

    print()
    print("=" * 68)
    print("MATCHED SAC VS PID")
    print("=" * 68)

    for scenario in args.scenarios:

        print(f"\n{scenario}")

        mass.matched_report(
            data[(scenario, "Baseline_30k")],
            data[(scenario, "PID")],
            "SAC vs PID",
        )

    # ========================================================
    # Wind degradation relative to nominal
    # ========================================================

    if "nominal" in args.scenarios:

        print()
        print("=" * 68)
        print("WIND DEGRADATION VS NOMINAL")
        print("=" * 68)

        for name in CONTROLLERS:

            for scenario in args.scenarios:

                if scenario == "nominal":
                    continue

                mass.matched_report(
                    data[(scenario, name)],
                    data[("nominal", name)],
                    f"{name}: {scenario} vs nominal",
                )

    # ========================================================
    # Gust recovery
    # ========================================================

    if "random_gust" in args.scenarios:

        print()
        print("=" * 68)
        print("RANDOM GUST TRANSIENTS")
        print("=" * 68)

        for name in CONTROLLERS:

            group = [
                row
                for row in data[
                    ("random_gust", name)
                ]
                if row["survived"]
            ]

            peaks = [
                row["gust_peak_3_to_5_m"]
                for row in group
                if np.isfinite(
                    row["gust_peak_3_to_5_m"]
                )
            ]

            excess = [
                row["gust_excess_peak_m"]
                for row in group
                if np.isfinite(
                    row["gust_excess_peak_m"]
                )
            ]

            recovery = [
                row["recovery_time_s"]
                for row in group
                if np.isfinite(
                    row["recovery_time_s"]
                )
            ]

            peak_cm = (
                100 * np.mean(peaks)
                if peaks
                else float("nan")
            )

            excess_cm = (
                100 * np.mean(excess)
                if excess
                else float("nan")
            )

            mean_recovery = (
                np.mean(recovery)
                if recovery
                else float("nan")
            )

            print(
                f"{name:<13} "
                f"peak={peak_cm:.3f}cm "
                f"excess={excess_cm:.3f}cm "
                f"recovered={len(recovery)}/{len(group)} "
                f"mean_recovery={mean_recovery:.3f}s"
            )

    print()
    print(
        f"[FINISHED] {outfile}"
    )

    print(
        "Wind-like forces, not aerodynamic wind speeds."
    )


if __name__ == "__main__":
    main()
