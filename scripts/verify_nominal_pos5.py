"""
Held-out SAC vs PID benchmark.

Includes:
    1. Independent circular tracking evaluation
    2. Hover retention evaluation
    3. Position-error transient plots
    4. Paired bootstrap confidence intervals

No training or model modification.
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import SAC

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel

from quadrotor_rl.envs.circle_tracking_env import CircleTrackingEnv
from quadrotor_rl.envs.quadrotor_tracking_env import QuadrotorTrackingEnv

# Reuse the SAME PID implementation as our previous benchmark.
from eval_controllers import pid_action


# ============================================================
# 1. Configuration
# ============================================================

OUT = Path(
    "results/evaluation/pos5_independent_confirmation"
)

OMEGAS = (0.2, 0.4, 0.6, 0.8, 1.0)

CIRCLE_SEEDS = range(2000, 2050)
HOVER_SEEDS = range(2100, 2150)

# Representative speeds for transient plots.
PLOT_OMEGAS = (0.4, 1.0)

MODEL_PATHS = {
    "Hover_80k": (
        "results/checkpoints/sac_hover_stage1/"
        "sac_hover_80000_steps.zip"
    ),

    "Pos5_100k": (
        "results/checkpoints/sac_reward_position5/"
        "sac_pos5_102000_steps.zip"
    ),
}


# ============================================================
# 2. Run a single evaluation episode
# ============================================================

def evaluate_episode(env, policy, seed):

    obs, _ = env.reset(seed=seed)

    # The PID controller has internal state.
    # Create a fresh instance for every episode.
    pid = (
        DSLPIDControl(drone_model=DroneModel.CF2X)
        if policy is None
        else None
    )

    errors = []
    vel_errors = []

    # Include the t=0 observation in plots.
    # RMSE metrics use post-step observations only.
    trace = [
        float(np.linalg.norm(obs[:3]))
    ]

    terminated = False
    truncated = False

    for _ in range(env.max_steps):

        if pid is None:

            action, _ = policy.predict(
                obs,
                deterministic=True,
            )

        else:

            action, _ = pid_action(pid, env)

        obs, _, terminated, truncated, _ = (
            env.step(action)
        )

        error = float(
            np.linalg.norm(obs[:3])
        )

        errors.append(error)

        vel_errors.append(
            float(np.linalg.norm(obs[3:6]))
        )

        trace.append(error)

        if terminated or truncated:
            break

    survived = bool(
        truncated
        and not terminated
        and len(errors) == env.max_steps
    )

    midpoint = env.max_steps // 2

    def rmse(values):

        if not values:
            return float("nan")

        return float(
            np.sqrt(np.mean(np.square(values)))
        )

    # Tracking RMSE is reported only for completed episodes.
    # Failures are counted separately.
    result = {
        "seed": seed,
        "survived": int(survived),

        "duration_s":
            len(errors) / env.control_freq_hz,

        "full_rmse_m":
            rmse(errors) if survived else float("nan"),

        "early_rmse_m":
            rmse(errors[:midpoint])
            if survived else float("nan"),

        "late_rmse_m":
            rmse(errors[midpoint:])
            if survived else float("nan"),

        "velocity_rmse_m_s":
            rmse(vel_errors)
            if survived else float("nan"),
    }

    return (
        result,
        np.asarray(trace) if survived else None,
    )


# ============================================================
# 3. Save results
# ============================================================

def save_csv(path, rows):

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 4. Print summary
# ============================================================

def print_summary(rows, names):

    for name in names:

        selected = [
            row
            for row in rows
            if row["controller"] == name
        ]

        survivors = [
            row
            for row in selected
            if row["survived"]
        ]

        survival = (
            100.0 * len(survivors) / len(selected)
        )

        def metric(key):

            if not survivors:
                return float("nan")

            return float(
                np.mean([
                    row[key]
                    for row in survivors
                ])
            )

        print(
            f"{name:12s} | "
            f"survival={survival:5.1f}% | "
            f"full={metric('full_rmse_m'):.4f}m | "
            f"early={metric('early_rmse_m'):.4f}m | "
            f"late={metric('late_rmse_m'):.4f}m | "
            f"velocity="
            f"{metric('velocity_rmse_m_s'):.4f}m/s",
            flush=True,
        )


# ============================================================
# 5. Plot error evolution over time
# ============================================================

def plot_transient(traces, output_path, title):

    plt.figure(figsize=(9, 5))

    for name, episodes in traces.items():

        if not episodes:
            continue

        arr = np.stack(
            episodes,
            axis=0,
        )

        t = np.arange(
            arr.shape[1]
        ) / 48.0

        # Pointwise RMS over evaluation seeds.
        instantaneous_rms = np.sqrt(
            np.mean(arr ** 2, axis=0)
        )

        plt.plot(
            t,
            instantaneous_rms * 100,
            label=f"{name} (n={len(arr)})",
        )

    plt.axvline(
        5.0,
        color="gray",
        linestyle="--",
        alpha=0.65,
        label="Late interval begins",
    )

    plt.xlabel("Time (s)")
    plt.ylabel(
        "Position error across seeds, RMS (cm)"
    )
    plt.title(title)

    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=160,
    )

    plt.close()


# ============================================================
# 6. Paired confidence interval: Pos5 vs PID
# ============================================================

def paired_bootstrap(
    circle_rows,
    key,
    n_boot=4000,
):

    indexed = {
        (
            row["controller"],
            row["omega_rad_s"],
            row["seed"],
        ): row

        for row in circle_rows
    }

    seed_mean_differences = []

    for seed in CIRCLE_SEEDS:

        differences = []

        for omega in OMEGAS:

            sac = indexed[
                ("Pos5_100k", omega, seed)
            ]

            pid = indexed[
                ("PID", omega, seed)
            ]

            if not (
                sac["survived"]
                and pid["survived"]
            ):

                print(
                    "Paired confidence interval unavailable: "
                    "at least one failed episode."
                )

                return

            differences.append(
                sac[key] - pid[key]
            )

        # Average across speeds for this seed.
        seed_mean_differences.append(
            np.mean(differences)
        )

    differences = np.asarray(
        seed_mean_differences
    )

    rng = np.random.default_rng(2026)

    indices = rng.integers(
        0,
        len(differences),
        size=(n_boot, len(differences)),
    )

    bootstrap_means = (
        differences[indices].mean(axis=1)
    )

    low, high = np.percentile(
        bootstrap_means,
        (2.5, 97.5),
    )

    print(
        f"Paired Pos5 - PID {key}: "
        f"{differences.mean()*100:+.3f} cm "
        f"(95% bootstrap CI "
        f"[{low*100:+.3f}, "
        f"{high*100:+.3f}] cm)"
    )


# ============================================================
# 7. Main
# ============================================================

def main():

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load policies
    # --------------------------------------------------------

    policies = {}

    for name, filename in MODEL_PATHS.items():

        if not Path(filename).is_file():
            raise FileNotFoundError(filename)

        policies[name] = SAC.load(
            filename,
            device="cpu",
        )

    # None represents our conventional PID controller.
    policies["PID"] = None

    names = list(policies)

    # ========================================================
    # STEP 1: Independent circular tracking
    # ========================================================

    circle_rows = []

    circle_traces = {
        omega: {
            name: []
            for name in names
        }
        for omega in PLOT_OMEGAS
    }

    print()
    print("=== HELD-OUT CIRCLE TRACKING ===")

    for omega in OMEGAS:

        env = CircleTrackingEnv(
            gui=False,
            physics_freq_hz=240,
            control_freq_hz=48,
            radius_m=0.30,
            angular_speed_rad_s=omega,
        )

        try:

            for name, policy in policies.items():

                for seed in CIRCLE_SEEDS:

                    row, trace = evaluate_episode(
                        env,
                        policy,
                        seed,
                    )

                    circle_rows.append({
                        "controller": name,
                        "omega_rad_s": omega,
                        **row,
                    })

                    if (
                        omega in circle_traces
                        and trace is not None
                    ):

                        circle_traces[omega][name].append(
                            trace
                        )

        finally:

            env.close()

        print()
        print(f"omega={omega:.1f} rad/s")

        speed_rows = [
            row
            for row in circle_rows
            if row["omega_rad_s"] == omega
        ]

        print_summary(
            speed_rows,
            names,
        )

    print()
    print("=== CIRCLE OVERALL ===")

    print_summary(
        circle_rows,
        names,
    )

    paired_bootstrap(
        circle_rows,
        "late_rmse_m",
    )

    paired_bootstrap(
        circle_rows,
        "full_rmse_m",
    )

    save_csv(
        OUT / "circle_episodes.csv",
        circle_rows,
    )

    for omega, traces in circle_traces.items():

        plot_transient(
            traces,
            OUT / (
                f"circle_transient_omega_{omega:.1f}.png"
            ),
            f"Circle tracking transient, "
            f"omega={omega:.1f} rad/s",
        )

    # ========================================================
    # STEP 2: Hover retention
    # ========================================================

    print()
    print("=== HELD-OUT HOVER RETENTION ===")

    hover_rows = []

    hover_traces = {
        name: []
        for name in names
    }

    env = QuadrotorTrackingEnv(
        gui=False,
        physics_freq_hz=240,
        control_freq_hz=48,
    )

    try:

        for name, policy in policies.items():

            for seed in HOVER_SEEDS:

                row, trace = evaluate_episode(
                    env,
                    policy,
                    seed,
                )

                hover_rows.append({
                    "controller": name,
                    **row,
                })

                if trace is not None:

                    hover_traces[name].append(
                        trace
                    )

    finally:

        env.close()

    print_summary(
        hover_rows,
        names,
    )

    save_csv(
        OUT / "hover_episodes.csv",
        hover_rows,
    )

    plot_transient(
        hover_traces,
        OUT / "hover_transient.png",
        "Hover retention and transient",
    )

    print()
    print("Evaluation completed.")
    print(f"Results saved in: {OUT}")

    print(
        "RMSE is averaged per successful episode. "
        "Failures are reported separately."
    )


if __name__ == "__main__":
    main()