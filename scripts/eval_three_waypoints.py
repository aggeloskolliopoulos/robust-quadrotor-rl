"""
Evaluate zero-shot three-waypoint tracking.

Reference:
    A: 0-4 seconds
    B: 4-7 seconds
    C: 7-10 seconds

Policy:
    Previously trained SB3 SAC hover controller.

No training takes place.
"""

import csv
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from quadrotor_rl.envs.three_waypoint_env import ThreeWaypointEnv


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = (
    "results/checkpoints/sac_hover_stage1/"
    "sac_hover_80000_steps.zip"
)

EVAL_SEEDS = range(1100, 1200)

CONTROL_FREQ = 48

SWITCH_STEPS = (192, 336)
PHASE_START_STEPS = (0, 192, 336)

MAX_EPISODE_STEPS = 480

POSITION_TOLERANCE = 0.10
VELOCITY_TOLERANCE = 0.15

HOLD_STEPS = round(0.5 * CONTROL_FREQ)

OUTPUT_PATH = Path(
    "results/evaluation/"
    "three_waypoints_zero_shot_episodes.csv"
)


# ============================================================
# Evaluate one episode
# ============================================================

def evaluate_episode(env, model, seed):

    observation, _ = env.reset(seed=seed)

    errors = [[], [], []]

    stable_steps = [0, 0, 0]
    reached = [False, False, False]
    reach_times = [None, None, None]

    stable_at_switch = [False, False]

    terminated = False
    truncated = False

    step = 0
    num_switches = 0

    while not (terminated or truncated):

        # ----------------------------------------------------
        # 1. Deterministic SAC action
        # ----------------------------------------------------

        action, _ = model.predict(
            observation,
            deterministic=True,
        )

        # ----------------------------------------------------
        # 2. Environment step
        # ----------------------------------------------------

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        step += 1

        # Target used during the step just completed.
        phase = info["completed_waypoint_index"]

        assert phase in (0, 1, 2)

        # ----------------------------------------------------
        # 3. Position error
        #
        # On a switching step, the returned observation
        # refers to the NEW waypoint.
        #
        # Recover the OLD target error because the completed
        # physical step belongs to the previous phase.
        # ----------------------------------------------------

        position_error = observation[:3].copy()

        if info["target_switched"]:

            num_switches += 1

            assert phase in (0, 1)

            assert step == SWITCH_STEPS[phase]

            old_target = info["previous_target_pos"]
            new_target = info["target_pos"]

            position_error -= (
                new_target - old_target
            )

        distance = float(
            np.linalg.norm(position_error)
        )

        errors[phase].append(distance)

        # ----------------------------------------------------
        # 4. Speed
        #
        # All three waypoints are stationary.
        # Therefore target_velocity = 0.
        #
        # ||velocity_error|| = ||actual_velocity||
        # ----------------------------------------------------

        speed = float(
            np.linalg.norm(observation[3:6])
        )

        # ----------------------------------------------------
        # 5. Stabilization criterion
        # ----------------------------------------------------

        inside_tolerance = (
            distance < POSITION_TOLERANCE
            and speed < VELOCITY_TOLERANCE
        )

        if inside_tolerance:
            stable_steps[phase] += 1
        else:
            stable_steps[phase] = 0

        if (
            stable_steps[phase] >= HOLD_STEPS
            and not reached[phase]
        ):

            reached[phase] = True

            # Time since activation of the phase.
            reach_times[phase] = (
                step - PHASE_START_STEPS[phase]
            ) / CONTROL_FREQ

        # ----------------------------------------------------
        # 6. Stability at the reference switch
        # ----------------------------------------------------

        if info["target_switched"]:

            stable_at_switch[phase] = (
                stable_steps[phase] >= HOLD_STEPS
            )

        if step > MAX_EPISODE_STEPS:
            raise RuntimeError(
                "Episode exceeded expected horizon."
            )

    # ========================================================
    # Episode summary
    # ========================================================

    survived = bool(
        truncated and not terminated
    )

    c_stable_at_end = bool(
        survived
        and stable_steps[2] >= HOLD_STEPS
    )

    all_reached = all(reached)

    strict_success = bool(
        survived
        and stable_at_switch[0]
        and stable_at_switch[1]
        and c_stable_at_end
    )

    def calculate_rmse(values):

        if not values:
            return None

        arr = np.asarray(
            values,
            dtype=np.float64,
        )

        return float(
            np.sqrt(np.mean(arr ** 2))
        )

    return {
        "seed": seed,

        "survived": int(survived),
        "num_switches": num_switches,

        "reached_A": int(reached[0]),
        "reached_B": int(reached[1]),
        "reached_C": int(reached[2]),

        "all_reached": int(all_reached),

        "A_stable_at_switch": int(
            stable_at_switch[0]
        ),

        "B_stable_at_switch": int(
            stable_at_switch[1]
        ),

        "C_stable_at_end": int(
            c_stable_at_end
        ),

        "strict_success": int(strict_success),

        "time_to_A_s": reach_times[0],
        "time_to_B_s": reach_times[1],
        "time_to_C_s": reach_times[2],

        "rmse_A_m": calculate_rmse(errors[0]),
        "rmse_B_m": calculate_rmse(errors[1]),
        "rmse_C_m": calculate_rmse(errors[2]),

        "final_error_m": float(
            np.linalg.norm(observation[:3])
        ),

        "duration_s": step / CONTROL_FREQ,
    }


# ============================================================
# Main
# ============================================================

def main():

    model = SAC.load(
        MODEL_PATH,
        device="cpu",
    )

    env = ThreeWaypointEnv(gui=False)

    results = []

    try:

        for seed in EVAL_SEEDS:

            result = evaluate_episode(
                env,
                model,
                seed,
            )

            results.append(result)

    finally:

        env.close()

    # --------------------------------------------------------
    # Save episode-level results
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )

        writer.writeheader()
        writer.writerows(results)

    # --------------------------------------------------------
    # Summary helpers
    # --------------------------------------------------------

    def percentage(key):

        return 100.0 * float(
            np.mean([
                row[key] for row in results
            ])
        )

    def mean_available(key):

        values = [
            row[key]
            for row in results
            if row[key] is not None
        ]

        if not values:
            return float("nan")

        return float(np.mean(values))

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print()
    print("=== Three-Waypoint Zero-Shot Evaluation ===")
    print(f"Episodes: {len(results)}")
    print()

    print(
        f"Survival: "
        f"{percentage('survived'):.1f}%"
    )

    print(
        f"Reached A: "
        f"{percentage('reached_A'):.1f}%"
    )

    print(
        f"Reached B: "
        f"{percentage('reached_B'):.1f}%"
    )

    print(
        f"Reached C: "
        f"{percentage('reached_C'):.1f}%"
    )

    print(
        f"All three reached: "
        f"{percentage('all_reached'):.1f}%"
    )

    print()
    print(
        f"A stable at 4s: "
        f"{percentage('A_stable_at_switch'):.1f}%"
    )

    print(
        f"B stable at 7s: "
        f"{percentage('B_stable_at_switch'):.1f}%"
    )

    print(
        f"C stable at end: "
        f"{percentage('C_stable_at_end'):.1f}%"
    )

    print()
    print(
        f"STRICT SEQUENCE SUCCESS: "
        f"{percentage('strict_success'):.1f}%"
    )

    print()

    for phase in ("A", "B", "C"):

        print(
            f"Waypoint {phase}: "
            f"time={mean_available('time_to_' + phase + '_s'):.2f}s, "
            f"RMSE={mean_available('rmse_' + phase + '_m'):.3f}m"
        )

    print()
    print(
        f"Mean final error: "
        f"{mean_available('final_error_m'):.3f}m"
    )

    print(
        f"Mean duration: "
        f"{mean_available('duration_s'):.2f}s"
    )

    print()
    print(f"Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()