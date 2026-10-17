"""
Zero-shot evaluation of sequential waypoint tracking.

Policy: SB3 hover SAC, trained for 80k steps.

Reference:
    0-4 s:  Random waypoint A
    4-10 s: Home waypoint B = [0, 0, 1]

No training takes place.
"""

import csv
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from quadrotor_rl.envs.sequential_waypoint_env import (
    SequentialWaypointEnv,
)


MODEL_PATH = (
    "results/checkpoints/sac_hover_stage1/"
    "sac_hover_80000_steps.zip"
)

EVAL_SEEDS = range(1100, 1200)

CONTROL_FREQ = 48
SWITCH_TIME_S = 4.0

POSITION_TOLERANCE = 0.10
VELOCITY_TOLERANCE = 0.15

HOLD_STEPS = round(0.5 * CONTROL_FREQ)

OUTPUT_PATH = Path(
    "results/evaluation/"
    "sequential_waypoints_zero_shot_episodes.csv"
)


def evaluate_episode(env, model, seed):

    observation, _ = env.reset(seed=seed)

    errors = {"A": [], "B": []}
    streak = {"A": 0, "B": 0}
    reached = {"A": False, "B": False}
    reach_time = {"A": None, "B": None}

    a_stable_at_switch = False
    target_switched = False

    terminated = False
    truncated = False

    step = 0

    while not (terminated or truncated):

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

        step += 1

        just_switched = info["target_switched"]

        # At the switching step, the returned observation
        # already contains the NEW target.
        #
        # But the completed step belongs to waypoint A.
        # Recover its position error relative to A.

        if just_switched:

            target_switched = True

            new_target = info["target_pos"]
            old_target = info["previous_target_pos"]

            position_error = (
                observation[:3]
                - (new_target - old_target)
            )

            phase = "A"

        else:

            position_error = observation[:3]

            phase = (
                "A"
                if info["waypoint_index"] == 0
                else "B"
            )

        distance = float(
            np.linalg.norm(position_error)
        )

        # Both targets are stationary.
        # Desired velocity is zero.
        speed = float(
            np.linalg.norm(observation[3:6])
        )

        errors[phase].append(distance)

        inside_tolerance = (
            distance < POSITION_TOLERANCE
            and speed < VELOCITY_TOLERANCE
        )

        if inside_tolerance:
            streak[phase] += 1
        else:
            streak[phase] = 0

        if (
            streak[phase] >= HOLD_STEPS
            and not reached[phase]
        ):

            reached[phase] = True

            absolute_time = step / CONTROL_FREQ

            # Time relative to activation of this target.
            if phase == "A":
                reach_time[phase] = absolute_time
            else:
                reach_time[phase] = (
                    absolute_time - SWITCH_TIME_S
                )

        if just_switched:
            a_stable_at_switch = (
                streak["A"] >= HOLD_STEPS
            )

        if step > 600:
            raise RuntimeError(
                "Episode exceeded expected horizon."
            )

    survived = bool(
        truncated and not terminated
    )

    b_stable_at_end = bool(
        survived
        and streak["B"] >= HOLD_STEPS
    )

    both_reached = bool(
        reached["A"] and reached["B"]
    )

    strict_success = bool(
        survived
        and a_stable_at_switch
        and b_stable_at_end
    )

    def phase_rmse(phase):
        if not errors[phase]:
            return None

        values = np.asarray(errors[phase])

        return float(
            np.sqrt(np.mean(values ** 2))
        )

    return {
        "seed": seed,
        "survived": int(survived),
        "target_switched": int(target_switched),
        "reached_A": int(reached["A"]),
        "reached_B": int(reached["B"]),
        "both_reached": int(both_reached),
        "A_stable_at_switch": int(
            a_stable_at_switch
        ),
        "B_stable_at_end": int(
            b_stable_at_end
        ),
        "strict_success": int(strict_success),
        "time_to_A_s": reach_time["A"],
        "time_to_B_s": reach_time["B"],
        "rmse_A_m": phase_rmse("A"),
        "rmse_B_m": phase_rmse("B"),
        "duration_s": step / CONTROL_FREQ,
    }


def main():

    model = SAC.load(
        MODEL_PATH,
        device="cpu",
    )

    env = SequentialWaypointEnv(
        gui=False,
        switch_time_s=SWITCH_TIME_S,
    )

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
            fieldnames=results[0].keys(),
        )

        writer.writeheader()
        writer.writerows(results)

    def percentage(key):
        return 100.0 * np.mean([
            row[key] for row in results
        ])

    def average_available(key):
        values = [
            row[key]
            for row in results
            if row[key] is not None
        ]

        return (
            float(np.mean(values))
            if values
            else float("nan")
        )

    print()
    print("=== Sequential Waypoint Zero-Shot ===")
    print(f"Episodes: {len(results)}")
    print()

    print(
        f"Survival: "
        f"{percentage('survived'):.1f}%"
    )

    print(
        f"Reached waypoint A: "
        f"{percentage('reached_A'):.1f}%"
    )

    print(
        f"Reached waypoint B: "
        f"{percentage('reached_B'):.1f}%"
    )

    print(
        f"Both reached: "
        f"{percentage('both_reached'):.1f}%"
    )

    print(
        f"A stable at switch: "
        f"{percentage('A_stable_at_switch'):.1f}%"
    )

    print(
        f"B stable at end: "
        f"{percentage('B_stable_at_end'):.1f}%"
    )

    print(
        f"Strict sequence success: "
        f"{percentage('strict_success'):.1f}%"
    )

    print()
    print(
        f"Mean time to A: "
        f"{average_available('time_to_A_s'):.2f} s"
    )

    print(
        f"Mean time to B (after switch): "
        f"{average_available('time_to_B_s'):.2f} s"
    )

    print(
        f"RMSE A: "
        f"{average_available('rmse_A_m'):.3f} m"
    )

    print(
        f"RMSE B: "
        f"{average_available('rmse_B_m'):.3f} m"
    )

    print()
    print(f"Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()