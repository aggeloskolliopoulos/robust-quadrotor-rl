import time
from pathlib import Path

from stable_baselines3 import SAC

from quadrotor_rl.envs.quadrotor_tracking_env import QuadrotorTrackingEnv


MODEL_PATH = Path(
    "results/checkpoints/sac_hover_stage1/"
    "sac_hover_80000_steps.zip"
)

# Already-used evaluation seeds.
WATCH_SEEDS = [1000, 1001, 1005]


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    # -------------------------------------------------------------
    # Load the trained SAC policy.
    # -------------------------------------------------------------
    model = SAC.load(
        str(MODEL_PATH),
        device="cpu",
    )

    # -------------------------------------------------------------
    # GUI-enabled environment.
    # -------------------------------------------------------------
    env = QuadrotorTrackingEnv(
        gui=True,
    )

    print("\nWatching Stage-1 SAC hover controller")
    print("=====================================")
    print(f"Model: {MODEL_PATH}\n")

    try:
        for seed in WATCH_SEEDS:
            obs, info = env.reset(seed=seed)

            print(f"\nSeed {seed}")
            print("----------------------")
            print(
                "Initial position:",
                info["initial_pos"],
            )
            print(
                "Initial RPY [rad]:",
                info["initial_rpy"],
            )

            terminated = False
            truncated = False
            step_count = 0
            episode_return = 0.0

            while not (terminated or truncated):
                # -------------------------------------------------
                # Deterministic controller:
                # no stochastic SAC exploration during evaluation.
                # -------------------------------------------------
                action, _ = model.predict(
                    obs,
                    deterministic=True,
                )

                obs, reward, terminated, truncated, info = env.step(
                    action
                )

                episode_return += reward
                step_count += 1

                # -------------------------------------------------
                # PyBullet simulation normally runs as fast as the
                # computer allows.
                #
                # Sleeping for one control period makes the GUI play
                # approximately in real time:
                #
                #     1 / 48 s per control step
                # -------------------------------------------------
                time.sleep(
                    1.0 / env.control_freq_hz
                )

            print(
                f"Finished after "
                f"{step_count / env.control_freq_hz:.2f} s"
            )
            print(
                f"Return: {episode_return:.3f}"
            )
            print(
                f"terminated={terminated}, "
                f"truncated={truncated}"
            )

            # Small pause between episodes so the reset is visible.
            time.sleep(1.0)

    finally:
        env.close()


if __name__ == "__main__":
    main()