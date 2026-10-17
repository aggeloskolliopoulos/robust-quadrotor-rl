import numpy as np

from quadrotor_rl.envs.circle_tracking_env import (
    CircleTrackingEnv,
)


class RandomizedCircleTrackingEnv(CircleTrackingEnv):
    """
    Continuous circular tracking with randomized angular speed.

    Each episode:
        1. Randomizes the initial quadrotor state.
        2. Samples omega uniformly from [omega_min, omega_max].
        3. Keeps omega constant throughout the episode.

    Circle radius and center remain fixed.

    Observation: 16D
    Action: 4D

    Reward and termination conditions are inherited.
    """

    def __init__(
        self,
        gui=False,
        physics_freq_hz=240,
        control_freq_hz=48,
        radius_m=0.30,
        omega_min=0.20,
        omega_max=1.00,
    ):

        if (
            not np.isfinite(omega_min)
            or not np.isfinite(omega_max)
            or omega_min <= 0
            or omega_max < omega_min
        ):
            raise ValueError(
                "Require 0 < omega_min <= omega_max"
            )

        super().__init__(
            gui=gui,
            physics_freq_hz=physics_freq_hz,
            control_freq_hz=control_freq_hz,
            radius_m=radius_m,
            angular_speed_rad_s=omega_min,
        )

        self.omega_min = float(omega_min)
        self.omega_max = float(omega_max)

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):

        # ----------------------------------------------------
        # 1. Reset the parent environment.
        #
        # This also initializes Gymnasium's random generator.
        #
        # The parent initially constructs an observation using
        # the angular speed from the preceding episode.
        # ----------------------------------------------------

        observation, info = super().reset(
            seed=seed,
            options=options,
        )

        old_target_pos = self.target_pos.copy()
        old_target_vel = self.target_vel.copy()

        # ----------------------------------------------------
        # 2. Sample a new angular speed.
        #
        # Use Gymnasium's seeded RNG for reproducibility.
        # ----------------------------------------------------

        self.angular_speed_rad_s = float(
            self.np_random.uniform(
                self.omega_min,
                self.omega_max,
            )
        )

        # ----------------------------------------------------
        # 3. Recompute the initial reference.
        #
        # At t=0:
        #
        # p_d = center + [R, 0, 0]
        # v_d = [0, R*omega, 0]
        #
        # Changing omega changes desired velocity.
        # ----------------------------------------------------

        self.elapsed_steps = 0

        new_target_pos, new_target_vel = (
            self.circle_reference(0.0)
        )

        self.target_pos = new_target_pos
        self.target_vel = new_target_vel

        # ----------------------------------------------------
        # 4. Correct the observation.
        #
        # Observation:
        #
        # [0:3] = desired_position - actual_position
        # [3:6] = desired_velocity - actual_velocity
        # [6:9] = desired_velocity
        #
        # The parent observation used the OLD omega.
        # We must immediately replace that reference.
        # ----------------------------------------------------

        observation = observation.copy()

        observation[0:3] += (
            new_target_pos - old_target_pos
        )

        observation[3:6] += (
            new_target_vel - old_target_vel
        )

        observation[6:9] = new_target_vel

        # ----------------------------------------------------
        # 5. Add experiment information.
        # ----------------------------------------------------

        info = dict(info)

        info["angular_speed_rad_s"] = (
            self.angular_speed_rad_s
        )

        info["target_pos"] = (
            self.target_pos.copy()
        )

        info["target_vel"] = (
            self.target_vel.copy()
        )

        info["reference_time_s"] = 0.0

        return observation, info