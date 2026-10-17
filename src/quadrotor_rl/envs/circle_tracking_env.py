import numpy as np

from quadrotor_rl.envs.quadrotor_tracking_env import (
    QuadrotorTrackingEnv,
)


class CircleTrackingEnv(QuadrotorTrackingEnv):
    """
    Continuous circular reference tracking.

    The desired position and velocity evolve over time:

        x_d(t) = cx + R*cos(omega*t)
        y_d(t) = cy + R*sin(omega*t)
        z_d(t) = cz

        vx_d(t) = -R*omega*sin(omega*t)
        vy_d(t) =  R*omega*cos(omega*t)
        vz_d(t) = 0

    Observation: 16D (unchanged).
    Action:       4D (unchanged).
    Reward:       inherited from QuadrotorTrackingEnv.
    """

    def __init__(
        self,
        gui=False,
        physics_freq_hz=240,
        control_freq_hz=48,
        radius_m=0.30,
        angular_speed_rad_s=0.40,
    ):
        super().__init__(
            gui=gui,
            physics_freq_hz=physics_freq_hz,
            control_freq_hz=control_freq_hz,
        )

        if radius_m <= 0:
            raise ValueError("radius_m must be positive")

        if angular_speed_rad_s <= 0:
            raise ValueError(
                "angular_speed_rad_s must be positive"
            )

        self.radius_m = float(radius_m)
        self.angular_speed_rad_s = float(
            angular_speed_rad_s
        )

        self.circle_center = np.asarray(
            self.initial_pos,
            dtype=np.float32,
        ).copy()

        self.elapsed_steps = 0

    # ========================================================
    # Desired circular reference
    # ========================================================

    def circle_reference(self, t):

        theta = self.angular_speed_rad_s * t

        r = self.radius_m
        omega = self.angular_speed_rad_s

        position = self.circle_center + np.array(
            [
                r * np.cos(theta),
                r * np.sin(theta),
                0.0,
            ],
            dtype=np.float32,
        )

        velocity = np.array(
            [
                -r * omega * np.sin(theta),
                 r * omega * np.cos(theta),
                 0.0,
            ],
            dtype=np.float32,
        )

        return (
            position.astype(np.float32),
            velocity,
        )

    # ========================================================
    # Reset
    # ========================================================

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):

        observation, info = super().reset(
            seed=seed,
            options=options,
        )

        # Reference used to construct the parent observation.
        old_target_pos = self.target_pos.copy()
        old_target_vel = self.target_vel.copy()

        self.elapsed_steps = 0

        # Initial circle reference at t=0.
        new_target_pos, new_target_vel = (
            self.circle_reference(0.0)
        )

        self.target_pos = new_target_pos
        self.target_vel = new_target_vel

        # The parent observation contains:
        #
        # obs[0:3] = target_pos - current_pos
        # obs[3:6] = target_vel - current_vel
        # obs[6:9] = target_vel
        #
        # Convert the observation to the new reference.

        observation = observation.copy()

        observation[0:3] += (
            new_target_pos - old_target_pos
        )

        observation[3:6] += (
            new_target_vel - old_target_vel
        )

        observation[6:9] = new_target_vel

        info = dict(info)

        info["reference_time_s"] = 0.0
        info["target_pos"] = self.target_pos.copy()
        info["target_vel"] = self.target_vel.copy()

        return observation, info

    # ========================================================
    # Step
    # ========================================================

    def step(self, action):

        # Compute the reference at the NEXT control instant.
        #
        # This makes the reward and next observation use
        # the same reference p_d(t+dt), v_d(t+dt).

        next_step = self.elapsed_steps + 1

        next_time_s = (
            next_step / self.control_freq_hz
        )

        next_target_pos, next_target_vel = (
            self.circle_reference(next_time_s)
        )

        # The inherited environment reads these reference
        # values while computing its next observation/reward.

        self.target_pos = next_target_pos
        self.target_vel = next_target_vel

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = super().step(action)

        self.elapsed_steps = next_step

        info = dict(info)

        info["reference_time_s"] = next_time_s
        info["target_pos"] = self.target_pos.copy()
        info["target_vel"] = self.target_vel.copy()

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )