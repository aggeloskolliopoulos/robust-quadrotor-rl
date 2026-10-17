import numpy as np

from quadrotor_rl.envs.quadrotor_tracking_env import (
    QuadrotorTrackingEnv,
)


class WaypointTrackingEnv(QuadrotorTrackingEnv):
    """
    Quadrotor environment for randomized waypoint regulation.

    Each episode:
        1. Starts near the nominal hover position.
        2. Samples a random target waypoint.
        3. Keeps that target fixed for the entire episode.

    The observation, action, reward, and failure definitions
    are inherited from QuadrotorTrackingEnv.
    """

    def __init__(
        self,
        gui: bool = False,
        physics_freq_hz: int = 240,
        control_freq_hz: int = 48,
    ):
        super().__init__(
            gui=gui,
            physics_freq_hz=physics_freq_hz,
            control_freq_hz=control_freq_hz,
        )

        # Waypoint sampling configuration.
        self.min_waypoint_radius = 0.30
        self.max_waypoint_radius = 0.60
        self.max_waypoint_z_offset = 0.15

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        """
        Reset the quadrotor and sample a new target waypoint.
        """

        # ---------------------------------------------------------
        # 1. Reset the original quadrotor environment.
        #
        # This handles:
        #   - Gymnasium RNG seeding
        #   - randomized initial position
        #   - randomized initial attitude
        #   - PyBullet simulation reset
        #
        # Initially the returned observation is computed using
        # the target from the preceding episode.
        # ---------------------------------------------------------

        previous_target = self.target_pos.copy()

        observation, info = super().reset(
            seed=seed,
            options=options,
        )

        # Recover the current position from:
        #
        # position_error = previous_target - position
        #
        # Therefore:
        # position = previous_target - position_error

        current_position = (
            previous_target - observation[0:3]
        )

        # ---------------------------------------------------------
        # 2. Sample a randomized target waypoint.
        #
        # The waypoint lies in a horizontal annulus around
        # the nominal hover position, with a small z offset.
        # ---------------------------------------------------------

        radius = self.np_random.uniform(
            self.min_waypoint_radius,
            self.max_waypoint_radius,
        )

        angle = self.np_random.uniform(
            -np.pi,
            np.pi,
        )

        z_offset = self.np_random.uniform(
            -self.max_waypoint_z_offset,
            self.max_waypoint_z_offset,
        )

        waypoint_offset = np.array(
            [
                radius * np.cos(angle),
                radius * np.sin(angle),
                z_offset,
            ],
            dtype=np.float32,
        )

        self.target_pos = (
            self.initial_pos + waypoint_offset
        ).astype(np.float32)

        self.target_vel = np.zeros(
            3,
            dtype=np.float32,
        )

        # ---------------------------------------------------------
        # 3. Update the position-error component of observation.
        #
        # The rest of the observation remains unchanged:
        #
        #   velocity error
        #   desired velocity
        #   quaternion
        #   angular velocity
        #
        # Since target velocity is still zero, those terms
        # do not need to change.
        # ---------------------------------------------------------

        observation = observation.copy()

        observation[0:3] = (
            self.target_pos - current_position
        )

        # ---------------------------------------------------------
        # 4. Return useful debugging information.
        # ---------------------------------------------------------

        info = dict(info)

        info["target_pos"] = self.target_pos.copy()

        return observation, info