import numpy as np

from quadrotor_rl.envs.waypoint_tracking_env import (
    WaypointTrackingEnv,
)


class ThreeWaypointEnv(WaypointTrackingEnv):
    """
    Three consecutive waypoint targets in a 10-second episode.

    0-4 seconds:
        A = random waypoint

    4-7 seconds:
        B = home position

    7-10 seconds:
        C = symmetric of A around home

    Uses the existing observation, action and reward.
    """

    def __init__(
        self,
        gui=False,
        physics_freq_hz=240,
        control_freq_hz=48,
    ):
        super().__init__(
            gui=gui,
            physics_freq_hz=physics_freq_hz,
            control_freq_hz=control_freq_hz,
        )

        self.switch_steps = (
            4 * control_freq_hz,
            7 * control_freq_hz,
        )

        self.elapsed_steps = 0
        self.waypoint_index = 0
        self.waypoints = None

    def reset(self, *, seed=None, options=None):

        observation, info = super().reset(
            seed=seed,
            options=options,
        )

        home = np.asarray(
            self.initial_pos,
            dtype=np.float32,
        ).copy()

        waypoint_a = self.target_pos.copy()

        waypoint_b = home

        waypoint_c = (
            2.0 * home - waypoint_a
        ).astype(np.float32)

        self.waypoints = np.stack(
            [
                waypoint_a,
                waypoint_b,
                waypoint_c,
            ]
        ).astype(np.float32)

        self.elapsed_steps = 0
        self.waypoint_index = 0

        info = dict(info)

        info["waypoint_index"] = 0
        info["target_switched"] = False
        info["target_pos"] = self.target_pos.copy()

        return observation, info

    def step(self, action):

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = super().step(action)

        self.elapsed_steps += 1
        info = dict(info)

        completed_index = self.waypoint_index
        switched = False

        # Switch only if the episode is still active.
        if not terminated and not truncated:

            if (
                self.waypoint_index < 2
                and self.elapsed_steps
                == self.switch_steps[self.waypoint_index]
            ):

                old_target = self.target_pos.copy()

                self.waypoint_index += 1

                new_target = self.waypoints[
                    self.waypoint_index
                ].copy()

                self.target_pos = new_target

                self.target_vel = np.zeros(
                    3,
                    dtype=np.float32,
                )

                # Parent returned observation relative to
                # the previous target. Correct position error:
                #
                # e_new = e_old + (new_target - old_target)

                observation = observation.copy()

                observation[:3] += (
                    new_target - old_target
                )

                switched = True

                info["previous_target_pos"] = old_target

        info["target_switched"] = switched

        # Target used for the control step just completed.
        info["completed_waypoint_index"] = completed_index

        # Target to be used for the next control step.
        info["waypoint_index"] = self.waypoint_index

        info["target_pos"] = self.target_pos.copy()

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )