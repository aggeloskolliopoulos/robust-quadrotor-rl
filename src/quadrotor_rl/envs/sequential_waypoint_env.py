import numpy as np

from quadrotor_rl.envs.waypoint_tracking_env import (
    WaypointTrackingEnv,
)


class SequentialWaypointEnv(WaypointTrackingEnv):
    """
    Two-waypoint tracking environment.

    Waypoint 0:
        Random waypoint sampled at reset.

    Waypoint 1:
        Nominal hover position [0, 0, 1].

    Reference switches after 4 seconds.

    Total episode duration remains 10 seconds.
    """

    def __init__(
        self,
        gui=False,
        physics_freq_hz=240,
        control_freq_hz=48,
        switch_time_s=4.0,
    ):
        super().__init__(
            gui=gui,
            physics_freq_hz=physics_freq_hz,
            control_freq_hz=control_freq_hz,
        )

        self.switch_time_s = switch_time_s

        self.switch_step = round(
            switch_time_s * control_freq_hz
        )

        self.elapsed_steps = 0
        self.waypoint_index = 0

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        # Parent samples the first random waypoint.
        observation, info = super().reset(
            seed=seed,
            options=options,
        )

        self.elapsed_steps = 0
        self.waypoint_index = 0

        info = dict(info)

        info["waypoint_index"] = 0
        info["target_switched"] = False
        info["target_pos"] = self.target_pos.copy()

        return observation, info

    def step(self, action):

        # Perform the physics/control step under the
        # currently active reference.
        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = super().step(action)

        self.elapsed_steps += 1

        info = dict(info)
        target_switched = False

        # ----------------------------------------------------
        # Switch reference after 4 seconds.
        # ----------------------------------------------------

        if (
            self.elapsed_steps == self.switch_step
            and not terminated
            and not truncated
        ):

            old_target = self.target_pos.copy()

            # Second waypoint = home position.
            self.target_pos = np.asarray(
                self.initial_pos,
                dtype=np.float32,
            ).copy()

            self.target_vel = np.zeros(
                3,
                dtype=np.float32,
            )

            self.waypoint_index = 1
            target_switched = True

            # ------------------------------------------------
            # IMPORTANT:
            #
            # The observation returned by the parent step
            # corresponds to the OLD target.
            #
            # Position error:
            #
            #   e_old = old_target - position
            #
            #   e_new = new_target - position
            #
            # Therefore:
            #
            #   e_new = e_old + (new_target - old_target)
            #
            # We must update the observation immediately
            # so that the actor receives the NEW reference
            # at its next decision.
            # ------------------------------------------------

            observation = observation.copy()

            observation[0:3] += (
                self.target_pos - old_target
            )

            info["previous_target_pos"] = (
                old_target.copy()
            )

        info["target_switched"] = target_switched

        info["waypoint_index"] = (
            self.waypoint_index
        )

        info["target_pos"] = (
            self.target_pos.copy()
        )

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )