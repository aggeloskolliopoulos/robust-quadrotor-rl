import gymnasium as gym
import numpy as np

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics


class QuadrotorTrackingEnv(gym.Env):
    """Minimal Gymnasium environment for single-quadrotor trajectory tracking."""

    def __init__(
        self,
        gui: bool = False,
        physics_freq_hz: int = 240,
        control_freq_hz: int = 48,
    ):
        super().__init__()

        # -------------------------------------------------------------
        # Fixed hover task for the first environment-validation stage
        # -------------------------------------------------------------
        self.target_pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.target_vel = np.zeros(3, dtype=np.float32)

        self.initial_pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.initial_rpy = np.zeros(3, dtype=np.float32)

        # -------------------------------------------------------------
        # Initial-state randomization for hover stabilization training
        #
        # Position:
        #     x, y: ±0.10 m around the nominal hover position
        #     z:    ±0.10 m around the nominal hover altitude
        #
        # Attitude:
        #     roll, pitch: ±5 degrees
        #     yaw:         ±10 degrees
        #
        # Initial linear and angular velocities remain zero for now.
        # -------------------------------------------------------------

        self.position_perturbation = np.array(
            [0.10, 0.10, 0.10],
            dtype=np.float32,
        )

        self.rpy_perturbation = np.deg2rad(
            np.array([5.0, 5.0, 10.0], dtype=np.float32)
        )
        self.physics_freq_hz = physics_freq_hz
        self.control_freq_hz = control_freq_hz

        self._step_count = 0
        self.episode_duration_s = 10.0
        self.max_steps = int(
            self.episode_duration_s * self.control_freq_hz
        )

        # -------------------------------------------------------------
        # Low-level simulator
        # -------------------------------------------------------------
        self.sim = CtrlAviary(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            initial_xyzs=self.initial_pos.reshape(1, 3),
            initial_rpys=self.initial_rpy.reshape(1, 3),
            physics=Physics.PYB,
            pyb_freq=self.physics_freq_hz,
            ctrl_freq=self.control_freq_hz,
            gui=gui,
            record=False,
            obstacles=False,
            user_debug_gui=False,
        )

        # Use the simulator's actual actuator constants.
        self.hover_rpm = float(self.sim.HOVER_RPM)
        self.max_rpm = float(self.sim.MAX_RPM)

        # -------------------------------------------------------------
        # RL action space:
        # one normalized command for each of the four motors
        # -------------------------------------------------------------
        self.action_space = gym.spaces.Box(
            low=np.float32(-1.0),
            high=np.float32(1.0),
            shape=(4,),
            dtype=np.float32,
        )

        # -------------------------------------------------------------
        # Observation:
        #
        # [position_error(3),
        #  velocity_error(3),
        #  desired_velocity(3),
        #  quaternion(4),
        #  angular_velocity(3)]
        #
        # Total = 16
        #
        # Quaternion order from PyBullet is [qx, qy, qz, qw].
        # We will canonicalize it later so that qw >= 0.
        # -------------------------------------------------------------
        obs_low = np.array(
            [-np.inf] * 9
            + [-1.0, -1.0, -1.0, 0.0]
            + [-np.inf] * 3,
            dtype=np.float32,
        )

        obs_high = np.array(
            [np.inf] * 9
            + [1.0, 1.0, 1.0, 1.0]
            + [np.inf] * 3,
            dtype=np.float32,
        )

        self.observation_space = gym.spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32,
        )

    def _get_reference(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Return the desired position and velocity at time t."""

        return self.target_pos.copy(), self.target_vel.copy()

    def _construct_observation(
        self,
        raw_state: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """Construct the 16D RL observation from the simulator state."""

        raw_state = np.asarray(raw_state)

        if raw_state.shape != (20,):
            raise ValueError(
                f"Expected raw state shape (20,), got {raw_state.shape}"
            )

        # Extract physical state from CtrlAviary's 20D state vector.
        position = raw_state[0:3]
        quaternion = raw_state[3:7].copy()
        velocity = raw_state[10:13]
        angular_velocity = raw_state[13:16]

        # Desired trajectory/reference at the same timestamp.
        target_pos, target_vel = self._get_reference(t)

        position_error = target_pos - position
        velocity_error = target_vel - velocity

        # q and -q represent the same physical orientation.
        # Choose a unique convention whenever possible: qw >= 0.
        if quaternion[3] < 0.0:
            quaternion = -quaternion

        observation = np.concatenate(
            [
                position_error,
                velocity_error,
                target_vel,
                quaternion,
                angular_velocity,
            ]
        ).astype(np.float32)

        return observation

    def _is_failure(
        self,
        raw_state: np.ndarray,
        t: float,
    ) -> bool:
        """Return True if the current state should terminate the episode."""

        raw_state = np.asarray(raw_state)

        if raw_state.shape != (20,):
            raise ValueError(
                f"Expected raw state shape (20,), got {raw_state.shape}"
            )

        # Numerical failure.
        if not np.all(np.isfinite(raw_state)):
            return True

        position = raw_state[0:3]
        rpy = raw_state[7:10]

        target_pos, _ = self._get_reference(t)
        position_error = target_pos - position

        # Drone is effectively on / extremely close to the ground.
        if position[2] < 0.10:
            return True

        # Drone has moved too far away from the reference.
        if np.linalg.norm(position_error) > 2.0:
            return True

        # Excessive roll or pitch means loss of attitude control.
        max_tilt = np.deg2rad(80.0)

        if abs(rpy[0]) > max_tilt or abs(rpy[1]) > max_tilt:
            return True

        return False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        """Reset the simulator with a randomized initial hover state."""

        # -------------------------------------------------------------
        # 1. Initialize Gymnasium's random-number generator.
        #
        # Using self.np_random makes reset(seed=...) reproducible:
        #
        #     reset(seed=42)
        #     reset(seed=42)
        #
        # should generate the same initial state.
        # -------------------------------------------------------------
        super().reset(seed=seed)

        # -------------------------------------------------------------
        # 2. Sample a small position perturbation around the nominal
        #    initial hover position.
        #
        # Each component is sampled independently and uniformly:
        #
        #     x, y, z perturbation ∈ [-0.10, +0.10] m
        # -------------------------------------------------------------
        position_offset = self.np_random.uniform(
            low=-self.position_perturbation,
            high=self.position_perturbation,
        )

        sampled_initial_pos = (
            self.initial_pos + position_offset
        ).astype(np.float32)

        # -------------------------------------------------------------
        # 3. Sample a small attitude perturbation.
        #
        # roll, pitch ∈ [-5°, +5°]
        # yaw         ∈ [-10°, +10°]
        #
        # Internally everything is stored in radians.
        # -------------------------------------------------------------
        rpy_offset = self.np_random.uniform(
            low=-self.rpy_perturbation,
            high=self.rpy_perturbation,
        )

        sampled_initial_rpy = (
            self.initial_rpy + rpy_offset
        ).astype(np.float32)

        # -------------------------------------------------------------
        # 4. Update the initial pose used internally by CtrlAviary.
        #
        # BaseAviary.reset() reloads the drone using INIT_XYZS and
        # INIT_RPYS, so these must be updated BEFORE self.sim.reset().
        # -------------------------------------------------------------
        self.sim.INIT_XYZS = sampled_initial_pos.reshape(1, 3)
        self.sim.INIT_RPYS = sampled_initial_rpy.reshape(1, 3)

        # -------------------------------------------------------------
        # 5. Reset the underlying PyBullet simulator.
        #
        # The newly sampled pose is now used when the drone URDF is
        # loaded again.
        # -------------------------------------------------------------
        raw_obs, _ = self.sim.reset(
            seed=seed,
            options=options,
        )

        # -------------------------------------------------------------
        # 6. Reset the RL/control-step counter.
        # -------------------------------------------------------------
        self._step_count = 0

        # Immediately after reset the simulation time is t = 0.
        t = 0.0

        # -------------------------------------------------------------
        # 7. Construct the 16D RL observation from the REAL simulator
        #    state returned after reset.
        # -------------------------------------------------------------
        observation = self._construct_observation(
            raw_state=raw_obs[0],
            t=t,
        )

        # -------------------------------------------------------------
        # 8. Return useful reset diagnostics.
        #
        # These values are not part of the RL observation. They are
        # useful for debugging and reproducibility checks.
        # -------------------------------------------------------------
        info = {
            "initial_pos": sampled_initial_pos.copy(),
            "initial_rpy": sampled_initial_rpy.copy(),
        }

        return observation, info

    def step(self, action: np.ndarray):
        """Advance the quadrotor by one control step."""

        # -------------------------------------------------------------
        # 1. Convert normalized RL action to physical motor RPMs
        #
        # RL action:
        #     shape = (4,)
        #     each value in [-1, 1]
        #
        # Mapping:
        #     -1 -> 0 RPM
        #      0 -> HOVER_RPM
        #     +1 -> MAX_RPM
        # -------------------------------------------------------------
        rpm = self._action_to_rpm(action)

        # -------------------------------------------------------------
        # 2. Advance the low-level simulator by one control interval
        #
        # CtrlAviary expects actions with shape:
        #     (num_drones, 4)
        #
        # Since we have one drone:
        #     (4,) -> (1, 4)
        #
        # Internally, CtrlAviary keeps this RPM command constant for
        # PYB_STEPS_PER_CTRL physics steps.
        #
        # With:
        #     physics_freq_hz = 240
        #     control_freq_hz = 48
        #
        # this corresponds to 5 physics steps per RL/control step.
        # -------------------------------------------------------------
        raw_obs, _, _, _, _ = self.sim.step(
            rpm.reshape(1, 4)
        )

        # -------------------------------------------------------------
        # 3. Update the RL/control-step counter
        #
        # This counter counts control steps, not PyBullet physics steps.
        # -------------------------------------------------------------
        self._step_count += 1

        # -------------------------------------------------------------
        # 4. Compute the current simulation time
        #
        # After k control steps:
        #     t = k / control_freq_hz
        #
        # Example:
        #     after the first step:
        #     t = 1 / 48 ≈ 0.020833 s
        # -------------------------------------------------------------
        t = self._step_count / self.control_freq_hz

        # -------------------------------------------------------------
        # 5. Convert the simulator's raw 20D state into our 16D
        #    RL observation
        #
        # Observation:
        #     [position_error,
        #      velocity_error,
        #      desired_velocity,
        #      quaternion,
        #      angular_velocity]
        #
        # The reference is evaluated at the same timestamp as the
        # returned simulator state.
        # -------------------------------------------------------------
        observation = self._construct_observation(
            raw_state=raw_obs[0],
            t=t,
        )

        # -------------------------------------------------------------
        # 6. Check whether the episode ended because of failure
        #
        # Failure conditions currently include:
        #     - invalid / non-finite state
        #     - very low altitude
        #     - excessive distance from target
        #     - excessive roll or pitch
        #
        # We compute termination before the reward because the reward
        # includes an additional terminal penalty when failure occurs.
        # -------------------------------------------------------------
        terminated = self._is_failure(
            raw_state=raw_obs[0],
            t=t,
        )

        # -------------------------------------------------------------
        # 7. Compute the dense RL reward
        #
        # The reward penalizes:
        #     - position tracking error
        #     - velocity tracking error
        #     - angular velocity
        #     - deviation of motor commands from the hover action
        #
        # If terminated=True, _compute_reward() also adds the
        # additional failure penalty.
        # -------------------------------------------------------------
        reward = self._compute_reward(
            observation=observation,
            action=action,
            terminated=terminated,
        )

        # -------------------------------------------------------------
        # 8. Check whether the episode ended normally due to time limit
        #
        # Reaching max_steps is NOT a failure.
        # Therefore:
        #     truncated = True
        #
        # We additionally require "not terminated" so that if a failure
        # happens exactly at the time limit, we classify the episode as
        # failure rather than normal truncation.
        # -------------------------------------------------------------
        truncated = (
            self._step_count >= self.max_steps
            and not terminated
        )

        # -------------------------------------------------------------
        # 9. Extra diagnostic information
        #
        # This does not become part of the RL observation.
        # It is useful for debugging, evaluation and logging.
        # -------------------------------------------------------------
        info = {
            "time": t,
            "rpm": rpm.copy(),
        }

        # -------------------------------------------------------------
        # Gymnasium step API:
        #
        # observation : next observation
        # reward      : scalar reward
        # terminated  : episode ended because of failure/task condition
        # truncated   : episode ended because of time limit
        # info        : auxiliary diagnostics
        # -------------------------------------------------------------
        return observation, reward, terminated, truncated, info

    def _action_to_rpm(self, action: np.ndarray) -> np.ndarray:
        """Map normalized motor commands in [-1, 1] to physical RPM values."""

        action = np.asarray(action, dtype=np.float32)

        if action.shape != (4,):
            raise ValueError(
                f"Expected action shape (4,), got {action.shape}"
            )

        if np.any(action < -1.0) or np.any(action > 1.0):
            raise ValueError(
                f"Action must lie in [-1, 1], got {action}"
            )

        rpm = np.where(
            action <= 0.0,
            (action + 1.0) * self.hover_rpm,
            self.hover_rpm
            + action * (self.max_rpm - self.hover_rpm),
        )

        return np.clip(rpm, 0.0, self.max_rpm).astype(np.float32)

    def _compute_reward(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        terminated: bool,
    ) -> float:
        """Compute the dense tracking reward for one control step."""

        observation = np.asarray(observation, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)

        if observation.shape != (16,):
            raise ValueError(
                f"Expected observation shape (16,), got {observation.shape}"
            )

        if action.shape != (4,):
            raise ValueError(
                f"Expected action shape (4,), got {action.shape}"
            )

        # -------------------------------------------------------------
        # Extract the quantities used by the reward.
        #
        # observation =
        # [position_error(3),
        #  velocity_error(3),
        #  desired_velocity(3),
        #  quaternion(4),
        #  angular_velocity(3)]
        # -------------------------------------------------------------
        position_error = observation[0:3]
        velocity_error = observation[3:6]
        angular_velocity = observation[13:16]

        # -------------------------------------------------------------
        # Reward weights.
        #
        # Tracking error is the main objective.
        # Velocity error, angular motion and actuator deviations from
        # hover are penalized more weakly.
        # -------------------------------------------------------------
        w_position = 1.0
        w_velocity = 0.1
        w_angular_velocity = 0.01
        w_action = 0.01

        # -------------------------------------------------------------
        # Dense quadratic penalties.
        #
        # Since action = 0 corresponds to HOVER_RPM, the action penalty
        # penalizes deviations from the nominal hover operating point,
        # not the hover thrust itself.
        # -------------------------------------------------------------
        reward = (
            -w_position * np.sum(position_error ** 2)
            -w_velocity * np.sum(velocity_error ** 2)
            -w_angular_velocity * np.sum(angular_velocity ** 2)
            -w_action * np.sum(action ** 2)
        )

        # -------------------------------------------------------------
        # Additional penalty for an actual failure.
        #
        # Normal time-limit truncation does not receive this penalty.
        # -------------------------------------------------------------
        if terminated:
            reward -= 10.0

        return float(reward)

    def close(self):
        self.sim.close()