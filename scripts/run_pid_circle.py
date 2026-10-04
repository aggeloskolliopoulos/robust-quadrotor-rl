import csv
import time
from pathlib import Path

import numpy as np

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.utils import sync

from quadrotor_rl.evaluation.metrics import (
    compute_position_rmse,
    compute_max_position_error,
    compute_control_effort,
    compute_action_smoothness,
)
from quadrotor_rl.tasks.trajectories import circle_reference


# ---------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------

DRONE_MODEL = DroneModel.CF2X

PHYSICS_FREQ_HZ = 240
CONTROL_FREQ_HZ = 48

# Circle parameters
CIRCLE_CENTER = np.array([0.0, 0.0])
CIRCLE_RADIUS = 0.5
CIRCLE_ALTITUDE = 1.0

CIRCLE_PERIOD = 8.0
CIRCLE_ANGULAR_SPEED = 2 * np.pi / CIRCLE_PERIOD

# Two complete circles
DURATION_SEC = 16

# At t = 0:
# x = center_x + R
# y = center_y
# z = altitude
INITIAL_POS = np.array([
    [
        CIRCLE_CENTER[0] + CIRCLE_RADIUS,
        CIRCLE_CENTER[1],
        CIRCLE_ALTITUDE,
    ]
])

INITIAL_RPY = np.array([[0.0, 0.0, 0.0]])

GUI = True


def main():

    # ---------------------------------------------------------
    # Create simulation environment
    # ---------------------------------------------------------

    env = CtrlAviary(
        drone_model=DRONE_MODEL,
        num_drones=1,
        initial_xyzs=INITIAL_POS,
        initial_rpys=INITIAL_RPY,
        physics=Physics.PYB,
        pyb_freq=PHYSICS_FREQ_HZ,
        ctrl_freq=CONTROL_FREQ_HZ,
        gui=GUI,
        record=False,
        obstacles=False,
        user_debug_gui=False,
    )

    # Reset simulator and get initial observation
    obs, info = env.reset()

    # ---------------------------------------------------------
    # Initial circle reference at t = 0
    # ---------------------------------------------------------

    target_pos, target_vel = circle_reference(
        t=0.0,
        center=CIRCLE_CENTER,
        radius=CIRCLE_RADIUS,
        angular_speed=CIRCLE_ANGULAR_SPEED,
        altitude=CIRCLE_ALTITUDE,
    )

    # Create PID controller
    controller = DSLPIDControl(drone_model=DRONE_MODEL)

    # CtrlAviary expects action shape (1, 4)
    action = np.zeros((1, 4))

    # Compute first motor command
    action[0], position_error, yaw_error = controller.computeControlFromState(
        control_timestep=env.CTRL_TIMESTEP,
        state=obs[0],
        target_pos=target_pos,
        target_vel=target_vel,
        target_rpy=np.zeros(3),
    )

    # ---------------------------------------------------------
    # Initial diagnostic information
    # ---------------------------------------------------------

    print("Observation shape:", obs.shape)
    print("Initial position:", obs[0, 0:3])

    print("Control timestep:", env.CTRL_TIMESTEP)
    print("Physics timestep:", env.PYB_TIMESTEP)
    print("Physics steps per control step:", env.PYB_STEPS_PER_CTRL)

    print("Initial target position:", target_pos)
    print("Initial target velocity:", target_vel)

    print("First PID motor RPMs:", action[0])
    print("Initial position error:", position_error)
    print("Yaw error:", yaw_error)
    print("Hover RPM:", env.HOVER_RPM)

    print("Circle radius:", CIRCLE_RADIUS)
    print("Circle period:", CIRCLE_PERIOD)
    print("Circle angular speed:", CIRCLE_ANGULAR_SPEED)
    print(
        "Circle linear speed:",
        CIRCLE_RADIUS * CIRCLE_ANGULAR_SPEED,
    )

    # ---------------------------------------------------------
    # Experiment data
    # ---------------------------------------------------------

    log_rows = []
    position_errors = []
    applied_actions = []

    num_control_steps = int(
        DURATION_SEC * CONTROL_FREQ_HZ
    )

    start_time = time.time()

    # ---------------------------------------------------------
    # Main control loop
    # ---------------------------------------------------------

    for step in range(num_control_steps):

        # Save the motor command that will be applied
        # during this control interval
        applied_action = action[0].copy()
        applied_actions.append(
            applied_action.copy()
        )
        # Advance the physics simulation
        obs, reward, terminated, truncated, info = env.step(action)

        # State at the END of this control interval
        state = obs[0]
        position = state[0:3]

        # Current simulation time
        timestamp = (step + 1) * env.CTRL_TIMESTEP

        # Circle reference at the SAME timestamp as the new state
        target_pos, target_vel = circle_reference(
            t=timestamp,
            center=CIRCLE_CENTER,
            radius=CIRCLE_RADIUS,
            angular_speed=CIRCLE_ANGULAR_SPEED,
            altitude=CIRCLE_ALTITUDE,
        )

        # Tracking error at the current timestamp
        position_error = target_pos - position

        # -----------------------------------------------------
        # Log current state, reference and applied motor command
        # -----------------------------------------------------

        log_rows.append(
            [
                timestamp,
                position[0],
                position[1],
                position[2],
                target_pos[0],
                target_pos[1],
                target_pos[2],
                target_vel[0],
                target_vel[1],
                target_vel[2],
                position_error[0],
                position_error[1],
                position_error[2],
                applied_action[0],
                applied_action[1],
                applied_action[2],
                applied_action[3],
            ]
        )

        position_errors.append(
            position_error.copy()
        )

        # -----------------------------------------------------
        # Compute RPM command for the NEXT control interval
        # -----------------------------------------------------

        action[0], _, _ = controller.computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=state,
            target_pos=target_pos,
            target_vel=target_vel,
            target_rpy=np.zeros(3),
        )

        # Print approximately once per second
        if (step + 1) % CONTROL_FREQ_HZ == 0:
            print(
                f"t = {timestamp:.1f} s | "
                f"position = {position} | "
                f"target = {target_pos} | "
                f"target_vel = {target_vel}"
            )

        if GUI:
            sync(
                step,
                start_time,
                env.CTRL_TIMESTEP,
            )

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    rmse_xyz, position_rmse = compute_position_rmse(
        position_errors
    )
    max_position_error = compute_max_position_error(
        position_errors
    )

    control_effort = compute_control_effort(
        applied_actions,
        env.HOVER_RPM,
    )

    action_smoothness = compute_action_smoothness(
        applied_actions,
        env.HOVER_RPM,
    )

    print("\nPID circle results")
    print("------------------")

    print(f"RMSE x: {rmse_xyz[0]:.6f} m")
    print(f"RMSE y: {rmse_xyz[1]:.6f} m")
    print(f"RMSE z: {rmse_xyz[2]:.6f} m")
    print(f"3D position RMSE: {position_rmse:.6f} m")

    print(
        f"Maximum position error: "
        f"{max_position_error:.6f} m"
    )

    print(
        f"Normalized control effort: "
        f"{control_effort:.6f}"
    )

    print(
        f"Normalized action smoothness cost: "
        f"{action_smoothness:.6f}"
    )
    
    # ---------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    csv_path = results_dir / "pid_circle.csv"

    header = [
        "time",
        "x",
        "y",
        "z",
        "target_x",
        "target_y",
        "target_z",
        "target_vx",
        "target_vy",
        "target_vz",
        "error_x",
        "error_y",
        "error_z",
        "rpm_1",
        "rpm_2",
        "rpm_3",
        "rpm_4",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(header)
        writer.writerows(log_rows)

    print(f"\nSaved log to: {csv_path}")

    print("\nFinal position:", obs[0, 0:3])
    print("Final target position:", target_pos)
    print(
        "Final position error:",
        target_pos - obs[0, 0:3],
    )

    env.close()


if __name__ == "__main__":
    main()