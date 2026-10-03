import csv
import time
import numpy as np
from pathlib import Path
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.utils import sync
from quadrotor_rl.evaluation.metrics import compute_position_rmse
from quadrotor_rl.tasks.trajectories import hover_reference

# Experiment configuration

DRONE_MODEL = DroneModel.CF2X

PHYSICS_FREQ_HZ = 240
CONTROL_FREQ_HZ = 48
DURATION_SEC = 10

TARGET_POS = np.array([0.0, 0.0, 1.0])

INITIAL_POS = np.array([[0.0, 0.0, 0.5]])
INITIAL_RPY = np.array([[0.0, 0.0, 0.0]])

GUI = True

def main():
    # Create the simulation environment
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

    # Reset the simulator and get the initial observation
    obs, info = env.reset()

    # Get the constant hover reference
    target_pos, target_vel = hover_reference(TARGET_POS)

    # Create the PID controller
    controller = DSLPIDControl(drone_model=DRONE_MODEL)

    # Allocate the action array expected by CtrlAviary: shape (1, 4)
    action = np.zeros((1, 4))

    # Compute the first motor command from the initial state
    action[0], position_error, yaw_error = controller.computeControlFromState(
        control_timestep=env.CTRL_TIMESTEP,
        state=obs[0],
        target_pos=target_pos,
        target_vel=target_vel,
        target_rpy=np.zeros(3),
    )

    print("Observation shape:", obs.shape)
    print("Initial position:", obs[0, 0:3])
    print("Control timestep:", env.CTRL_TIMESTEP)
    print("Physics timestep:", env.PYB_TIMESTEP)
    print("Physics steps per control step:", env.PYB_STEPS_PER_CTRL)

    print("First PID motor RPMs:", action[0])
    print("Position error:", position_error)
    print("Yaw error:", yaw_error)
    print("Hover RPM:", env.HOVER_RPM)

    # Containers for experiment data
    log_rows = []
    position_errors = []

    # Number of controller updates during the experiment
    num_control_steps = int(DURATION_SEC * CONTROL_FREQ_HZ)

    start_time = time.time()

    for step in range(num_control_steps):

        # Save the RPM command that is about to be applied
        applied_action = action[0].copy()

        # Apply the current motor RPMs to the simulator
        obs, reward, terminated, truncated, info = env.step(action)

        # State of the single drone after this control interval
        state = obs[0]

        position = state[0:3]

        # Position tracking error: target - actual
        position_error = target_pos - position

        # Time corresponding to the new state
        timestamp = (step + 1) * env.CTRL_TIMESTEP

        # Store experiment data
        log_rows.append(
            [
                timestamp,
                position[0],
                position[1],
                position[2],
                target_pos[0],
                target_pos[1],
                target_pos[2],
                position_error[0],
                position_error[1],
                position_error[2],
                applied_action[0],
                applied_action[1],
                applied_action[2],
                applied_action[3],
            ]
        )

        position_errors.append(position_error.copy())

        # Compute the RPM command for the next control interval
        action[0], _, _ = controller.computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=state,
            target_pos=target_pos,
            target_vel=target_vel,
            target_rpy=np.zeros(3),
        )

        # Print one status update per second
        if step % CONTROL_FREQ_HZ == 0:
            print(
                f"t = {timestamp:.1f} s | "
                f"position = {position} | "
                f"error = {position_error}"
            )

        if GUI:
            sync(step, start_time, env.CTRL_TIMESTEP)

    rmse_xyz, position_rmse = compute_position_rmse(position_errors)

    print("\nPID hover results")
    print("-----------------")
    print(f"RMSE x: {rmse_xyz[0]:.6f} m")
    print(f"RMSE y: {rmse_xyz[1]:.6f} m")
    print(f"RMSE z: {rmse_xyz[2]:.6f} m")
    print(f"3D position RMSE: {position_rmse:.6f} m")

    # Create results directory if it does not already exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    csv_path = results_dir / "pid_hover_transient.csv"

    header = [
        "time",
        "x",
        "y",
        "z",
        "target_x",
        "target_y",
        "target_z",
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
    print("Final position error:", target_pos - obs[0, 0:3])

    env.close()


if __name__ == "__main__":
    main()