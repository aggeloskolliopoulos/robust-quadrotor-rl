import numpy as np


def hover_reference(position):
    """
    Constant hover reference.

    Parameters
    ----------
    position : array-like, shape (3,)
        Desired fixed hover position [x, y, z].

    Returns
    -------
    target_pos : np.ndarray, shape (3,)
        Desired position.

    target_vel : np.ndarray, shape (3,)
        Desired linear velocity.
    """
    target_pos = np.asarray(position, dtype=float)
    target_vel = np.zeros(3)

    return target_pos, target_vel


def circle_reference(t, center, radius, angular_speed, altitude):
    """
    Circular trajectory in the horizontal x-y plane.

    Parameters
    ----------
    t : float
        Time in seconds.

    center : array-like, shape (2,)
        Circle center [cx, cy].

    radius : float
        Circle radius in meters.

    angular_speed : float
        Angular speed in rad/s.

    altitude : float
        Constant z altitude.

    Returns
    -------
    target_pos : np.ndarray, shape (3,)
        Desired position.

    target_vel : np.ndarray, shape (3,)
        Desired linear velocity.
    """
    center = np.asarray(center, dtype=float)

    theta = angular_speed * t

    target_pos = np.array([
        center[0] + radius * np.cos(theta),
        center[1] + radius * np.sin(theta),
        altitude,
    ])

    target_vel = np.array([
        -radius * angular_speed * np.sin(theta),
        radius * angular_speed * np.cos(theta),
        0.0,
    ])

    return target_pos, target_vel

def waypoint_reference(t, waypoints, hold_time):
    """
    Piecewise-constant waypoint reference.

    Parameters
    ----------
    t : float
        Current time in seconds.

    waypoints : array-like, shape (N, 3)
        Sequence of desired XYZ positions.

    hold_time : float
        Time in seconds to hold each waypoint.

    Returns
    -------
    target_pos : np.ndarray, shape (3,)
        Current desired position.

    target_vel : np.ndarray, shape (3,)
        Desired velocity. Zero for waypoint regulation.

    waypoint_index : int
        Index of the active waypoint.
    """
    waypoints = np.asarray(waypoints, dtype=float)

    waypoint_index = min(
        int(t // hold_time),
        len(waypoints) - 1,
    )

    target_pos = waypoints[waypoint_index]
    target_vel = np.zeros(3)

    return target_pos, target_vel, waypoint_index