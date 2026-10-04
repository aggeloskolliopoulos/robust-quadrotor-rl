import numpy as np


def compute_position_rmse(position_errors):
    """
    Compute per-axis and overall 3D position RMSE.

    Parameters
    ----------
    position_errors : array-like, shape (N, 3)
        Position tracking errors [ex, ey, ez].

    Returns
    -------
    rmse_xyz : np.ndarray, shape (3,)
        RMSE for x, y and z.

    rmse_3d : float
        Overall 3D position RMSE.
    """
    errors = np.asarray(position_errors, dtype=float)

    rmse_xyz = np.sqrt(
        np.mean(errors**2, axis=0)
    )

    rmse_3d = np.sqrt(
        np.mean(
            np.sum(errors**2, axis=1)
        )
    )

    return rmse_xyz, rmse_3d


def compute_max_position_error(position_errors):
    """
    Compute the maximum 3D position error magnitude.

    Parameters
    ----------
    position_errors : array-like, shape (N, 3)
        Position tracking errors.

    Returns
    -------
    max_error : float
        Maximum Euclidean position error.
    """
    errors = np.asarray(position_errors, dtype=float)

    error_norms = np.linalg.norm(
        errors,
        axis=1,
    )

    return np.max(error_norms)


def compute_control_effort(actions, hover_rpm):
    """
    Compute normalized control effort.

    The metric measures the mean squared deviation
    of the motor commands from hover RPM.

    Parameters
    ----------
    actions : array-like, shape (N, 4)
        Applied motor RPM commands.

    hover_rpm : float
        Hover RPM of the quadrotor.

    Returns
    -------
    effort : float
        Dimensionless normalized control-effort metric.
    """
    actions = np.asarray(actions, dtype=float)

    normalized_actions = (
        actions - hover_rpm
    ) / hover_rpm

    effort = np.mean(
        np.sum(
            normalized_actions**2,
            axis=1,
        )
    )

    return effort


def compute_action_smoothness(actions, hover_rpm):
    """
    Compute normalized action variation between
    consecutive control steps.

    Parameters
    ----------
    actions : array-like, shape (N, 4)
        Applied motor RPM commands.

    hover_rpm : float
        Hover RPM used for normalization.

    Returns
    -------
    smoothness_cost : float
        Mean squared normalized change in motor commands.
        Lower values indicate smoother actuator commands.
    """
    actions = np.asarray(actions, dtype=float)

    action_differences = np.diff(
        actions,
        axis=0,
    )

    normalized_differences = (
        action_differences / hover_rpm
    )

    smoothness_cost = np.mean(
        np.sum(
            normalized_differences**2,
            axis=1,
        )
    )

    return smoothness_cost
