import numpy as np


def compute_position_rmse(position_errors):
    """
    Compute per-axis and overall 3D position RMSE.

    Parameters
    ----------
    position_errors : array-like, shape (N, 3)
        Position tracking errors [ex, ey, ez] for N time steps.

    Returns
    -------
    rmse_xyz : np.ndarray, shape (3,)
        RMSE independently for x, y, and z.

    rmse_3d : float
        Overall Euclidean position RMSE.
    """
    errors = np.asarray(position_errors)

    rmse_xyz = np.sqrt(
        np.mean(errors**2, axis=0)
    )

    rmse_3d = np.sqrt(
        np.mean(np.sum(errors**2, axis=1))
    )

    return rmse_xyz, rmse_3d