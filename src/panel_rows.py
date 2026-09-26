"""
panel_rows.py

Purpose:
    Helper functions for defining approximate solar panel row locations and
    converting them from image pixel coordinates to ground-grid coordinates.

Main uses:
    - Store manually selected panel row endpoints.
    - Convert panel row endpoints from image pixels to world coordinates.
    - Match projected panel row endpoints to the nearest ground-grid cells.
    - Overlay panel rows on final ground-grid shadow maps.

Coordinate systems:
    Image coordinates:
        u, v are pixel coordinates.
        u increases left to right.
        v increases top to bottom.

    World / ground coordinates:
        X, Y are measured in centimeters on the ground plane.
        Z is assumed to be 0.

Notes:
    The panel row endpoints are manually selected in image pixel coordinates.
    To update them for a different image/camera view, use:

        findCoordinates.py

    The helper script lets you click on an image and record pixel coordinates.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Default panel row endpoint coordinates
# ---------------------------------------------------------------------

# Approximate solar panel row endpoints in image pixel coordinates.
#
# These values can be changed if the camera view changes or if the panel
# row lines need to be adjusted.
#
# To find new coordinates, use:
#
#     findCoordinates.py
#
# Each row is defined by a start pixel and an end pixel:
#     (u_start, v_start) -> (u_end, v_end)
#
# Coordinates are in pixels:
#     u = horizontal pixel coordinate
#     v = vertical pixel coordinate
DEFAULT_PANEL_ROWS_PIXELS = [
    {
        "row_id": 1,
        "u_start": 226,
        "v_start": 1050,
        "u_end": 1207,
        "v_end": 664,
    },
    {
        "row_id": 2,
        "u_start": 1280,
        "v_start": 1097,
        "u_end": 1763,
        "v_end": 636,
    },
    {
        "row_id": 3,
        "u_start": 2645,
        "v_start": 1140,
        "u_end": 2236,
        "v_end": 648,
    },
]


def find_nearest_grid_cell(
    X_cm: float,
    Y_cm: float,
    grid_df: pd.DataFrame,
) -> pd.Series:
    """
    Finds the nearest ground-grid cell center to a world coordinate.

    Parameters
    ----------
    X_cm : float
        World X coordinate in centimeters.
    Y_cm : float
        World Y coordinate in centimeters.
    grid_df : pd.DataFrame
        Ground grid created by create_ground_grid().

    Returns
    -------
    pd.Series
        Row from grid_df corresponding to the nearest grid cell.

    Notes
    -----
    This is used after panel row endpoints are projected onto the ground plane.
    The nearest grid cell lets the panel rows be drawn on grid-based maps.
    """

    temp = grid_df.copy()

    # Squared distance is enough for nearest-neighbor comparison.
    # No need to take square root because it preserves the same ordering.
    temp["dist2"] = (
        (temp["x_center_cm"] - X_cm) ** 2
        + (temp["y_center_cm"] - Y_cm) ** 2
    )

    nearest = temp.loc[temp["dist2"].idxmin()]

    return nearest


def create_panel_rows_grid(
    panel_rows_pixels: list[dict],
    grid_df: pd.DataFrame,
    pixel_to_world_fn,
    K: np.ndarray,
    dist: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> pd.DataFrame:
    """
    Converts panel row endpoints from image pixels to ground-grid coordinates.

    Parameters
    ----------
    panel_rows_pixels : list of dict
        Panel row endpoints in image pixel coordinates.
    grid_df : pd.DataFrame
        Ground grid created by create_ground_grid().
    pixel_to_world_fn : callable
        Function that converts one pixel point to world coordinates on
        the ground plane.
    K : np.ndarray
        Camera intrinsic matrix.
    dist : np.ndarray
        Camera distortion coefficients.
    R : np.ndarray
        Camera rotation matrix.
    t : np.ndarray
        Camera translation vector.

    Returns
    -------
    pd.DataFrame
        Table containing panel row endpoints in both world coordinates and
        nearest grid-cell coordinates.

    Output columns
    --------------
    row_id : int
        Panel row ID.
    start_world_x_cm, start_world_y_cm : float
        Start endpoint in world coordinates.
    end_world_x_cm, end_world_y_cm : float
        End endpoint in world coordinates.
    start_cell_x, start_cell_y : int
        Nearest grid cell for the start endpoint.
    end_cell_x, end_cell_y : int
        Nearest grid cell for the end endpoint.

    Notes
    -----
    The output is mainly used for overlaying panel row locations on final
    ground-grid maps.
    """

    panel_rows_grid_list = []

    for row in panel_rows_pixels:
        # Convert the start endpoint from image pixel coordinates to
        # world coordinates on the ground plane.
        X1, Y1 = pixel_to_world_fn(
            row["u_start"],
            row["v_start"],
            K,
            dist,
            R,
            t,
            z_plane=0.0,
        )

        # Convert the end endpoint from image pixel coordinates to
        # world coordinates on the ground plane.
        X2, Y2 = pixel_to_world_fn(
            row["u_end"],
            row["v_end"],
            K,
            dist,
            R,
            t,
            z_plane=0.0,
        )

        # Find the nearest grid cells so the row can be shown on grid maps.
        start_cell = find_nearest_grid_cell(X1, Y1, grid_df)
        end_cell = find_nearest_grid_cell(X2, Y2, grid_df)

        panel_rows_grid_list.append(
            {
                "row_id": row["row_id"],
                "start_world_x_cm": X1,
                "start_world_y_cm": Y1,
                "end_world_x_cm": X2,
                "end_world_y_cm": Y2,
                "start_cell_x": int(start_cell["cell_x"]),
                "start_cell_y": int(start_cell["cell_y"]),
                "end_cell_x": int(end_cell["cell_x"]),
                "end_cell_y": int(end_cell["cell_y"]),
            }
        )

    return pd.DataFrame(panel_rows_grid_list)