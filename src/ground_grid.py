"""
ground_grid.py

Purpose:
    Helper functions for creating a ground-coordinate grid and sampling
    shadow results inside each grid cell.

Main uses:
    - Create a regular grid on the ground plane.
    - Project each grid cell into the camera image.
    - Calculate shadow probability and binary shadow coverage per cell.

Coordinate system:
    Ground/world coordinates are in centimeters.
    Image coordinates are in pixels.

Notes:
    This module assumes that each ground cell lies on the plane Z = 0.
    The actual projection from ground coordinates to image pixels is passed in
    as world_to_pixel_fn, usually from camera_geometry.py.
"""

import cv2
import numpy as np
import pandas as pd


def create_ground_grid(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    cell_size: float,
) -> pd.DataFrame:
    """
    Creates a rectangular grid on the ground plane.

    Parameters
    ----------
    x_min : float
        Minimum X coordinate in centimeters.
    x_max : float
        Maximum X coordinate in centimeters.
    y_min : float
        Minimum Y coordinate in centimeters.
    y_max : float
        Maximum Y coordinate in centimeters.
    cell_size : float
        Width and height of each grid cell in centimeters.

    Returns
    -------
    pd.DataFrame
        One row per grid cell, including cell indices, cell bounds,
        and cell center coordinates.

    Notes
    -----
    cell_x and cell_y are grid indices.
    x0_cm, x1_cm, y0_cm, y1_cm are physical bounds in centimeters.
    """

    cells = []

    x_edges = np.arange(x_min, x_max + cell_size, cell_size)
    y_edges = np.arange(y_min, y_max + cell_size, cell_size)

    for iy in range(len(y_edges) - 1):
        for ix in range(len(x_edges) - 1):
            x0 = x_edges[ix]
            x1 = x_edges[ix + 1]
            y0 = y_edges[iy]
            y1 = y_edges[iy + 1]

            cells.append(
                {
                    "cell_x": ix,
                    "cell_y": iy,
                    "x0_cm": x0,
                    "x1_cm": x1,
                    "y0_cm": y0,
                    "y1_cm": y1,
                    "x_center_cm": (x0 + x1) / 2,
                    "y_center_cm": (y0 + y1) / 2,
                }
            )

    return pd.DataFrame(cells)


def process_one_image_to_ground_grid(
    image_name: str,
    shadow_prob: np.ndarray,
    pred_shadow: np.ndarray,
    ground_mask: np.ndarray,
    grid_df: pd.DataFrame,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    image_shape: tuple,
    world_to_pixel_fn,
    min_visible_fraction: float = 0.2,
    binary_cell_threshold: float = 0.1,
) -> pd.DataFrame:
    """
    Projects each ground-grid cell into one image and computes shadow statistics.

    Parameters
    ----------
    image_name : str
        Base image name used for the output table.
    shadow_prob : np.ndarray
        2D shadow probability map for this image.
    pred_shadow : np.ndarray
        Binary predicted shadow mask for this image.
    ground_mask : np.ndarray
        Binary mask showing valid visible ground pixels.
    grid_df : pd.DataFrame
        Ground grid created by create_ground_grid().
    rvec : np.ndarray
        Camera rotation vector.
    tvec : np.ndarray
        Camera translation vector.
    K : np.ndarray
        Camera intrinsic matrix.
    dist : np.ndarray
        Camera distortion coefficients.
    image_shape : tuple
        Shape of the original image.
    world_to_pixel_fn : callable
        Function that projects world ground coordinates to image pixels.
    min_visible_fraction : float
        Minimum fraction of projected cell pixels that must be visible inside
        the ground mask for the cell to be analyzed.
    binary_cell_threshold : float
        Cell is marked as shaded if the fraction of binary shadow pixels inside
        the visible cell area is above this threshold.

    Returns
    -------
    pd.DataFrame
        One row per grid cell with visibility and shadow statistics.

    Notes
    -----
    The shadow statistics are calculated only using pixels that are both:
        1. inside the projected grid cell
        2. inside the valid ground mask

    This prevents panels, sky, image edges, and non-ground regions from
    affecting the per-cell estimates.
    """

    rows = []
    height, width = image_shape[:2]

    for _, cell in grid_df.iterrows():
        # Define the four ground-plane corners of this cell in centimeters.
        corners_world = np.array(
            [
                [cell["x0_cm"], cell["y0_cm"]],
                [cell["x1_cm"], cell["y0_cm"]],
                [cell["x1_cm"], cell["y1_cm"]],
                [cell["x0_cm"], cell["y1_cm"]],
            ],
            dtype=np.float32,
        )

        # Project the ground-cell corners into image pixel coordinates.
        corners_pixel = world_to_pixel_fn(
            corners_world,
            rvec,
            tvec,
            K,
            dist,
        )

        # Rasterize the projected cell polygon into a binary image mask.
        mask_cell = np.zeros((height, width), dtype=np.uint8)
        pts = corners_pixel.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillConvexPoly(mask_cell, pts, 255)

        cell_pixels = mask_cell > 0

        # Only count pixels that are inside the projected cell and inside
        # the visible ground mask.
        visible_pixels = cell_pixels & (ground_mask > 0)

        n_cell_pixels = int(np.sum(cell_pixels))
        n_visible_pixels = int(np.sum(visible_pixels))

        if n_cell_pixels > 0:
            visible_fraction = n_visible_pixels / n_cell_pixels
        else:
            visible_fraction = 0

        # Skip cells that are mostly outside the valid visible ground area.
        # This avoids unstable estimates from tiny visible cell fragments.
        if visible_fraction > min_visible_fraction:
            values_prob = shadow_prob[visible_pixels]
            values_pred = pred_shadow[visible_pixels]

            mean_shadow_prob = float(np.nanmean(values_prob))
            binary_shadow_fraction = float(np.mean(values_pred > 0))

            cell_is_shaded = binary_shadow_fraction > binary_cell_threshold

        else:
            mean_shadow_prob = np.nan
            binary_shadow_fraction = np.nan
            cell_is_shaded = np.nan

        rows.append(
            {
                "image_name": image_name,
                "cell_x": int(cell["cell_x"]),
                "cell_y": int(cell["cell_y"]),
                "x_center_cm": cell["x_center_cm"],
                "y_center_cm": cell["y_center_cm"],
                "visible_fraction": visible_fraction,
                "mean_shadow_prob": mean_shadow_prob,
                "binary_shadow_fraction": binary_shadow_fraction,
                "cell_is_shaded": cell_is_shaded,
                "n_cell_pixels": n_cell_pixels,
                "n_visible_pixels": n_visible_pixels,
            }
        )

    return pd.DataFrame(rows)


def make_grid_map(
    df: pd.DataFrame,
    value_col: str,
) -> np.ndarray:
    """
    Converts a long-format grid table into a 2D grid map.

    Parameters
    ----------
    df : pd.DataFrame
        Grid table containing cell_x, cell_y, and the selected value column.
    value_col : str
        Name of the column to place into the 2D grid map.

    Returns
    -------
    np.ndarray
        2D array where rows correspond to cell_y and columns correspond
        to cell_x.

    Notes
    -----
    This format is useful for plotting heatmaps of grid-level results.
    """

    n_cols = int(df["cell_x"].max()) + 1
    n_rows = int(df["cell_y"].max()) + 1

    grid_map = np.full((n_rows, n_cols), np.nan)

    for _, row in df.iterrows():
        x = int(row["cell_x"])
        y = int(row["cell_y"])
        grid_map[y, x] = row[value_col]

    return grid_map