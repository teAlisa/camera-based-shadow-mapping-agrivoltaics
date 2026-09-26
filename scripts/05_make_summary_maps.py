"""
05_visualize_ground_grid_results.py

Purpose:
    Creates final visualizations of ground-grid shadow results and overlays
    approximate solar panel row locations.

Inputs:
    - agg_shadow_probability.csv from 04_project_shadow_to_ground_grid.py
    - agg_binary_shade_frequency.csv from 04_project_shadow_to_ground_grid.py
    - Camera model from sample_data/camera_model
    - Raw images from sample_data/images

Outputs:
    - panel_rows_grid.csv
    - average_shadow_probability_with_panel_rows.png
    - shade_frequency_with_panel_rows.png
    - mean_shadow_coverage_with_panel_rows.png
    - shade_frequency_with_panel_rows_meters.png
    - mean_shadow_coverage_with_panel_rows_meters.png

Notes:
    Panel rows are defined in image pixel coordinates and then projected
    onto the ground grid using the camera model. This allows the final maps
    to show shadow patterns relative to the physical panel row locations.
"""

import os
import sys

import cv2
import numpy as np
import pandas as pd

# This allows the script to import from src/ when run from the project root.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.camera_geometry import (
    make_K,
    pixel_to_world_on_plane,
)

from src.ground_grid import create_ground_grid

from src.panel_rows import (
    DEFAULT_PANEL_ROWS_PIXELS,
    create_panel_rows_grid,
)

from src.visualization import (
    plot_map_with_panel_rows,
    plot_map_with_panel_rows_meters,
)


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

# Project root (the folder that contains scripts/, src/, sample_data/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGE_DIR = os.path.join(
    BASE_DIR,
    "sample_data",
    "images",
)

CAMERA_MODEL_PATH = os.path.join(
    BASE_DIR,
    "sample_data",
    "camera_model",
    "APS_SP_camera_model.npz",
)

GRID_RESULTS_DIR = os.path.join(
    BASE_DIR,
    "sample_outputs",
    "grid_results",
)

FIGURES_DIR = os.path.join(
    BASE_DIR,
    "sample_outputs",
    "figures",
)

os.makedirs(FIGURES_DIR, exist_ok=True)

AGG_PROB_PATH = os.path.join(
    GRID_RESULTS_DIR,
    "agg_shadow_probability.csv",
)

AGG_BIN_PATH = os.path.join(
    GRID_RESULTS_DIR,
    "agg_binary_shade_frequency.csv",
)


# ---------------------------------------------------------------------
# Camera / projection settings
# ---------------------------------------------------------------------

# Approximate horizontal field of view used to build the camera matrix.
HFOV_DEG = 70

# Distortion coefficients used when projecting pixel-defined panel rows
# onto the ground plane.
DIST_COEFFS = np.array(
    [[-0.28], [0.0], [0.0], [0.0], [0.0]],
    dtype=np.float32,
)


# ---------------------------------------------------------------------
# Ground grid settings
# ---------------------------------------------------------------------

# These bounds must match the grid used in 04_project_shadow_to_ground_grid.py.
GRID_X_MIN = -550
GRID_X_MAX = 550
GRID_Y_MIN = 50
GRID_Y_MAX = 2000
GRID_CELL_SIZE = 50


def find_reference_image(image_dir: str) -> str:
    """
    Finds the first available image to use for image size and camera matrix.

    Parameters
    ----------
    image_dir : str
        Directory containing raw images.

    Returns
    -------
    str
        Full path to the first image file found.

    Raises
    ------
    FileNotFoundError
        If no image file is found.
    """

    for filename in sorted(os.listdir(image_dir)):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            return os.path.join(image_dir, filename)

    raise FileNotFoundError(
        f"No reference image found in: {image_dir}"
    )


def validate_input_files() -> None:
    """
    Checks that all required input files and folders exist before plotting.
    """

    if not os.path.exists(AGG_PROB_PATH):
        raise FileNotFoundError(
            f"Could not find: {AGG_PROB_PATH}"
        )

    if not os.path.exists(AGG_BIN_PATH):
        raise FileNotFoundError(
            f"Could not find: {AGG_BIN_PATH}"
        )

    if not os.path.exists(CAMERA_MODEL_PATH):
        raise FileNotFoundError(
            f"Could not find camera model: {CAMERA_MODEL_PATH}"
        )

    if not os.path.exists(IMAGE_DIR):
        raise FileNotFoundError(
            f"Could not find image directory: {IMAGE_DIR}"
        )


def load_camera_model(camera_model_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads camera parameters needed to project panel rows onto the ground plane.

    Parameters
    ----------
    camera_model_path : str
        Path to the saved .npz camera model file.

    Returns
    -------
    tuple of np.ndarray
        Rotation matrix and translation vector.
    """

    camera_data = np.load(camera_model_path)

    required_keys = {"rot_mat", "tvec"}
    missing_keys = required_keys - set(camera_data.files)

    if missing_keys:
        raise ValueError(
            f"Camera model is missing required keys: {missing_keys}"
        )

    return camera_data["rot_mat"], camera_data["tvec"]


def create_panel_row_overlay(
    image_dir: str,
    camera_model_path: str,
) -> tuple[pd.DataFrame, str]:
    """
    Projects panel row locations from image pixels onto the ground grid.

    Parameters
    ----------
    image_dir : str
        Directory containing raw images.
    camera_model_path : str
        Path to the saved camera model.

    Returns
    -------
    tuple
        panel_rows_grid : pd.DataFrame
            Projected panel row coordinates in the same coordinate system
            as the ground grid.
        reference_image_path : str
            Image path used to get image width and height.
    """

    R, t = load_camera_model(camera_model_path)

    reference_image_path = find_reference_image(image_dir)
    reference_image = cv2.imread(reference_image_path)

    if reference_image is None:
        raise FileNotFoundError(
            f"Could not read reference image: {reference_image_path}"
        )

    height, width = reference_image.shape[:2]

    # Build camera matrix using the reference image size.
    K = make_K(
        width=width,
        height=height,
        hfov_deg=HFOV_DEG,
    )

    # Create the same ground grid used for shadow aggregation.
    grid_df = create_ground_grid(
        x_min=GRID_X_MIN,
        x_max=GRID_X_MAX,
        y_min=GRID_Y_MIN,
        y_max=GRID_Y_MAX,
        cell_size=GRID_CELL_SIZE,
    )

    # Convert panel row lines from image pixel coordinates to ground-grid
    # coordinates so they can be overlaid on the final maps.
    panel_rows_grid = create_panel_rows_grid(
        panel_rows_pixels=DEFAULT_PANEL_ROWS_PIXELS,
        grid_df=grid_df,
        pixel_to_world_fn=pixel_to_world_on_plane,
        K=K,
        dist=DIST_COEFFS,
        R=R,
        t=t,
    )

    return panel_rows_grid, reference_image_path


def save_shadow_maps(
    agg_prob_df: pd.DataFrame,
    agg_bin_df: pd.DataFrame,
    panel_rows_grid: pd.DataFrame,
) -> None:
    """
    Saves all final ground-grid shadow visualizations.

    Parameters
    ----------
    agg_prob_df : pd.DataFrame
        Aggregated probability-based shadow results.
    agg_bin_df : pd.DataFrame
        Aggregated binary-based shade frequency results.
    panel_rows_grid : pd.DataFrame
        Projected panel row coordinates.
    """

    # 1. Average shadow probability map.
    # Shows the mean predicted shadow probability for each grid cell.
    plot_map_with_panel_rows(
        df=agg_prob_df,
        value_col="mean_shadow_prob",
        panel_rows_grid=panel_rows_grid,
        title="Average shadow probability with panel rows",
        colorbar_label="Mean shadow probability",
        save_path=os.path.join(
            FIGURES_DIR,
            "average_shadow_probability_with_panel_rows.png",
        ),
        cmap_name="viridis_r",
        panel_color="red",
    )

    # 2. Shade frequency map.
    # Shows how often each cell was classified as shaded.
    plot_map_with_panel_rows(
        df=agg_bin_df,
        value_col="shade_frequency",
        panel_rows_grid=panel_rows_grid,
        title="Shade frequency with panel rows",
        colorbar_label="Shade frequency",
        save_path=os.path.join(
            FIGURES_DIR,
            "shade_frequency_with_panel_rows.png",
        ),
        cmap_name="viridis_r",
        panel_color="red",
    )

    # 3. Mean binary shadow coverage map.
    # Shows the average fraction of visible pixels in each cell that were
    # classified as shadow.
    plot_map_with_panel_rows(
        df=agg_bin_df,
        value_col="mean_binary_shadow_fraction",
        panel_rows_grid=panel_rows_grid,
        title="Mean shadow coverage with panel rows",
        colorbar_label="Mean fraction of visible pixels classified as shadow",
        save_path=os.path.join(
            FIGURES_DIR,
            "mean_shadow_coverage_with_panel_rows.png",
        ),
        cmap_name="viridis_r",
        panel_color="red",
    )

    # 4. Shade frequency map with axes converted from centimeters to meters.
    plot_map_with_panel_rows_meters(
        df=agg_bin_df,
        value_col="shade_frequency",
        panel_rows_grid=panel_rows_grid,
        title="Shade frequency with panel rows",
        colorbar_label="Shade frequency",
        save_path=os.path.join(
            FIGURES_DIR,
            "shade_frequency_with_panel_rows_meters.png",
        ),
        cmap_name="viridis_r",
        panel_color="red",
    )

    # 5. Mean binary shadow coverage map with axes converted to meters.
    plot_map_with_panel_rows_meters(
        df=agg_bin_df,
        value_col="mean_binary_shadow_fraction",
        panel_rows_grid=panel_rows_grid,
        title="Mean shadow coverage with panel rows",
        colorbar_label="Mean fraction of visible pixels classified as shadow",
        save_path=os.path.join(
            FIGURES_DIR,
            "mean_shadow_coverage_with_panel_rows_meters.png",
        ),
        cmap_name="viridis_r",
        panel_color="red",
    )


def main() -> None:
    """
    Loads aggregated grid results, creates the panel-row overlay, and saves
    final map figures.
    """

    validate_input_files()

    agg_prob_df = pd.read_csv(AGG_PROB_PATH)
    agg_bin_df = pd.read_csv(AGG_BIN_PATH)

    panel_rows_grid, reference_image_path = create_panel_row_overlay(
        image_dir=IMAGE_DIR,
        camera_model_path=CAMERA_MODEL_PATH,
    )

    panel_rows_path = os.path.join(
        FIGURES_DIR,
        "panel_rows_grid.csv",
    )

    panel_rows_grid.to_csv(panel_rows_path, index=False)

    save_shadow_maps(
        agg_prob_df=agg_prob_df,
        agg_bin_df=agg_bin_df,
        panel_rows_grid=panel_rows_grid,
    )

    print("Done.")
    print("Reference image:", reference_image_path)
    print("Saved figures to:", FIGURES_DIR)
    print("Saved panel rows to:", panel_rows_path)


if __name__ == "__main__":
    main()