"""
04_project_shadow_to_ground_grid.py

Purpose:
    Projects shadow results from image pixels onto a ground-level grid.
    For each image, the script computes shadow statistics for each ground
    grid cell.

Inputs:
    - Shadow probability arrays from 03_generate_shadow_probability_maps.py
    - Binary predicted shadow masks from 03_generate_shadow_probability_maps.py
    - SAM+ROI ground masks from 02_create_sam_roi_masks.py
    - Camera model from sample_data/camera_model

Outputs:
    - multi_ground_grid_results.csv
        Per-image, per-cell shadow results.

    - agg_shadow_probability.csv
        Average shadow probability for each ground cell across all images.

    - agg_binary_shade_frequency.csv
        Frequency of each ground cell being classified as shaded.

    - grid_processing_skipped_files.csv
        Log of files that could not be processed.

Notes:
    The grid is defined in world coordinates, in centimeters. Each grid cell
    is projected into the camera image using the saved camera model.

    A cell is only analyzed if enough of it is visible inside the ground mask.
    This avoids unstable estimates for cells that are mostly outside the image,
    blocked by panels, or outside the usable ground region.
"""

import os
import sys
import glob

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# This allows the script to import from src/ when run from the project root.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.camera_geometry import (
    make_K,
    world_to_pixel_points,
)

from src.ground_grid import (
    create_ground_grid,
    process_one_image_to_ground_grid,
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

GROUND_MASK_DIR = os.path.join(
    BASE_DIR,
    "sample_data",
    "masks",
)

PROB_MAP_DIR = os.path.join(
    BASE_DIR,
    "sample_outputs",
    "shadow_probability_maps",
)

PRED_MASK_DIR = os.path.join(
    BASE_DIR,
    "sample_outputs",
    "pred_masks",
)

CAMERA_MODEL_PATH = os.path.join(
    BASE_DIR,
    "sample_data",
    "camera_model",
    "APS_SP_camera_model.npz",
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "sample_outputs",
    "grid_results",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# Camera / projection settings
# ---------------------------------------------------------------------

# Approximate horizontal field of view for the camera.
# This is used to build the intrinsic camera matrix K.
HFOV_DEG = 70

# Distortion coefficients for the camera model.
# These are used when projecting world grid points into image pixels.
DIST_COEFFS = np.array(
    [[-0.28], [0.0], [0.0], [0.0], [0.0]],
    dtype=np.float32,
)


# ---------------------------------------------------------------------
# Ground grid settings
# ---------------------------------------------------------------------

# Ground grid bounds in world coordinates, in centimeters.
# X controls left/right position and Y controls distance from the camera.
GRID_X_MIN = -550
GRID_X_MAX = 550
GRID_Y_MIN = 50
GRID_Y_MAX = 2000

# Each grid cell is 50 cm x 50 cm.
GRID_CELL_SIZE = 50

# A grid cell is ignored if less than this fraction is visible inside
# the valid ground mask.
MIN_VISIBLE_FRACTION = 0.2

# A cell is classified as shaded if its binary shadow fraction is above
# this threshold.
BINARY_CELL_THRESHOLD = 0.1


def find_image_path(image_dir: str, base: str) -> str | None:
    """
    Finds the raw image corresponding to a base filename.

    Parameters
    ----------
    image_dir : str
        Directory containing raw images.
    base : str
        Image base name without extension.

    Returns
    -------
    str or None
        Full image path if found, otherwise None.
    """

    for ext in [".jpg", ".jpeg", ".png"]:
        path = os.path.join(image_dir, base + ext)

        if os.path.exists(path):
            return path

    return None


def append_skipped_row(
    skipped_rows: list[dict],
    base: str,
    status: str,
    error_message: str | None = None,
) -> None:
    """
    Adds one row to the skipped-files log.

    Parameters
    ----------
    skipped_rows : list of dict
        List where skipped file records are stored.
    base : str
        Image base name.
    status : str
        Reason the image was skipped.
    error_message : str or None
        Optional error message for processing errors.
    """

    row = {
        "base": base,
        "status": status,
    }

    if error_message is not None:
        row["error_message"] = error_message

    skipped_rows.append(row)


def load_camera_model(camera_model_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads camera extrinsic parameters from the saved camera model file.

    Parameters
    ----------
    camera_model_path : str
        Path to the .npz file containing rvec and tvec.

    Returns
    -------
    tuple of np.ndarray
        Rotation vector and translation vector.
    """

    if not os.path.exists(camera_model_path):
        raise FileNotFoundError(
            f"Could not find camera model file: {camera_model_path}"
        )

    camera_data = np.load(camera_model_path)

    required_keys = {"rvec", "tvec"}
    missing_keys = required_keys - set(camera_data.files)

    if missing_keys:
        raise ValueError(
            f"Camera model is missing required keys: {missing_keys}"
        )

    return camera_data["rvec"], camera_data["tvec"]


def process_all_probability_maps() -> None:
    """
    Processes all saved shadow probability arrays and projects each result
    onto the ground grid.
    """

    shadow_prob_files = sorted(
        glob.glob(os.path.join(PROB_MAP_DIR, "*_shadow_prob.npy"))
    )

    if len(shadow_prob_files) == 0:
        raise FileNotFoundError(
            f"No shadow probability files found in: {PROB_MAP_DIR}"
        )

    rvec, tvec = load_camera_model(CAMERA_MODEL_PATH)

    # Create the world-coordinate grid once. The same grid is used for
    # every image because the ground plane and camera position are fixed.
    grid_df = create_ground_grid(
        x_min=GRID_X_MIN,
        x_max=GRID_X_MAX,
        y_min=GRID_Y_MIN,
        y_max=GRID_Y_MAX,
        cell_size=GRID_CELL_SIZE,
    )

    all_results = []
    skipped_rows = []

    for shadow_prob_path in tqdm(shadow_prob_files):
        filename = os.path.basename(shadow_prob_path)
        base = filename.replace("_shadow_prob.npy", "")

        image_path = find_image_path(IMAGE_DIR, base)

        ground_mask_path = os.path.join(
            GROUND_MASK_DIR,
            base + "_sam_roi_mask.png",
        )

        pred_shadow_path = os.path.join(
            PRED_MASK_DIR,
            base + "_pred_shadow.png",
        )

        if image_path is None:
            append_skipped_row(
                skipped_rows=skipped_rows,
                base=base,
                status="missing_image",
            )
            continue

        if not os.path.exists(ground_mask_path):
            append_skipped_row(
                skipped_rows=skipped_rows,
                base=base,
                status="missing_ground_mask",
            )
            continue

        if not os.path.exists(pred_shadow_path):
            append_skipped_row(
                skipped_rows=skipped_rows,
                base=base,
                status="missing_pred_shadow",
            )
            continue

        image = cv2.imread(image_path)
        ground_mask = cv2.imread(ground_mask_path, cv2.IMREAD_GRAYSCALE)
        pred_shadow = cv2.imread(pred_shadow_path, cv2.IMREAD_GRAYSCALE)
        shadow_prob = np.load(shadow_prob_path)

        if image is None:
            append_skipped_row(
                skipped_rows=skipped_rows,
                base=base,
                status="read_image_failed",
            )
            continue

        if ground_mask is None:
            append_skipped_row(
                skipped_rows=skipped_rows,
                base=base,
                status="read_ground_mask_failed",
            )
            continue

        if pred_shadow is None:
            append_skipped_row(
                skipped_rows=skipped_rows,
                base=base,
                status="read_pred_shadow_failed",
            )
            continue

        # Convert the saved binary image into a boolean mask.
        pred_shadow = pred_shadow > 0

        # Resize masks if needed. Nearest-neighbor interpolation keeps masks
        # discrete instead of creating artificial gray values.
        if ground_mask.shape != image.shape[:2]:
            ground_mask = cv2.resize(
                ground_mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        if pred_shadow.shape != image.shape[:2]:
            pred_shadow = (
                cv2.resize(
                    pred_shadow.astype(np.uint8),
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                > 0
            )

        height, width = image.shape[:2]

        # Build the intrinsic camera matrix for this image size.
        K = make_K(
            width=width,
            height=height,
            hfov_deg=HFOV_DEG,
        )

        try:
            # Project each ground-grid cell into the image and compute shadow
            # statistics using the probability map, binary mask, and ground mask.
            one_df = process_one_image_to_ground_grid(
                image_name=base,
                shadow_prob=shadow_prob,
                pred_shadow=pred_shadow,
                ground_mask=ground_mask,
                grid_df=grid_df,
                rvec=rvec,
                tvec=tvec,
                K=K,
                dist=DIST_COEFFS,
                image_shape=image.shape,
                world_to_pixel_fn=world_to_pixel_points,
                min_visible_fraction=MIN_VISIBLE_FRACTION,
                binary_cell_threshold=BINARY_CELL_THRESHOLD,
            )

            all_results.append(one_df)

        except Exception as error:
            append_skipped_row(
                skipped_rows=skipped_rows,
                base=base,
                status="processing_error",
                error_message=str(error),
            )
            continue

    if len(all_results) == 0:
        raise RuntimeError("No images were successfully processed.")

    multi_df = pd.concat(all_results, ignore_index=True)

    # -----------------------------------------------------------------
    # Save per-image, per-cell results
    # -----------------------------------------------------------------

    multi_save_path = os.path.join(
        OUTPUT_DIR,
        "multi_ground_grid_results.csv",
    )

    multi_df.to_csv(multi_save_path, index=False)

    # -----------------------------------------------------------------
    # Aggregate probability-based results
    # -----------------------------------------------------------------

    # This table shows the average shadow probability for each cell over time.
    # It is useful for visualizing which ground areas are generally more shaded.
    agg_prob_df = (
        multi_df
        .groupby(
            ["cell_x", "cell_y", "x_center_cm", "y_center_cm"],
            as_index=False,
        )
        .agg(
            mean_shadow_prob=("mean_shadow_prob", "mean"),
            mean_visible_fraction=("visible_fraction", "mean"),
            n_observations=("mean_shadow_prob", lambda x: x.notna().sum()),
        )
    )

    agg_prob_save_path = os.path.join(
        OUTPUT_DIR,
        "agg_shadow_probability.csv",
    )

    agg_prob_df.to_csv(agg_prob_save_path, index=False)

    # -----------------------------------------------------------------
    # Aggregate binary-based results
    # -----------------------------------------------------------------

    # Convert True/False shaded labels into 1/0 so the mean becomes
    # the fraction of observations where the cell was shaded.
    multi_df["cell_is_shaded_float"] = multi_df["cell_is_shaded"].map(
        {
            True: 1.0,
            False: 0.0,
        }
    )

    # This table shows how often each cell was classified as shaded.
    # It is easier to interpret than raw probability for some summaries.
    agg_bin_df = (
        multi_df
        .groupby(
            ["cell_x", "cell_y", "x_center_cm", "y_center_cm"],
            as_index=False,
        )
        .agg(
            shade_frequency=("cell_is_shaded_float", "mean"),
            mean_binary_shadow_fraction=("binary_shadow_fraction", "mean"),
            n_observations=("cell_is_shaded_float", lambda x: x.notna().sum()),
        )
    )

    agg_bin_save_path = os.path.join(
        OUTPUT_DIR,
        "agg_binary_shade_frequency.csv",
    )

    agg_bin_df.to_csv(agg_bin_save_path, index=False)

    # -----------------------------------------------------------------
    # Save skipped-file log
    # -----------------------------------------------------------------

    skipped_df = pd.DataFrame(skipped_rows)

    skipped_save_path = os.path.join(
        OUTPUT_DIR,
        "grid_processing_skipped_files.csv",
    )

    skipped_df.to_csv(skipped_save_path, index=False)

    print("Done.")
    print()
    print("Shadow probability files found:", len(shadow_prob_files))
    print("Successfully processed images:", multi_df["image_name"].nunique())
    print("Total grid rows:", len(multi_df))
    print("Observed grid rows:", multi_df["mean_shadow_prob"].notna().sum())
    print("Skipped files:", len(skipped_df))
    print()
    print("Saved:")
    print(" -", multi_save_path)
    print(" -", agg_prob_save_path)
    print(" -", agg_bin_save_path)
    print(" -", skipped_save_path)


def main() -> None:
    """
    Runs the full ground-grid projection step.
    """

    process_all_probability_maps()


if __name__ == "__main__":
    main()