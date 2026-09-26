"""
sam_masks.py

Purpose:
    Helper functions for creating and checking SAM + ROI ground masks.

Main uses:
    - Store default SAM prompt points.
    - Create a rectangular region-of-interest (ROI) mask.
    - Run SAM on each image and combine the SAM mask with the ROI.
    - Save one ground mask per image.
    - Run basic quality control checks on saved masks.

Notes:
    The SAM prompt points are manually selected image pixel coordinates.
    They are intended to guide SAM toward the visible ground area.

    If the camera view changes, update DEFAULT_SAM_POINTS using:

        findCoordinates.py

    The ROI rectangle is used to limit the mask to the useful ground region.
    This helps remove panels, sky, image borders, and unrelated background.
"""

import os

import cv2
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Default SAM prompt points
# ---------------------------------------------------------------------

# Positive SAM prompt points in image pixel coordinates.
#
# These points tell SAM which region should be included in the mask.
# In this project, they are selected on visible ground areas.
#
# To update these points for a new camera view, use:
#
#     findCoordinates.py
#
# Coordinates are written as [x, y], where:
#   x = horizontal pixel coordinate
#   y = vertical pixel coordinate
DEFAULT_SAM_POINTS = [
    [1523, 834],
    [1635, 737],
    [1702, 688],
    [1761, 655],
    [1943, 652],
    [2132, 655],
    [2260, 711],
    [2347, 796],
    [2017, 729],
    [2007, 834],
    [2350, 1013],
    [1978, 1003],
    [1651, 1000],
    [606, 970],
    [1031, 988],
    [691, 808],
    [947, 747],
    [1139, 675],
    [1047, 791],
]

# SAM labels:
#   1 = positive point / include this region
#   0 = negative point / exclude this region
#
# Here all default points are positive because they mark ground pixels.
DEFAULT_SAM_LABELS = [1] * len(DEFAULT_SAM_POINTS)


def make_roi_mask(
    shape: tuple,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> np.ndarray:
    """
    Creates a rectangular ROI mask.

    Parameters
    ----------
    shape : tuple
        Image or mask shape, usually (height, width).
    x1 : int
        Left x-coordinate of the ROI.
    y1 : int
        Top y-coordinate of the ROI.
    x2 : int
        Right x-coordinate of the ROI.
    y2 : int
        Bottom y-coordinate of the ROI.

    Returns
    -------
    np.ndarray
        Binary ROI mask with 255 inside the ROI and 0 outside.

    Notes
    -----
    Coordinates are clipped to the image boundaries so the function does not
    fail if the ROI is slightly outside the image.
    """

    roi_mask = np.zeros(shape[:2], dtype=np.uint8)

    height, width = shape[:2]

    # Clip ROI coordinates to image boundaries.
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(int(x2), width)
    y2 = min(int(y2), height)

    roi_mask[y1:y2, x1:x2] = 255

    return roi_mask


def get_best_sam_roi_mask(
    image_path: str,
    sam_model,
    points: list[list[int]],
    labels: list[int],
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> tuple[np.ndarray | None, float | None]:
    """
    Runs SAM on one image and returns the best SAM + ROI ground mask.

    Parameters
    ----------
    image_path : str
        Path to the input image.
    sam_model
        Loaded SAM model.
    points : list of list of int
        SAM prompt points in image pixel coordinates.
    labels : list of int
        SAM labels for each point. Usually 1 for positive points.
    x1, y1, x2, y2 : int
        ROI rectangle coordinates.

    Returns
    -------
    tuple
        best_mask : np.ndarray or None
            Best binary SAM + ROI mask, or None if SAM failed.
        best_score : float or None
            White fraction inside the ROI for the selected mask.

    Notes
    -----
    SAM may return multiple candidate masks. This function selects the mask
    with the largest white fraction inside the ROI.

    The final mask is:

        SAM mask AND rectangular ROI mask

    This keeps only the part of the SAM mask inside the area we want to analyze.
    """

    results = sam_model(
        image_path,
        points=points,
        labels=labels,
        verbose=False,
    )

    if results[0].masks is None:
        return None, None

    masks = results[0].masks.data.cpu().numpy()

    best_mask = None
    best_score = -1

    for mask in masks:
        # Convert SAM probability/logit mask to a binary 0/255 mask.
        sam_mask = (mask > 0.5).astype(np.uint8) * 255

        roi_mask = make_roi_mask(
            sam_mask.shape,
            x1,
            y1,
            x2,
            y2,
        )

        # Limit the SAM mask to the fixed ROI.
        final_mask = cv2.bitwise_and(
            sam_mask,
            roi_mask,
        )

        inside_roi = roi_mask > 0
        roi_pixels = np.sum(inside_roi)

        if roi_pixels == 0:
            white_fraction_roi = 0
        else:
            white_fraction_roi = np.sum(final_mask > 0) / roi_pixels

        # Pick the SAM result that covers the most ground area inside the ROI.
        if white_fraction_roi > best_score:
            best_score = white_fraction_roi
            best_mask = final_mask

    return best_mask, best_score


def build_sam_result_row(
    base: str,
    image_path: str,
    mask_save_path: str | None,
    status: str,
    white_fraction_roi: float | None = np.nan,
) -> dict:
    """
    Builds a consistent status row for SAM mask creation.

    Parameters
    ----------
    base : str
        Image base name without extension.
    image_path : str
        Path to the image.
    mask_save_path : str or None
        Path where the mask is saved, or None if no mask was saved.
    status : str
        SAM processing status.
    white_fraction_roi : float or None
        Fraction of the ROI covered by the selected SAM mask.

    Returns
    -------
    dict
        One row for the SAM mask creation summary table.
    """

    return {
        "base": base,
        "image_path": image_path,
        "sam_roi_mask_path": mask_save_path,
        "sam_white_fraction_roi": white_fraction_roi,
        "sam_status": status,
    }


def create_sam_roi_masks_from_metadata(
    metadata_df: pd.DataFrame,
    output_mask_dir: str,
    sam_model,
    points: list[list[int]],
    labels: list[int],
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    skip_existing: bool = True,
) -> pd.DataFrame:
    """
    Creates SAM + ROI masks for all images listed in a metadata table.

    Parameters
    ----------
    metadata_df : pd.DataFrame
        Metadata table. Must contain image_path and base columns.
    output_mask_dir : str
        Directory where output masks will be saved.
    sam_model
        Loaded SAM model.
    points : list of list of int
        SAM prompt points in image pixel coordinates.
    labels : list of int
        SAM labels for each point.
    x1, y1, x2, y2 : int
        ROI rectangle coordinates.
    skip_existing : bool
        If True, existing masks are not recomputed.

    Returns
    -------
    pd.DataFrame
        Summary table with one row per image.

    Output masks
    ------------
    Each output mask is saved as:

        <base>_sam_roi_mask.png

    Notes
    -----
    This function is usually called by 02_create_sam_roi_masks.py.
    """

    os.makedirs(output_mask_dir, exist_ok=True)

    required_cols = {"image_path", "base"}
    missing_cols = required_cols - set(metadata_df.columns)

    if missing_cols:
        raise ValueError(
            f"metadata_df is missing required columns: {missing_cols}"
        )

    results = []

    for _, row in metadata_df.iterrows():
        image_path = row["image_path"]
        base = row["base"]

        mask_save_path = os.path.join(
            output_mask_dir,
            base + "_sam_roi_mask.png",
        )

        if not os.path.exists(image_path):
            results.append(
                build_sam_result_row(
                    base=base,
                    image_path=image_path,
                    mask_save_path=mask_save_path,
                    status="missing_image",
                )
            )
            continue

        if skip_existing and os.path.exists(mask_save_path):
            results.append(
                build_sam_result_row(
                    base=base,
                    image_path=image_path,
                    mask_save_path=mask_save_path,
                    status="already_exists",
                )
            )
            continue

        final_mask, score = get_best_sam_roi_mask(
            image_path=image_path,
            sam_model=sam_model,
            points=points,
            labels=labels,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
        )

        if final_mask is None:
            results.append(
                build_sam_result_row(
                    base=base,
                    image_path=image_path,
                    mask_save_path=None,
                    status="sam_failed",
                )
            )
            continue

        # Save as a clean binary 0/255 PNG mask.
        cv2.imwrite(
            mask_save_path,
            (final_mask > 0).astype(np.uint8) * 255,
        )

        results.append(
            build_sam_result_row(
                base=base,
                image_path=image_path,
                mask_save_path=mask_save_path,
                status="saved",
                white_fraction_roi=score,
            )
        )

    return pd.DataFrame(results)


def run_mask_qc(
    mask_dir: str,
    valid_bases: set[str] | None = None,
    x1: int = 225,
    y1: int = 384,
    x2: int = 2606,
    y2: int = 1044,
) -> pd.DataFrame:
    """
    Runs simple QC on saved SAM + ROI masks.

    Parameters
    ----------
    mask_dir : str
        Folder containing *_sam_roi_mask.png files.
    valid_bases : set of str or None
        Optional set of image base names to include. If provided, masks not
        matching these base names are ignored.
    x1, y1, x2, y2 : int
        ROI rectangle coordinates.

    Returns
    -------
    pd.DataFrame
        QC table with white_fraction_roi and is_suspicious columns.

    Notes
    -----
    QC is based on the fraction of ROI pixels covered by the mask.

    A mask is flagged as suspicious if its white fraction is much smaller or
    much larger than the median white fraction across all masks. This is a
    simple way to catch masks that are incomplete, overly large, or failed.
    """

    mask_files = sorted(
        [
            os.path.join(mask_dir, filename)
            for filename in os.listdir(mask_dir)
            if filename.lower().endswith(".png")
        ]
    )

    qc_rows = []

    for mask_path in mask_files:
        mask_name = os.path.basename(mask_path)
        base = mask_name.replace("_sam_roi_mask.png", "")

        if valid_bases is not None and base not in valid_bases:
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        if mask is None:
            qc_rows.append(
                {
                    "base": base,
                    "mask_path": mask_path,
                    "white_fraction_roi": np.nan,
                    "white_pixels_roi": np.nan,
                    "roi_pixels": np.nan,
                    "read_ok": False,
                }
            )
            continue

        # Make sure mask is 2D.
        # Some saved PNG masks may load as (H, W, 1) or color images.
        if mask.ndim == 3:
            if mask.shape[2] == 1:
                mask = mask[:, :, 0]
            else:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        mask = mask.astype(np.uint8)

        roi_mask = make_roi_mask(
            mask.shape,
            x1,
            y1,
            x2,
            y2,
        )

        inside_roi = roi_mask > 0
        white_inside_roi = (mask > 0) & inside_roi

        roi_pixels = np.sum(inside_roi)

        if roi_pixels == 0:
            white_fraction_roi = np.nan
        else:
            white_fraction_roi = np.sum(white_inside_roi) / roi_pixels

        qc_rows.append(
            {
                "base": base,
                "mask_path": mask_path,
                "white_fraction_roi": white_fraction_roi,
                "white_pixels_roi": int(np.sum(white_inside_roi)),
                "roi_pixels": int(roi_pixels),
                "read_ok": True,
            }
        )

    mask_qc_df = pd.DataFrame(qc_rows)

    if len(mask_qc_df) == 0:
        return mask_qc_df

    median_white = mask_qc_df["white_fraction_roi"].median()

    low_threshold = median_white * 0.60
    high_threshold = median_white * 1.40

    # Flag masks that differ strongly from the typical mask size.
    # These should be inspected manually before trusting downstream results.
    mask_qc_df["is_suspicious"] = (
        (mask_qc_df["white_fraction_roi"] < low_threshold)
        | (mask_qc_df["white_fraction_roi"] > high_threshold)
        | (~mask_qc_df["read_ok"])
    )

    return mask_qc_df