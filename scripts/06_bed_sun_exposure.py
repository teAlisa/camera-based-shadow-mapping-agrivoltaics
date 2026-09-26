"""
06_analyze_bed_sun_exposure.py

Purpose:
    Calculates how long a selected plant row / flower bed was sunlit or shaded
    during the analyzed day.

Inputs:
    - image_metadata.csv from 01_create_image_metadata.py
    - Binary predicted shadow masks from 03_generate_shadow_probability_maps.py
    - Shadow probability arrays from 03_generate_shadow_probability_maps.py

Outputs:
    - bed_sun_exposure_timeseries.csv
        Per-image shadow statistics for the selected bed polygon.

    - bed_sun_exposure_summary.csv
        Total sunlit/shaded hours and summary statistics.

    - bed_sunlit_intervals.csv
        Continuous time intervals when the selected bed was classified as sunlit.

    - bed_polygon_overlay_preview.png
        Preview image showing the selected polygon on top of a raw image.

Notes:
    The selected bed is defined manually using pixel coordinates in BED_POLYGON.
    These coordinates can be changed to analyze a different plant row or bed.

    To find new pixel coordinates, use the helper script:

        findCoordinates.py

    The helper script lets you click on an image and record the coordinates
    for a new polygon.
"""

import os
import sys

import cv2
import numpy as np
import pandas as pd

# This allows the script to import from src/ when run from the project root.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

# Project root (the folder that contains scripts/, src/, sample_data/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

METADATA_PATH = os.path.join(
    BASE_DIR,
    "sample_data",
    "metadata",
    "image_metadata.csv",
)

IMAGE_DIR = os.path.join(
    BASE_DIR,
    "sample_data",
    "images",
)

PRED_MASK_DIR = os.path.join(
    BASE_DIR,
    "sample_outputs",
    "pred_masks",
)

PROB_MAP_DIR = os.path.join(
    BASE_DIR,
    "sample_outputs",
    "shadow_probability_maps",
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "sample_outputs",
    "bed_sun_exposure",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# Bed / plant row ROI settings
# ---------------------------------------------------------------------

# Flower bed polygon in image pixel coordinates.
#
# These coordinates can be changed to analyze a different plant row or bed.
# To find new coordinates, use:
#
#     findCoordinates.py
#
# Important:
#   - Points should be listed in order around the bed boundary.
#   - Do not cross diagonally.
#   - The polygon should cover only the plant row / bed area you want to analyze.
#   - Coordinates are written as [x, y], where x is horizontal pixel position
#     and y is vertical pixel position.
BED_POLYGON = [
    [1681, 765],  # start left
    [1702, 765],  # start right
    [1743, 729],  # end right
    [1715, 727],  # end left
]


# If less than 20% of the bed area is classified as shadow,
# we treat the bed as sunlit.
#
# Increase this value if you want the bed to still count as sunlit even with
# more partial shade. Decrease it if you want a stricter sunlit definition.
SUNLIT_SHADOW_FRACTION_THRESHOLD = 0.2


# Large gaps probably mean missing images, overnight periods, or camera gaps.
# These intervals are set to unknown and are not counted as continuous
# sun/shade exposure time.
MAX_INTERVAL_MINUTES = 30


def create_polygon_roi_mask(
    image_shape: tuple,
    polygon_points: list[list[int]],
) -> np.ndarray:
    """
    Creates a binary ROI mask from polygon points.

    Parameters
    ----------
    image_shape : tuple
        Shape of the image/probability map, usually (height, width).
    polygon_points : list of list of int
        List of [x, y] pixel coordinates ordered around the polygon boundary.

    Returns
    -------
    np.ndarray
        Boolean mask where True marks the selected bed region.
    """

    height, width = image_shape[:2]
    roi_mask = np.zeros((height, width), dtype=np.uint8)

    polygon = np.array(
        polygon_points,
        dtype=np.int32,
    ).reshape((-1, 1, 2))

    cv2.fillPoly(
        roi_mask,
        [polygon],
        color=255,
    )

    return roi_mask > 0


def calculate_time_intervals(exposure_df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimates how many minutes each image represents using the time difference
    to the next image.

    The last image receives the median interval. Very large gaps are set to NaN
    and are not counted as continuous sun/shade exposure time.

    Parameters
    ----------
    exposure_df : pd.DataFrame
        Per-image exposure table with timestamps.

    Returns
    -------
    pd.DataFrame
        Exposure table with next_timestamp and interval_minutes columns.
    """

    df = exposure_df.sort_values("timestamp").copy()

    df["next_timestamp"] = df["timestamp"].shift(-1)

    df["interval_minutes"] = (
        df["next_timestamp"] - df["timestamp"]
    ).dt.total_seconds() / 60

    median_interval = df["interval_minutes"].median()
    df["interval_minutes"] = df["interval_minutes"].fillna(median_interval)

    df.loc[
        df["interval_minutes"] > MAX_INTERVAL_MINUTES,
        "interval_minutes",
    ] = np.nan

    return df


def build_failed_result(
    base: str,
    image_name: str,
    status: str,
    n_bed_pixels: int | float = np.nan,
    n_valid_pixels: int | float = np.nan,
) -> dict:
    """
    Builds a consistent output row for images that cannot be processed.

    Parameters
    ----------
    base : str
        Image base name without extension.
    image_name : str
        Full image filename.
    status : str
        Processing status.
    n_bed_pixels : int or float
        Number of pixels in the selected bed polygon.
    n_valid_pixels : int or float
        Number of valid pixels used for analysis.

    Returns
    -------
    dict
        One result row for the exposure time series.
    """

    return {
        "base": base,
        "image_name": image_name,
        "status": status,
        "n_bed_pixels": n_bed_pixels,
        "n_valid_pixels": n_valid_pixels,
        "mean_shadow_prob": np.nan,
        "binary_shadow_fraction": np.nan,
        "is_sunlit": np.nan,
        "is_shaded": np.nan,
    }


def process_one_image(
    base: str,
    image_name: str,
) -> dict:
    """
    Samples shadow values inside the selected bed polygon for one image.

    Parameters
    ----------
    base : str
        Image base name without extension.
    image_name : str
        Full image filename.

    Returns
    -------
    dict
        Shadow statistics for the selected bed polygon in this image.
    """

    pred_shadow_path = os.path.join(
        PRED_MASK_DIR,
        base + "_pred_shadow.png",
    )

    shadow_prob_path = os.path.join(
        PROB_MAP_DIR,
        base + "_shadow_prob.npy",
    )

    if not os.path.exists(pred_shadow_path):
        return build_failed_result(
            base=base,
            image_name=image_name,
            status="missing_pred_shadow",
        )

    if not os.path.exists(shadow_prob_path):
        return build_failed_result(
            base=base,
            image_name=image_name,
            status="missing_shadow_prob",
        )

    pred_shadow = cv2.imread(pred_shadow_path, cv2.IMREAD_GRAYSCALE)
    shadow_prob = np.load(shadow_prob_path)

    if pred_shadow is None:
        return build_failed_result(
            base=base,
            image_name=image_name,
            status="read_pred_shadow_failed",
        )

    # Make sure the binary shadow mask and probability map have the same size.
    # Nearest-neighbor interpolation keeps the binary mask discrete.
    if pred_shadow.shape != shadow_prob.shape:
        pred_shadow = cv2.resize(
            pred_shadow,
            (shadow_prob.shape[1], shadow_prob.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    bed_roi = create_polygon_roi_mask(
        image_shape=shadow_prob.shape,
        polygon_points=BED_POLYGON,
    )

    # Only use pixels inside the bed polygon where the probability map
    # contains a valid value.
    valid_pixels = bed_roi & np.isfinite(shadow_prob)

    n_bed_pixels = int(np.sum(bed_roi))
    n_valid_pixels = int(np.sum(valid_pixels))

    if n_valid_pixels == 0:
        return build_failed_result(
            base=base,
            image_name=image_name,
            status="no_valid_pixels",
            n_bed_pixels=n_bed_pixels,
            n_valid_pixels=n_valid_pixels,
        )

    pred_shadow_bool = pred_shadow > 0

    mean_shadow_prob = float(np.nanmean(shadow_prob[valid_pixels]))
    binary_shadow_fraction = float(np.mean(pred_shadow_bool[valid_pixels]))

    # The bed is considered sunlit if only a small fraction of the selected
    # polygon is classified as shadow.
    is_sunlit = binary_shadow_fraction < SUNLIT_SHADOW_FRACTION_THRESHOLD
    is_shaded = not is_sunlit

    return {
        "base": base,
        "image_name": image_name,
        "status": "processed",
        "n_bed_pixels": n_bed_pixels,
        "n_valid_pixels": n_valid_pixels,
        "mean_shadow_prob": mean_shadow_prob,
        "binary_shadow_fraction": binary_shadow_fraction,
        "is_sunlit": is_sunlit,
        "is_shaded": is_shaded,
    }


def summarize_sunlit_intervals(exposure_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates continuous time intervals when the bed was classified as sunlit.

    A sunlit interval starts when is_sunlit becomes True and ends at the next
    timestamp after the last sunlit image in that run.

    Parameters
    ----------
    exposure_df : pd.DataFrame
        Per-image exposure table with timestamps and sunlit labels.

    Returns
    -------
    pd.DataFrame
        Table of continuous sunlit intervals.
    """

    df = exposure_df.sort_values("timestamp").copy()

    # Only use rows with valid interval times.
    df = df[df["interval_minutes"].notna()].copy()

    intervals = []

    current_start = None
    current_end = None
    current_minutes = 0

    for _, row in df.iterrows():
        is_sunlit = row["is_sunlit"] is True or row["is_sunlit"] == True

        start_time = row["timestamp"]
        end_time = row["next_timestamp"]

        if pd.isna(end_time):
            continue

        if is_sunlit:
            if current_start is None:
                current_start = start_time

            current_end = end_time
            current_minutes += row["interval_minutes"]

        else:
            if current_start is not None:
                intervals.append(
                    {
                        "sunlit_start": current_start,
                        "sunlit_end": current_end,
                        "sunlit_minutes": current_minutes,
                        "sunlit_hours": current_minutes / 60,
                    }
                )

                current_start = None
                current_end = None
                current_minutes = 0

    # Save the final open interval if the day ended during a sunlit period.
    if current_start is not None:
        intervals.append(
            {
                "sunlit_start": current_start,
                "sunlit_end": current_end,
                "sunlit_minutes": current_minutes,
                "sunlit_hours": current_minutes / 60,
            }
        )

    return pd.DataFrame(intervals)


def save_bed_polygon_overlay(
    image_path: str,
    polygon_points: list[list[int]],
    save_path: str,
) -> None:
    """
    Draws the selected bed polygon on the original image and saves the preview.

    Parameters
    ----------
    image_path : str
        Path to the raw reference image.
    polygon_points : list of list of int
        Polygon coordinates in image pixel coordinates.
    save_path : str
        Output path for the overlay preview image.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    overlay = image.copy()
    output = image.copy()

    polygon = np.array(
        polygon_points,
        dtype=np.int32,
    ).reshape((-1, 1, 2))

    # Semi-transparent yellow fill.
    cv2.fillPoly(
        overlay,
        [polygon],
        color=(0, 255, 255),
    )

    alpha = 0.35

    cv2.addWeighted(
        overlay,
        alpha,
        output,
        1 - alpha,
        0,
        output,
    )

    # Red polygon outline.
    cv2.polylines(
        output,
        [polygon],
        isClosed=True,
        color=(0, 0, 255),
        thickness=2,
    )

    # Mark polygon corners so the coordinate order can be checked visually.
    for i, (x, y) in enumerate(polygon_points, start=1):
        cv2.circle(
            output,
            (int(x), int(y)),
            radius=4,
            color=(255, 0, 0),
            thickness=-1,
        )

        cv2.putText(
            output,
            str(i),
            (int(x) + 5, int(y) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        output,
        "Plant row ROI",
        (polygon_points[0][0] + 10, polygon_points[0][1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.imwrite(save_path, output)


def validate_metadata(metadata: pd.DataFrame) -> None:
    """
    Checks that the metadata table contains the columns required for this step.

    Parameters
    ----------
    metadata : pd.DataFrame
        Metadata table loaded from image_metadata.csv.

    Raises
    ------
    ValueError
        If required columns are missing.
    """

    required_cols = {"base", "image_name", "timestamp"}
    missing_cols = required_cols - set(metadata.columns)

    if missing_cols:
        raise ValueError(
            f"Metadata file is missing required columns: {missing_cols}"
        )


def build_summary_table(exposure_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a one-row summary table for total sunlit and shaded time.

    Parameters
    ----------
    exposure_df : pd.DataFrame
        Per-image exposure table with interval times and sunlit/shaded labels.

    Returns
    -------
    pd.DataFrame
        Summary table with total sunlit and shaded exposure.
    """

    valid_time = exposure_df["interval_minutes"].notna()

    sunlit_minutes = exposure_df.loc[
        valid_time & (exposure_df["is_sunlit"] == True),
        "interval_minutes",
    ].sum()

    shaded_minutes = exposure_df.loc[
        valid_time & (exposure_df["is_shaded"] == True),
        "interval_minutes",
    ].sum()

    unknown_minutes = exposure_df.loc[
        valid_time & (exposure_df["status"] != "processed"),
        "interval_minutes",
    ].sum()

    total_counted_minutes = sunlit_minutes + shaded_minutes

    if total_counted_minutes > 0:
        sunlit_fraction = sunlit_minutes / total_counted_minutes
    else:
        sunlit_fraction = np.nan

    summary_df = pd.DataFrame(
        [
            {
                "bed_polygon": str(BED_POLYGON),
                "sunlit_shadow_fraction_threshold": (
                    SUNLIT_SHADOW_FRACTION_THRESHOLD
                ),
                "max_interval_minutes": MAX_INTERVAL_MINUTES,
                "processed_images": int(
                    (exposure_df["status"] == "processed").sum()
                ),
                "total_images": len(exposure_df),
                "sunlit_minutes": sunlit_minutes,
                "shaded_minutes": shaded_minutes,
                "unknown_minutes": unknown_minutes,
                "sunlit_hours": sunlit_minutes / 60,
                "shaded_hours": shaded_minutes / 60,
                "sunlit_fraction": sunlit_fraction,
                "mean_shadow_prob_over_time": (
                    exposure_df["mean_shadow_prob"].mean()
                ),
                "mean_binary_shadow_fraction_over_time": (
                    exposure_df["binary_shadow_fraction"].mean()
                ),
            }
        ]
    )

    return summary_df


def save_polygon_preview(
    exposure_df: pd.DataFrame,
    metadata: pd.DataFrame,
) -> str | None:
    """
    Saves a preview image showing the selected bed polygon.

    Parameters
    ----------
    exposure_df : pd.DataFrame
        Per-image exposure table.
    metadata : pd.DataFrame
        Image metadata table.

    Returns
    -------
    str or None
        Path to the saved preview image, or None if no processed images exist.
    """

    processed_rows = exposure_df[exposure_df["status"] == "processed"]

    if len(processed_rows) == 0:
        return None

    # Use the first processed image for the preview so the polygon is drawn
    # on an image that was actually included in the analysis.
    first_image_name = processed_rows.iloc[0]["image_name"]

    first_image_path = os.path.join(
        IMAGE_DIR,
        first_image_name,
    )

    overlay_preview_path = os.path.join(
        OUTPUT_DIR,
        "bed_polygon_overlay_preview.png",
    )

    save_bed_polygon_overlay(
        image_path=first_image_path,
        polygon_points=BED_POLYGON,
        save_path=overlay_preview_path,
    )

    return overlay_preview_path


def main() -> None:
    """
    Runs the full bed sun-exposure analysis.
    """

    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            f"Could not find metadata: {METADATA_PATH}"
        )

    metadata = pd.read_csv(METADATA_PATH)
    validate_metadata(metadata)

    metadata["timestamp"] = pd.to_datetime(metadata["timestamp"])

    rows = []

    for _, row in metadata.iterrows():
        result = process_one_image(
            base=row["base"],
            image_name=row["image_name"],
        )

        result["timestamp"] = row["timestamp"]
        result["elevation"] = row.get("elevation", np.nan)
        result["azimuth"] = row.get("azimuth", np.nan)

        rows.append(result)

    exposure_df = pd.DataFrame(rows)

    # Add interval_minutes so each image contributes time instead of just
    # counting as one observation.
    exposure_df = calculate_time_intervals(exposure_df)

    sunlit_intervals_df = summarize_sunlit_intervals(exposure_df)
    summary_df = build_summary_table(exposure_df)

    timeseries_path = os.path.join(
        OUTPUT_DIR,
        "bed_sun_exposure_timeseries.csv",
    )

    summary_path = os.path.join(
        OUTPUT_DIR,
        "bed_sun_exposure_summary.csv",
    )

    sunlit_intervals_path = os.path.join(
        OUTPUT_DIR,
        "bed_sunlit_intervals.csv",
    )

    exposure_df.to_csv(timeseries_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    sunlit_intervals_df.to_csv(sunlit_intervals_path, index=False)

    overlay_preview_path = save_polygon_preview(
        exposure_df=exposure_df,
        metadata=metadata,
    )

    print("Done.")
    print("Saved time series to:", timeseries_path)
    print("Saved summary to:", summary_path)
    print("Saved sunlit intervals to:", sunlit_intervals_path)

    if overlay_preview_path is not None:
        print("Saved bed polygon overlay preview to:", overlay_preview_path)

    print()
    print("Sunlit intervals:")
    print(sunlit_intervals_df)

    print()
    print("Status summary:")
    print(exposure_df["status"].value_counts(dropna=False))

    print()
    print(summary_df)


if __name__ == "__main__":
    main()