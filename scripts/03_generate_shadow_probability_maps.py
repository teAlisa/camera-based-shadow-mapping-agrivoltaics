"""
03_generate_shadow_probability_maps.py

Purpose:
    Runs shadow detection for each daytime image using the existing SAM+ROI
    ground masks. For each image, the script creates a binary predicted shadow
    mask, a visual shadow probability map, and a saved NumPy probability array.

Inputs:
    - image_metadata.csv from 01_create_image_metadata.py
    - Raw images from sample_data/images
    - SAM+ROI masks from 02_create_sam_roi_masks.py

Outputs:
    - Binary predicted shadow masks in sample_outputs/pred_masks
    - Shadow probability map images in sample_outputs/shadow_probability_maps
    - Shadow probability arrays (.npy) in sample_outputs/shadow_probability_maps
    - shadow_detection_summary.csv

Notes:
    The model uses an illumination-normalized image ratio. This helps reduce
    the effect of uneven brightness across the image before estimating shadow
    probability.

    The saved .npy probability arrays are used later for polygon/grid-level
    sunlight and shade analysis.
"""

import os
import sys

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# This allows the script to import from src/ when run from the project root.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.shadow_model import (
    compute_ratio,
    predict_shadow_from_ratio,
)

from src.visualization import (
    save_probability_map,
    save_probability_array,
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

FINAL_MASK_DIR = os.path.join(
    BASE_DIR,
    "sample_data",
    "masks",
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "sample_outputs",
)

PRED_MASK_DIR = os.path.join(
    OUTPUT_DIR,
    "pred_masks",
)

PROB_MAP_DIR = os.path.join(
    OUTPUT_DIR,
    "shadow_probability_maps",
)

METADATA_PATH = os.path.join(
    BASE_DIR,
    "sample_data",
    "metadata",
    "image_metadata.csv",
)

for directory in [OUTPUT_DIR, PRED_MASK_DIR, PROB_MAP_DIR]:
    os.makedirs(directory, exist_ok=True)


# ---------------------------------------------------------------------
# Shadow model settings
# ---------------------------------------------------------------------

# Threshold used in the sigmoid shadow probability model.
# Lower values make the model more likely to classify darker pixels as shadow.
T = 0.45

# Controls how sharp the transition is between sunlit and shaded pixels.
# Higher values make the probability change more abruptly around T.
K = 30

# Probability threshold used to convert the shadow probability map into
# a binary shadow mask.
BINARY_THRESHOLD = 0.80

# Kernel size used to estimate smooth illumination across the image.
# A large kernel helps capture broad lighting changes while preserving
# local dark regions caused by shadows.
ILLUM_KERNEL_SIZE = 301

# Small cleanup kernel used to remove tiny noisy regions from the binary mask.
CLEANUP_KERNEL_SIZE = 3


def build_failure_result(
    image_name: str,
    status: str,
    image_path: str,
    mask_path: str,
    error_message: str | None = None,
) -> dict:
    """
    Creates a consistent result row for images that cannot be processed.

    Parameters
    ----------
    image_name : str
        Name of the image file.
    status : str
        Processing status, such as missing_image or read_mask_failed.
    image_path : str
        Full path to the image.
    mask_path : str
        Full path to the corresponding SAM+ROI mask.
    error_message : str or None
        Optional error message if processing failed with an exception.

    Returns
    -------
    dict
        One row for the processing summary table.
    """

    result = {
        "image_name": image_name,
        "status": status,
        "image_path": image_path,
        "mask_path": mask_path,
        "illumination_contrast": np.nan,
        "warmth": np.nan,
    }

    if error_message is not None:
        result["error_message"] = error_message

    return result


def process_one_image_existing_mask(
    image_name: str,
    save_outputs: bool = True,
) -> dict:
    """
    Processes one image using an existing SAM+ROI ground mask.

    Inputs:
        - image_name.jpg
        - image_name_sam_roi_mask.png

    Outputs:
        - image_name_pred_shadow.png
        - image_name_shadow_probability.png
        - image_name_shadow_prob.npy

    Parameters
    ----------
    image_name : str
        Name of the image file listed in the metadata table.
    save_outputs : bool
        If True, saves the predicted binary mask, probability map image,
        and probability array.

    Returns
    -------
    dict
        Processing status and output paths/statistics for this image.
    """

    image_path = os.path.join(IMAGE_DIR, image_name)
    base = os.path.splitext(image_name)[0]

    final_mask_path = os.path.join(
        FINAL_MASK_DIR,
        base + "_sam_roi_mask.png",
    )

    # Check files before cv2.imread so missing files are handled cleanly.
    if not os.path.exists(image_path):
        return build_failure_result(
            image_name=image_name,
            status="missing_image",
            image_path=image_path,
            mask_path=final_mask_path,
        )

    if not os.path.exists(final_mask_path):
        return build_failure_result(
            image_name=image_name,
            status="missing_mask",
            image_path=image_path,
            mask_path=final_mask_path,
        )

    image = cv2.imread(image_path)
    final_mask = cv2.imread(final_mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        return build_failure_result(
            image_name=image_name,
            status="read_image_failed",
            image_path=image_path,
            mask_path=final_mask_path,
        )

    if final_mask is None:
        return build_failure_result(
            image_name=image_name,
            status="read_mask_failed",
            image_path=image_path,
            mask_path=final_mask_path,
        )

    # The mask should match the image size. If not, resize it using nearest
    # neighbor so the mask remains binary.
    if final_mask.shape != image.shape[:2]:
        final_mask = cv2.resize(
            final_mask,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    # If the mask has no visible ground pixels, shadow analysis is impossible.
    if np.sum(final_mask > 0) == 0:
        return build_failure_result(
            image_name=image_name,
            status="empty_mask",
            image_path=image_path,
            mask_path=final_mask_path,
        )

    try:
        # Compute illumination-normalized ratio image. This helps separate
        # true shadows from broad brightness gradients in the camera view.
        ratio, illumination = compute_ratio(
            image,
            final_mask,
            illum_kernel_size=ILLUM_KERNEL_SIZE,
        )

        # Convert the ratio image into a shadow probability map and a binary
        # predicted shadow mask.
        shadow_prob, pred_shadow, illumination_contrast, warmth = (
            predict_shadow_from_ratio(
                ratio=ratio,
                final_mask=final_mask,
                image_bgr=image,
                t=T,
                k=K,
                binary_threshold=BINARY_THRESHOLD,
                cleanup_kernel_size=CLEANUP_KERNEL_SIZE,
            )
        )

        pred_save_path = os.path.join(
            PRED_MASK_DIR,
            base + "_pred_shadow.png",
        )

        prob_save_path = os.path.join(
            PROB_MAP_DIR,
            base + "_shadow_probability.png",
        )

        prob_array_save_path = os.path.join(
            PROB_MAP_DIR,
            base + "_shadow_prob.npy",
        )

        if save_outputs:
            cv2.imwrite(pred_save_path, pred_shadow * 255)
            save_probability_map(shadow_prob, prob_save_path)
            save_probability_array(shadow_prob, prob_array_save_path)

        return {
            "image_name": image_name,
            "status": "processed",
            "image_path": image_path,
            "mask_path": final_mask_path,
            "pred_mask_path": pred_save_path if save_outputs else None,
            "prob_map_path": prob_save_path if save_outputs else None,
            "prob_array_path": prob_array_save_path if save_outputs else None,
            "illumination_contrast": illumination_contrast,
            "warmth": warmth,
            "mean_shadow_probability": float(np.nanmean(shadow_prob)),
            "binary_shadow_fraction": float(np.mean(pred_shadow > 0)),
        }

    except Exception as error:
        return build_failure_result(
            image_name=image_name,
            status="error",
            image_path=image_path,
            mask_path=final_mask_path,
            error_message=str(error),
        )


def validate_metadata(metadata: pd.DataFrame) -> None:
    """
    Checks that the metadata table contains the image_name column.

    Parameters
    ----------
    metadata : pd.DataFrame
        Metadata table loaded from image_metadata.csv.

    Raises
    ------
    ValueError
        If the image_name column is missing.
    """

    if "image_name" not in metadata.columns:
        raise ValueError(
            "Metadata file must contain an 'image_name' column."
        )


def main():
    """
    Loads daytime metadata, processes each image, and saves the shadow
    detection summary table.
    """

    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            f"Could not find metadata file: {METADATA_PATH}"
        )

    metadata = pd.read_csv(METADATA_PATH)
    validate_metadata(metadata)

    results = []

    for image_name in tqdm(metadata["image_name"]):
        result = process_one_image_existing_mask(
            image_name=image_name,
            save_outputs=True,
        )
        results.append(result)

    results_df = pd.DataFrame(results)

    results_path = os.path.join(
        OUTPUT_DIR,
        "shadow_detection_summary.csv",
    )

    results_df.to_csv(results_path, index=False)

    print("Done.")
    print()
    print("Status summary:")
    print(results_df["status"].value_counts(dropna=False))
    print()
    print("Total rows in metadata:", len(metadata))
    print("Processed images:", (results_df["status"] == "processed").sum())
    print("Saved summary to:", results_path)


if __name__ == "__main__":
    main()