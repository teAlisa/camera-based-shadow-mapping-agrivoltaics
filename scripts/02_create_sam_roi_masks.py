"""
02_create_sam_roi_masks.py

Purpose:
    Creates ground/ROI masks for all daytime images using SAM and a fixed
    region of interest (ROI), then runs quality control checks on the masks.

Inputs:
    - image_metadata.csv from 01_create_image_metadata.py
    - Raw images from sample_data/images

Outputs:
    - Binary masks in sample_data/masks
    - sam_roi_creation_summary.csv
    - sam_roi_mask_qc.csv

Notes:
    SAM is used to detect the visible ground region. A fixed rectangular ROI
    is applied to limit the analysis to the area of interest and avoid panels,
    sky, and irrelevant background areas.

    The QC file helps identify masks that may be incomplete or suspicious.
"""

import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import SAM

from src.sam_masks import (
    DEFAULT_SAM_POINTS,
    DEFAULT_SAM_LABELS,
    create_sam_roi_masks_from_metadata,
    run_mask_qc,
)


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

MASK_DIR = os.path.join(
    BASE_DIR,
    "sample_data",
    "masks",
)

QC_DIR = os.path.join(
    BASE_DIR,
    "sample_outputs",
    "qc",
)

os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs(QC_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# SAM and ROI settings
# ---------------------------------------------------------------------

# SAM model used for automatic mask generation.
SAM_MODEL_PATH = "sam2_b.pt"

# Fixed region of interest for this camera view.
# The ROI removes areas that should not be analyzed, such as sky, panels,
# image borders, and unrelated background.
ROI_X1 = 225
ROI_Y1 = 384
ROI_X2 = 2606
ROI_Y2 = 1044

# If True, existing masks are not recomputed. This is useful because SAM
# can be slow, and masks usually do not need to be regenerated every time.
SKIP_EXISTING = True


def validate_metadata(metadata: pd.DataFrame) -> None:
    """
    Checks that the metadata file contains the columns required for mask
    creation.

    Parameters
    ----------
    metadata : pd.DataFrame
        Metadata table loaded from image_metadata.csv.

    Raises
    ------
    ValueError
        If required columns are missing.
    """

    required_cols = {"image_path", "base"}
    missing_cols = required_cols - set(metadata.columns)

    if missing_cols:
        raise ValueError(
            f"Metadata file is missing required columns: {missing_cols}"
        )


def main():
    """
    Loads metadata, creates SAM + ROI masks, runs QC checks, and saves
    summary tables.
    """

    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            f"Could not find metadata file: {METADATA_PATH}"
        )

    metadata = pd.read_csv(METADATA_PATH)
    validate_metadata(metadata)

    print("Loading SAM model:", SAM_MODEL_PATH)
    sam_model = SAM(SAM_MODEL_PATH)

    print("Creating SAM + ROI masks...")

    # Create one binary mask per image.
    # The final mask combines the SAM-selected region with the fixed ROI.
    sam_results_df = create_sam_roi_masks_from_metadata(
        metadata_df=metadata,
        output_mask_dir=MASK_DIR,
        sam_model=sam_model,
        points=DEFAULT_SAM_POINTS,
        labels=DEFAULT_SAM_LABELS,
        x1=ROI_X1,
        y1=ROI_Y1,
        x2=ROI_X2,
        y2=ROI_Y2,
        skip_existing=SKIP_EXISTING,
    )

    sam_results_path = os.path.join(
        QC_DIR,
        "sam_roi_creation_summary.csv",
    )

    sam_results_df.to_csv(sam_results_path, index=False)

    # Only masks corresponding to images in the metadata file should be
    # included in the QC summary.
    valid_bases = set(metadata["base"])

    # Run basic QC checks to detect masks that are too small, too large,
    # missing, or otherwise suspicious.
    mask_qc_df = run_mask_qc(
        mask_dir=MASK_DIR,
        valid_bases=valid_bases,
        x1=ROI_X1,
        y1=ROI_Y1,
        x2=ROI_X2,
        y2=ROI_Y2,
    )

    mask_qc_path = os.path.join(
        QC_DIR,
        "sam_roi_mask_qc.csv",
    )

    mask_qc_df.to_csv(mask_qc_path, index=False)

    print("Done.")
    print()
    print("SAM creation status:")
    print(sam_results_df["sam_status"].value_counts(dropna=False))
    print()
    print("QC masks:", len(mask_qc_df))

    if "is_suspicious" in mask_qc_df.columns:
        print("Suspicious masks:", int(mask_qc_df["is_suspicious"].sum()))

    print()
    print("Saved:")
    print(" -", sam_results_path)
    print(" -", mask_qc_path)


if __name__ == "__main__":
    main()