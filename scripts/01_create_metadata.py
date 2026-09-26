"""
01_create_image_metadata.py

Purpose:
    Creates a metadata table for all input images, adds solar position
    information, and filters images to keep only daytime images.

Inputs:
    - Raw camera images from sample_data/images

Outputs:
    - image_metadata_all.csv : metadata for all images
    - image_metadata.csv     : metadata for daytime images only

Notes:
    Daytime images are selected using solar elevation. Images with very low
    solar elevation are excluded because shadows are less reliable and the
    camera may capture sunrise/sunset or low-light conditions.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metadata import (
    create_image_metadata,
    add_solar_position,
    filter_daytime_images,
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

METADATA_DIR = os.path.join(
    BASE_DIR,
    "sample_data",
    "metadata",
)

os.makedirs(METADATA_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# Site information
# ---------------------------------------------------------------------

# Site location:
# 47°10'11.1"N, 88°30'25.7"W
LATITUDE = 47 + 10 / 60 + 11.1 / 3600
LONGITUDE = -(88 + 30 / 60 + 25.7 / 3600)
TIMEZONE = "America/Detroit"


# Minimum solar elevation used to keep an image for daytime analysis.
# Images below this threshold are removed because very low sun angles
# can create long/unclear shadows and low-light images.
MIN_SOLAR_ELEVATION = 10


def main():
    """
    Builds image metadata, adds solar position, filters daytime images,
    and saves both full and filtered metadata tables.
    """

    # Create metadata from image filenames, including parsed timestamps.
    metadata = create_image_metadata(IMAGE_DIR)

    # Add sun azimuth and elevation for each image timestamp.
    metadata = add_solar_position(
        image_meta=metadata,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        timezone=TIMEZONE,
        altitude=0,
    )

    # Keep only images where the sun is high enough for reliable analysis.
    daytime_metadata = filter_daytime_images(
        metadata,
        min_elevation=MIN_SOLAR_ELEVATION,
    )

    full_metadata_path = os.path.join(
        METADATA_DIR,
        "image_metadata_all.csv",
    )

    daytime_metadata_path = os.path.join(
        METADATA_DIR,
        "image_metadata.csv",
    )

    metadata.to_csv(full_metadata_path, index=False)
    daytime_metadata.to_csv(daytime_metadata_path, index=False)

    print("Done.")
    print("Images found:", len(metadata))
    print("Parsed timestamps:", metadata["timestamp"].notna().sum())
    print("Missing timestamps:", metadata["timestamp"].isna().sum())
    print("Daytime images:", len(daytime_metadata))
    print()
    print("Saved full metadata to:", full_metadata_path)
    print("Saved daytime metadata to:", daytime_metadata_path)


if __name__ == "__main__":
    main()