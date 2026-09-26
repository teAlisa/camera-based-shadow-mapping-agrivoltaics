"""
metadata.py

Purpose:
    Helper functions for creating image metadata and adding solar position
    information.

Main uses:
    - Find image files in the input image folder.
    - Parse timestamps from image filenames.
    - Add solar azimuth/elevation using pvlib.
    - Filter images to keep only daytime images.

Expected filename format:
    The image filename should contain a timestamp in this format:

        YYYYMMDDHHMMSS

    Example:
        RLN16-41024_09_20260705072106.jpg

    In this example, the timestamp is:

        2026-07-05 07:21:06

Notes:
    The parsed timestamp is treated as local naive time first, then localized
    to the provided timezone before calculating solar position.
"""

import os
import glob
import re

import numpy as np
import pandas as pd
import pvlib


def find_image_files(image_dir: str) -> list[str]:
    """
    Finds image files in a folder.

    Parameters
    ----------
    image_dir : str
        Directory containing input images.

    Returns
    -------
    list of str
        Sorted list of image file paths.

    Notes
    -----
    Supported extensions:
        - .jpg
        - .jpeg
        - .png
    """

    image_files = sorted(
        glob.glob(os.path.join(image_dir, "*.jpg"))
        + glob.glob(os.path.join(image_dir, "*.jpeg"))
        + glob.glob(os.path.join(image_dir, "*.png"))
    )

    return image_files


def parse_timestamp_from_base(base: str) -> pd.Timestamp:
    """
    Parses a timestamp from an image base filename.

    Parameters
    ----------
    base : str
        Image filename without extension.

    Returns
    -------
    pd.Timestamp
        Parsed timestamp if found. Otherwise pd.NaT.

    Expected timestamp format
    -------------------------
    YYYYMMDDHHMMSS

    Example
    -------
    Input:
        RLN16-41024_09_20260705072106

    Parsed timestamp:
        2026-07-05 07:21:06

    Notes
    -----
    The regular expression searches for a 14-digit timestamp starting with 20.
    """

    match = re.search(r"(20\d{12})", base)

    if match:
        ts_str = match.group(1)

        return pd.to_datetime(
            ts_str,
            format="%Y%m%d%H%M%S",
        )

    return pd.NaT


def create_image_metadata(image_dir: str) -> pd.DataFrame:
    """
    Creates a metadata table from image files.

    Parameters
    ----------
    image_dir : str
        Directory containing input images.

    Returns
    -------
    pd.DataFrame
        Metadata table with one row per image.

    Output columns
    --------------
    image_path : str
        Full path to the image.
    image_name : str
        Image filename with extension.
    base : str
        Image filename without extension.
    timestamp : pd.Timestamp
        Timestamp parsed from the image filename.
    """

    image_files = find_image_files(image_dir)

    image_meta = pd.DataFrame(
        {
            "image_path": image_files,
        }
    )

    image_meta["image_name"] = image_meta["image_path"].apply(
        os.path.basename
    )

    image_meta["base"] = image_meta["image_name"].apply(
        lambda filename: os.path.splitext(filename)[0]
    )

    # Extract timestamp from the base filename.
    image_meta["timestamp"] = image_meta["base"].apply(
        parse_timestamp_from_base
    )

    return image_meta


def add_solar_position(
    image_meta: pd.DataFrame,
    latitude: float,
    longitude: float,
    timezone: str,
    altitude: float = 0,
) -> pd.DataFrame:
    """
    Adds solar position columns using pvlib.

    Parameters
    ----------
    image_meta : pd.DataFrame
        Metadata table created by create_image_metadata().
    latitude : float
        Site latitude in decimal degrees.
    longitude : float
        Site longitude in decimal degrees.
    timezone : str
        Local timezone name, for example "America/Detroit".
    altitude : float
        Site altitude in meters.

    Returns
    -------
    pd.DataFrame
        Metadata table with added solar position columns.

    Added columns
    -------------
    timestamp_local : timezone-aware datetime
        Timestamp localized to the site timezone.
    apparent_zenith : float
        Solar apparent zenith angle in degrees.
    zenith : float
        Solar zenith angle in degrees.
    apparent_elevation : float
        Solar apparent elevation angle in degrees.
    elevation : float
        Solar elevation angle in degrees.
    azimuth : float
        Solar azimuth angle in degrees.
    equation_of_time : float
        Equation of time from pvlib.

    Notes
    -----
    The input timestamp is assumed to be local naive time. This means the
    filename timestamp is treated as already being in the local site timezone,
    not UTC.

    Rows with missing timestamps are kept, but solar position values remain NaN.
    """

    image_meta = image_meta.copy()

    if "timestamp" not in image_meta.columns:
        raise ValueError("Metadata must contain a 'timestamp' column.")

    valid = image_meta["timestamp"].notna()

    # Create as object first so timezone-aware datetimes do not cause
    # dtype conflicts when assigned into the DataFrame.
    image_meta["timestamp_local"] = pd.NaT
    image_meta["timestamp_local"] = image_meta["timestamp_local"].astype("object")

    # Localize valid timestamps to the site timezone before calling pvlib.
    localized_times = image_meta.loc[valid, "timestamp"].dt.tz_localize(
        timezone
    )

    image_meta.loc[valid, "timestamp_local"] = localized_times

    # Calculate solar position only for images with valid timestamps.
    solar_position = pvlib.solarposition.get_solarposition(
        time=localized_times,
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
    )

    # Assign solar position columns directly by row index.
    # This avoids merge/index issues and keeps the original metadata order.
    solar_cols = [
        "apparent_zenith",
        "zenith",
        "apparent_elevation",
        "elevation",
        "azimuth",
        "equation_of_time",
    ]

    for col in solar_cols:
        image_meta[col] = np.nan
        image_meta.loc[valid, col] = solar_position[col].values

    return image_meta


def filter_daytime_images(
    image_meta: pd.DataFrame,
    min_elevation: float = 10,
) -> pd.DataFrame:
    """
    Filters metadata to daytime images using solar elevation.

    Parameters
    ----------
    image_meta : pd.DataFrame
        Metadata table with an elevation column.
    min_elevation : float
        Minimum solar elevation angle in degrees.

    Returns
    -------
    pd.DataFrame
        Filtered metadata table containing only images where solar elevation
        is greater than min_elevation.

    Notes
    -----
    Images with low solar elevation are excluded because sunrise/sunset and
    low-light conditions can produce long, unstable, or unclear shadows.
    """

    if "elevation" not in image_meta.columns:
        raise ValueError("Metadata must contain an 'elevation' column.")

    return image_meta[image_meta["elevation"] > min_elevation].copy()