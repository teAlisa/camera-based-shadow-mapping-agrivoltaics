"""
shadow_model.py

Purpose:
    Helper functions for estimating shadow probability from camera images.

Main uses:
    - Normalize image brightness using an estimated illumination field.
    - Convert normalized brightness ratio into shadow probability.
    - Adjust shadow prediction for cloudy / low-contrast scenes.
    - Smooth probability maps inside the valid ground mask.
    - Convert probability maps into clean binary shadow masks.

Model idea:
    Shadows are usually darker than nearby sunlit ground. Instead of using raw
    brightness directly, this module estimates a smooth illumination background
    and compares each pixel's brightness to that local illumination estimate.

    ratio = pixel_lightness / estimated_illumination

    Lower ratio values usually indicate darker, more shadow-like pixels.

Notes:
    All calculations are limited to the final_mask region. Pixels outside the
    valid ground mask are set to NaN in the probability map and 0 in the binary
    shadow mask.
"""

import cv2
import numpy as np


def compute_ratio(
    image_bgr: np.ndarray,
    final_mask: np.ndarray,
    illum_kernel_size: int = 301,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes an illumination-normalized brightness ratio.

    Parameters
    ----------
    image_bgr : np.ndarray
        Input image in OpenCV BGR format.
    final_mask : np.ndarray
        Binary valid ground mask. Pixels equal to 0 are excluded.
    illum_kernel_size : int
        Kernel size used to estimate smooth illumination. Larger values create
        a smoother illumination field.

    Returns
    -------
    tuple
        ratio : np.ndarray
            Lightness divided by estimated illumination. Pixels outside the
            valid mask are set to NaN.
        illumination : np.ndarray
            Estimated smooth illumination image.

    Notes
    -----
    The image is converted to LAB color space, and the L channel is used as
    the brightness/lightness channel.

    Morphological closing estimates the broad illumination field. This helps
    reduce the effect of uneven lighting across the camera view.
    """

    # Convert from BGR to LAB and use the L channel as brightness/lightness.
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)

    # Large elliptical kernel used to estimate broad illumination patterns.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (illum_kernel_size, illum_kernel_size),
    )

    # Morphological closing fills darker regions and gives a smooth estimate
    # of local illumination.
    illumination = cv2.morphologyEx(
        L.astype(np.uint8),
        cv2.MORPH_CLOSE,
        kernel,
    ).astype(np.float32)

    # Avoid division by zero or extremely small illumination values.
    illumination[illumination < 1] = 1

    # Ratio close to 1 means similar to local illumination.
    # Lower ratio values usually indicate shadow.
    ratio = L / illumination

    # Ignore pixels outside the valid ground mask.
    ratio[final_mask == 0] = np.nan

    return ratio, illumination


def get_scene_stats(
    image_bgr: np.ndarray,
    ratio: np.ndarray,
    final_mask: np.ndarray,
) -> tuple[float | None, float | None]:
    """
    Computes simple scene-level statistics used for cloudy-image adjustment.

    Parameters
    ----------
    image_bgr : np.ndarray
        Input image in OpenCV BGR format.
    ratio : np.ndarray
        Illumination-normalized ratio image.
    final_mask : np.ndarray
        Binary valid ground mask.

    Returns
    -------
    tuple
        illumination_contrast : float or None
            Difference between the 90th and 10th percentile of ratio values.
            Lower values suggest a flatter, lower-contrast scene.
        warmth : float or None
            Mean red channel divided by mean blue channel inside the valid mask.
            Lower values can indicate cooler/cloudier lighting.

    Notes
    -----
    These values are used to detect cloudy or low-contrast images where shadow
    boundaries are less obvious.
    """

    valid = (final_mask > 0) & np.isfinite(ratio)
    ratio_vals = ratio[valid]

    if ratio_vals.size == 0:
        return None, None

    # Contrast estimate based on percentile spread.
    # This is more robust than max-min because it ignores extreme outliers.
    p10 = np.percentile(ratio_vals, 10)
    p90 = np.percentile(ratio_vals, 90)

    illumination_contrast = p90 - p10

    b_channel = image_bgr[:, :, 0].astype(np.float32)
    r_channel = image_bgr[:, :, 2].astype(np.float32)

    mean_r = np.mean(r_channel[valid])
    mean_b = np.mean(b_channel[valid])

    # Warmth is a simple red/blue ratio.
    # Cloudy scenes can look cooler and have lower warmth.
    warmth = mean_r / (mean_b + 1e-6)

    return illumination_contrast, warmth


def remove_small_components(
    binary_mask: np.ndarray,
    min_area: int = 300,
) -> np.ndarray:
    """
    Removes small connected components from a binary mask.

    Parameters
    ----------
    binary_mask : np.ndarray
        Binary mask where shadow pixels are 1 and non-shadow pixels are 0.
    min_area : int
        Minimum connected-component area in pixels to keep.

    Returns
    -------
    np.ndarray
        Cleaned binary mask.

    Notes
    -----
    This removes tiny isolated shadow detections that are likely noise.
    """

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask.astype(np.uint8),
        connectivity=8,
    )

    clean_mask = np.zeros(binary_mask.shape, dtype=np.uint8)

    # Label 0 is background, so start from 1.
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= min_area:
            clean_mask[labels == i] = 1

    return clean_mask


def smooth_probability_masked(
    shadow_prob: np.ndarray,
    final_mask: np.ndarray,
    ksize: tuple[int, int] = (15, 3),
) -> np.ndarray:
    """
    Smooths a shadow probability map only inside the valid ground mask.

    Parameters
    ----------
    shadow_prob : np.ndarray
        Shadow probability map with values from 0 to 1. Pixels outside the
        valid mask may be NaN.
    final_mask : np.ndarray
        Binary valid ground mask.
    ksize : tuple of int
        Blur kernel size.

    Returns
    -------
    np.ndarray
        Smoothed probability map. Pixels outside the valid mask are NaN.

    Notes
    -----
    A normal blur would mix valid ground pixels with invalid background pixels.
    This function avoids that by normalizing the blurred probability by the
    blurred valid-mask weights.
    """

    # Replace NaN with 0 for convolution, but keep a separate valid-mask weight.
    prob = np.nan_to_num(shadow_prob, nan=0).astype(np.float32)
    mask_float = (final_mask > 0).astype(np.float32)

    # Blur the weighted probability and the mask separately.
    prob_blur = cv2.blur(prob * mask_float, ksize)
    mask_blur = cv2.blur(mask_float, ksize)

    smoothed = np.zeros_like(prob_blur, dtype=np.float32)

    # Only divide where the blurred mask has enough valid pixels.
    valid = mask_blur > 1e-6
    smoothed[valid] = prob_blur[valid] / mask_blur[valid]

    # Preserve NaN outside the valid ground mask.
    smoothed[final_mask == 0] = np.nan

    return smoothed


def predict_shadow_from_ratio(
    ratio: np.ndarray,
    final_mask: np.ndarray,
    image_bgr: np.ndarray,
    t: float = 0.45,
    k: float = 30,
    binary_threshold: float = 0.80,
    cleanup_kernel_size: int = 3,
    min_component_area: int = 300,
) -> tuple[np.ndarray, np.ndarray, float | None, float | None]:
    """
    Converts an illumination-normalized ratio image into shadow predictions.

    Parameters
    ----------
    ratio : np.ndarray
        Illumination-normalized brightness ratio.
    final_mask : np.ndarray
        Binary valid ground mask.
    image_bgr : np.ndarray
        Original image in OpenCV BGR format.
    t : float
        Ratio threshold used in the sigmoid probability model.
        Lower ratio values become more shadow-like.
    k : float
        Sigmoid steepness. Higher values make the transition sharper.
    binary_threshold : float
        Probability threshold used to convert shadow probability to a binary
        shadow mask.
    cleanup_kernel_size : int
        Kernel size for morphological cleanup. If None or <= 1, cleanup is
        skipped.
    min_component_area : int
        Minimum connected-component area in pixels to keep.

    Returns
    -------
    tuple
        shadow_prob : np.ndarray
            Smoothed shadow probability map with values from 0 to 1.
        pred_shadow : np.ndarray
            Clean binary shadow mask where 1 = shadow and 0 = non-shadow.
        illumination_contrast : float or None
            Scene contrast statistic.
        warmth : float or None
            Scene warmth statistic.

    Notes
    -----
    Shadow probability is calculated using a sigmoid function:

        shadow_prob = 1 / (1 + exp(k * (ratio - t)))

    This means:
        - ratio < t gives higher shadow probability
        - ratio > t gives lower shadow probability

    For cloudy, low-contrast images, the model slightly boosts probabilities
    and lowers the binary threshold so soft shadows are not completely missed.
    """

    illumination_contrast, warmth = get_scene_stats(
        image_bgr,
        ratio,
        final_mask,
    )

    # Convert ratio values to shadow probability.
    # Darker-than-local-illumination pixels get higher probability.
    shadow_prob = 1 / (1 + np.exp(k * (ratio - t)))

    # Cloudy / low-contrast adjustment.
    # In cloudy scenes, shadows are softer and the ratio contrast is smaller.
    # This adjustment makes the model slightly more sensitive in those cases.
    if illumination_contrast is not None and warmth is not None:
        if illumination_contrast < 0.35 and warmth < 1.25:
            cloudy_strength = (0.35 - illumination_contrast) / 0.35
            cloudy_strength = np.clip(cloudy_strength, 0, 1)

            # Slightly boost shadow probabilities for cloudy scenes.
            shadow_prob = shadow_prob ** (1 - 0.20 * cloudy_strength)

            # Lower the binary threshold, but keep a minimum threshold so the
            # model does not become too aggressive.
            binary_threshold = binary_threshold - 0.10 * cloudy_strength
            binary_threshold = max(binary_threshold, 0.74)

    # Exclude pixels outside the valid ground mask.
    shadow_prob[final_mask == 0] = np.nan

    # Smooth probabilities inside the valid mask while avoiding background bleed.
    shadow_prob = smooth_probability_masked(
        shadow_prob,
        final_mask,
        ksize=(15, 3),
    )

    # Convert probability map into binary shadow mask.
    pred_shadow = (shadow_prob > binary_threshold).astype(np.uint8)
    pred_shadow[final_mask == 0] = 0

    # Remove tiny isolated components.
    pred_shadow = remove_small_components(
        pred_shadow,
        min_area=min_component_area,
    )

    # Morphological cleanup:
    #   - OPEN removes small noise
    #   - CLOSE fills small holes
    if cleanup_kernel_size is not None and cleanup_kernel_size > 1:
        kernel_cleanup = np.ones(
            (cleanup_kernel_size, cleanup_kernel_size),
            np.uint8,
        )

        pred_shadow = cv2.morphologyEx(
            pred_shadow,
            cv2.MORPH_OPEN,
            kernel_cleanup,
        )

        pred_shadow = cv2.morphologyEx(
            pred_shadow,
            cv2.MORPH_CLOSE,
            kernel_cleanup,
        )

    return shadow_prob, pred_shadow, illumination_contrast, warmth