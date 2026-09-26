"""
findCoordinates.py

Purpose:
    Helper script for manually selecting pixel coordinates on an image.

Main uses:
    - Find BED_POLYGON coordinates for bed sun-exposure analysis.
    - Find DEFAULT_SAM_POINTS for SAM ground-mask prompts.
    - Find DEFAULT_PANEL_ROWS_PIXELS for panel row overlays.
    - Check ROI corner coordinates.

How to use:
    1. Change IMAGE_PATH to the image you want to use.
    2. Run this script.
    3. Click points on the image.
    4. Press 'q' to quit.
    5. Copy the printed coordinates into the relevant script/module.

Notes:
    The displayed image may be resized to fit the screen, but the printed
    coordinates are converted back to the original image size.
"""

import os

import cv2


# ---------------------------------------------------------------------
# Image path
# ---------------------------------------------------------------------

# Change this path to the image you want to use for coordinate selection.
IMAGE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "sample_data", "images", "RLN16-41024_09_20260705100111.jpg",
)


# ---------------------------------------------------------------------
# Display settings
# ---------------------------------------------------------------------

# Maximum display size for the OpenCV window.
# The image will only be scaled down if it is larger than these limits.
MAX_WIDTH = 1500
MAX_HEIGHT = 1000


coordinates = []


def get_coordinates(event, x, y, flags, param):
    """
    Mouse callback function for recording clicked coordinates.

    Parameters
    ----------
    event : int
        OpenCV mouse event type.
    x, y : int
        Click coordinates on the displayed resized image.
    flags :
        OpenCV event flags. Not used here.
    param : dict
        Dictionary containing scale_x and scale_y.

    Notes
    -----
    If the image is resized for display, the clicked coordinates are converted
    back to the original image coordinate system before being saved.
    """

    if event == cv2.EVENT_LBUTTONDOWN:
        # Convert displayed-image coordinates back to original-image coordinates.
        orig_x = int(x * param["scale_x"])
        orig_y = int(y * param["scale_y"])

        print(f"Clicked at: ({orig_x}, {orig_y})")
        coordinates.append([orig_x, orig_y])


def main():
    """
    Opens an image and records clicked pixel coordinates.
    """

    image = cv2.imread(IMAGE_PATH)

    if image is None:
        print("Failed to load image. Check the file path:")
        print(IMAGE_PATH)
        return

    # Original image dimensions.
    orig_height, orig_width = image.shape[:2]

    # Scale the image to fit on screen, but never upscale small images.
    scale = min(
        MAX_WIDTH / orig_width,
        MAX_HEIGHT / orig_height,
        1,
    )

    new_width = int(orig_width * scale)
    new_height = int(orig_height * scale)

    resized_image = cv2.resize(
        image,
        (new_width, new_height),
    )

    # Scale factors used to convert displayed coordinates back to original size.
    scale_x = 1 / scale
    scale_y = 1 / scale

    cv2.imshow("Image", resized_image)

    cv2.setMouseCallback(
        "Image",
        get_coordinates,
        {
            "scale_x": scale_x,
            "scale_y": scale_y,
        },
    )

    print("Click points on the image to record coordinates.")
    print("Press 'q' to quit.")
    print()

    while True:
        cv2.imshow("Image", resized_image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

    print()
    print("All recorded coordinates relative to the original image:")
    print(coordinates)
    print()
    print("Image height and width:", image.shape[:2])


if __name__ == "__main__":
    main()