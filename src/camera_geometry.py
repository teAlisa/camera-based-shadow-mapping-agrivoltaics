"""
camera_geometry.py

Purpose:
    Helper functions for camera projection geometry.

    This module converts between:
        - world coordinates on the ground plane, in centimeters
        - image pixel coordinates, in pixels

Main uses:
    - Project ground-grid cells into the camera image.
    - Convert manually selected panel-row pixels back onto the ground plane.
    - Build an approximate camera intrinsic matrix from image size and HFOV.

Coordinate systems:
    World coordinates:
        X, Y are measured on the ground plane in centimeters.
        Z is assumed to be 0 for ground-plane points.

    Image coordinates:
        u, v are pixel coordinates.
        u increases left to right.
        v increases top to bottom.

Notes:
    The camera model uses OpenCV conventions.
"""

import math

import cv2
import numpy as np


def make_K(
    width: int,
    height: int,
    hfov_deg: float,
    cx: float | None = None,
    cy: float | None = None,
) -> np.ndarray:
    """
    Creates an approximate camera intrinsic matrix K.

    The focal length is estimated from the image width and the horizontal
    field of view. The same focal length is used for both x and y directions.

    Parameters
    ----------
    width : int
        Image width in pixels.
    height : int
        Image height in pixels.
    hfov_deg : float
        Horizontal field of view in degrees.
    cx : float or None
        Principal point x-coordinate. If None, the image center is used.
    cy : float or None
        Principal point y-coordinate. If None, the image center is used.

    Returns
    -------
    np.ndarray
        3x3 camera intrinsic matrix.

    Notes
    -----
    K has the form:

        [fx,  0, cx]
        [ 0, fy, cy]
        [ 0,  0,  1]
    """

    # Estimate focal length from horizontal field of view.
    fx = (width / 2) / math.tan(math.radians(hfov_deg) / 2)

    # Assume square pixels, so fy is the same as fx.
    fy = fx

    # Use image center as principal point unless custom values are provided.
    cx = width / 2 if cx is None else cx
    cy = height / 2 if cy is None else cy

    return np.array(
        [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )


def world_to_pixel_points(
    world_xy_cm: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> np.ndarray:
    """
    Projects world ground-plane points into image pixel coordinates.

    Parameters
    ----------
    world_xy_cm : np.ndarray
        Array of shape (N, 2), where each row is [X, Y] in centimeters.
        All points are assumed to lie on the ground plane Z = 0.
    rvec : np.ndarray
        Camera rotation vector from the camera model.
    tvec : np.ndarray
        Camera translation vector from the camera model.
    K : np.ndarray
        Camera intrinsic matrix.
    dist : np.ndarray
        Distortion coefficients.

    Returns
    -------
    np.ndarray
        Array of shape (N, 2), where each row is [u, v] in pixel coordinates.

    Notes
    -----
    This function is used when projecting ground-grid cell corners into the
    image so that each cell can be matched with shadow pixels.
    """

    n_points = len(world_xy_cm)

    # Convert 2D ground-plane coordinates [X, Y] into 3D points [X, Y, Z].
    # Since all points are on the ground, Z is set to 0.
    world_points_3d = np.zeros((n_points, 3), dtype=np.float32)
    world_points_3d[:, 0] = world_xy_cm[:, 0]
    world_points_3d[:, 1] = world_xy_cm[:, 1]
    world_points_3d[:, 2] = 0.0

    # OpenCV projects 3D world points into 2D image pixels using the camera
    # rotation, translation, intrinsic matrix, and distortion coefficients.
    img_points, _ = cv2.projectPoints(
        objectPoints=world_points_3d,
        rvec=rvec,
        tvec=tvec,
        cameraMatrix=K,
        distCoeffs=dist,
    )

    return img_points.reshape(n_points, 2)


def pixel_to_world_on_plane(
    u: float,
    v: float,
    K: np.ndarray,
    dist: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    z_plane: float = 0.0,
) -> tuple[float, float]:
    """
    Converts one image pixel coordinate to world X, Y coordinates on a plane.

    Parameters
    ----------
    u : float
        Pixel x-coordinate.
    v : float
        Pixel y-coordinate.
    K : np.ndarray
        Camera intrinsic matrix.
    dist : np.ndarray
        Distortion coefficients.
    R : np.ndarray
        Camera rotation matrix.
    t : np.ndarray
        Camera translation vector.
    z_plane : float
        World Z coordinate of the plane. For the ground plane, this is 0.

    Returns
    -------
    tuple of float
        World coordinates (X, Y) where the pixel ray intersects the plane.

    Notes
    -----
    This function casts a ray from the camera through the selected pixel and
    finds where that ray intersects the ground plane.

    In this project, it is used to convert manually selected panel-row points
    from image pixels into ground-grid coordinates.
    """

    # Put the pixel into OpenCV's expected shape.
    pts = np.array([[[u, v]]], dtype=np.float32)

    # Remove lens distortion and convert the pixel to normalized camera
    # coordinates. The result is not in pixels anymore.
    undist = cv2.undistortPoints(pts, K, dist)

    x_cam = float(undist[0, 0, 0])
    y_cam = float(undist[0, 0, 1])

    # Ray direction in camera coordinates.
    ray_cam = np.array([x_cam, y_cam, 1.0], dtype=np.float64)

    # Convert camera center and ray direction into world coordinates.
    Rt = R.T
    C_world = -Rt @ t.reshape(3, 1)
    ray_world = Rt @ ray_cam.reshape(3, 1)

    Cz = C_world[2, 0]
    rz = ray_world[2, 0]

    # If rz is almost zero, the ray is parallel to the plane and will never
    # intersect it reliably.
    if abs(rz) < 1e-10:
        raise ValueError("Ray is parallel to the ground plane.")

    # Solve for the scale factor s where:
    #     C_world + s * ray_world
    # reaches the requested plane height z_plane.
    s = (z_plane - Cz) / rz
    P_world = C_world + s * ray_world

    X = float(P_world[0, 0])
    Y = float(P_world[1, 0])

    return X, Y


def batch_pixels_to_world(
    img_pts: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    z_plane: float = 0.0,
) -> np.ndarray:
    """
    Converts multiple image pixel points to world X, Y coordinates on a plane.

    Parameters
    ----------
    img_pts : np.ndarray
        Array of shape (N, 2), where each row is [u, v] in pixel coordinates.
    K : np.ndarray
        Camera intrinsic matrix.
    dist : np.ndarray
        Distortion coefficients.
    R : np.ndarray
        Camera rotation matrix.
    t : np.ndarray
        Camera translation vector.
    z_plane : float
        World Z coordinate of the plane. For the ground plane, this is 0.

    Returns
    -------
    np.ndarray
        Array of shape (N, 2), where each row is [X, Y] in world coordinates.

    Notes
    -----
    This is the vectorized version of pixel_to_world_on_plane().
    It is faster when many pixel points need to be converted at once.
    """

    # Convert input pixels to OpenCV shape: (N, 1, 2).
    pts = img_pts.astype(np.float32).reshape(-1, 1, 2)

    # Remove lens distortion and convert to normalized camera coordinates.
    undist = cv2.undistortPoints(pts, K, dist)

    Rt = R.T

    # Camera center in world coordinates.
    C_world = -Rt @ t.reshape(3, 1)

    n_points = undist.shape[0]

    # Build one camera ray for each undistorted pixel.
    rays_cam = np.hstack(
        [
            undist[:, 0, :],
            np.ones((n_points, 1), dtype=np.float32),
        ]
    )

    # Rotate all rays from camera coordinates into world coordinates.
    rays_world = (Rt @ rays_cam.T).T

    Cz = C_world[2, 0]
    rz = rays_world[:, 2]

    if np.any(np.abs(rz) < 1e-10):
        raise ValueError("One or more rays are parallel to the ground plane.")

    # Find where each ray intersects the target plane.
    s = (z_plane - Cz) / rz
    P_world = C_world.reshape(1, 3) + s[:, None] * rays_world

    return P_world[:, :2]