"""Problem Set 4: Geometry."""

import numpy as np
import cv2

import os

# I/O directories
input_dir = "input"
output_dir = "output"

# Input files
PIC_A = "pic_a.jpg"
PIC_A_2D = "pts2d-pic_a.txt"
PIC_A_2D_NORM = "pts2d-norm-pic_a.txt"
PIC_B = "pic_b.jpg"
PIC_B_2D = "pts2d-pic_b.txt"
SCENE = "pts3d.txt"
SCENE_NORM = "pts3d-norm.txt"

# Utility code
def read_points(filename):
    """Read point data from given file and return as NumPy array."""
    with open(filename) as f:
        lines = f.readlines()
        pts = []
        for line in lines:
            pts.append(map(float, line.split()))
    return np.array(pts)


# Assignment code
def solve_least_squares(pts3d, pts2d):
    """Solve for transformation matrix M that maps each 3D point to corresponding 2D point using the least squares method.

    Parameters
    ----------
        pts3d: 3D (object) points, NumPy array of shape (N, 3)
        pts2d: corresponding 2D (image) points, NumPy array of shape (N, 2)

    Returns
    -------
        M: transformation matrix, NumPy array of shape (3, 4)
        error: sum of squared residuals of all points
    """

    # TODO: Your code here
    return M, error


def project_points(pts3d, M):
    """Project each 3D point to 2D using matrix M.

    Parameters
    ----------
        pts3d: 3D (object) points, NumPy array of shape (N, 3)
        M: projection matrix, NumPy array of shape (3, 4)

    Returns
    -------
        pts2d_projected: projected 2D points, NumPy array of shape (N, 2)
    """

    # TODO: Your code here
    return pts2d_projected    


def get_residuals(pts2d, pts2d_projected):
    """Compute residual error for each point.

    Parameters
    ----------
        pts2d: observed 2D (image) points, NumPy array of shape (N, 2)
        pts2d_projected: 3D (object) points projected to 2D, NumPy array of shape (N, 3)

    Returns
    -------
        residuals: residual error for each point (L2 distance between observed and projected 2D points)
    """

    # TODO: Your code here
    return residuals


def calibrate_camera(pts3d, pts2d):
    """Find the best camera projection matrix given corresponding 3D and 2D points.

    Parameters
    ----------
        pts3d: 3D (object) points, NumPy array of shape (N, 3)
        pts2d: corresponding 2D (image) points, NumPy array of shape (N, 2)

    Returns
    -------
        bestM: best transformation matrix, NumPy array of shape (3, 4)
        error: sum of squared residuals of all points for bestM
    """

    # TODO: Your code here
    # NOTE: Use the camera calibration procedure in the problem set
    return bestM, error


def compute_fundamental_matrix(pts2d_a, pts2d_b):
    """Compute fundamental matrix given corresponding points from 2 images of a scene.

    Parameters
    ----------
        pts2d_a: 2D points from image A, NumPy array of shape (N, 2)
        pts2d_b: corresponding 2D points from image B, NumPy array of shape (N, 2)

    Returns
    -------
        F: the fundamental matrix
    """

    # TODO: Your code here
    return F


# Driver code
def main():
    """Driver code."""

    # 1a
    # Read points
    pts3d_norm = read_points(os.path.join(input_dir, SCENE_NORM))
    pts2d_norm_pic_a = read_points(os.path.join(input_dir, PIC_A_2D_NORM))

    # Solve for transformation matrix using least squares
    M, error = solve_least_squares(pts3d_norm, pts2d_norm_pic_a)  # TODO: implement this

    # Project 3D points to 2D
    pts2d_projected = project_points(pts3d_norm, M)  # TODO: implement this

    # Compute residual error for each point
    residuals = get_residuals(pts2d_norm_pic_a, pts2d_projected)  # TODO: implement this

    # TODO: Print the <u, v> projection of the last point, and the corresponding residual

    # 1b
    # Read points
    pts3d = read_points(os.path.join(input_dir, SCENE))
    pts2d_pic_b = read_points(os.path.join(input_dir, PIC_B_2D))
    # NOTE: These points are not normalized

    # TODO: Use the functions from 1a to implement calibrate_camera() and find the best transform (bestM)

    # 1c
    # TODO: Compute the camera location using bestM

    # 2a
    # TODO: Implement compute_fundamental_matrix() to find the raw fundamental matrix

    # 2b
    # TODO: Reduce the rank of the fundamental matrix

    # 2c
    # TODO: Draw epipolar lines


if __name__ == '__main__':
    main()
