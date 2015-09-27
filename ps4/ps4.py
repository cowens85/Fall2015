"""Problem Set 4: Geometry."""

import numpy as np
import cv2
import random

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

    N = pts2d.shape[0]
    A = np.zeros((2 * N, 11))
    B = np.zeros((2 * N, 1))


    for i in range(0, N):
        u,v = pts2d[i]
        X, Y, Z = pts3d[i]
        A[2 * i] = np.array([X, Y, Z, 1, 0, 0, 0, 0, -u * X, -u * Y, -u * Z])
        B[2 * i] = u

        A[2 * i + 1] = np.array([ 0, 0, 0, 0,X, Y, Z, 1, -v * X, -v * Y, -v * Z])
        B[2 * i + 1] = v

    m, residuals, rank, singular_values = np.linalg.lstsq(A,B)

    error = ((B - (A * m.reshape((m.shape[1], m.shape[0])))) ** 2).sum()

    m_prime = np.zeros((m.shape[0]+1, m.shape[1]))
    m_prime[:-1] = m
    m_prime[-1] = 1
    M = m_prime.reshape((3, 4))

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

    N = (pts3d.shape[0])
    pts2d_proj = np.zeros((N, 2))

    #tack on a column of ones
    last_col = np.ones((N, 1))
    new_pts = np.concatenate((pts3d, last_col),1)

    #loop, dot product with M, normalize, set value in pointa2d
    for i in range(0, N):
        homo_proj_pt = np.dot(M, new_pts[i])
        inhomo_proj_pt = homo_proj_pt / homo_proj_pt[2]
        pts2d_proj[i] = np.array([inhomo_proj_pt[0], inhomo_proj_pt[1]])

    return pts2d_proj


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

    return np.linalg.norm(pts2d - pts2d_projected, axis=1)


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


    # NOTE: Use the camera calibration procedure in the problem set
    k_choices = [12, 16, 20]
    indexes = np.arange(pts3d.shape[0])

    lowest_avg_residual = np.infty
    bestM, error = None, None
    print "\n###\nPart 1b\n###\n\nAverage Residual per trial K:"
    for k in k_choices:
        lowest_avg_residual_k = np.inf
        bestM_k, error_k = None, None

        # subtract 4 due to the test points
        print "k:", k - 4
        for n in range(10):
            #choose k point indexes
            all_pt_indexes = random.sample(indexes, k)

            proj_pt_indexes = all_pt_indexes[5:]

            proj_pts2d = pts2d[proj_pt_indexes]
            proj_pts3d = pts3d[proj_pt_indexes]
            M, error = solve_least_squares(proj_pts3d, proj_pts2d)

            proj_pt_test_indexes = all_pt_indexes[:4]
            proj_pts2d = pts2d[proj_pt_test_indexes]
            proj_pts3d = pts3d[proj_pt_test_indexes]

            avg_residual = np.mean(get_residuals(proj_pts2d,project_points(proj_pts3d, M)))

            if avg_residual < lowest_avg_residual_k:
                lowest_avg_residual_k = avg_residual
                bestM_k = M
                error_k = error

            print lowest_avg_residual_k

        if lowest_avg_residual_k < lowest_avg_residual:
            lowest_avg_residual = lowest_avg_residual_k
            bestM = bestM_k
            error = error_k

        print ""

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
    N = pts2d_a.shape[0]
    A = np.zeros((N,8))
    B = np.ones((N,1)) * -1

    for i in range(N):
        u = pts2d_a[i][0]
        v = pts2d_a[i][1]
        u_prime = pts2d_b[i][0]
        v_prime = pts2d_b[i][1]

        A[i] = np.array([u * u_prime, v * u_prime, u_prime, u * v_prime, v * v_prime, v_prime, u, v])

    m, residuals, rank, singular_values = np.linalg.lstsq(A, B)

    m_prime = np.zeros((m.shape[0]+1, m.shape[1]))
    m_prime[:-1] = m
    m_prime[-1] = 1

    return m_prime.reshape((3, 3))

def get_trans_and_F(pts2d_a, pts2d_b):
    #
    # Compute T_a
    #
    T_avg = np.zeros((3, 3), dtype=np.float)
    np.fill_diagonal(T_avg, 1.0)
    u_avg = np.mean(pts2d_a[:, 0:1])
    v_avg = np.mean(pts2d_a[:, 1:2])
    T_avg[0][2] = -u_avg
    T_avg[1][2] = -v_avg

    T_scale = np.zeros((3, 3),dtype=np.float)
    T_scale[0][0] = 1.0 / np.std(pts2d_a[:, 0:1] - u_avg)
    T_scale[1][1] = 1.0 / np.std(pts2d_a[:, 1:2] - v_avg)
    T_scale[2][2] = 1.0

    T_a = np.dot(T_scale, T_avg)

    new_pts_a = np.concatenate((pts2d_a, np.ones((pts2d_a.shape[0], 1))), 1)
    pts_a_transformed = pts2d_a.copy()
    for i in range(pts2d_a.shape[0]):
        pts_a_transformed[i] = np.dot(T_a, new_pts_a[i])[:-1]

    #
    # Compute T_b
    #
    T_avg = np.zeros((3, 3), dtype=np.float)
    np.fill_diagonal(T_avg, 1.0)
    u_avg = np.mean(pts2d_b[:, 0:1])
    v_avg = np.mean(pts2d_b[:, 1:2])
    T_avg[0][2] = -u_avg
    T_avg[1][2] = -v_avg

    T_scale = np.zeros((3, 3), dtype=np.float)
    T_scale[0][0] = 1.0 / np.std(pts2d_b[:, 0:1] - u_avg)
    T_scale[1][1] = 1.0 / np.std(pts2d_b[:, 1:2] - v_avg)
    T_scale[2][2] = 1.0

    T_b = np.dot(T_scale, T_avg)

    new_pts_b = np.concatenate((pts2d_b, np.ones((pts2d_b.shape[0], 1))), 1)
    pts_b_transformed = pts2d_b.copy()
    for i in range(pts2d_b.shape[0]):
        tmp = np.dot(T_b, new_pts_b[i])
        pts_b_transformed[i] = tmp[:-1]

    return T_a, T_b, compute_fundamental_matrix(pts_a_transformed, pts_b_transformed)


# Driver code
def draw_epipolar(F, pts2d, pic):
    num_row = pic.shape[0]
    num_col = pic.shape[1]
    color = (255, 0, 0)

    last_col = np.ones((pts2d.shape[0], 1))
    new_pts_pic_a = np.concatenate((pts2d, last_col), 1)

    for pt in new_pts_pic_a:
        l_b = np.dot(F, pt)
        l_L = np.cross([0, 0, 1], [0, num_row, 1])
        l_R = np.cross([num_col, 0, 1], [num_col, num_row, 1])
        P_i_L = np.cross(l_b, l_L)
        P_i_R = np.cross(l_b, l_R)

        x1 = int(P_i_L[0] / P_i_L[2])
        y1 = int(P_i_L[1] / P_i_L[2])
        x2 = int(P_i_R[0] / P_i_R[2])
        y2 = int(P_i_R[1] / P_i_R[2])

        cv2.line(pic, (x1, y1), (x2, y2), color, 1)

    return pic


def main():
    """Driver code."""

    # 1a
    # Read points
    pts3d_norm = read_points(os.path.join(input_dir, SCENE_NORM))
    pts2d_norm_pic_a = read_points(os.path.join(input_dir, PIC_A_2D_NORM))

    # Solve for transformation matrix using least squares
    M, error = solve_least_squares(pts3d_norm, pts2d_norm_pic_a)

    #Print M
    print "\n###\nPart 1a\n###\n\nThe matrix M recovered from the normalized points:\n", M, "\n\n"

    # Project 3D points to 2D
    pts2d_projected = project_points(pts3d_norm, M)

    # Compute residual error for each point
    residuals = get_residuals(pts2d_norm_pic_a, pts2d_projected)

    # Print the <u, v> projection of the last point, and the corresponding residual
    print "\nThe < u, v > projection of the last point given your M matrix:\n", pts2d_projected[-1], "\n\n"
    print "\nThe residual between that projected location and the actual one given\n", residuals[-1], "\n\n"

    # 1b
    # Read points
    pts3d = read_points(os.path.join(input_dir, SCENE))
    pts2d_pic_b = read_points(os.path.join(input_dir, PIC_B_2D))
    # print "shape:", pts2d_pic_b.shape
    # NOTE: These points are not normalized

    # find the best transform (bestM)
    bestM, error = calibrate_camera(pts3d, pts2d_pic_b)
    print "\n\nbestM:\n", bestM, "\n\n"

    # 1c
    # Compute the camera location using bestM
    Q = bestM[:3,:3]
    m4 = bestM[:3, -1].reshape((3,1))
    center = -1 * np.dot(np.linalg.inv(Q), m4)
    print "\n###\nPart 1c\n###\n\nCamera Center in the world:\n", center, "\n\n"

    # 2a
    # find the raw fundamental matrix
    pts2d_pic_a = read_points(os.path.join(input_dir, PIC_A_2D))
    F = compute_fundamental_matrix(pts2d_pic_a, pts2d_pic_b)
    print "\n###\nPart 2a\n###\n\nF:\n", F, "\n\n"

    # 2b
    # Reduce the rank of the fundamental matrix
    u, s, v = np.linalg.svd(F)
    s[s.argmin()] = 0
    s = np.diag(s)
    new_F = np.dot(np.dot(u, s), v)
    print "###\nPart 2b\n###\n\n F:\n", new_F, "\n\n"


    # 2c
    # Draw epipolar lines
    pic_a = draw_epipolar(F.T, read_points(os.path.join(input_dir, PIC_B_2D)), cv2.imread("input/pic_a.jpg"))
    pic_b = draw_epipolar(F, read_points(os.path.join(input_dir, PIC_A_2D)), cv2.imread("input/pic_b.jpg"))

    cv2.imwrite(os.path.join(output_dir, "ps3-2-c-1.png"), pic_a)
    cv2.imwrite(os.path.join(output_dir, "ps3-2-c-2.png"), pic_b)

    # cv2.imshow("pic_b", pic_b)
    # cv2.imshow("pic_a", pic_a)
    # cv2.waitKey(8000)

    # 2d
    """
    Create two matrices Ta and Tb for the set of points defined in the
    files pts2d-pic_a.txt and pts2d-pic_b.txt respectively.
    Use these matrices to transform the two sets of points.
    Then, use these normalized points to create a new Fundamental matrix F.
    Compute it as above, including making the smaller singular value zero.

    Output:
    - The matrices Ta, Tb and F [text response]

    """
    T_a, T_b, F_hat = get_trans_and_F(pts2d_pic_a, pts2d_pic_b)
    print "###\nPart 2d\n###\n\nT_a:\n", T_a, " \n\nT_b:\n", T_b, "\n\nF_hat\n", F_hat, "\n\n"

    # 2e
    #
    #
    # A little help from the forums:
    #
    #  Jeff Copeland:
    #  I was throwing my normalized coordinates from 2d through F and F^T
    #  when generating my epipolar lines when I should have been using the
    #  original non-normalized points.  Normalized points are just used for
    #  F matrix construction.
    #
    #  Philip Glau:
    #  Example: Point_a = [u, v, 1] in 'Pixel coordinates' gets transformed
    #  by T_a into a normalized space, then passes through the normalized
    #  fundamental matrix F_hat. This results is then converted from
    #  'normalized space' back to Image B pixel space by the T_b^T
    #  transformation
    #
    #  F=T_b^T * F_hat * T_a

    F = np.dot(np.dot(T_b.T, F_hat), T_a)
    print "###\n Part 2e\n###\n\nNew F:\n", F, "\n\n"
    pic_a = draw_epipolar(F.T, read_points(os.path.join(input_dir, PIC_B_2D)), cv2.imread("input/pic_a.jpg"))
    pic_b = draw_epipolar(F, read_points(os.path.join(input_dir, PIC_A_2D)), cv2.imread("input/pic_b.jpg"))

    cv2.imwrite(os.path.join(output_dir, "ps3-2-e-1.png"), pic_a)
    cv2.imwrite(os.path.join(output_dir, "ps3-2-e-2.png"), pic_b)

    # cv2.imshow("pic_b", pic_b)
    # cv2.imshow("pic_a", pic_a)
    # cv2.waitKey(8000)


if __name__ == '__main__':
    main()
