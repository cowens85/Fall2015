"""Problem Set 6: Optic Flow."""

import numpy as np
import cv2
import scipy as sp
import scipy.signal

import os

# I/O directories
input_dir = "input"
output_dir = "output"


# Assignment code
def optic_flow_LK(A, B):
    """Compute optic flow using the Lucas-Kanade method.

    Parameters
    ----------
        A: grayscale floating-point image, values in [0.0, 1.0]
        B: grayscale floating-point image, values in [0.0, 1.0]

    Returns
    -------
        U: raw displacement (in pixels) along X-axis, same size as image, floating-point type
        V: raw displacement (in pixels) along Y-axis, same size and type as U
    """
    A = cv2.GaussianBlur(A, (7, 7), 11)
    B = cv2.GaussianBlur(B, (7, 7), 11)

    # I_x = np.zeros(A.shape)
    # I_y = np.zeros(A.shape)
    # I_t = np.zeros(A.shape)
    # I_x[1:-1, 1:-1] = (A[1:-1, 2:] - A[1:-1, :-2]) / 2
    # I_y[1:-1, 1:-1] = (A[2:, 1:-1] - A[:-2, 1:-1]) / 2
    # I_t[1:-1, 1:-1] = A[1:-1, 1:-1] - B[1:-1, 1:-1]

    I_x = cv2.Sobel(A,cv2.CV_64F,1,0,ksize=5)#(A[1:-1, 2:] - A[1:-1, :-2]) / 2
    I_y = cv2.Sobel(A,cv2.CV_64F,0,1,ksize=5)#(A[2:, 1:-1] - A[:-2, 1:-1]) / 2
    I_t = A - B

    params = np.zeros(A.shape + (5,)) #Ix2, Iy2, Ixy, Ixt, Iyt
    params[..., 0] = I_x * I_x # I_x2
    params[..., 1] = I_y * I_y # I_y2
    params[..., 2] = I_x * I_y # I_xy
    params[..., 3] = I_x * I_t # I_xt
    params[..., 4] = I_y * I_t # I_yt

    cum_params = np.cumsum(np.cumsum(params, axis=0), axis=1)

    win = 5
    win_params = (cum_params[2 * win + 1:, 2 * win + 1:] -
                  cum_params[2 * win + 1:, :-1 - 2 * win] -
                  cum_params[:-1 - 2 * win, 2 * win + 1:] +
                  cum_params[:-1 - 2 * win, :-1 - 2 * win])
    det = win_params[...,0] * win_params[..., 1] - win_params[..., 2] **2
    U = np.where(det != 0,
                 (win_params[..., 1] * win_params[..., 3] -
                  win_params[..., 2] * win_params[..., 4]) / det,
                 0)
    V = np.where(det != 0,
                 (win_params[..., 0] * win_params[..., 4] -
                  win_params[..., 2] * win_params[..., 3]) / det,
                 0)
    return U, V

def generatingKernel(parameter):
  """ Return a 5x5 generating kernel based on an input parameter.

  Note: This function is provided for you, do not change it.

  Args:
    parameter (float): Range of value: [0, 1].

  Returns:
    numpy.ndarray: A 5x5 kernel.

  """
  kernel = np.array([0.25 - parameter / 2.0, 0.25, parameter,
                     0.25, 0.25 - parameter /2.0])
  return np.outer(kernel, kernel)

def reduce(image):
    """Reduce image to the next smaller level.

    Parameters
    ----------
        image: grayscale floating-point image, values in [0.0, 1.0]

    Returns
    -------
        reduced_image: same type as image, half size
    """
    reduced_image = scipy.signal.convolve2d(image.copy(),generatingKernel(0.4), mode="same")

    reduced_image = reduced_image[0::2,0::2]

    # TODO: Your code here
    return reduced_image


def gaussian_pyramid(image, levels):
    """Create a Gaussian pyramid of given image.

    Parameters
    ----------
        image: grayscale floating-point image, values in [0.0, 1.0]
        levels: number of levels in the resulting pyramid

    Returns
    -------
        g_pyr: Gaussian pyramid, with g_pyr[0] = image
    """

    # TODO: Your code here
    g_pyr = [image]
    # WRITE YOUR CODE HERE

    for i in range(1,levels+1):
      g_pyr.append(reduce(g_pyr[i-1]))

    return g_pyr


def expand(image):
    """Expand image to the next larger level.

    Parameters
    ----------
        image: grayscale floating-point image, values in [0.0, 1.0]

    Returns
    -------
        reduced_image: same type as image, double size
    """

    shape = image.shape

    expanded_image = np.zeros((shape[0]*2,shape[1]*2),dtype="float")

    np.copyto(expanded_image[0::2,0::2],image)

    expanded_image = scipy.signal.convolve2d(expanded_image,generatingKernel(0.4), mode="same")*4

    return expanded_image

    # TODO: Your code here
    return expanded_image


def laplacian_pyramid(g_pyr):
    """Create a Laplacian pyramid from a given Gaussian pyramid.

    Parameters
    ----------
        g_pyr: Gaussian pyramid, as returned by gaussian_pyramid()

    Returns
    -------
        l_pyr: Laplacian pyramid, with l_pyr[-1] = g_pyr[-1]
    """
    l_pyr = []
    # WRITE YOUR CODE HERE.

    for i in range(len(g_pyr)-1):
      gauss_layer = g_pyr[i]
      gauss_row = gauss_layer.shape[0]
      gauss_col = gauss_layer.shape[1]
      expand_layer = expand(g_pyr[i + 1])
      expand_row = expand_layer.shape[0]
      expand_col = expand_layer.shape[1]

      if gauss_row < expand_row and gauss_col < expand_col:
        expand_layer = expand_layer[0:-1, 0:-1]
      elif gauss_row < expand_row:
        expand_layer = expand_layer[0:-1, :]
      elif gauss_col < expand_col:
        expand_layer = expand_layer[:, 0:-1]

      l_pyr.append(gauss_layer - expand_layer)

    l_pyr.append(g_pyr[-1])

    return l_pyr


def warp(image, U, V):
    """Warp image using X and Y displacements (U and V).

    Parameters
    ----------
        image: grayscale floating-point image, values in [0.0, 1.0]

    Returns
    -------
        warped: warped image, such that warped[y, x] = image[y + V[y, x], x + U[y, x]]

    """

    # TODO: Your code here
    return warped


def hierarchical_LK(A, B):
    """Compute optic flow using the Hierarchical Lucas-Kanade method.

    Parameters
    ----------
        A: grayscale floating-point image, values in [0.0, 1.0]
        B: grayscale floating-point image, values in [0.0, 1.0]

    Returns
    -------
        U: raw displacement (in pixels) along X-axis, same size as image, floating-point type
        V: raw displacement (in pixels) along Y-axis, same size and type as U
    """

    # TODO: Your code here
    return U, V

def scale_to_img(matrix):
    tmp = matrix.copy()
    min = np.min(tmp)
    tmp += abs(min)
    max = np.max(tmp)
    tmp *= 255.0 / max
    return tmp.astype(np.uint8)


def save_image_pair(image1, image2, name, scale=False, apply_color=False):
    tmp1 = image1.copy()
    tmp2 = image2.copy()
    if scale:
        tmp1 = scale_to_img(tmp1)
        tmp2 = scale_to_img(tmp2)
    write_image(np.concatenate((tmp1, tmp2), axis=1), name, apply_color=True)


def write_image(image, name, scale=False, apply_color=False):
    tmp = image.copy()
    if scale:
        tmp = scale_to_img(tmp)
    if apply_color:
        tmp = cv2.applyColorMap(tmp, cv2.COLORMAP_BONE)

    cv2.imwrite(os.path.join(output_dir, name), tmp)


def one_a():
    Shift0 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'Shift0.png'), 0).astype(np.float) / 255.0

    ShiftR2 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR2.png'), 0).astype(np.float) / 255.0
    # TODO: Optionally, smooth the images if LK doesn't work well on raw images
    U, V = optic_flow_LK(Shift0, ShiftR2)  # TODO: implement this
    # TODO: Save U, V as side-by-side false-color image or single quiver plot
    save_image_pair(U, V, "ps6-1-a-1.png", scale=True, apply_color=True)

    ShiftR5 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR5U5.png'), 0).astype(np.float) / 255.0
    # TODO: Optionally, smooth the images if LK doesn't work well on raw images
    U, V = optic_flow_LK(Shift0, ShiftR5)  # TODO: implement this
    # TODO: Save U, V as side-by-side false-color image or single quiver plot
    save_image_pair(U, V, "ps6-1-a-2.png", scale=True, apply_color=True)


def one_b():
    Shift0 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'Shift0.png'), 0).astype(np.float) / 255.0
    ShiftR = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR10.png'), 0).astype(np.float) / 255.0
    # TODO: Optionally, smooth the images if LK doesn't work well on raw images
    U, V = optic_flow_LK(Shift0, ShiftR)  # TODO: implement this
    # TODO: Save U, V as side-by-side false-color image or single quiver plot
    save_image_pair(U, V, "ps6-1-b-1.png", scale=True, apply_color=True)

    ShiftR = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR20.png'), 0).astype(np.float) / 255.0
    # TODO: Optionally, smooth the images if LK doesn't work well on raw images
    U, V = optic_flow_LK(Shift0, ShiftR)  # TODO: implement this
    # TODO: Save U, V as side-by-side false-color image or single quiver plot
    save_image_pair(U, V, "ps6-1-b-2.png", scale=True, apply_color=True)

    ShiftR = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR40.png'), 0).astype(np.float) / 255.0
    # TODO: Optionally, smooth the images if LK doesn't work well on raw images
    U, V = optic_flow_LK(Shift0, ShiftR)  # TODO: implement this
    # TODO: Save U, V as side-by-side false-color image or single quiver plot
    save_image_pair(U, V, "ps6-1-b-3.png", scale=True, apply_color=True)


def two_a():
    yos_img_01 = cv2.imread(os.path.join(input_dir, 'DataSeq1', 'yos_img_01.jpg'), 0) / 255.0
    yos_img_01_g_pyr = gaussian_pyramid(yos_img_01, 4)  # TODO: implement this
    # TODO: Save pyramid images as a single side-by-side image (write a utility function?)

# Driver code
def main():
    # Note: Comment out parts of this code as necessary

    # 1a
    one_a()

    # 1b
    # TSimilarly for ShiftR10, ShiftR20 and ShiftR40
    one_b()


    # 2a
    # yos_img_01 = cv2.imread(os.path.join(input_dir, 'DataSeq1', 'yos_img_01.jpg'), 0) / 255.0
    # yos_img_01_g_pyr = gaussian_pyramid(yos_img_01, 4)  # TODO: implement this
    # # TODO: Save pyramid images as a single side-by-side image (write a utility function?)
    #
    # # 2b
    # yos_img_01_l_pyr = laplacian_pyramid(yos_img_01_g_pyr)  # TODO: implement this
    # # TODO: Save pyramid images as a single side-by-side image
    #
    # # 3a
    # yos_img_02 = cv2.imread(os.path.join(input_dir, 'DataSeq1', 'yos_img_02.jpg'), 0) / 255.0
    # yos_img_02_g_pyr = gaussian_pyramid(yos_img_02, 4)
    # # TODO: Select appropriate pyramid *level* that leads to best optic flow estimation
    # U, V = optic_flow_LK(yos_img_01_g_pyr[level], yos_img_02_g_pyr[level])
    # # TODO: Scale up U, V to original image size (note: don't forget to scale values as well!)
    # # TODO: Save U, V as side-by-side false-color image or single quiver plot
    #
    # yos_img_02_warped = warp(yos_img_02, U, V)  # TODO: implement this
    # # TODO: Save difference image between yos_img_02_warped and original yos_img_01
    # # Note: Scale values such that zero difference maps to neutral gray, max -ve to black and max +ve to white
    #
    # # Similarly, you can compute displacements for yos_img_02 and yos_img_03 (but no need to save images)
    #
    # # TODO: Repeat for DataSeq2 (save images)
    #
    # # 4a
    # ShiftR10 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR10.png'), 0) / 255.0
    # ShiftR20 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR20.png'), 0) / 255.0
    # ShiftR40 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR40.png'), 0) / 255.0
    # U10, V10 = hierarchical_LK(Shift0, ShiftR10)  # TODO: implement this
    # U20, V20 = hierarchical_LK(Shift0, ShiftR20)
    # U40, V40 = hierarchical_LK(Shift0, ShiftR40)
    # # TODO: Save displacement image pairs (U, V), stacked
    # # Hint: You can use np.concatenate()
    # ShiftR10_warped = warp(ShiftR10, U10, V10)
    # ShiftR20_warped = warp(ShiftR20, U20, V20)
    # ShiftR40_warped = warp(ShiftR40, U40, V40)
    # # TODO: Save difference between each warped image and original image (Shift0), stacked
    #
    # # 4b
    # # TODO: Repeat for DataSeq1 (use yos_img_01.png as the original)
    #
    # # 4c
    # # TODO: Repeat for DataSeq1 (use 0.png as the original)


if __name__ == "__main__":
    main()
