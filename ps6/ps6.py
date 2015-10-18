"""Problem Set 6: Optic Flow."""

import numpy as np
import cv2
import scipy as sp
import scipy.signal

import os

# I/O directories
input_dir = "input"
output_dir = "output"

def grad_x(image, naive=False):
    if naive:
        ret_val = np.zeros(image.shape)
        for i in range(0, len(image)):
            for j in range(0, len(image[0])-1):
                ret_val[i, j] = float(image[i, j+1]) - float(image[i, j])

        return ret_val
    else:
        return cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)

def grad_y(image, naive=False):
    if naive:
        ret_val = np.zeros(image.shape)

        for i in range(0,len(image)-1):
            for j in range(0, len(image[0])):
                ret_val[i,j] = float(image[i+1, j]) - float(image[i, j])
        return ret_val
    else:
        return cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

# Assignment code
def optic_flow_LK(A, B, win=5, naive_grad=False, blur=True, blur_size=7, blur_sigma=11):
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
    if blur:
        kernel = (blur_size, blur_size)
        A = cv2.GaussianBlur(A, kernel, blur_sigma)
        B = cv2.GaussianBlur(B, kernel, blur_sigma)

    # I_x = np.zeros(A.shape)
    # I_y = np.zeros(A.shape)
    # I_t = np.zeros(A.shape)
    # I_x[1:-1, 1:-1] = (A[1:-1, 2:] - A[1:-1, :-2]) / 2
    # I_y[1:-1, 1:-1] = (A[2:, 1:-1] - A[:-2, 1:-1]) / 2
    # I_t[1:-1, 1:-1] = A[1:-1, 1:-1] - B[1:-1, 1:-1]

    I_x = grad_x(A, naive=naive_grad)#(A[1:-1, 2:] - A[1:-1, :-2]) / 2
    I_y = grad_y(A, naive=naive_grad)#(A[2:, 1:-1] - A[:-2, 1:-1]) / 2
    I_t = A - B

    # params = np.zeros(A.shape + (5,)) #Ix2, Iy2, Ixy, Ixt, Iyt
    # params[..., 0] = I_x * I_x # I_x2
    # params[..., 1] = I_y * I_y # I_y2
    # params[..., 2] = I_x * I_y # I_xy
    # params[..., 3] = I_x * I_t # I_xt
    # params[..., 4] = I_y * I_t # I_yt
    #
    # cum_params = np.cumsum(np.cumsum(params, axis=0), axis=1)
    #
    # win_params = (cum_params[2 * win + 1:, 2 * win + 1:] -
    #               cum_params[2 * win + 1:, :-1 - 2 * win] -
    #               cum_params[:-1 - 2 * win, 2 * win + 1:] +
    #               cum_params[:-1 - 2 * win, :-1 - 2 * win])
    # det = win_params[...,0] * win_params[..., 1] - win_params[..., 2] **2
    # U = np.where(det != 0,
    #              (win_params[..., 1] * win_params[..., 3] -
    #               win_params[..., 2] * win_params[..., 4]) / det,
    #              0)
    # V = np.where(det != 0,
    #              (win_params[..., 0] * win_params[..., 4] -
    #               win_params[..., 2] * win_params[..., 3]) / det,
    #              0)
    # return U, V

    Ixx = I_x*I_x
    Iyy = I_y*I_y
    Ixy = I_x*I_y
    Ixt = I_x*I_t
    Iyt = I_y*I_t


    kernel = np.ones((win,win), np.float32)/(win * win)
    Sxx = cv2.filter2D(Ixx, -1, kernel)
    Syy = cv2.filter2D(Iyy, -1, kernel)
    Sxy = cv2.filter2D(Ixy, -1, kernel)
    Sxt = cv2.filter2D(Ixt, -1, kernel)
    Syt = cv2.filter2D(Iyt, -1, kernel)

    U = np.zeros(A.shape).astype(np.float64)
    V = np.zeros(B.shape).astype(np.float64)

    for y, x in np.ndindex(A.shape):
        calc_A = np.array([[Sxx[y][x], Sxy[y][x]],
                     [Sxy[y][x], Syy[y][x]]])
        calc_B = np.array([-Sxt[y][x],
                     -Syt[y][x]])

        if np.linalg.det(calc_A) < .000001:
            U[y][x] = 0
            V[y][x] = 0
        else:
            solution, residuals, rank, s = np.linalg.lstsq(calc_A, calc_B)
            U[y][x] = solution[0]
            V[y][x] = solution[1]

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

    g_pyr = [image]

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
    shape = image.shape
    warped = np.zeros(shape)

    warped[0] = image[0]
    for y in range(shape[0]):
        for x in range(shape[1]):
            try:
                warped[y][x] = image[y + V[y, x]][x + U[y, x]]
            except:
                # ignore the Nan
                a = ""

    return warped


def hierarchical_LK(A, B, max_level=4):
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


    # 1. Given input images A and B. Initialize k = n where n is the max level.
    # 2. REDUCE both input images to level k. Call these images Ak and Bk.
    # 3. If k = n initialize U and V to be zero images the size of Ak;
    #    otherwise expand the flow field and double to get to the next level: U = 2*EXPAND(U), V = 2* EXPAND(V).
    # 4. Warp Bk using U and V to form Ck.
    # 5. Perform LK on Ak and Ck to yield two incremental flow fields Dx and Dy.
    # 6. Add these to original flow: U = U + Dx and V = V + Dy.
    # 7. If k > 0 let k = k - 1 and goto (2).
    # 8. Return U and V


    k = max_level
    while k >= 0:
        # for i in range(k):
        A_k = reduce(A)
        B_k = reduce(B)
        if k == max_level:
            U = np.zeros(A_k.shape)
            V = np.zeros(A_k.shape)
        else:
            U = 2*expand(U)
            V = 2*expand(V)
        ck = warp(B_k, U, V)
        dx, dy = optic_flow_LK(A_k, ck, naive_grad=True, blur=False)

        # print "shape dx, dy ", dx.shape, ",", dy.shape
        # print "U shape ", U.shape
        # print "k = ", k
        for i in range(max_level - k):
            dx = expand(dx)
            dy = expand(dy)
        U += dx
        V += dy
        k -= 1

    return U, V

def scale_to_img(matrix):
    tmp = matrix.copy()
    min = np.min(tmp)
    tmp += abs(min)
    max = np.max(tmp)
    tmp *= 255.0 / max
    return tmp.astype(np.uint8)

def cat_images(images, stacked=False):
    original = images[0].copy()
    orig_shape = original.shape
    axis = 1
    if stacked:
        axis = 0
    for img in images[1:]:
        shape = img.shape
        tmp = np.zeros((orig_shape[0], shape[1]))
        # pad = (orig_shape[1] - shape[1]) / 2
        # if pad > 0:
        #     img = np.pad(img,((pad,0),(0,0)),'constant',constant_values=0)
        tmp[0:img.shape[0], 0:img.shape[1]] = img

        original = np.concatenate((original, tmp), axis=axis)

    return original


def write_quiver_image(U, V, name, scale=5):
    stride = 15  # plot every so many rows, columns
    color = (0, 255, 0)  # green
    img_out = np.zeros((V.shape[0], U.shape[1], 3), dtype=np.uint8)
    print 'drawing'
    print U
    print V
    for y in xrange(0, V.shape[0], stride):
        for x in xrange(0, U.shape[1], stride):
            cv2.arrowedLine(img_out, (x, y), (x + int(U[y, x] * scale), y + int(V[y, x] * scale)), color, 1, tipLength=.3)

    write_image(img_out, name, scale=False)


def save_cat_images(images, name, scale=True, apply_color=False, stacked=False):
    write_image(cat_images(images, stacked=stacked), name, scale=scale, apply_color=apply_color)


def write_image(image, name, scale=True, apply_color=False):
    tmp = image.copy()
    if scale:
        tmp = scale_to_img(tmp)
    if apply_color:
        tmp = cv2.applyColorMap(tmp, cv2.COLORMAP_JET)

    cv2.imwrite(os.path.join(output_dir, name), tmp)


def one_a():
    Shift0 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'Shift0.png'), 0).astype(np.float) / 255.0

    ShiftR2 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR2.png'), 0).astype(np.float) / 255.0
    # TODO: Optionally, smooth the images if LK doesn't work well on raw images
    U, V = optic_flow_LK(Shift0, ShiftR2)  # TODO: implement this
    # TODO: Save U, V as side-by-side false-color image or single quiver plot
    save_cat_images([U, V], "ps6-1-a-1.png", scale=True, apply_color=True)

    ShiftR5 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR5U5.png'), 0).astype(np.float) / 255.0
    # TODO: Optionally, smooth the images if LK doesn't work well on raw images
    U, V = optic_flow_LK(Shift0, ShiftR5)  # TODO: implement this
    # TODO: Save U, V as side-by-side false-color image or single quiver plot
    save_cat_images([U, V], "ps6-1-a-2.png", scale=True, apply_color=True)


def one_b():
    Shift0 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'Shift0.png'), 0).astype(np.float) / 255.0
    ShiftR = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR10.png'), 0).astype(np.float) / 255.0
    # TODO: Optionally, smooth the images if LK doesn't work well on raw images
    U, V = optic_flow_LK(Shift0, ShiftR)  # TODO: implement this
    # TODO: Save U, V as side-by-side false-color image or single quiver plot
    save_cat_images([U, V], "ps6-1-b-1.png", scale=True, apply_color=True)

    ShiftR = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR20.png'), 0).astype(np.float) / 255.0
    # TODO: Optionally, smooth the images if LK doesn't work well on raw images
    U, V = optic_flow_LK(Shift0, ShiftR)  # TODO: implement this
    # TODO: Save U, V as side-by-side false-color image or single quiver plot
    save_cat_images([U, V], "ps6-1-b-2.png", scale=True, apply_color=True)

    ShiftR = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR40.png'), 0).astype(np.float) / 255.0
    # TODO: Optionally, smooth the images if LK doesn't work well on raw images
    U, V = optic_flow_LK(Shift0, ShiftR)  # TODO: implement this
    # TODO: Save U, V as side-by-side false-color image or single quiver plot
    save_cat_images([U, V], "ps6-1-b-3.png", scale=True, apply_color=True)


def two():

    # # 2a
    yos_img_01 = cv2.imread(os.path.join(input_dir, 'DataSeq1', 'yos_img_01.jpg'), 0) / 255.0
    yos_img_01_g_pyr = gaussian_pyramid(yos_img_01, 4)
    # # TODO: Save pyramid images as a single side-by-side image (write a utility function?)
    save_cat_images(yos_img_01_g_pyr, "ps6-2-a-1.png", scale=True)

    #
    # # 2b
    yos_img_01_l_pyr = laplacian_pyramid(yos_img_01_g_pyr)
    # # TODO: Save pyramid images as a single side-by-side image
    save_cat_images(yos_img_01_l_pyr, "ps6-2-b-1.png", scale=True)


def three_a():
    yos_img_01 = cv2.imread(os.path.join(input_dir, 'DataSeq1', 'yos_img_01.jpg'), 0) / 255.0
    yos_img_02 = cv2.imread(os.path.join(input_dir, 'DataSeq1', 'yos_img_02.jpg'), 0) / 255.0
    yos_img_01_g_pyr = gaussian_pyramid(yos_img_01, 4)
    yos_img_02_g_pyr = gaussian_pyramid(yos_img_02, 4)

    """Select appropriate pyramid *level* that leads to best optic flow estimation"""
    level = 2
    U, V = optic_flow_LK(yos_img_01_g_pyr[level], yos_img_02_g_pyr[level],win=21, naive_grad=True, blur=False)

    """Scale up U, V to original image size (note: don't forget to scale values as well!)"""
    for i in range(0, level):
        U = expand(U) * 2
        V = expand(V) * 2

    """Save U, V as side-by-side false-color image or single quiver plot"""
    save_cat_images([U, V], "ps6-3-a-1.png", scale=True, apply_color=True)

    yos_img_02_warped = warp(yos_img_02, U, V)

    """Save difference image between yos_img_02_warped and original yos_img_01"""
    write_image(yos_img_01, "pr3-img1.png")
    write_image(yos_img_02_warped, "pr3-im2-warp.png")
    save_cat_images([yos_img_01, yos_img_02_warped], "pr3-pair.png")

    diff = yos_img_02_warped - yos_img_01
    write_image(diff, "diff.png")

def four_a():
    # 4a
    Shift0 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'Shift0.png'), 0) / 255.0
    ShiftR10 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR10.png'), 0) / 255.0
    ShiftR20 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR20.png'), 0) / 255.0
    ShiftR40 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR40.png'), 0) / 255.0

    U10, V10 = hierarchical_LK(Shift0, ShiftR10)
    U20, V20 = hierarchical_LK(Shift0, ShiftR20)
    U40, V40 = hierarchical_LK(Shift0, ShiftR40)
    """Save displacement image pairs (U, V), stacked"""
    u_v_10_cat = cat_images([U10,V10])
    u_v_20_cat = cat_images([U20,V20])
    u_v_40_cat = cat_images([U40,V40])
    save_cat_images([u_v_10_cat, u_v_20_cat, u_v_40_cat], "ps6-4-a-1.png", stacked=True, apply_color=True)


    ShiftR10_warped = warp(ShiftR10, U10, V10)
    ShiftR20_warped = warp(ShiftR20, U20, V20)
    ShiftR40_warped = warp(ShiftR40, U40, V40)

    """Save difference between each warped image and original image (Shift0), stacked"""
    diff_10 = ShiftR10_warped - Shift0
    diff_20 = ShiftR20_warped - Shift0
    diff_40 = ShiftR40_warped - Shift0
    save_cat_images([diff_10, diff_20, diff_40], "ps6-4-a-2.png", stacked=True, apply_color=True)








# Driver code
def main():
    # Note: Comment out parts of this code as necessary

    # 1a
    # one_a()

    # 1b
    # TSimilarly for ShiftR10, ShiftR20 and ShiftR40
    # one_b()

    # 2a
    # 2b
    # two()

    # 3a
    # three_a()

    # 4a
    four_a()


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
