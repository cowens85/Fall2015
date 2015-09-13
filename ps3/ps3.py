"""Problem Set 3: Window-based Stereo Matching."""

import numpy as np
import cv2

from numpy.lib.stride_tricks import as_strided

import os

input_dir = "input"  # read images from os.path.join(input_dir, <filename>)
output_dir = "output"  # write images to os.path.join(output_dir, <filename>)

"""
Kina pointles now that the offsets have been moved out, but this does the array magic to create the patch
"""
def get_patch(matrix, y_up_offset, y_down_offset, x_left_offset, x_right_offset, y, x):
    return matrix[(y - y_up_offset):(y + y_down_offset + 1), (x - x_left_offset):(x + x_right_offset + 1)]

def disparity_ssd(L, R, window_size = 21):
    """Compute disparity map D(y, x) such that: L(y, x) = R(y, x + D(y, x))

    Params:
    L: Grayscale left image, in range [0.0, 1.0]
    R: Grayscale right image, same size as L

    Returns: Disparity map, same size as L, R
    """

    # kernel_size = 5
    # sigma = 3
    # cv2.GaussianBlur(L.copy(), (kernel_size,kernel_size), sigma)
    # cv2.GaussianBlur(R.copy(), (kernel_size,kernel_size), sigma)

    D = np.zeros(L.shape, dtype=np.float)

    # subtract 1 due to the starting pixel
    offset = (window_size - 1) / 2
    shape = L.shape
    height = shape[0]
    width = shape[1]
    left_shift = False
    right_shift = False
    search_range = width / 3

    r_shape = (R.shape[0]-(window_size-1), R.shape[1]-(window_size-1), window_size, window_size)
    r_strides = (R.shape[1] * R.itemsize, R.itemsize, R.itemsize * R.shape[1], R.itemsize)
    r_strips = as_strided(R, r_shape, r_strides)

    for y in range(offset, height - offset):
        """ Compute Y Offsets """
        # y_up_offset = offset if y >= offset else y
        # y_down_offset = offset if y + offset < height else height - y - 1
        # print "y d off:", y_down_offset
        r_strip = r_strips[y]
        for x in range(offset,width-offset):
            """ Compute X Offsets """
            # x_left_offset = offset if x >= offset else x
            # x_right_offset = offset if x + offset < width else width - x - 1

            # l_patch = get_patch(L, y_up_offset, y_down_offset, x_left_offset, x_right_offset, y, x)
            l_patch = get_patch(L, offset, offset, offset, offset, y, x)
            l_strip = as_strided(l_patch, r_strip.shape, (0, l_patch.itemsize*window_size, l_patch.itemsize))


            ssd = ((l_strip - r_strip)**2).sum((1,2))

            x_prime = np.argmin(ssd) + offset
            D[y][x] = x_prime - x

    return D

# def disparity_ssd(L, R, window_size = 21):
#     """Compute disparity map D(y, x) such that: L(y, x) = R(y, x + D(y, x))
#
#     Params:
#     L: Grayscale left image, in range [0.0, 1.0]
#     R: Grayscale right image, same size as L
#
#     Returns: Disparity map, same size as L, R
#     """
#
#     # kernel_size = 5
#     # sigma = 3
#     # cv2.GaussianBlur(L.copy(), (kernel_size,kernel_size), sigma)
#     # cv2.GaussianBlur(R.copy(), (kernel_size,kernel_size), sigma)
#
#     D = np.zeros(L.shape, dtype=np.float)
#
#     # subtract 1 due to the starting pixel
#     offset = (window_size - 1) / 2
#     shape = L.shape
#     height = shape[0]
#     width = shape[1]
#     left_shift = False
#     right_shift = False
#     search_range = width / 3
#     for y in range(height):
#         """ Compute Y Offsets """
#         y_up_offset = offset if y >= offset else y
#         y_down_offset = offset if y + offset < height else height - y - 1
#         # print "y d off:", y_down_offset
#
#         for x in range(width):
#             """ Compute X Offsets """
#             x_left_offset = offset if x >= offset else x
#             x_right_offset = offset if x + offset < width else width - x - 1
#
#             l_patch = get_patch(L, y_up_offset, y_down_offset, x_left_offset, x_right_offset, y, x)
#
#             # print "x", x
#             min_ssd = np.infty
#             x_prime_start = x_left_offset
#             x_prime_end = width-x_right_offset
#
#             # If we've figured out which way we're shifting,
#             # Stop looking for matches as if it was shifted the other way
#             # This adds speed and may decrease mistakes with repeating patterns
#             if left_shift:
#                 x_prime_end = min(x + offset,x_prime_end)
#             elif right_shift:
#                 x_prime_start = max(x - offset, x_prime_start)
#
#             # Take a guess on how far off things can shift and restrict to that area
#             x_prime_start = max (x_prime_start,x - search_range)
#             x_prime_end = min (x_prime_end,x + search_range)
#
#             for x_prime in range(x_prime_start, x_prime_end):
#                 # print "x'", x_prime
#                 r_patch = get_patch(R, y_up_offset, y_down_offset, x_left_offset, x_right_offset, y, x_prime)
#                 # print "sum:", np.sum(r_patch)
#                 ssd = np.sum((l_patch - r_patch)**2)
#                 if ssd < min_ssd:
#                     min_ssd = ssd
#                     D[y][x] = x_prime - x
#
#         # Try to figure out if we're shifting left or right.
#         # Assuming no motion in the image, there should only be one shift
#         if left_shift is False and right_shift is False:
#             sum = np.sum(D[y])
#             if sum < 0:
#                 left_shift = True
#             elif sum > 0:
#                 right_shift = True
#
#     return D

def disparity_ncorr(L, R, window_size=19):
    """Compute disparity map D(y, x) such that: L(y, x) = R(y, x + D(y, x))
    
    Params:
    L: Grayscale left image, in range [0.0, 1.0]
    R: Grayscale right image, same size as L

    Returns: Disparity map, same size as L, R
    """

    D = np.zeros(L.shape, dtype=np.float)

    # subtract 1 due to the starting pixel
    offset = (window_size - 1) / 2
    shape = L.shape
    height = shape[0]
    width = shape[1]
    left_shift = False
    right_shift = False
    search_range = width / 3
    for y in range(height):
        """ Compute Y Offsets """
        y_up_offset = offset if y >= offset else y
        y_down_offset = offset if y + offset < height else height - y - 1
        # print "y d off:", y_down_offset

        for x in range(width):
            """ Compute X Offsets """
            x_left_offset = offset if x >= offset else x
            x_right_offset = offset if x + offset < width else width - x - 1

            l_patch = get_patch(L, y_up_offset, y_down_offset, x_left_offset, x_right_offset, y, x)

            # print "x", x
            # max_norm = -np.infty
            # x_prime_start = x_left_offset
            # x_prime_end = width-x_right_offset
            #
            # # If we've figured out which way we're shifting,
            # # Stop looking for matches as if it was shifted the other way
            # # This adds speed and may decrease mistakes with repeating patterns
            # if left_shift:
            #     x_prime_end = min(x + offset,x_prime_end)
            # elif right_shift:
            #     x_prime_start = max(x - offset, x_prime_start)
            #
            # # Take a guess on how far off things can shift and restrict to that area
            # x_prime_start = max (x_prime_start,x - search_range)
            # x_prime_end = min (x_prime_end,x + search_range)

            strip = R[(y - y_up_offset):(y + y_down_offset + 1), :]
            print l_patch
            print strip
            result = cv2.matchTemplate(strip, l_patch, method=cv2.TM_CCOEFF_NORMED)
            upper_left = cv2.minMaxLoc(result)[3]
            # Add the left offset because that will get us the x coordinate in the patch
            x_prime = upper_left[1] + x_left_offset
            D[y][x] = x_prime - x


        #     for x_prime in range(x_prime_start, x_prime_end):
        #         r_patch = get_patch(R, y_up_offset, y_down_offset, x_left_offset, x_right_offset, y, x_prime)
        #         norm = cv2.matchTemplate(l_patch, r_patch, method=cv2.TM_CCOEFF_NORMED)
        #         if norm < max_norm:
        #             max_norm = norm
        #             D[y][x] = x_prime - x
        #
        # # Try to figure out if we're shifting left or right.
        # # Assuming no motion in the image, there should only be one shift
        # if left_shift is False and right_shift is False:
        #     sum = np.sum(D[y])
        #     if sum < 0:
        #         left_shift = True
        #     elif sum > 0:
        #         right_shift = True

    return D

def apply_disparity_ssd(l_image, r_image, problem, window_size = 21):
    L = cv2.imread(os.path.join('input', l_image), 0) * (1 / 255.0)  # grayscale, scale to [0.0, 1.0]
    R = cv2.imread(os.path.join('input', r_image), 0) * (1 / 255.0)

    # Compute disparity (using method disparity_ssd defined in disparity_ssd.py)
    D_L = disparity_ssd(L, R, window_size)  # TODO: implemenet disparity_ssd()
    D_R = disparity_ssd(R, L, window_size)

    min = D_L.min()
    if min < 0:
        D_L += np.abs(min)
    max = D_L.max()
    D_L *= 255.0/max

    min = D_R.min()
    if min < 0:
        D_R += np.abs(min)
    D_R *= 255.0/D_R.max()

    cv2.imwrite(os.path.join("output", "ps3-" + problem + "-a-1.png"), np.clip(D_L, 0, 255).astype(np.uint8))
    cv2.imwrite(os.path.join("output", "ps3-" + problem + "-a-2.png"), np.clip(D_R, 0, 255).astype(np.uint8))

def apply_disparity_norm(l_image, r_image, problem, window_size = 21):
    L = cv2.imread(os.path.join('input', l_image), 0)
    R = cv2.imread(os.path.join('input', r_image), 0)

    # Compute disparity (using method disparity_ssd defined in disparity_ssd.py)
    D = disparity_ncorr(L, R, window_size)

    min = D.min()
    if min < 0:
        D += np.abs(min)
    max = D.max()
    D *= 255.0/max

    cv2.imwrite(os.path.join("output", "ps3-" + problem + "-a-1.png"), np.clip(D, 0, 255).astype(np.uint8))

def main():

    """Run code/call functions to solve problems."""
    # 1
    # """
    apply_disparity_ssd("pair0-L.png", "pair0-R.png", "1", 11)
    # """
    # apply_disparity_norm("pair0-L.png", "pair0-R.png", "test", 3)

    # out = internet_ssd(cv2.imread(os.path.join('input', "pair0-L.png"), 0) * (1 / 255.0), cv2.imread(os.path.join('input', "pair0-R.png"), 0) * (1 / 255.0))
    # print out
    # cv2.imwrite(os.path.join("output","ps3-1-a-1.png"),out)

    # 1-a
    # Read images
    # L = cv2.imread(os.path.join('input', 'pair0-L.png'), 0) * (1 / 255.0)  # grayscale, scale to [0.0, 1.0]
    # R = cv2.imread(os.path.join('input', 'pair0-R.png'), 0) * (1 / 255.0)
    #
    # # Compute disparity (using method disparity_ssd defined in disparity_ssd.py)
    # D_L = disparity_ssd(L, R)  # TODO: implemenet disparity_ssd()
    # D_R = disparity_ssd(R, L)
    #
    # # TODO: Save output images (D_L as output/ps3-1-a-1.png and D_R as output/ps3-1-a-2.png)
    # # Note: They may need to be scaled/shifted before saving to show results properly
    # cv2.imwrite(os.path.join("output","ps3-1-a-1.png"),D_L)
    # cv2.imwrite(os.path.join("output","ps3-1-a-2.png"),D_R)

    # 2
    # TODO: Apply disparity_ssd() to pair1-L.png and pair1-R.png (in both directions)
    """
    apply_disparity_ssd("pair1-L.png", "pair1-R.png", "2", 15)
    """

    # 3
    # TODO: Apply disparity_ssd() to noisy versions of pair1 images
    # TODO: Boost contrast in one image and apply again

    # 4
    # TODO: Implement disparity_ncorr() and apply to pair1 images (original, noisy and contrast-boosted)

    # 5
    # TODO: Apply stereo matching to pair2 images, try pre-processing the images for best results


if __name__ == "__main__":
    main()
