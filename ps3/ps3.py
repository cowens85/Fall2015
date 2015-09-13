"""Problem Set 3: Window-based Stereo Matching."""

import numpy as np
import cv2

from numpy.lib.stride_tricks import as_strided

import os

input_dir = "input"  # read images from os.path.join(input_dir, <filename>)
output_dir = "output"  # write images to os.path.join(output_dir, <filename>)

"""
Kinda pointles now that the offsets have been moved out, but this does the array magic to create the patch
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

    D = np.zeros(L.shape, dtype=np.float)

    # subtract 1 due to the starting pixel
    offset = (window_size - 1) / 2

    L_padded = cv2.copyMakeBorder(L, offset, offset, offset, offset, cv2.BORDER_CONSTANT, value=0)
    R_padded = cv2.copyMakeBorder(R, offset, offset, offset, offset, cv2.BORDER_CONSTANT, value=0)

    r_shape = (R_padded.shape[0] - (offset * 2), R_padded.shape[1] - (offset * 2), window_size, window_size)
    # r_shape = (R_padded.shape[0], R_padded.shape[1], window_size, window_size)
    r_strides = (R_padded.shape[1] * R_padded.itemsize, R_padded.itemsize, R_padded.itemsize * R_padded.shape[1], R_padded.itemsize)
    r_strips = as_strided(R_padded, r_shape, r_strides)

    shape = L_padded.shape
    height = shape[0]
    width = shape[1]


    for y in range(offset, height - offset):
        r_strip = r_strips[y - offset]

        for x in range(offset,width-offset):
            l_patch = get_patch(L_padded, offset, offset, offset, offset, y, x)
            l_strip = np.tile(l_patch, (r_strip.shape[0],1,1))

            ssd = ((l_strip - r_strip)**2).sum((1,2))
            x_prime = np.argmin(ssd)

            D[y-offset][x-offset] = x_prime - x

    return D

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

    L_padded = cv2.copyMakeBorder(L, offset, offset, offset, offset, cv2.BORDER_CONSTANT, value=0)
    R_padded = cv2.copyMakeBorder(R, offset, offset, offset, offset, cv2.BORDER_CONSTANT, value=0)

    r_shape = (R_padded.shape[0]- (offset * 2), window_size, R_padded.shape[1])
    r_strides = (R_padded.shape[1] * R_padded.itemsize, R_padded.itemsize * R_padded.shape[1], R_padded.itemsize)
    r_strips = as_strided(R_padded, r_shape, r_strides)

    shape = L_padded.shape
    height = shape[0]
    width = shape[1]


    for y in range(offset, height - offset):
        r_strip = r_strips[y - offset]

        for x in range(offset,width-offset):
            l_patch = get_patch(L_padded, offset, offset, offset, offset, y, x)
            result = cv2.matchTemplate(r_strip, l_patch, method=cv2.TM_CCOEFF_NORMED)

            upper_left = cv2.minMaxLoc(result)[3]
            x_prime = upper_left[1] + offset

            D[y-offset][x-offset] = x_prime - x

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

def apply_disparity_norm(l_image, r_image, problem, window_size=21):
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

def part_3_noise(window_size=9,kernel_size=5,sigma=5):
    L = cv2.imread(os.path.join('input', "pair1-L.png"), 0) * (1 / 255.0)  # grayscale, scale to [0.0, 1.0]
    R = cv2.imread(os.path.join('input', "pair1-R.png"), 0) * (1 / 255.0)

    L = cv2.GaussianBlur(L.copy(), (kernel_size,kernel_size), sigma)
    R = cv2.GaussianBlur(R.copy(), (kernel_size,kernel_size), sigma)
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

    cv2.imwrite(os.path.join("output", "ps3-3-a-1.png"), np.clip(D_L, 0, 255).astype(np.uint8))
    cv2.imwrite(os.path.join("output", "ps3-3-a-2.png"), np.clip(D_R, 0, 255).astype(np.uint8))

def part_3_contrast(window_size=9,kernel_size=5,sigma=5):
    L = cv2.imread(os.path.join('input', "pair1-L.png"), 0) * (1 / 255.0) * 1.1  # grayscale, scale to [0.0, 1.0]
    R = cv2.imread(os.path.join('input', "pair1-R.png"), 0) * (1 / 255.0) * 1.1

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

    cv2.imwrite(os.path.join("output", "ps3-3-b-1.png"), np.clip(D_L, 0, 255).astype(np.uint8))
    cv2.imwrite(os.path.join("output", "ps3-3-b-2.png"), np.clip(D_R, 0, 255).astype(np.uint8))

def main():

    """Run code/call functions to solve problems."""
    # 1
    # apply_disparity_ssd("pair0-L.png", "pair0-R.png", "1", 15)

    # 2
    # apply_disparity_ssd("pair1-L.png", "pair1-R.png", "2", 9)

    # 3
    # TODO: Apply disparity_ssd() to noisy versions of pair1 images
    # part_3_noise()

    # TODO: Boost contrast in one image and apply again
    # part_3_contrast()

    # 4
    # TODO: Implement disparity_ncorr() and apply to pair1 images (original, noisy and contrast-boosted)
    apply_disparity_norm("pair1-L.png", "pair1-R.png", "4", 9)

    # 5
    # TODO: Apply stereo matching to pair2 images, try pre-processing the images for best results


if __name__ == "__main__":
    main()



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
