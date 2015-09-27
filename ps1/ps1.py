__author__ = 'chris.owens@gatech.edu'

import numpy as np
import scipy as sp
import scipy.signal
import cv2

blue_channel = 0
green_channel = 1
red_channel = 2

img_1_loc = "output/ps1-1-a-1.png"
img_2_loc = "output/ps1-1-a-2.png"

img_1_swap_rb = "output/ps1-2-a-1.png"

img_1_red_mono_loc = "output/ps1-2-b-1.png"
img_1_green_mono_loc = "output/ps1-2-c-1.png"

img_1_rep_loc = "output/ps1-3-a-1.png"

img_1_green_crazy_math = "output/ps1-4-b-1.png"

img_1_green_left_shift = "output/ps1-4-c-1.png"

img_1_green_minus_left = "output/ps1-4-d-1.png"


def swap_red_and_blue(image):
    red = np.copy(image[:,:,red_channel])
    blue = np.copy(image[:,:,blue_channel])

    image[:,:,red_channel] = blue
    image[:,:,blue_channel] = red

    return image

def monochrome(image,channel):
    return image[:,:,channel]

def replace_center_100_square(img_square,img_replace):
    half_height = img_square.shape[0] / 2
    half_width = img_square.shape[1] / 2

    center_100 = np.copy(img_square[half_height - 50: half_height + 50,half_width - 50:half_width + 50])

    half_height = img_replace.shape[0] / 2
    half_width = img_replace.shape[1] / 2

    img_replace[half_height - 50: half_height + 50,half_width - 50:half_width + 50] = center_100

    return img_replace

def compute_min_max_mean_sd(image):
    min = np.min(image)
    max = np.max(image)
    mean = np.mean(image)
    stand_dev = np.std(image)

    print "min: ", min
    print "max: ", max
    print "mean: ", mean
    print "stand dev: ", stand_dev

def omg_crazy_math(image):
    image = image.astype(np.float64)
    mean = np.mean(image)
    stand_dev = np.std(image)
    # Subtract the mean from all pixels
    image = image - mean
    # divide by standard deviation
    image = image / stand_dev
    # multiply by 10
    image = image * 10
    # add the mean back in.
    image = image + mean

    return image

def shift_left(image, px_amount):
    image = image.astype(np.float64)
    # Remove the left side
    image = image[:, px_amount:]

    # Add zeroed pixels to the right side
    image = np.concatenate((image,np.zeros((image.shape[0],px_amount),dtype = int)),axis=1)

    return image

def subtract(image, sub_image):
    image = image.astype(np.float64)
    sub_image = sub_image.astype(np.float64)
    return np.clip((image - sub_image), 0, 255).astype(np.uint8)

def add_noise(image, channel, sigma=4):
    tmp = image[:,:,channel].copy()
    tmp = tmp.astype(np.float64)
    rand_matrix = np.random.randn(image.shape[0],image.shape[1]) * sigma
    tmp += rand_matrix
    image[:, :, channel] = tmp.clip(0, 255).astype(np.uint8)

    return image

list_of_func =  "0  - All\n" + \
                "1  - 2.a Swap Red and Blue\n" + \
                "2  - 2.b Green Monochrome\n" + \
                "3  - 2.c Red Monochrome\n" + \
                "4  - 3.a Replace center 100\n" + \
                "5  - 4.a Compute Min, Max, Mean, and Std Dev\n" + \
                "6  - 4.b Lots of Maths on the Image\n" + \
                "7  - 4.c Shift Left by 2\n" + \
                "8  - 4.d Subtract Shifted\n" + \
                "9  - 5.a Add Noise to Green Channel\n" + \
                "10 - 5.b Add Noise to Blue Channel\n"
input = int(raw_input("\n" + list_of_func + "\nChoose a number:"))


if input == 1 or input == 0:
    # 2.a
    cv2.imwrite(img_1_swap_rb, swap_red_and_blue(cv2.imread(img_1_loc)))

if input == 2 or input == 0:
    # 2.b mono-green
    cv2.imwrite(img_1_green_mono_loc, monochrome(cv2.imread(img_1_loc),green_channel))

if input == 3 or input == 0:
    # 2.c mono-red
    cv2.imwrite(img_1_red_mono_loc, monochrome(cv2.imread(img_1_loc),red_channel))

if input == 4 or input == 0:
    # 3.a
    cv2.imwrite(img_1_rep_loc, replace_center_100_square(cv2.imread(img_1_red_mono_loc)[:,:,red_channel],monochrome(cv2.imread(img_2_loc),red_channel)))

if input == 5 or input == 0:
    # 4.a
    compute_min_max_mean_sd(cv2.imread(img_1_green_mono_loc)[:,:,green_channel])

if input == 6 or input == 0:
    # 4.b
    cv2.imwrite(img_1_green_crazy_math,omg_crazy_math(cv2.imread(img_1_green_mono_loc)[:,:,green_channel]))

if input == 7 or input == 0:
    # 4.c
    if input == 0:
        num_shift = 2
    else:
        num_shift = int(raw_input("How many pixels?:"))
    cv2.imwrite(img_1_green_left_shift,shift_left(cv2.imread(img_1_green_mono_loc)[:,:,green_channel],num_shift))

if input == 8 or input == 0:
    # 4.d
    cv2.imwrite(img_1_green_minus_left,subtract(cv2.imread(img_1_green_mono_loc)[:,:,green_channel],cv2.imread(img_1_green_left_shift)[:,:,green_channel]))

if input == 9 or input == 0:
    # 5.a
    # for sigma in range(1,5):
    #     cv2.imwrite("output/ps1-5-a-1-" + str(sigma) +".png", add_noise(cv2.imread(img_1_loc),green_channel,sigma))
    if input == 0:
        sigma = 4
    else:
        sigma = int(raw_input("Sigma:"))
    cv2.imwrite("output/ps1-5-a-1.png", add_noise(cv2.imread(img_1_loc),green_channel,sigma))

if input == 10 or input == 0:
    # 5.b
    # for sigma in range(1,15,2):
    #     cv2.imwrite("output/ps1-5-b-1-" + str(sigma) +".png", add_noise(cv2.imread(img_1_loc),blue_channel,sigma))
    if input == 0:
        sigma = 4
    else:
        sigma = int(raw_input("Sigma:"))
    cv2.imwrite("output/ps1-5-b-1.png", add_noise(cv2.imread(img_1_loc),blue_channel,sigma))