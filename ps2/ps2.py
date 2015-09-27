"""Problem Set 2: Edges and Lines."""

import numpy as np
import cv2

import os
from math import pi

input_dir = "input"  # read images from os.path.join(input_dir, <filename>)
output_dir = "output"  # write images to os.path.join(output_dir, <filename>)

def hough_lines_acc(img_edges, rho_res=1, theta_res=pi/90):
    """Compute Hough Transform for lines on edge image.

    Parameters
    ----------
        img_edges: binary edge image
        rho_res: rho resolution (in pixels)
        theta_res: theta resolution (in radians)

    Returns
    -------
        H: Hough accumulator array
        rho: vector of rho values, one for each row of H
        theta: vector of theta values, one for each column of H
    """
    # Find all the points in our edge matrix
    # points = np.argwhere(img_edges.max() == img_edges)

    num_rows, num_cols = img_edges.shape

    theta = np.linspace(0.0, 90.0, np.ceil(90.0 / theta_res) + 1.0) * pi / 90.0

    diagonal_len = np.sqrt((num_rows - 1)**2 + (num_cols - 1)**2)
    q = np.ceil(diagonal_len / rho_res)
    num_rho = 2*q + 1
    rho = np.linspace(-q * rho_res, q * rho_res, num_rho)
    H = np.zeros((len(rho), len(theta)),dtype=np.int)

    for y, x in zip(*np.nonzero(img_edges)):
        for theta_idx in range(len(theta)):
            rho_val = x * np.cos(theta[theta_idx]) + \
                     y * np.sin(theta[theta_idx])
            rho_idx = np.nonzero(np.abs(rho - rho_val) == np.min(np.abs(rho - rho_val)))[0]
            H[rho_idx[0], theta_idx] += 1

    return H, rho, theta


def hough_peaks(H, Q):
    """Find peaks (local maxima) in accumulator array.

    Parameters
    ----------
        H: Hough accumulator array
        Q: number of peaks to find (max)

    Returns
    -------
        peaks: Px2 matrix (P <= Q) where each row is a (rho_idx, theta_idx) pair
    """
    # This finds all the maxes, but is slow
    # peaks = np.argwhere(H.max() == H)
    # copy H just in case it's used elsewhere since we modify it
    H_copy = np.copy(H)
    indices = H_copy.ravel().argsort()[-Q:]
    indices = (np.unravel_index(i, H_copy.shape) for i in indices)
    peaks = [(i) for i in indices]

    # Fails with 3D H :-\
    # H_copy = H.copy()
    # peaks = np.zeros((Q,2),dtype=np.int)
    # for i in range(Q - 1):
    #     peaks[i] = np.unravel_index(H_copy.argmax(),H_copy.shape)
    #     # set the current max to -1 so it isn't a max again (-1 should be invalid)
    #     H_copy[peaks[i][0]][peaks[i][1]] = -1

    return peaks


def hough_lines_draw(img_out, peaks, rho, theta):
    """Draw lines on an image corresponding to accumulator peaks.

    Parameters
    ----------
        img_out: 3-channel (color) image
        peaks: Px2 matrix where each row is a (rho_idx, theta_idx) index pair
        rho: vector of rho values, such that rho[rho_idx] is a valid rho value
        theta: vector of theta values, such that theta[theta_idx] is a valid theta value
    """
    for rho_idx, theta_idx in peaks:
        a = np.cos(theta[theta_idx])
        b = np.sin(theta[theta_idx])
        x0 = a*rho[rho_idx]
        y0 = b*rho[rho_idx]
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(img_out,(x1,y1),(x2,y2),(0,0,255),2)
    pass  # TODO: Your code here (nothing to return, just draw on img_out directly)

def get_edges(image, sigma=.33):
    channels = cv2.split(image)
    if len(channels) == 3:
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))


    return cv2.Canny(gray, threshold1=lower, threshold2=upper)

def highlight_peaks(image, peaks):
    channels = cv2.split(image)
    if len(channels) < 3:
        image = cv2.merge((image.copy(), image.copy(), image.copy()))

    for point in peaks:
        cv2.circle(image, (point[1], point[0]), 5, (0, 255, 0), thickness=-1)

    return image

def compute_hough(img, edge_out_loc, peaks_out_loc, line_out_loc, num_peaks=30, rho_res=1, theta_res=pi/90):
    # Compute edge image (img_edges)
    img_edges = get_edges(img)
    write_img(edge_out_loc, img_edges)  # save as ps2-1-a-1.png

    # 2-a
    # Compute Hough Transform for lines on edge image
    H, rho, theta = hough_lines_acc(img_edges, rho_res, theta_res)

    # 2-b
    # Find peaks (local maxima) in accumulator array
    peaks = hough_peaks(H, num_peaks)

    # Store a copy of accumulator array image (from 2-a), with peaks highlighted, as ps2-2-b-1.png
    write_img(peaks_out_loc,np.clip(highlight_peaks(H,peaks), 0, 255).astype(np.uint8))

    # 2-c
    # Draw lines corresponding to accumulator peaks
    img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # copy & convert to color image
    hough_lines_draw(img_out, peaks, rho, theta)
    write_img(line_out_loc, img_out)

def compute_hough_write_last(img, line_out_loc, num_peaks=30, rho_res=1, theta_res=pi/90):
    # Compute edge image (img_edges)
    img_edges = get_edges(img)

    # Compute Hough Transform for lines on edge image
    H, rho, theta = hough_lines_acc(img_edges, rho_res, theta_res)

    # Find peaks (local maxima) in accumulator array
    peaks = hough_peaks(H, num_peaks)

    # Draw lines corresponding to accumulator peaks
    img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # copy & convert to color image
    hough_lines_draw(img_out, peaks, rho, theta)
    write_img(line_out_loc, img_out)

def compute_hough_enahnce_peaks(img, line_out_loc, num_peaks=30, rho_res=1, theta_res=pi/90):
    # Compute edge image (img_edges)
    img_edges = get_edges(img)

    # Compute Hough Transform for lines on edge image
    H, rho, theta = hough_lines_acc(img_edges, rho_res, theta_res)

    # Find peaks (local maxima) in accumulator array
    peaks = hough_peaks(H, num_peaks)

    #for this, only draw lines with theta's that are not close to 90 or 0
    real_peaks = []
    for peak in peaks:
        if not (80<(theta[peak[1]] * 180/pi)<100 or -10<(theta[peak[1]] * 180/pi)<10 or 175<(theta[peak[1]] * 180/pi)<190):
            real_peaks.append(peak)

    # Draw lines corresponding to accumulator peaks
    img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # copy & convert to color image
    hough_lines_draw(img_out, real_peaks, rho, theta)
    write_img(line_out_loc, img_out)


"""

Hough
Circles

[START]

"""

def hough_circles_acc(img_edges, min_radius, max_radius):
    # Add 1 to the max since range is inclusive and our depth of H would need 1 more as well
    max_radius += 1
    radii = np.arange(min_radius, max_radius)
    thetas = np.arange(90)
    H = np.zeros((img_edges.shape[0], img_edges.shape[1], np.abs(max_radius - min_radius)))
    for y, x in zip(*np.nonzero(img_edges)):
        for radius in radii:
            for theta in thetas:
                a = x - radius * np.cos(theta)
                b = y + radius * np.sin(theta)
                if a < H.shape[0] and b < H.shape[1] \
                    and a >= 0 and b >=0:
                    H[a,b,radius - min_radius] += 1

    return H, radii, thetas


"""

Hough
Circles

[END]

"""





def write_img(name, img):
    cv2.imwrite(os.path.join(output_dir, name + '.png'), img)

def read_img(name):
    return cv2.imread(os.path.join(input_dir, name), 0)

def smooth_img(img, sigma=3, kernel_size=5):
    return cv2.GaussianBlur(img.copy(), (kernel_size,kernel_size), sigma)

def main():
    """Run code/call functions to solve problems."""

    # 1-a
    # Load the input grayscale image
    img = cv2.imread(os.path.join(input_dir, 'ps2-input0.png'), 0)  # flags=0 ensures grayscale

    #  Compute edge image (img_edges)
    img_edges = get_edges(img)

    cv2.imwrite(os.path.join(output_dir, 'ps2-1-a-1.png'), img_edges)  # save as ps2-1-a-1.png

    # # 2-a
    # # Compute Hough Transform for lines on edge image
    H, rho, theta = hough_lines_acc(img_edges,theta_res=pi/45)  #  implement this, try calling with different parameters
    #
    # #  Store accumulator array (H) as ps2-2-a-1.png
    # # Note: Write a normalized uint8 version, mapping min value to 0 and max to 255
    cv2.imwrite("output/ps2-2-a-1.png",np.clip(H, 0, 255).astype(np.uint8))

    # 2-b
    # Find peaks (local maxima) in accumulator array
    peaks = hough_peaks(H, 30)  #  implement this, try different parameters

    #  Store a copy of accumulator array image (from 2-a), with peaks highlighted, as ps2-2-b-1.png

    #  Highlight peaks
    write_img("ps2-2-b-1",np.clip(highlight_peaks(H,peaks), 0, 255).astype(np.uint8))

    # 2-c
    # Draw lines corresponding to accumulator peaks
    img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # copy & convert to color image
    hough_lines_draw(img_out, peaks, rho, theta)  #  implement this
    cv2.imwrite(os.path.join(output_dir, 'ps2-2-c-1.png'), img_out)  # save as ps2-2-c-1.png

    # 3-a
    #  Read ps2-input0-noise.png, compute smoothed image using a Gaussian filter
    img = read_img("ps2-input0-noise.png")
    smoothed_img = cv2.GaussianBlur(img, (15,15), 2)
    write_img("ps2-3-a-1", smoothed_img)

    # # 3-b
    # #  Compute binary edge images for both original image and smoothed version
    img_edges = get_edges(img)
    write_img("ps2-3-b-1", img_edges)
    # smoothed_img_edges = get_edges(smoothed_img)

    # 3-c
    #  Apply Hough methods to smoothed image, tweak parameters to find best lines
    compute_hough(smoothed_img, "ps2-3-b-2", "ps2-3-c-1", "ps2-3-c-2", num_peaks=50, theta_res=pi/30)

    # 4
    #  Like problem 3 above, but using ps2-input1.png
    img = read_img("ps2-input1.png")
    smoothed_img = cv2.GaussianBlur(img, (15,15), 2)
    write_img("ps2-4-a-1", smoothed_img)
    #
    # img_edges = get_edges(img)
    # write_img("ps2-4-b-1", img_edges)
    #
    compute_hough(smoothed_img, "ps2-4-b-1", "ps2-4-c-1", "ps2-4-c-2", num_peaks=50, theta_res=pi/30)

    # # 5
    # #  Implement Hough Transform for circles
    #   5-a implement hough_circles_acc()
    #       Using the same original image (monochrome) as above (ps2-input1.png),
    #       smooth it, find the edges (or directly use edge image from 4-b above),
    #       and try calling your function with radius = 20:
    #
    #       Output:
    #           Smoothed image: ps2-5-a-1.png (this may be identical to  ps2-4-a-1.png)
    #           Edge image: ps2-5-a-2.png (this may be identical to  ps2-4-b-1.png)
    #           Original monochrome image with the circles drawn in color:  ps2-5-a-3.png

    img = read_img("ps2-input1.png")

    smoothed_img = smooth_img(img)
    write_img("ps2-5-a-1", smoothed_img)

    img_edges = get_edges(smoothed_img)
    write_img("ps2-5-a-2", img_edges)

    radius = 20
    H, radii, theta = hough_circles_acc(img_edges, radius, radius)

    # Find peaks (local maxima) in accumulator array
    peaks = hough_peaks(H, 20)

    # Draw lines corresponding to accumulator peaks
    img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # copy & convert to color image
    for peak in peaks:
        cv2.circle(img_out, (peak[0], peak[1]), peak[2] + radius, (0, 255, 0), thickness=2)
    write_img("ps2-5-a-3", img_out)




    #   5-b Implement a function  find_circles() that combines the above two steps,
    #       searching for circles within a given (inclusive) radius range,
    #       and returning circle centers along with their radii:
    #
    #       Output:
    #           Original monochrome image with the circles drawn in color:  ps2-5-b-1.png
    #           Text response: Describe what you had to do to find circles.

    img = read_img("ps2-input1.png")

    smoothed_img = smooth_img(img)

    img_edges = get_edges(smoothed_img)

    min_radius = 15
    max_radius = 50
    H, radii, theta = hough_circles_acc(img_edges, min_radius, max_radius)

    # Find peaks (local maxima) in accumulator array
    peaks = hough_peaks(H,20)

    # Draw lines corresponding to accumulator peaks
    img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # copy & convert to color image
    for peak in peaks:
        cv2.circle(img_out, (peak[0], peak[1]), peak[2] + min_radius, (0, 255, 0), thickness=2)
    write_img("ps2-5-b-1", img_out)




    # # 6
    # # TODO: Find lines a more realtistic image, ps2-input2.png
    #
    #   6-a Apply your line finder. Use whichever smoothing filter and edge detector
    #       that seems to work best for finding all pen edges. Don't worry (until 6b)
    #       about whether you are finding other lines as well.
    #
    #       Output: Smoothed image you used with the Hough lines drawn on them: ps2-6-a-1.png
    img = read_img("ps2-input2.png")
    smoothed_img = smooth_img(img,11,3)

    compute_hough_write_last(smoothed_img, "ps2-6-a-1",15,theta_res=.5)

    #   6-b Likely, the last step found lines that are not the boundaries of the pens.
    #       What are the problems present?
    #
    #       Output: Text response


    #   6-c Attempt to find only the lines that are the *boundaries* of the pen.
    #       Three operations you need to try are: better thresholding in finding the
    #       lines (look for stronger edges); checking the minimum length of the line;
    #       and looking for nearby parallel lines.
    #
    #       Output: Smoothed image with new Hough lines drawn: ps2-6-c-1.png

    img = read_img("ps2-input2.png")
    smoothed_img = smooth_img(img,11,3)

    compute_hough_enahnce_peaks(smoothed_img, "ps2-6-c-1",15,theta_res=.5)



    # # 7
    # # TODO: Find circles in the same realtistic image, ps2-input2.png
    #

    #   7-a Apply your circle finder. Use a smoothing filter that you think seems to
    #       work best in terms of finding all the coins.
    #
    #       Output: ps2-7-a-1.png the smoothed image you used with the circles drawn on them

    img = read_img("ps2-input2.png")

    smoothed_img = smooth_img(img)

    img_edges = get_edges(smoothed_img)

    min_radius = 22
    max_radius = 50
    H, radii, theta = hough_circles_acc(img_edges, min_radius, max_radius)

    # Find peaks (local maxima) in accumulator array
    peaks = hough_peaks(H,20)

    # Draw lines corresponding to accumulator peaks
    img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # copy & convert to color image
    for peak in peaks:
        cv2.circle(img_out, (peak[0], peak[1]), peak[2] + min_radius, (0, 255, 0), thickness=2)
    write_img("ps2-7-a-1", img_out)


    #   7-b Are there any false positives? How would/did you get rid of them?
    #
    #       Output: Text response (if you did these steps, mention where they are in
    #       the code by file, line no., and also include brief snippets)

    """
    There were many false positives. I got rid of most of them by adjusting the radius range from 15-45 to 22-50.
    I also tried to blur out the text, which was giving most of my false positives, but that didn't work.
    """

    # # 8
    # # TODO: Find lines and circles in distorted image, ps2-input3.png

    #   8-a Apply the line and circle finders to the distorted image. Can you find lines? Circles?
    #
    #       Output: Monochrome image with lines and circles (if any) found: ps2-8-a-1.png
    img = read_img("ps2-input3.png")


    smooth_img(img, 5, 3)
    # Compute edge image (img_edges)
    img_edges = get_edges(img)

    # Compute Hough Transform for lines on edge image
    H, rho, theta = hough_lines_acc(img_edges)

    # Find peaks (local maxima) in accumulator array
    peaks = hough_peaks(H, 10)

    # Draw lines corresponding to accumulator peaks
    img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # copy & convert to color image
    hough_lines_draw(img_out, peaks, rho, theta)


    smoothed_img = smooth_img(img)

    img_edges = get_edges(smoothed_img)

    min_radius = 20
    max_radius = 45
    H, radii, theta = hough_circles_acc(img_edges, min_radius, max_radius)

    # Find peaks (local maxima) in accumulator array
    peaks = hough_peaks(H,20)

    # Draw lines corresponding to accumulator peaks
    for peak in peaks:
        cv2.circle(img_out, (peak[0], peak[1]), peak[2] + min_radius, (0, 255, 0), thickness=2)

    write_img("ps2-8-a-1", img_out)

    #   8-b What might you do to fix the circle problem?
    #
    #       Output: Text response describing what you might try


    #   8-c EXTRA CREDIT:  Try to fix the circle problem (THIS IS HARD).
    #   Output: Image that is the best shot at fixing the circle problem, with circles found: ps2-8-c-1.png
    #           Text response describing what tried and what worked best (with snippets).


if __name__ == "__main__":
    main()
