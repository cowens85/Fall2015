"""Problem Set 5: Harris, SIFT, RANSAC."""

import numpy as np
import cv2
import scipy as sp
import scipy.signal as sig

import os

# I/O directories
input_dir = "input"
output_dir = "output"

"""
    Faster/better gradients?
    [START]
"""
def gauss_derivative_kernels(size, size_y=None):
    """ returns x and y derivatives of a 2D
        gauss kernel array for convolutions """
    size = int(size)
    if not size_y:
        size_y = size
    else:
        size_y = int(size_y)
    y, x = np.mgrid[-size:size+1, -size_y:size_y+1]

    #x and y derivatives of a 2D gaussian with standard dev half of size
    # (ignore scale factor)
    gx = - x * np.exp(-(x**2/float((0.5*size)**2)+y**2/float((0.5*size_y)**2)))
    gy = - y * np.exp(-(x**2/float((0.5*size)**2)+y**2/float((0.5*size_y)**2)))

    return gx,gy

def gauss_derivatives(img, n, ny=None):
    """ returns x and y derivatives of an image using gaussian
        derivative filters of size n. The optional argument
        ny allows for a different size in the y direction."""

    gx, gy = gauss_derivative_kernels(n, size_y=ny)

    imx = np.convolve(img, gx, mode='same')
    imy = np.convolve(img, gy, mode='same')

    return imx,imy

"""
    Faster/better gradients?
    [START]
"""

# Assignment code
def gradientX(image):
    """Compute image gradient in X direction.

    Parameters
    ----------
        image: grayscale floating-point image, values in [0.0, 1.0]

    Returns
    -------
        Ix: image gradient in X direction, values in [-1.0, 1.0]
    """
    return  cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)
    # shape = image.shape
    # rows = shape[0]
    # cols = shape[1]
    #
    # grad_x = np.zeros([rows,cols], dtype=np.float)
    #
    # for row in range(rows):
    #     for col in range(cols-1):
    #         grad_x[row][col] = abs(image[row][col+1] - image[row][col])
    #
    # return grad_x

def gradientY(image):
    """Compute image gradient in Y direction.

    Parameters
    ----------
        image: grayscale floating-point image, values in [0.0, 1.0]

    Returns
    -------
        Iy: image gradient in Y direction, values in [-1.0, 1.0]
    """
    return cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5)
    # shape = image.shape
    # rows = shape[0]
    # cols = shape[1]
    #
    # grad_y = np.zeros([rows,cols], dtype=np.float)
    #
    # for row in range(rows-1):
    #     for col in range(cols):
    #         grad_y[row][col] = abs(image[row+1][col] - image[row][col])
    #
    # return grad_y


def make_image_pair(image1, image2):
    """Adjoin two images side-by-side to make a single new image.

    Parameters
    ----------
        image1: first image, could be grayscale or color (BGR)
        image2: second image, same type as first

    Returns
    -------
        image_pair: combination of both images, side-by-side, same type
    """
    tmp1 = image1.copy()
    tmp2 = image2.copy()

    min = np.min(tmp1)
    tmp1 += abs(min)
    max = np.max(tmp1)
    tmp1 *= 255.0 / max

    min = np.min(tmp2)
    tmp2 += abs(min)
    max = np.max(tmp2)
    tmp2 *= 255.0 / max


    return np.concatenate((tmp1, tmp2), axis=1)


def harris_response(Ix, Iy, kernel, alpha):
    """Compute Harris reponse map using given image gradients.

    Parameters
    ----------
        Ix: image gradient in X direction, values in [-1.0, 1.0]
        Iy: image gradient in Y direction, same size and type as Ix
        kernel: 2D windowing kernel with weights, typically square
        alpha: Harris detector parameter multiplied with square of trace

    Returns
    -------
        R: Harris response map, same size as inputs, floating-point
    """

    # M = "something"
    #
    # R = np.linalg.det(M) - alpha * (np.sum(np.diag(M))**2)

    #compute components of the structure tensor
    w_xx = sig.convolve2d(Ix * Ix, kernel, mode='same')
    # Wxy = sig.convolve2d(Ix * Iy,kernel, mode='same')
    w_yy = sig.convolve2d(Iy * Iy, kernel, mode='same')

    R = w_xx *w_yy - alpha* ((w_xx - w_yy)**2)

    # # determinant and trace
    # det = w_xx*w_yy - Wxy**2
    # trace = w_xx + w_yy
    #
    # R = det / trace

    return R


def find_corners(R, threshold, radius):
    """Find corners in given response map.

    Parameters
    ----------
        R: floating-point response map, e.g. output from the Harris detector
        threshold: response values less than this should not be considered plausible corners
        radius: radius of circular region for non-maximal suppression (could be half the side of square instead)

    Returns
    -------
        corners: peaks found in response map R, as a sequence (list) of (x, y) coordinates
    """

    #find top corner candidates above a threshold
    corner_threshold = max(R.ravel()) * threshold
    R_t = (R > corner_threshold) * 1

    M, N = R_t.shape
    for x in range(0,M-radius+1):
        for y in range(0,N-radius+1):
            window = R_t[x:x+radius, y:y+radius]
            if np.sum(window)==0:
                localMax=0
            else:
                localMax = np.amax(window)
            maxCoord = np.argmax(window)
            # zero all but the localMax in the window
            window[:] = 0
            window.flat[maxCoord] = localMax

    #get coordinates of candidates
    candidates = R_t.nonzero()
    corner_coords = [(candidates[0][c],candidates[1][c]) for c in range(len(candidates[0]))]
    #...and their values
    candidate_values = [R[c[0]][c[1]] for c in corner_coords]



    #sort candidates
    # index = np.argsort(candidate_values)

    # #store allowed point locations in array
    # allowed_locations = np.zeros(R.shape)
    # allowed_locations[radius:-radius,radius:-radius] = 1
    #
    # #select the best points taking min_distance into account
    # filtered_corner_coords = []
    # for i in index:
    #     if allowed_locations[corner_coords[i][0]][corner_coords[i][1]] == 1:
    #         filtered_corner_coords.append(corner_coords[i])
    #         allowed_locations[(corner_coords[i][0] - radius):(corner_coords[i][0] + radius), (corner_coords[i][1] - radius):(corner_coords[i][1] + radius)] = 0

    return corner_coords


def draw_corners(image, corners):
    """Draw corners on (a copy of) given image.

    Parameters
    ----------
        image: grayscale floating-point image, values in [0.0, 1.0]
        corners: peaks found in response map R, as a sequence (list) of (x, y) coordinates

    Returns
    -------
        image_out: copy of image with corners drawn on it, color (BGR), uint8, values in [0, 255]
    """
    image_out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for y, x in corners:
        cv2.circle(image_out, (x, y),2,(0, 255, 0))

    return image_out


def gradient_angle(Ix, Iy):
    """Compute angle (orientation) image given X and Y gradients.

    Parameters
    ----------
        Ix: image gradient in X direction
        Iy: image gradient in Y direction, same size and type as Ix

    Returns
    -------
        angle: gradient angle image, each value in degrees [0, 359)
    """

    # Note: +ve X axis points to the right (0 degrees), +ve Y axis points down (90 degrees)

    # angle = np.zeros(Ix.shape)
    # for row in range(Ix.shape[0]):
    #     for col in range(Ix.shape[1]):
    #         angle[row][col] = cv2.fastAtan2(Iy[row][col], Ix[row][col])

    return np.degrees(np.arctan2(Ix, Iy))

def get_keypoints(points, R, angle, _size, _octave=0):
    """Create OpenCV KeyPoint objects given interest points, response and angle images.

    Parameters
    ----------
        points: interest points (e.g. corners), as a sequence (list) of (x, y) coordinates
        R: floating-point response map, e.g. output from the Harris detector
        angle: gradient angle (orientation) image, each value in degrees [0, 359)
        _size: fixed _size parameter to pass to cv2.KeyPoint() for all points
        _octave: fixed _octave parameter to pass to cv2.KeyPoint() for all points

    Returns
    -------
        keypoints: a sequence (list) of cv2.KeyPoint objects
    """

    # Note: You should be able to plot the keypoints using cv2.drawKeypoints() in OpenCV 2.4.9+
    keypoints = []
    for i in range(len(points)):
        point = points[i]
        y, x = point
        keypoints.append(cv2.KeyPoint(x=x, y=y, _size=_size, _angle=angle[y][x], _response=R[y][x], _octave=_octave))

    return keypoints


def get_descriptors(image, keypoints):
    """Extract feature descriptors from image at each keypoint.

    Parameters
    ----------
        keypoints: a sequence (list) of cv2.KeyPoint objects

    Returns
    -------
        descriptors: 2D NumPy array of shape (len(keypoints), 128)
    """
    # Note: You can use OpenCV's SIFT.compute() method to extract descriptors, or write your own!
    sift = cv2.SIFT()
    return sift.compute(image.astype(np.uint8), keypoints)[1]


def match_descriptors(desc1, desc2):
    """Match feature descriptors obtained from two images.

    Parameters
    ----------
        desc1: descriptors from image 1, as returned by SIFT.compute()
        desc2: descriptors from image 2, same format as desc1

    Returns
    -------
        matches: a sequence (list) of cv2.DMatch objects containing corresponding descriptor indices
    """

    # Note: You can use OpenCV's descriptor matchers, or roll your own!
    bf_matcher = cv2.BFMatcher(normType=cv2.HAMMING_NORM_TYPE, crossCheck=True)

    # Compute the matches between both images.
    matches = bf_matcher.match(desc1, desc2)

    return matches


def draw_matches(image1, image2, kp1, kp2, matches):
    """Show matches by drawing lines connecting corresponding keypoints.

    Parameters
    ----------
        image1: first image
        image2: second image, same type as first
        kp1: list of keypoints (cv2.KeyPoint objects) found in image1
        kp2: list of keypoints (cv2.KeyPoint objects) found in image2
        matches: list of matching keypoint index pairs (as cv2.DMatch objects)

    Returns
    -------
        image_out: image1 and image2 joined side-by-side with matching lines; color image (BGR), uint8, values in [0, 255]
    """

    # Note: DO NOT use OpenCV's match drawing function(s)! Write your own :)
    # Compute number of channels.
    num_channels = 1
    if len(image1.shape) == 3:
        num_channels = image1.shape[2]
    # Separation between images.
    margin = 10
    # Create an array that will fit both images (with a margin of 10 to separate
    # the two images)
    joined_image = np.zeros((max(image1.shape[0], image2.shape[0]), image1.shape[1] + image2.shape[1] + margin, 3))
    if num_channels == 1:
        for channel_idx in range(3):
            joined_image[:image1.shape[0], :image1.shape[1], channel_idx] = image1
            joined_image[:image2.shape[0], image1.shape[1] + margin:, channel_idx] = image2
    else:
        joined_image[:image1.shape[0], :image1.shape[1]] = image1
        joined_image[:image2.shape[0], image1.shape[1] + margin:] = image2

    for match in matches:
        image_1_point = (int(kp1[match.queryIdx].pt[0]),
                         int(kp1[match.queryIdx].pt[1]))
        image_2_point = (int(kp2[match.trainIdx].pt[0] + image1.shape[1] + margin),
                       int(kp2[match.trainIdx].pt[1]))

        cv2.circle(joined_image, image_1_point, 5, (0, 0, 255), thickness=-1)
        cv2.circle(joined_image, image_2_point, 5, (0, 255, 0), thickness=-1)
        cv2.line(joined_image, image_1_point, image_2_point, (255, 0, 0), thickness=3)

    return joined_image


def compute_translation_RANSAC(kp1, kp2, matches, threshold=15):
    """Compute best translation vector using RANSAC given keypoint matches.

    Parameters
    ----------
        kp1: list of keypoints (cv2.KeyPoint objects) found in image1
        kp2: list of keypoints (cv2.KeyPoint objects) found in image2
        matches: list of matches (as cv2.DMatch objects)

    Returns
    -------
        translation: translation/offset vector <x, y>, NumPy array of shape (2, 1)
        good_matches: consensus set of matches that agree with this translation
    """

    # Brain stopped working...
    # Store all the things separately

    num_matches = len(matches)
    x_deltas = np.zeros(num_matches)
    y_deltas = np.zeros(num_matches)
    translations = np.zeros((num_matches,2,1))

    for i in range(num_matches):
        match = matches[i]
        x = int(kp1[match.queryIdx].pt[0])
        y = int(kp1[match.queryIdx].pt[1])
        x_prime = int(kp2[match.trainIdx].pt[0])
        y_prime = int(kp2[match.trainIdx].pt[1])

        x_delta = x_prime - x
        y_delta = y_prime - y

        x_deltas[i] = x_delta
        y_deltas[i] = y_delta

        translations[i][0] = x_delta
        translations[i][1] = y_delta

    max_consensus = 0
    # print "x_deltas ", x_deltas
    # print "y_deltas ", y_deltas
    for i in range(num_matches):
        x_deltas_copy = x_deltas.copy()
        x_delta = x_deltas_copy[i]

        x_deltas_copy[x_deltas_copy <= x_delta - threshold] = 0
        x_deltas_copy[x_deltas_copy >= x_delta + threshold] = 0

        x_consensus_indices = np.where(x_deltas_copy != 0)[0]


        y_deltas_copy = y_deltas.copy()
        y_delta = y_deltas_copy[i]

        y_deltas_copy[y_deltas_copy <= y_delta - threshold] = 0
        y_deltas_copy[y_deltas_copy >= y_delta + threshold] = 0

        y_consensus_indices = np.where(y_deltas_copy != 0)[0]


        # print "x_consensus ", x_consensus_indices
        # print "y_consensus ", y_consensus_indices

        consensus_indices = np.intersect1d(y_consensus_indices, x_consensus_indices)

        num_consensus = len(consensus_indices)
        if num_consensus > max_consensus:
            max_consensus = num_consensus
            max_consensus_indices = consensus_indices
            max_translation = translations[i]

    good_matches = np.asarray(matches)[max_consensus_indices]

    # print "num matches", num_matches
    # print "num good matches", len(good_matches)
    for match in good_matches:
        x = int(kp1[match.queryIdx].pt[0])
        y = int(kp1[match.queryIdx].pt[1])
        x_prime = int(kp2[match.trainIdx].pt[0])
        y_prime = int(kp2[match.trainIdx].pt[1])
        x_delta = x_prime - x
        y_delta = y_prime - y

        delta = np.sqrt(x_delta**2 + y_delta**2)
        # print "x,y: ", x, " ", y
        # print "x',y': ", x_prime, " ", y_prime
        # print "delta", delta

    # print "translation", max_translation

    return max_translation, good_matches


def find_best_transform(max_consensus, num_matches, threshold, transforms, x_deltas, y_deltas):
    for i in range(num_matches):
        x_deltas_copy = x_deltas.copy()
        x_delta = x_deltas_copy[i]

        x_deltas_copy[x_deltas_copy <= x_delta - threshold] = 0
        x_deltas_copy[x_deltas_copy >= x_delta + threshold] = 0

        x_consensus_indices = np.where(x_deltas_copy != 0)[0]

        y_deltas_copy = y_deltas.copy()
        y_delta = y_deltas_copy[i]

        y_deltas_copy[y_deltas_copy <= y_delta - threshold] = 0
        y_deltas_copy[y_deltas_copy >= y_delta + threshold] = 0

        y_consensus_indices = np.where(y_deltas_copy != 0)[0]

        consensus_indices = np.intersect1d(y_consensus_indices, x_consensus_indices)
        num_consensus = len(consensus_indices)
        # print "len(consensus_indices) ", num_consensus
        if num_consensus > max_consensus:
            max_consensus = num_consensus
            max_consensus_indices = consensus_indices
            max_transform = transforms[i]

    return max_consensus_indices, max_transform


def compute_similarity_RANSAC(kp1, kp2, matches, threshold=15):
    """Compute best similarity transform using RANSAC given keypoint matches.

    Parameters
    ----------
        kp1: list of keypoints (cv2.KeyPoint objects) found in image1
        kp2: list of keypoints (cv2.KeyPoint objects) found in image2
        matches: list of matches (as cv2.DMatch objects)

    Returns
    -------
        transform: similarity transform matrix, NumPy array of shape (2, 3)
        good_matches: consensus set of matches that agree with this transform
    """

    num_matches = len(matches)
    indices = np.arange(num_matches)

    x_deltas = np.zeros(num_matches)
    y_deltas = np.zeros(num_matches)
    transforms = np.zeros((num_matches,2,3))

    np.random.shuffle(indices)

    while len(indices) > 1:

        index_1 = indices[0]
        index_2 = indices[1]

        # remove the used indices so we don't use them again
        indices = indices[2:]

        match_1 = matches[index_1]
        point_1_query = kp1[match_1.queryIdx].pt
        point_1_train = kp2[match_1.trainIdx].pt
        u = point_1_query[0]
        v = point_1_query[1]
        u_prime = point_1_train[0]
        v_prime = point_1_train[1]

        match_2 = matches[index_2]
        point_2_query = kp1[match_2.queryIdx].pt
        point_2_train = kp2[match_2.trainIdx].pt
        x = point_2_query[0]
        y = point_2_query[1]
        x_prime = point_2_train[0]
        y_prime = point_2_train[1]

        # Get into the form 'Ax = b'
        # A = np.array([[u, v, x, y], [-v, u, -y, x], [1, 0, 1, 0], [0, 1, 0, 1]])
        # b_tmp = np.array([u_prime, v_prime, x_prime, y_prime])

        A = np.array([[u, -v, 1, 0], [v, u, 0, 1], [x, -y, 1, 0], [y, x, 0, 1]])
        b_tmp = np.array([u_prime, v_prime, x_prime, y_prime])

        # Solve for 'x'
        a, b, c, d = np.linalg.solve(A, b_tmp)
        # print "solution: ", np.linalg.solve(A, b_tmp), "\n\n"

        """
        A little help from the forums:

            Alexander Stephen Burch

            Just now From my understanding, you'll calculate T. So you found a and b from linalg.solve,
            then you will get c and d. Next, you'll loop through each point, and do matrix multiplication
            between T and the point to get a new point. Then, we should compare this new point with the point
            found from the match in the 2nd image (ie use, trainIdx in KP2).
        """

        transform = np.array([[a, -b, c],
                              [b,  a, d]], dtype=np.float)

        # make sure to add the one to the point so the multiplication will work
        t_point_1_query = np.dot(transform, np.append(point_1_query, 1))

        # print "transform: ", transform
        # print "transformed pt: ",t_point_1_query
        # print "matching pt: ", point_1_query, "\n\n"

        x_delta = point_1_query[0] - t_point_1_query[0]
        y_delta = point_1_query[1] - t_point_1_query[1]

        # deltas[index_1] = delta
        # deltas[index_2] = delta


        # delta = np.sqrt((x_delta)**2 + (y_delta)**2)
        # print "delta", delta

        x_deltas[index_1] = x_delta
        y_deltas[index_1] = y_delta

        x_deltas[index_2] = x_delta
        y_deltas[index_2] = y_delta


        transforms[index_1] = transform
        transforms[index_2] = transform

    max_consensus = 0
    max_consensus_indices, max_transform = find_best_transform(max_consensus, num_matches, threshold, transforms,
                                                               x_deltas, y_deltas)

    good_matches = np.asarray(matches)[max_consensus_indices]

    return max_transform, good_matches


def compute_affine(kp1, kp2, matches, threshold=15):
    """Compute best similarity transform using RANSAC given keypoint matches.

    Parameters
    ----------
        kp1: list of keypoints (cv2.KeyPoint objects) found in image1
        kp2: list of keypoints (cv2.KeyPoint objects) found in image2
        matches: list of matches (as cv2.DMatch objects)

    Returns
    -------
        transform: similarity transform matrix, NumPy array of shape (2, 3)
        good_matches: consensus set of matches that agree with this transform
    """

    num_matches = len(matches)
    indices = np.arange(num_matches)

    x_deltas = np.zeros(num_matches)
    y_deltas = np.zeros(num_matches)
    transforms = np.zeros((num_matches,2,3))

    np.random.shuffle(indices)

    while len(indices) > 2:

        index_1 = indices[0]
        index_2 = indices[1]
        index_3 = indices[2]

        # remove the used indices so we don't use them again
        indices = indices[3:]

        match_1 = matches[index_1]
        point_1_query = kp1[match_1.queryIdx].pt
        point_1_train = kp2[match_1.trainIdx].pt
        u = point_1_query[0]
        v = point_1_query[1]
        u_prime = point_1_train[0]
        v_prime = point_1_train[1]

        match_2 = matches[index_2]
        point_2_query = kp1[match_2.queryIdx].pt
        point_2_train = kp2[match_2.trainIdx].pt
        x = point_2_query[0]
        y = point_2_query[1]
        x_prime = point_2_train[0]
        y_prime = point_2_train[1]

        match_3 = matches[index_3]
        point_3_query = kp1[match_3.queryIdx].pt
        point_3_train = kp2[match_3.trainIdx].pt
        n = point_3_query[0]
        m = point_3_query[1]
        n_prime = point_3_train[0]
        m_prime = point_3_train[1]

        # Get into the form 'Ax = b'
        # A = np.array([[u, v, x, y], [-v, u, -y, x], [1, 0, 1, 0], [0, 1, 0, 1]])
        # b_tmp = np.array([u_prime, v_prime, x_prime, y_prime])

        A = np.array([[u, v, 1, 0, 0, 0], [0, 0, 0, u, v, 1], [x, y, 1, 0, 0, 0], [0, 0, 0, x, y, 1], [n, m, 1, 0, 0, 0], [0, 0, 0, n, m, 1]])
        b_tmp = np.array([u_prime, v_prime, x_prime, y_prime, n_prime, m_prime])

        # Solve for 'x'
        a, b, c, d, e, f = np.linalg.solve(A, b_tmp)
        # print "A:\n", A
        # print "b:\n", b_tmp
        # solution = np.linalg.solve(A, b_tmp)
        # print "solution: ", solution, "\n\n"
        # a, b, c, d, e, f = solution

        """
        A little help from the forums:

            Alexander Stephen Burch

            Just now From my understanding, you'll calculate T. So you found a and b from linalg.solve,
            then you will get c and d. Next, you'll loop through each point, and do matrix multiplication
            between T and the point to get a new point. Then, we should compare this new point with the point
            found from the match in the 2nd image (ie use, trainIdx in KP2).
        """

        transform = np.array([[a, b, c],
                              [d,  e, f]], dtype=np.float)

        # make sure to add the one to the point so the multiplication will work
        t_point_1_query = np.dot(transform, np.append(point_1_query, 1))

        # print "transform: ", transform
        # print "transformed pt: ",t_point_1_query
        # print "matching pt: ", point_1_query, "\n\n"

        x_delta = point_1_query[0] - t_point_1_query[0]
        y_delta = point_1_query[1] - t_point_1_query[1]

        # deltas[index_1] = delta
        # deltas[index_2] = delta


        # delta = np.sqrt((x_delta)**2 + (y_delta)**2)
        # print "delta", delta

        x_deltas[index_1] = x_delta
        y_deltas[index_1] = y_delta

        x_deltas[index_2] = x_delta
        y_deltas[index_2] = y_delta


        transforms[index_1] = transform
        transforms[index_2] = transform

    max_consensus = 0
    max_consensus_indices, max_transform = find_best_transform(max_consensus, num_matches, threshold, transforms,
                                                               x_deltas, y_deltas)

    good_matches = np.asarray(matches)[max_consensus_indices]

    return max_transform, good_matches


def scale_to_img(matrix):
    max = matrix.max()
    matrix *= 255.0/max
    return matrix.astype(np.uint8)


def write_image(image, name, scale=False):
    if scale:
        image = scale_to_img(image)
    cv2.imwrite(os.path.join(output_dir, name), image)
    return image

"""

1a

"""
def one_a(img_name, run="foo"):
    img = cv2.imread(os.path.join(input_dir, img_name), cv2.IMREAD_GRAYSCALE).astype(np.float_) / 255.0
    Ix = gradientX(img)
    Iy = gradientY(img)
    pair = make_image_pair(Ix, Iy)

    if run == "transA":
        write_image(pair, "ps5-1-a-1.png", scale=True)
    elif run == "simA":
        write_image(pair, "ps5-1-a-2.png", scale=True)

    return img, Ix, Iy

"""

1b

transA, transB, simA and simB

"""
def one_b(Ix, Iy, run="foo"):
    kernel = cv2.getGaussianKernel(3,0.3)
    # kernel = np.ones((3, 3), dtype=np.float) / 9.0
    R = harris_response(Ix, Iy, kernel, 0.04)

    # Scale/type-cast response map and write to file
    if run == "transA":
        write_image(R, "ps5-1-b-1.png", scale=True)
    elif run == "transB":
        write_image(R, "ps5-1-b-2.png", scale=True)
    elif run == "simA":
        write_image(R, "ps5-1-b-3.png", scale=True)
    elif run == "simB":
        write_image(R, "ps5-1-b-4.png", scale=True)

    return R

"""

1c

transA, transB, simA and simB

"""
def one_c(R, img, threshold=0.3, radius=5.0, run="foo"):
    corners = find_corners(R, threshold, radius)
    img_out = draw_corners(scale_to_img(img), corners)

    if run == "transA":
        write_image(img_out, "ps5-1-c-1.png", scale=True)
    elif run == "transB":
        write_image(img_out, "ps5-1-c-2.png", scale=True)
    elif run == "simA":
        write_image(img_out, "ps5-1-c-3.png", scale=True)
    elif run == "simB":
        write_image(img_out, "ps5-1-c-4.png", scale=True)

    return corners

"""

2a

transA, transB and simA, simB

"""
def two_a(a_img, a_Ix, a_Iy, a_corners, a_R, b_img, b_Ix, b_Iy, b_corners, b_R, a_run="foo", _size=5.0, _octave=0):
    a_angle = gradient_angle(a_Ix, a_Iy)
    a_kps = get_keypoints(a_corners, a_R, a_angle, _size, _octave)

    # TODO: Draw keypoints on transA

    b_angle = gradient_angle(b_Ix, b_Iy)
    b_kps = get_keypoints(b_corners, b_R, b_angle, _size, _octave)

    # TODO: Similarly, find keypoints for transB and draw them
    # TODO: Combine transA and transB images (with keypoints drawn) using make_image_pair() and write to file
    # make_image_pair(imgA, imgB)

    # print a_img.shape

    a_img_match = cv2.drawKeypoints(cv2.cvtColor(a_img.copy().astype(np.uint8), cv2.COLOR_GRAY2BGR), a_kps,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    b_img_match = cv2.drawKeypoints(cv2.cvtColor(b_img.copy().astype(np.uint8), cv2.COLOR_GRAY2BGR), b_kps,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    combined_img = make_image_pair(a_img_match, b_img_match)

    if a_run == "transA":
        write_image(combined_img, "ps5-2-a-1.png")
    elif a_run == "simA":
        write_image(combined_img, "ps5-2-a-2.png")


    # TODO: Ditto for (simA, simB) pair

    return a_kps, b_kps

"""

2b

"""
def two_b(a_img, a_kps, b_img, b_kps, a_run="foo"):
    a_descs = get_descriptors(a_img, a_kps)
    b_descs = get_descriptors(b_img, b_kps)

    matches = match_descriptors(a_descs.astype(np.uint8), b_descs.astype(np.uint8))
    match_img = draw_matches(a_img, b_img, a_kps, b_kps, matches)

    if a_run == "transA":
        write_image(match_img, "ps5-2-b-1.png")
    if a_run == "simA":
        write_image(match_img, "ps5-2-b-2.png")

    return a_descs, b_descs, matches

"""

3a

Compute translation vector using RANSAC for (transA, transB) pair, draw biggest consensus set
"""
def three_a(a_kps, b_kps, matches, a_img, b_img, a_run="foo", threshold=50):
    translation, good_matches = compute_translation_RANSAC(a_kps, b_kps, matches, threshold)

    match_img = draw_matches(a_img, b_img, a_kps, b_kps, good_matches)
    print "3a."
    print "Translation: ", translation
    print "% Good Matches: ", (len(good_matches)*1.0 / len(matches)) * 100

    # if a_run == "transA":
    write_image(match_img, "ps5-3-a-1.png")
    # elif a_run == "transB":
    #     write_image(match_img, "ps5-3-a-2.png", scale=True)
    # elif a_run == "simA":
    #     write_image(match_img, "ps5-3-a-3.png", scale=True)
    # elif a_run == "simB":

    return translation, good_matches

"""

3b

"""
def three_b(a_kps, b_kps, matches, a_img, b_img, a_run="foo", threshold=15):
    transform, good_matches = compute_similarity_RANSAC(a_kps, b_kps, matches, threshold)

    match_img = draw_matches(a_img, b_img, a_kps, b_kps, good_matches)
    write_image(match_img, "ps5-3-b-1.png")
    print "3b."
    print "Transform: ", transform
    print "% Good Matches: ", (len(good_matches)*1.0 / len(matches)) * 100

    return transform, good_matches

"""

3c

"""
def three_c(a_kps, b_kps, matches, a_img, b_img, a_run="foo", threshold=20):
    transform, good_matches = compute_affine(a_kps, b_kps, matches, threshold)

    match_img = draw_matches(a_img, b_img, a_kps, b_kps, good_matches)
    write_image(match_img, "ps5-3-c-1.png")
    print "3c."
    print "Transform: ", transform
    print "% Good Matches: ", (len(good_matches)*1.0 / len(matches)) * 100

    return transform, good_matches

"""

3d

"""
def three_d(transform, a_img, b_img):
    print "HERE"
    print transform
    rand = np.zeros((b_img.shape[0], b_img.shape[1]))
    out_warp = cv2.invertAffineTransform(transform)
    warped_image = cv2.warpAffine(b_img.astype(np.uint8), out_warp, (b_img.shape[1], b_img.shape[0]), flags=cv2.INTER_LINEAR)
    write_image(warped_image, "ps5-3-d-1.png")
    merged = cv2.merge((rand.astype(np.uint8),warped_image.astype(np.uint8),a_img.astype(np.uint8)))
    write_image(merged, "ps5-3-d-2.png")

    return out_warp
    
"""

3e

"""
def three_e(affine, a_img, b_img):
    rand = np.zeros((a_img.shape[0], a_img.shape[1]))
    out_warp = cv2.invertAffineTransform(affine)
    warped_image = cv2.warpAffine(b_img.astype(np.uint8),out_warp, (b_img.shape[1], b_img.shape[0]), flags=cv2.INTER_LINEAR)
    write_image(warped_image, "ps5-3-e-1.png")
    merged = cv2.merge((rand.astype(np.uint8),warped_image.astype(np.uint8),a_img.astype(np.uint8)))
    write_image(merged, "ps5-3-e-2.png")
    
    return out_warp

# Driver code
def main():
    # image_pairs = np.array([["transA", "transB"]])
    # image_pairs = np.array([["simA", "simB"]])
    image_pairs = np.array([["transA", "transB"], ["simA", "simB"]])

    for img_pair in image_pairs:
        a_run = img_pair[0]
        b_run = img_pair[1]

        """ 1a """
        a_img, a_Ix, a_Iy = one_a(a_run + ".jpg", run=a_run)
        b_img, b_Ix, b_Iy = one_a(b_run + ".jpg", run=b_run)

        # """ 1b """
        # a_R = one_b(a_Ix, a_Iy, run=a_run)
        # b_R = one_b(b_Ix, b_Iy, run=b_run)
        #
        # """ 1c """
        # if a_run == "transA":
        #     threshold=0.1
        #     radius=15
        # else:
        #     threshold=0.1
        #     radius=20
        # a_corners = one_c(a_R, a_img, threshold, radius, run=a_run)
        # b_corners = one_c(b_R, b_img, threshold, radius, run=b_run)
        #
        # # print "num A corners: ", len(a_corners)
        # # print "num B corners: ", len(b_corners)
        #
        # """ 2a """
        # a_kps, b_kps = two_a(a_img, a_Ix, a_Iy, a_corners, a_R, b_img, b_Ix, b_Iy, b_corners, b_R, a_run=a_run)
        #
        # """ 2b """
        # a_descs, b_descs, matches = two_b(a_img, a_kps, b_img, b_kps, a_run=a_run)
        #
        # """
        # 3a  -  Compute translation vector using RANSAC for (transA, transB) pair, draw biggest consensus set
        # """
        # if a_run == "transA":
        #     translation, good_matches = three_a(a_kps, b_kps, matches, a_img, b_img, a_run=a_run, threshold=35)
        #
        # """
        # 3b  -  Compute similarity transform for (simA, simB) pair, draw biggest consensus set
        # """
        # if a_run == "simA":
        #     similarity_transform, good_matches = three_b(a_kps, b_kps, matches, a_img, b_img, a_run=a_run, threshold=20)
        #     # print "transform: ", transform
        #     # print "matches: ", good_matches
        #
        # """
        # 3c
        # """
        # if a_run == "simA":
        #     affine_transform, good_matches = three_c(a_kps, b_kps, matches, a_img, b_img, a_run=a_run, threshold=15)
        #
        # """
        # 3d
        # """
        # if a_run == "simA":
        #     three_d(similarity_transform, a_img, b_img)
        #
        # """
        # 3e
        # """
        # if a_run == "simA":
        #     three_e(affine_transform, a_img, b_img)
            


# def main():
#     # Note: Comment out parts of this code as necessary
#
#     """ 1a """
#     transA = cv2.imread(os.path.join(input_dir, "transA.jpg"), cv2.IMREAD_GRAYSCALE).astype(np.float_) / 255.0
#     transA_Ix = gradientX(transA)  # TODO: implement this
#     transA_Iy = gradientY(transA)  # TODO: implement this
#     transA_pair = make_image_pair(transA_Ix, transA_Iy)
#     write_image(transA_pair, "ps5-1-a-1.png", scale = True)
#
#     # TODO: Similarly for simA.jpg
#
#     """ 1b """
#     transA_R = harris_response(transA_Ix, transA_Iy, np.ones((3, 3), dtype=np.float) / 9.0, 0.04)
#     # TODO: Scale/type-cast response map and write to file
#     write_image(transA_R, "ps5-1-b-1.png", scale=True)
#
#     # TODO: Similarly for transB, simA and simB (you can write a utility function for grouping operations on each image)
#
#     """ 1c """
#     transA_corners = find_corners(transA_R, 0.15, 2.5)  # TODO: implement this, tweak parameters till you get good corners
#     transA_out = draw_corners(scale_to_img(transA), transA_corners)  # TODO: implement this
#     # TODO: Write image to file
#     write_image(transA_out, "ps5-1-c-1.png")
#
#     # TODO: Similarly for transB, simA and simB (write a utility function if you want)
#
#     """ 2a """
#     transA_angle = gradient_angle(transA_Ix, transA_Iy)
#     transA_kp = get_keypoints(transA_corners, transA_R, transA_angle, _size=5.0, _octave=0)  # TODO: implement this, update parameters
#     # TODO: Draw keypoints on transA
#
#     # TODO: Similarly, find keypoints for transB and draw them
#     # TODO: Combine transA and transB images (with keypoints drawn) using make_image_pair() and write to file
#
#     # TODO: Ditto for (simA, simB) pair
#
#     # """ 2b """
#     # transA_desc = get_descriptors(transA, transA_kp)  # TODO: implement this
#     # # TODO: Similarly get transB_desc
#     # # TODO: Find matches: trans_matches = match_descriptors(transA_desc, transB_desc)
#     # # TODO: Draw matches and write to file: draw_matches(transA, transB, transA_kp, transB_kp, trans_matches)
#     #
#     # # TODO: Ditto for (simA, simB) pair (may have to vary some parameters along the way?)
#     #
#     # """ 3a """
#     # # TODO: Compute translation vector using RANSAC for (transA, transB) pair, draw biggest consensus set
#     #
#     # """ 3b """
#     # # TODO: Compute similarity transform for (simA, simB) pair, draw biggest consensus set
#
#     # Extra credit: 3c, 3d, 3e


if __name__ == "__main__":
    main()
