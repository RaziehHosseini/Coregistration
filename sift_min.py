import numpy as np
from numba import jit     # 0.44.0

@jit(nopython=True)
def patch_coordinates_rotated(x, y, N, width, theta):
    coordinates = np.zeros((N, N, 2), dtype=np.int64)
    st = np.sin(theta)
    ct = np.cos(theta)
    half_N = N // 2

    for x_ in range(N):
        for y_ in range(N):
            dx = (x_ - half_N) * width / (N - 1)
            dy = (y_ - half_N) * width / (N - 1)
            coordinates[x_, y_, 0] = round(x + ct * dx - st * dy)
            coordinates[x_, y_, 1] = round(y + st * dx + ct * dy)

    return coordinates

@jit(nopython=True)
def patch_coordinates(x, y, N, width):
    coordinates = np.zeros((N, N, 2), dtype=np.int64)
    half_N = N // 2

    for x_ in range(N):
        for y_ in range(N):
            coordinates[x_, y_, 0] = round(x + (x_ - half_N) * width / (N - 1))
            coordinates[x_, y_, 1] = round(y + (y_ - half_N) * width / (N - 1))

    return coordinates

@jit(nopython=True)
def gauss_filter_N(sigma, N):
    filter = np.zeros((N, N), dtype=np.float64)
    offset = N / 2 - 0.5
    two_sigma_sq = 2 * sigma * sigma
    for x_ in range(N):
        for y_ in range(N):
            x = x_ - offset
            y = y_ - offset
            G_sigma = 1.0 / (np.pi * two_sigma_sq) * np.e ** (-(x * x + y * y) / (two_sigma_sq))
            filter[x_, y_] = G_sigma
    return filter


@jit(nopython=True)
def derivative_of_gaussian(sigma, axis=0):
    N = int(6 * sigma)
    if N % 2 == 0:
        N += 1
    filter = np.zeros((N, N), dtype=np.float64)                                # TO ASK: filter as the variable name?
    offset = int(N / 2)
    two_sigma_sq = 2 * sigma * sigma
    two_pi_sigma_4 = 2 * np.pi * sigma ** 4

    if axis == 0:
        for x_ in range(N):
            for y_ in range(N):
                x = x_ - offset
                y = y_ - offset
                G_sigma = - x / two_pi_sigma_4 *  np.exp(-(x * x + y * y) / two_sigma_sq)
                filter[x_, y_] = G_sigma
    else:
        for x_ in range(N):
            for y_ in range(N):
                x = x_ - offset
                y = y_ - offset
                G_sigma = - y / two_pi_sigma_4 * \
                          np.exp(-(x * x + y * y) / two_sigma_sq)
                filter[x_, y_] = G_sigma
    return filter


@jit(nopython=True)
def convolve_at(image, filter, x, y):
    h, w = image.shape
    filter_size = filter.shape[0]
    half_ks = int(filter_size / 2)
    x_low = x - half_ks
    y_low = y - half_ks

    v = 0.0
    for x_ in range(filter_size):
        for y_ in range(filter_size):
            xi = min(max(0, x_low + x_), h - 1)
            yi = min(max(0, y_low + y_), w - 1)
            v += image[xi, yi] * filter[x_, y_]

    return v

# SIFT

def sift_descriptor(image, features):
    num_features = features.shape[0]
    N_o = 5                                                               # TO ASK: N_o =5 ?
    rad2deg = 180 / np.pi
    deg2rad = np.pi / 180

    ### MAIN FEATURE ORIENTATION

    gauss_o = gauss_filter_N(N_o / 6, N_o)
    orientations = np.zeros((num_features), dtype=np.float64)
    for i in range(num_features):
        bins = np.zeros((36), dtype=float) # 360/10 = 36 bins
        x, y = features[i]
        sigma = 2
        window_width = 10 * sigma
        coordinates = patch_coordinates(x, y, N_o, window_width)

        fdgx = derivative_of_gaussian(sigma, 0)
        fdgy = derivative_of_gaussian(sigma, 1)

        for xc in range(N_o):
            for yc in range(N_o):
                xp, yp = coordinates[xc, yc, :]  # resampled coordinates of patch
                w = gauss_o[xc, yc]

                dx = convolve_at(image, fdgx, xp, yp)
                dy = convolve_at(image, fdgy, xp, yp)

                amplitude = np.sqrt(dx * dx + dy * dy)
                orientation = np.arctan2(dy, dx) * rad2deg
                if orientation < 0:
                    orientation += 360

                bin_index = int(orientation // 36)
                bins[bin_index] += w * amplitude
        orientations[i] = (np.argmax(bins) * 36.0 + 5) * deg2rad               # TO ASK: why 36.0 + 5

    ### FEATURE DESCRIPTORS

    N_d = 16
    gauss_d = gauss_filter_N(N_d / 2, N_d)

    descriptors = np.zeros((num_features, 128), dtype=np.float64)
    for i in range(num_features):
        bins_d = np.zeros((4, 4, 8))
        x, y = features[i]
        sigma = 2
        # print(sigma)
        window_width = max(16, 8 * sigma)  # TO ASK: why (6 also works well)
        coordinates = patch_coordinates_rotated(x, y, N_d, window_width, orientations[i])
        fdgx = derivative_of_gaussian(sigma, 0)
        fdgy = derivative_of_gaussian(sigma, 1)

        for xc in range(N_d):
            for yc in range(N_d):
                xp, yp = coordinates[xc, yc, :]  # resampled coordinates of patch
                w = gauss_d[xc, yc]

                dx = convolve_at(image, fdgx, xp, yp)
                dy = convolve_at(image, fdgy, xp, yp)

                amplitude = np.sqrt(dx * dx + dy * dy)
                orientation = (np.arctan2(dy, dx) - orientations[i]) * rad2deg
                if orientation < 0:
                    orientation += 360

                bin_index = int(orientation // 45)
                bins_d[xc // 4, yc // 4, bin_index] += amplitude  # * w

        for x_ in range(4):
            for y_ in range(4):
                bins_d[x_, y_, :] /= np.sum(bins_d[x_, y_, :])

        descriptors[i, :] = bins_d.reshape((128))

    return descriptors, orientations
	
