import cv2
import numpy as np
import math
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image
import requests
from scipy.signal import convolve2d

# https://github.com/jktr/matplotlib-backend-kitty
try:
    matplotlib.use("module://matplotlib-backend-kitty2")
except:
    pass


def task_3_4():
    """
    This function applys the edge detection to an image in three different sizes.
    """
    # Read an image from an URL.
    url = "https://external-content.duckduckgo.com/iu/?u=http%3A%2F%2Fupload.wikimedia.org%2Fwikipedia%2Fcommons%2F4%2F44%2FWood_fence.jpg&f=1&nofb=1&ipt=7074d53ba8040b9b73789433353f6f578e483804ca0d6dbc03d93a87b851590b&ipo=images"
    img_orig = np.array(Image.open(requests.get(url, stream=True).raw))
    number_of_levels = 5
    # Construct gaussian pyramid.
    pyramid_levels = [img_orig]
    for _ in range(number_of_levels - 1):
        pyramid_levels.append(gaussian_pyramid_get_next_level(pyramid_levels[-1]))
    # Reconstruct based on gaussian pyramid.
    pyramid_levels_reconstructed = [pyramid_levels[-1]]
    for _ in range(number_of_levels - 1):
        pyramid_levels_reconstructed.append(
            gaussian_pyramid_get_prev_level(pyramid_levels_reconstructed[-1])
        )
    pyramid_levels_reconstructed.reverse()
    # Construct laplace pyramid.
    laplace_pyramid = []
    for idx in range(number_of_levels - 1):
        laplace_pyramid.append(
            laplacian_pyramid_get_level(pyramid_levels[idx], pyramid_levels[idx + 1])
        )
    # Reconstruct using the laplace pyramid.
    pyramid_levels_reconstructed_laplace = pyramid_levels_reconstructed
    for idx in range(len(laplace_pyramid)):
        print(
            pyramid_levels_reconstructed_laplace[idx].shape, laplace_pyramid[idx].shape
        )
        pyramid_levels_reconstructed_laplace[idx] = cv2.add(
            pyramid_levels_reconstructed_laplace[idx], laplace_pyramid[idx]
        )
    # Plot.
    fig, axs = plt.subplots(4, len(pyramid_levels))
    for idx, img in enumerate(pyramid_levels):
        axs[0][idx].imshow(img)
        axs[0][idx].axis("off")
        axs[0][idx].set_title(f"Level {idx}")
    for idx, img in enumerate(pyramid_levels_reconstructed):
        axs[1][idx].imshow(img)
        axs[1][idx].axis("off")
        axs[1][idx].set_title(f"Level {idx} reconstructed")
    for idx, img in enumerate(laplace_pyramid):
        axs[2][idx].imshow(img)
        axs[2][idx].axis("off")
        axs[2][idx].set_title(f"Laplace level {idx}")
    fig.delaxes(axs[2][-1])
    for idx, img in enumerate(pyramid_levels_reconstructed_laplace):
        axs[3][idx].imshow(img)
        axs[3][idx].axis("off")
        axs[3][idx].set_title(f"Laplace level {idx} reconstructed")
    plt.show()


def gaussian_pyramid_get_next_level(img):
    """
    This function calculates the next level of the gaussian pyramid. It first smoothes the image using
    a binomial kernel and the downsamples it to half the size.
    The function returns the new level.
    """
    img_smoothed = cv2.filter2D(img, -1, get_binomial_kernel(3))
    img_next_level = cv2.resize(
        img_smoothed, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR
    )
    return img_next_level


def gaussian_pyramid_get_prev_level(img):
    """
    This function calculates the previous level of the gaussian pyramid. It upsamples it.
    The function returns the new level.
    """
    img_prev_level = cv2.resize(
        img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR
    )
    return img_prev_level


def laplacian_pyramid_get_level(img_current_level, img_next_level):
    img_next_level_upsampled = cv2.resize(
        img_next_level, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR
    )
    diff = cv2.subtract(img_current_level, img_next_level_upsampled)
    return diff


def get_binomial_kernel(size):
    """
    This function returns a binomial kernel for blurring the image / reducing the noise.
    """
    # https://stackoverflow.com/questions/15580291/how-to-efficiently-calculate-a-row-in-pascals-triangle#15580400
    kernel_1d = np.ones((size,), dtype=np.float64)
    for k in range(1, size):
        kernel_1d[k] = kernel_1d[k - 1] * (size - k) / k
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    return kernel_2d / np.sum(kernel_2d)


if __name__ == "__main__":
    task_3_4()
