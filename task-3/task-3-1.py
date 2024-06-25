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
    matplotlib.use("module://matplotlib-backend-kitty")
except:
    pass


def task_3_1():
    """
    This function applys the edge detection to an image in three different sizes.
    """
    # Read an image from an URL.
    url = "https://assets.serlo.org/legacy/1538.png"
    img_orig = np.array(Image.open(requests.get(url, stream=True).raw))
    img_half_size = cv2.resize(
        img_orig, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR
    )
    img_double_size = cv2.resize(
        img_orig, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR
    )
    imgs_orig = apply_edge_detection(img_orig, "original image (original size)")
    imgs_50 = apply_edge_detection(img_half_size, "original image (50% size)")
    imgs_200 = apply_edge_detection(img_double_size, "original image (200% size)")
    compare_on_different_sizes([imgs_orig, imgs_50, imgs_200])


def apply_edge_detection(img, img_description):
    """
    This function applys different edge detection algorithms to the given image.
    It returns the images with their respective descriptions for later use and comparison.
    """
    img_smoothed = cv2.filter2D(img, -1, get_binomial_kernel(3))
    # Laplace.
    img_laplace = cv2.filter2D(img_smoothed, -1, get_laplace_edge_detector_kernel())
    # Sobel.
    (kernel_sobel_x, kernel_sobel_y) = get_sobel_edge_detector_kernel()
    img_sobel_x = cv2.filter2D(img_smoothed, -1, kernel_sobel_x)
    img_sobel_y = cv2.filter2D(img_smoothed, -1, kernel_sobel_y)
    img_sobel = np.sqrt(
        np.square(img_sobel_x.astype(np.float32))
        + np.square(img_sobel_y.astype(np.float32))
    ).astype(np.uint8)
    # LoG.
    img_log_filter = cv2.filter2D(img_smoothed, -1, get_log_kernel())
    # DoG.
    img_dog_filter = cv2.filter2D(img_smoothed, -1, get_dog_kernel())
    # What images do we want to show?
    # img, description
    img_plot_informations = [
        (img, img_description),
        (img_smoothed, "original image smoothed using a binomial kernel"),
        (img_laplace, "Laplace"),
        (img_sobel_x, "Sobel horizontal"),
        (img_sobel_y, "Sobel vertical"),
        (img_sobel, "Sobel magnitude"),
        (img_log_filter, "LoG filter"),
        (img_dog_filter, "DoG filter"),
    ]
    fig, axs = plt.subplots(math.ceil(len(img_plot_informations) / 2), 2)
    axs = axs.flatten()
    for idx, (img, description) in enumerate(img_plot_informations):
        axs[idx].imshow(img)
        axs[idx].axis("off")
        axs[idx].set_title(description)
    if len(img_plot_informations) % 2 == 1:
        fig.delaxes(axs[-1])
    plt.show()
    return img_plot_informations


def compare_on_different_sizes(imgs):
    """
    This function plots the images of different sizes analyzed with the same edge detection
    algorithm next to eachother to make it easier to compare.
    """
    _, axs = plt.subplots(math.ceil(len(imgs[0])), len(imgs))
    for col, img_plot_informations in enumerate(imgs):
        for row, (img, description) in enumerate(img_plot_informations):
            axs[row][col].imshow(img)
            axs[row][col].axis("off")
            axs[row][col].set_title(description)
    plt.show()


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


def get_laplace_edge_detector_kernel():
    """
    This function returns the kernel for the laplace edge detection.
    """
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    return kernel


def get_sobel_edge_detector_kernel():
    """
    This function returns two kernels, the horizontal and the vertical sobel operator.
    """
    kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    kernel_y = np.transpose(kernel_x)
    return (kernel_x, kernel_y)


def get_log_kernel():
    """
    This function returns the kernel for the LoG filter.
    """
    kernel_typed = (
        np.array(
            [
                [0, 1, 2, 1, 0],
                [1, 0, -2, 0, 1],
                [2, -2, -8, -2, 2],
                [1, 0, -2, 0, 1],
                [0, 1, 2, 1, 0],
            ]
        )
        / 16
    )
    # https://stackoverflow.com/questions/58477255/how-to-do-convolution-in-opencv
    kernel_calculated = (
        convolve2d(
            np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
            np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]),
        )
        / 16
    )
    # Check if it is correct.
    assert np.array_equal(kernel_typed, kernel_calculated)
    return kernel_calculated


def get_dog_kernel():
    """
    This function returns the kernel for the DoG filter.
    """
    kernel_typed = np.array(
        [
            [1, 4, 6, 4, 1],
            [4, 0, -8, 0, 4],
            [6, -8, -28, -8, 6],
            [4, 0, -8, 0, 4],
            [1, 4, 6, 4, 1],
        ]
    )
    # Calculate it by yourself.
    kernel_binomial = get_binomial_kernel(3)
    kernel_i = np.zeros((3, 3))
    kernel_i[1][1] = 1
    kernel_calculated = convolve2d(4 * (kernel_binomial - kernel_i), kernel_binomial)
    # Check if it is correct.
    # assert np.array_equal(kernel_typed, kernel_calculated)
    return kernel_calculated


if __name__ == "__main__":
    task_3_1()
