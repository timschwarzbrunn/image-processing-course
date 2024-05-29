import cv2
import numpy as np
import math
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image
import requests

# https://github.com/jktr/matplotlib-backend-kitty
try:
    matplotlib.use("module://matplotlib-backend-kitty")
except:
    pass


def task_3_2(kernel_size):
    # Read an image from an URL.
    url = "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2F3.bp.blogspot.com%2F-MulwSvCmKmU%2FWvyOwge3PzI%2FAAAAAAAAEes%2FUqp41AKCXlYAdiveNkrDOh0V2ERHdZ5zACLcBGAs%2Fs1600%2Ftree.jpg&f=1&nofb=1&ipt=fba54ef924ef0ce9d7005dbf5461d95bc8987a5929631e91b39c1addead5de95&ipo=images"
    img_orig = np.array(Image.open(requests.get(url, stream=True).raw))
    img_smoothed = cv2.filter2D(img_orig, -1, get_binomial_kernel(3))
    # Create a grayscale image.
    weights = [0.2989, 0.5870, 0.1140]
    img_grayscale = np.dot(img_smoothed[..., :3], weights)
    # We anticipate the background to be white / lighter than the object.
    # So the object in the binary image should contain False values / 0.0 and the
    # background True values / 1.0 values (white).
    threshold = 220
    img_binary = 1.0 * (img_grayscale > threshold)
    # Erode the image.
    img_erosion = cv2.erode(
        img_binary, np.ones((kernel_size, kernel_size)), iterations=1
    )
    # Dilate the image.
    img_dilation = cv2.dilate(
        img_binary, np.ones((kernel_size, kernel_size)), iterations=1
    )
    # Opening: erode and then dilate.
    img_opening = cv2.dilate(
        cv2.erode(img_binary, np.ones((kernel_size, kernel_size)), iterations=1),
        np.ones((kernel_size, kernel_size)),
        iterations=1,
    )
    # Closing: dilate and then erode.
    img_closing = cv2.erode(
        cv2.dilate(img_binary, np.ones((kernel_size, kernel_size)), iterations=1),
        np.ones((kernel_size, kernel_size)),
        iterations=1,
    )
    # Plot the results.
    img_plot_informations = [
        (img_orig, "original image"),
        (img_smoothed, "original image smoothed using a binomial kernel"),
        (img_grayscale, "grayscale image"),
        (img_binary, "binary image"),
        (
            img_erosion,
            "binary image after applying an erosion (kernel {}x{})".format(
                kernel_size, kernel_size
            ),
        ),
        (
            img_dilation,
            "binary image after applying an dilation (kernel {}x{})".format(
                kernel_size, kernel_size
            ),
        ),
        (
            img_opening,
            "binary image after applying an opening (kernel {}x{})".format(
                kernel_size, kernel_size
            ),
        ),
        (
            img_closing,
            "binary image after applying an closing (kernel {}x{})".format(
                kernel_size, kernel_size
            ),
        ),
    ]
    fig, axs = plt.subplots(math.ceil(len(img_plot_informations) / 2), 2)
    axs = axs.flatten()
    for idx, (img, description) in enumerate(img_plot_informations):
        # The first two are rgb images, the next one is a grayscale image,
        # the rest is binary.
        cmap = (
            plt.get_cmap("viridis")
            if idx < 2
            else plt.get_cmap("gray")
            if idx == 2
            else plt.get_cmap("binary_r")
        )
        axs[idx].imshow(img, interpolation="nearest", cmap=cmap)
        axs[idx].axis("off")
        axs[idx].set_title(description)
    if len(img_plot_informations) % 2 == 1:
        fig.delaxes(axs[-1])
    plt.show()
    return img_plot_informations


def compare_on_different_sizes(imgs):
    _, axs = plt.subplots(math.ceil(len(imgs[0])), len(imgs))
    for col, img_plot_informations in enumerate(imgs):
        for row, (img, description) in enumerate(img_plot_informations):
            cmap = (
                plt.get_cmap("viridis")
                if row < 2
                else plt.get_cmap("gray")
                if row == 2
                else plt.get_cmap("binary_r")
            )
            axs[row][col].imshow(img, interpolation="nearest", cmap=cmap)
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


if __name__ == "__main__":
    imgs_3x3 = task_3_2(3)
    imgs_5x5 = task_3_2(5)
    compare_on_different_sizes([imgs_3x3, imgs_5x5])
