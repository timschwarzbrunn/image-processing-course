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


def task_3_3():
    # Read an image from an URL.
    url = "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2F3.bp.blogspot.com%2F-MulwSvCmKmU%2FWvyOwge3PzI%2FAAAAAAAAEes%2FUqp41AKCXlYAdiveNkrDOh0V2ERHdZ5zACLcBGAs%2Fs1600%2Ftree.jpg&f=1&nofb=1&ipt=fba54ef924ef0ce9d7005dbf5461d95bc8987a5929631e91b39c1addead5de95&ipo=images"
    img_orig = np.array(Image.open(requests.get(url, stream=True).raw))
    img_smoothed = cv2.filter2D(img_orig, -1, get_binomial_kernel(3))
    # Create a grayscale image.
    img_grayscale = cv2.cvtColor(img_smoothed, cv2.COLOR_RGB2GRAY)
    # We anticipate the background to be white / lighter than the object.
    # So the object in the binary image should contain False values / 0.0 and the
    # background True values / 1.0 values (white).
    threshold = 220
    img_binary = cv2.threshold(img_grayscale, threshold, 255, cv2.THRESH_BINARY)[1]
    # Get the skeleton.
    kernel_v1 = np.array([[-1, 0, 1], [-1, 1, 1], [-1, 0, 1]])
    kernel_45deg_v1 = np.array([[-1, -1, 0], [-1, 1, 1], [0, 1, 1]])
    kernel_v2 = np.array([[-1, 0, 1], [-1, 1, 0], [-1, 0, 1]])
    kernel_45deg_v2 = np.array([[-1, -1, 0], [-1, 1, 1], [0, 1, 0]])
    img_skel_v1 = get_skeleton_of_image(img_binary, kernel_v1, kernel_45deg_v1)
    img_skel_v2 = get_skeleton_of_image(img_binary, kernel_v2, kernel_45deg_v2)
    # Plot the results.
    img_plot_informations = [
        (img_orig, "original image"),
        (img_smoothed, "original image smoothed using a binomial kernel"),
        (img_grayscale, "grayscale image"),
        (img_binary, "binary image"),
        (img_skel_v1, "skeleton of binary image (kernel version 1)"),
        (img_skel_v2, "skeleton of binary image (kernel version 2)"),
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
        axs[idx].imshow(img, cmap=cmap)
        axs[idx].axis("off")
        axs[idx].set_title(description)
    if len(img_plot_informations) % 2 == 1:
        fig.delaxes(axs[-1])
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


def get_skeleton_of_image(img, kernel, kernel_45deg):
    """
    This function returns the skeleton of an image.

    https://stackoverflow.com/questions/58790429/skeletonization-with-thinning-and-hit-or-miss-implementation-never-stops

    https://homepages.inf.ed.ac.uk/rbf/HIPR2/thin.htm
    """
    # Invert it so the function works on the False values.
    img = cv2.bitwise_not(img)
    img_skel = img
    while True:
        # Get a mask of all zeros.
        mask = np.zeros(img.shape, np.uint8)
        # The two basic kernels that will be rotated afterwards to obtain
        # eight kernels in total.
        for _ in range(4):
            kernel = np.rot90(kernel)
            kernel_45deg = np.rot90(kernel_45deg)
            img_morph = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel)
            mask = cv2.bitwise_or(img_morph, mask)
            img_morph = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel_45deg)
            mask = cv2.bitwise_or(img_morph, mask)
        img = img - mask
        if np.array_equal(img, img_skel):
            break
        img_skel = img
    # Invert it again so the skeleton is black.
    return cv2.bitwise_not(img)


if __name__ == "__main__":
    task_3_3()
