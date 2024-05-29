import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image
import requests

# https://github.com/jktr/matplotlib-backend-kitty
try:
    matplotlib.use("module://matplotlib-backend-kitty")
except:
    pass


def task_2_3():
    # Read an image from an URL.
    url = "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fimages.wallpapersden.com%2Fimage%2Fdownload%2F4k-starry-sky-stars-milky-way-galaxy_bGttbWeUmZqaraWkpJRqZmetamZn.jpg&f=1&nofb=1&ipt=326b18ddc7ff8b32ebd4ef7918a46a818be42e96b3df8ea0b33a8608fa25e0cb&ipo=images"
    img = np.array(Image.open(requests.get(url, stream=True).raw))
    # Create kernels of size 3, 7 and 15.
    kernel_sizes = [3, 7, 15]
    gaussian_kernel_sigma = 1
    for kernel_size in kernel_sizes:
        kernel_gaussian = generate_gaussian_kernel(kernel_size, gaussian_kernel_sigma)
        kernel_binomial = generate_binomial_kernel(kernel_size)
        img_gaussian = cv2.filter2D(img, -1, kernel_gaussian)
        img_binomial = cv2.filter2D(img, -1, kernel_binomial)
        fig, (ax_orig, ax_gaussian, ax_binomial) = plt.subplots(1, 3)
        ax_orig.imshow(img)
        ax_orig.set_xticks([])
        ax_orig.set_yticks([])
        ax_orig.set_title("original image")
        ax_gaussian.imshow(img_gaussian)
        ax_gaussian.set_xticks([])
        ax_gaussian.set_yticks([])
        ax_gaussian.set_title(
            "image after applying a gaussian filter of size "
            + str(kernel_size)
            + "x"
            + str(kernel_size)
        )
        ax_binomial.imshow(img_binomial)
        ax_binomial.set_xticks([])
        ax_binomial.set_yticks([])
        ax_binomial.set_title(
            "image after applying a binomial filter of size "
            + str(kernel_size)
            + "x"
            + str(kernel_size)
        )
        plt.show()


def generate_gaussian_kernel(size, sigma):
    # Determine the center of the kernel. We compute it also for even sized kernels,
    # even if we will not use it.
    center = size // 2 if size % 2 == 1 else size // 2 - 0.5
    # Calculate the kernel based on the gaussian function.
    # g(x, y) = 1 / (2 * pi * sigma^2) * e^(-(x^2 + y^2) / (2 * sigma^2))
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma**2))
        * np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma**2)),
        (size, size),
    )
    kernel = kernel / np.sum(kernel)
    return kernel


def generate_binomial_kernel(size):
    # https://stackoverflow.com/questions/15580291/how-to-efficiently-calculate-a-row-in-pascals-triangle#15580400
    kernel_1d = np.ones((size,), dtype=np.float64)
    for k in range(1, size):
        kernel_1d[k] = kernel_1d[k - 1] * (size - k) / k
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    kernel = kernel_2d / np.sum(kernel_2d)
    return kernel


if __name__ == "__main__":
    task_2_3()
    # The greater the kernel, the stronger the blur effect.
