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


def task_2_2():
    # Read an image from an URL.
    url = "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fimages.wallpapersden.com%2Fimage%2Fdownload%2F4k-starry-sky-stars-milky-way-galaxy_bGttbWeUmZqaraWkpJRqZmetamZn.jpg&f=1&nofb=1&ipt=326b18ddc7ff8b32ebd4ef7918a46a818be42e96b3df8ea0b33a8608fa25e0cb&ipo=images"
    img = np.array(Image.open(requests.get(url, stream=True).raw))
    # Generate a 5x5 box kernel and manipulate the center value.
    kernel_size = 5
    center_values = [1, 5, 10, 20, 100, 500]
    for center_value in center_values:
        kernel = generate_kernel(kernel_size, center_value)
        img_filtered = cv2.filter2D(img, -1, kernel)
        plt.imshow(img_filtered)
        plt.xticks([])
        plt.yticks([])
        plt.title(
            "Image after applying box filter of size "
            + str(kernel_size)
            + "x"
            + str(kernel_size)
            + " and an center value of "
            + str(center_value)
            + " (before normalization)"
        )
        plt.show()
    # Question: What is noticeable with values of increasing size?
    # The greater the value in the center of the kernel, the less the neighbors
    # are taken into account. If the value is a lot greater than the values in the kernel
    # for the neighboring pixels, it seems that no kernel at all was applied.
    # (no blur effect is noticable when the value at the center of the kernel is much
    # greater than the others)


def generate_kernel(size, center_value):
    kernel = np.ones((size, size), dtype=np.float32)
    # Set the center value.
    kernel[size // 2, size // 2] = center_value
    # Normalize the kernel.
    kernel = kernel / np.sum(kernel)
    return kernel


if __name__ == "__main__":
    task_2_2()
