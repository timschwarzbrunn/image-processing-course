import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image
import requests

from sift.sift_params import SIFT_Params
from sift.sift_algo import SIFT_Algorithm
from sift.sift_visualization import visualize_scale_space, visualize_keypoints

# https://github.com/jktr/matplotlib-backend-kitty
try:
    matplotlib.use("module://matplotlib-backend-kitty")
except:
    pass


def task_4_1():
    """
    This function loads an image, resizes it to a reasonable size, converts it into a grayscale image,
    and applies the SIFT algorithm to it.
    """
    # Read an image from an URL.
    url = "https://skyhookcontentful.imgix.net/6MPvB1nbHtL2AQbxMi2D7y/af0829fe9fc4733a754e15705d99d33d/pixabay-pehrlich-himalayas.jpg?auto=compress%2Cformat%2Cenhance%2Credeye&crop=faces%2Ccenter&fit=crop&ar=1%3A1&ixlib=react-9.7.0"
    img_orig = np.array(Image.open(requests.get(url, stream=True).raw))
    # Resize it to a square image of a size of a power of 2. Recommended is 128x128 for run-time-reasons.
    img_resized = cv2.resize(img_orig, (128, 128))
    # Convert it to a grayscale image with values in the range [0 ; 1].
    img_grayscale = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    print(
        f"Values in grayscale image range from {img_grayscale.min()} to {img_grayscale.max()} before conversion."
    )
    img_grayscale = img_grayscale.astype(np.float32) / 255.0
    print(
        f"Values in grayscale image range from {img_grayscale.min()} to {img_grayscale.max()} after conversion."
    )
    # Show these three images.
    _, axs = plt.subplots(1, 3)
    axs[0].imshow(img_orig)
    axs[0].axis("off")
    axs[0].set_title(f"original image of size {img_orig.shape}")
    axs[1].imshow(img_resized)
    axs[1].axis("off")
    axs[1].set_title(f"resized image of size {img_resized.shape}")
    axs[2].imshow(img_grayscale, cmap="gray")
    axs[2].axis("off")
    axs[2].set_title(f"grayscale image of size {img_grayscale.shape}")
    plt.show()

    # Get the parameters. Just use the default ones from the constructor.
    sift_params = SIFT_Params()
    print("Parameters for the SIFT algorithm:")
    print(str(sift_params))

    # Create the scale space.
    scale_space_comparison, deltas_comparison, sigmas_comparison = (
        SIFT_Algorithm.create_scale_space(img_grayscale, sift_params)
    )
    visualize_scale_space(
        scale_space_comparison, "Scale Space - used algorithm: comparison algorithm"
    )

    # Calculate the DoG images.
    dogs_comparison = SIFT_Algorithm.create_dogs(scale_space_comparison, sift_params)
    visualize_scale_space(
        dogs_comparison, "DoG's - used algorithm: comparison algorithm"
    )

    # Find the local extremas in the DoG images.
    keypoints_comparison = SIFT_Algorithm.find_discrete_extremas(
        dogs_comparison, sift_params, sigmas_comparison, deltas_comparison
    )
    visualize_keypoints(
        scale_space_comparison,
        keypoints_comparison,
        deltas_comparison,
        "Keypoints - used algorithm: comparison algorithm",
        use_keypoint_coordinates=False,
        show_orientations=False,
        show_descriptors=False,
    )


def create_scale_space():
    """"""
    pass


def create_dogs():
    """"""
    pass


def find_discrete_extremas():
    """"""
    pass


if __name__ == "__main__":
    task_4_1()
