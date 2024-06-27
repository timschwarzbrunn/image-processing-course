import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image
import requests

from sift.sift_params import SIFT_Params
from sift.sift_algo import SIFT_Algorithm
from sift.sift_keypoint import SIFT_KeyPoint
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

    # Get the reference results from the given functions in sift_algo.
    (scale_space_ref, dogs_ref, deltas_ref, sigmas_ref, keypoints_ref) = (
        get_reference_results(img_grayscale, sift_params)
    )
    # Visualize the reference values.
    visualize_results(
        scale_space_ref, dogs_ref, deltas_ref, keypoints_ref, "reference algorithm"
    )

    # Get the results from our own function.
    scale_space, deltas, sigmas = create_scale_space(img_grayscale, sift_params)
    dogs = create_dogs(scale_space)
    keypoints = find_discrete_extremas(dogs, sift_params, sigmas, deltas)
    # Visualize our own results.
    visualize_results(scale_space, dogs, deltas, keypoints, "own algorithm")

    # Plot the delta and sigma values next to each other.
    for idx_octave in range(len(deltas)):
        print(
            f"Delta of octave {idx_octave}: {deltas[idx_octave]} (own algorithm) ; {deltas_ref[idx_octave]} (referene)"
        )
    for idx_octave in range(len(sigmas)):
        print(
            f"Sigmas of octave {idx_octave}:\n\t{sigmas[idx_octave]} (own algorithm)\n\t{sigmas_ref[idx_octave]} (referene)"
        )
    print(
        f"Number of found keypoints: {len(keypoints)} (own algorithm) ; {len(keypoints_ref)} (reference)"
    )

    # Simply check if the values are correct by assertion.
    assert deltas == deltas_ref
    assert sigmas == sigmas_ref
    assert len(keypoints) == len(keypoints_ref)


def create_scale_space(img, sift_params: SIFT_Params):
    """
    This function calculates the scale space and the corresponding delta and sigma values.
    It takes in the image and the sift parameters and returns the tuple of scale_space, deltas and sigmas.

    Octave:
    An octave represents a series of images where the resolution is reduced by a factor of 2 from the previous octave.
    Each octave starts with the image at a specific resolution and progressively blurs it.

    Interval:
    Within each octave, the image is blurred with Gaussian kernels of different sigma values.
    The number of intervals determines the number of intermediate blurs between octaves. Typically, SIFT uses 3 intervals per octave.

    Sigmas:
    Sigma is the standard deviation of the Gaussian kernel used for blurring.
    SIFT starts with an initial sigma value and increases it for each interval.

    Deltas:
    Delta in SIFT context is related to the sigma values used for blurring at each interval within an octave.

    The scale space therefore consists of multiple octaves that on the other hand consist of multiple images of the
    same size but different blurring. The images of the next octave are half the size of the current octave.
    """
    # The default sampling distance delta_min is 0.5, so we double the size of the image here initially.
    img = cv2.resize(
        img,
        (0, 0),
        fx=1.0 / sift_params.delta_min,
        fy=1.0 / sift_params.delta_min,
        interpolation=cv2.INTER_LINEAR,
    )
    # We assume that the image is already blurred a little bit. So we will not apply the full blur to the image
    # at the start but only the difference between the desired blur and the currently assumed blur.
    sigma_diff = (
        np.sqrt(sift_params.sigma_min**2 - sift_params.sigma_in**2)
        / sift_params.delta_min
    )
    img = cv2.GaussianBlur(img, (0, 0), sigma_diff)
    # Create the scale space.
    scale_space = []
    deltas = []
    sigmas = []
    for idx_octave in range(sift_params.n_octaves):
        # The images of the current octave. We either start the the given image or with one from the previous octave
        # at half size.
        imgs_of_current_octave = [
            img
            if idx_octave == 0
            else cv2.resize(
                scale_space[-1][sift_params.n_scales_per_octave - 1],
                (0, 0),
                fx=0.5,
                fy=0.5,
                interpolation=cv2.INTER_LINEAR,
            )
        ]
        # What delta to use for this octave?
        # We double the delta each time but
        deltas.append(sift_params.delta_min if idx_octave == 0 else deltas[-1] * 2)
        # Calculate the sigmas for this octave.
        # It depends on the number of intervals / scales per octave. But since we will create the dogs later on,
        # we will need to create a few more images and therefore need also a few more sigmas.
        sigmas_of_current_octave = [
            sift_params.sigma_min
            if idx_octave == 0
            else sigmas[idx_octave - 1][sift_params.n_scales_per_octave - 1]
        ]
        for idx_interval in range(1, sift_params.n_scales_per_octave + 3):
            sigmas_of_current_octave.append(
                (deltas[-1] / sift_params.delta_min)
                * sift_params.sigma_min
                * (2 ** (idx_interval / sift_params.n_scales_per_octave))
            )
            if idx_interval > 0:
                imgs_of_current_octave.append(
                    cv2.GaussianBlur(
                        imgs_of_current_octave[-1],
                        (0, 0),
                        np.sqrt(
                            sigmas_of_current_octave[-1] ** 2
                            - sigmas_of_current_octave[-2] ** 2
                        )
                        / deltas[-1],
                    )
                )
        sigmas.append(sigmas_of_current_octave)
        scale_space.append(imgs_of_current_octave)
    return scale_space, deltas, sigmas


def create_dogs(scale_space):
    """
    This function calculates the DoG images of the scale space that was calculated using the 'create_scale_space' function.
    It takes in the scale_space and sift parameters and returns the DoGs as a list of images.
    """
    # For each octave calculate the difference of two consecutive images. This is the difference of gaussian since
    # all the images are differently blurred.
    dogs = []
    for idx_octave, imgs_of_current_octave in enumerate(scale_space):
        print(f"Calculate dogs of octave {idx_octave} (own algorithm)...", end=" ")
        dogs_of_current_octave = []
        for idx_interval in range(len(imgs_of_current_octave) - 1):
            dogs_of_current_octave.append(
                cv2.subtract(
                    imgs_of_current_octave[idx_interval + 1],
                    imgs_of_current_octave[idx_interval],
                )
            )
        dogs.append(dogs_of_current_octave)
        print("Done.")
    return dogs


def find_discrete_extremas(dogs, sift_params: SIFT_Params, sigmas, deltas):
    """
    This function finds the local extremas in the DoG images calculated using the 'create_dogs' function.
    It also needs the deltas and sigmas calculated by the 'create_scale_space' function.
    It takes in the dogs, sift parameters, sigmas and deltas and returns a list of keypoints.
    """
    keypoints = []
    for idx_octave, dogs_of_current_octave in enumerate(dogs):
        print(f"Calculate keypoints of octave {idx_octave} (own algorithm)...", end=" ")
        # Go through all intervals and take the previous, current and next interval to search for local extremas.
        for idx_interval in range(1, len(dogs_of_current_octave) - 1):
            dog_prev_interval = dogs_of_current_octave[idx_interval - 1]
            dog_current_interval = dogs_of_current_octave[idx_interval]
            dog_next_interval = dogs_of_current_octave[idx_interval + 1]
            # Check for local extremas.
            for x_coord in range(1, dog_current_interval.shape[1] - 1):
                for y_coord in range(1, dog_current_interval.shape[0] - 1):
                    window_dog_prev_interval = dog_prev_interval[
                        y_coord - 1 : y_coord + 2, x_coord - 1 : x_coord + 2
                    ]
                    window_dog_current_interval = dog_current_interval[
                        y_coord - 1 : y_coord + 2, x_coord - 1 : x_coord + 2
                    ].copy()  # copy this one because we want to change a value.
                    window_dog_next_interval = dog_next_interval[
                        y_coord - 1 : y_coord + 2, x_coord - 1 : x_coord + 2
                    ]
                    # The value that we want to compare to all of its neighbors is in the middle of our cube.
                    center_value = window_dog_current_interval[1][1]
                    # To not compare against it, set it to zero.
                    window_dog_current_interval[1][1] = 0.0
                    # Is it a keypoint? If so it should be local extrema.
                    if (
                        center_value < 0
                        and np.all(center_value < window_dog_prev_interval)
                        and np.all(center_value < window_dog_current_interval)
                        and np.all(center_value < window_dog_next_interval)
                    ) or (
                        center_value > 0
                        and np.all(center_value > window_dog_prev_interval)
                        and np.all(center_value > window_dog_current_interval)
                        and np.all(center_value > window_dog_next_interval)
                    ):
                        # It is a keypoint.
                        keypoints.append(
                            SIFT_KeyPoint(
                                idx_octave,
                                idx_interval,
                                x_coord,
                                y_coord,
                                sigmas[idx_octave][idx_interval],
                                deltas[idx_octave] * x_coord,
                                deltas[idx_octave] * y_coord,
                            )
                        )
        print("Done.")
    return keypoints


def get_reference_results(img, sift_params):
    """
    This function uses the given functions from the sift_algo class to create results that are known to be correct.
    These can be used for comparison.
    It returns (scale_space, dogs, deltas, sigmas, keypoints)
    """
    scale_space, deltas, sigmas = SIFT_Algorithm.create_scale_space(img, sift_params)
    dogs = SIFT_Algorithm.create_dogs(scale_space, sift_params)
    keypoints = SIFT_Algorithm.find_discrete_extremas(dogs, sift_params, sigmas, deltas)
    return (scale_space, dogs, deltas, sigmas, keypoints)


def visualize_results(scale_space, dogs, deltas, keypoints, used_algorithm):
    """ """
    visualize_scale_space(
        scale_space, f"scale space - used algorithm: {used_algorithm}"
    )
    visualize_scale_space(dogs, f"DoG's - used algorithm: {used_algorithm}")
    visualize_keypoints(
        scale_space,
        keypoints,
        deltas,
        f"keypoints - used algorithm: {used_algorithm}",
        use_keypoint_coordinates=False,
        show_orientations=False,
        show_descriptors=False,
    )


if __name__ == "__main__":
    task_4_1()
