import matplotlib.pyplot as plt
from numpy.typing import NDArray
import numpy as np
from sift_keypoint import SIFT_KeyPoint


def visualize_scale_space(array: list[list[NDArray[np.float32]]], title: str):
    px = 1 / plt.rcParams["figure.dpi"]  # pixel in inches
    plt.figure(num=title, figsize=(1200 * px, 800 * px))
    for o, octave in enumerate(array):
        for s, scale in enumerate(octave):
            plt.subplot(len(array), len(array[0]), o * len(array[0]) + s + 1)
            plt.title(f"O {o} S {s}")
            plt.imshow(scale, cmap="gray")
    plt.show()


def visualize_keypoints(
    scale_space: list[list[NDArray[np.float32]]],
    keypoints: list[SIFT_KeyPoint],
    deltas: list[float],
    title: str,
    use_keypoint_coordinates: bool = False,
    show_orientations: bool = False,
    show_descriptors: bool = False,
):
    px = 1 / plt.rcParams["figure.dpi"]  # pixel in inches
    plt.figure(num=title, figsize=(1200 * px, 800 * px))

    for idx_octave, octave in enumerate(scale_space):
        for idx_scale, scale in enumerate(octave):
            plt.subplot(
                len(scale_space),
                len(scale_space[0]),
                idx_octave * len(scale_space[0]) + idx_scale + 1,
            )
            plt.title(f"O {idx_octave} S {idx_scale}")
            plt.imshow(scale, cmap="gray")
            keypoint_x = []
            keypoint_y = []
            quiver_x = []
            quiver_y = []
            descriptors = []
            if use_keypoint_coordinates:
                for keypoint in keypoints:
                    if (
                        keypoint.octave == idx_octave
                        and keypoint.scale_level == idx_scale
                    ):
                        keypoint_x.append(
                            round(
                                keypoint.x_coord_in_scale_space
                                / deltas[keypoint.octave]
                            )
                        )
                        keypoint_y.append(
                            round(
                                keypoint.y_coord_in_scale_space
                                / deltas[keypoint.octave]
                            )
                        )
                        quiver_x.append(np.cos(keypoint.theta))  # *keypoint.magnitude)
                        quiver_y.append(np.sin(keypoint.theta))  # *keypoint.magnitude)
                        descriptors.append(keypoint.descriptor)
            else:
                keypoint_x = [
                    keypoint.x_coord
                    for keypoint in keypoints
                    if keypoint.octave == idx_octave
                    and keypoint.scale_level == idx_scale
                ]
                keypoint_y = [
                    keypoint.y_coord
                    for keypoint in keypoints
                    if keypoint.octave == idx_octave
                    and keypoint.scale_level == idx_scale
                ]
                quiver_x = [
                    np.cos(keypoint.theta) * keypoint.magnitude
                    for keypoint in keypoints
                    if keypoint.octave == idx_octave
                    and keypoint.scale_level == idx_scale
                ]
                quiver_y = [
                    np.sin(keypoint.theta) * keypoint.magnitude
                    for keypoint in keypoints
                    if keypoint.octave == idx_octave
                    and keypoint.scale_level == idx_scale
                ]
                descriptors = [
                    keypoint.descriptor
                    for keypoint in keypoints
                    if keypoint.octave == idx_octave
                    and keypoint.scale_level == idx_scale
                ]
            plt.scatter(keypoint_x, keypoint_y, c="r", s=1)
            if show_orientations:
                plt.quiver(keypoint_x, keypoint_y, quiver_x, quiver_y)
            if show_descriptors:
                print(f"Octave: {idx_octave}, extremum: {idx_scale}")
                print(np.matrix(descriptors))
    plt.show()
