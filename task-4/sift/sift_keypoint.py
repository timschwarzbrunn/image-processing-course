from numpy.typing import NDArray
import numpy as np


class SIFT_KeyPoint:
    """
    Represents Extrema / Keypoint.
    """

    def __init__(
        self,
        octave: int,
        scale_level: int,
        x_coord: int,
        y_coord: int,
        sigma: float = 0.0,
        x_coord_in_scale_space: int = 0,
        y_coord_in_scale_space: int = 0,
        omega: float = 0.0,
        theta: float = 0.0,
        magnitude: float = 0.0,
        descriptor: NDArray[np.float32] = np.array([]),
    ):
        """
        Represents Extrema / Keypoint.
        Args:
            octave (int): The octave - 0-based
            scale_level (int): The scale level from Dog image - 0-based
            x_coord (int): The x - coordinate in Dog image - 0-based
            y_coord (int): The y - coordinate in Dog image - 0-based
            sigma (float, optional): The scale level in scale space - 0-based. Defaults to 0.0.
            x_coord_in_scale_space (int, optional): The x - coordinate in scale space - 0-based. Defaults to 0.
            y_coord_in_scale_space (int, optional): the y - coordinate in scale space - 0-based. Defaults to 0.
            omega (float, optional): The scale value in scale space. Defaults to 0.0.
            theta (float, optional): The Orientation angle in radians. Defaults to 0.0.
            magnitude (float, optional): The magnitude of the orientation. Defaults to 0.0.
            descriptor (NDArray[np.float32], optional): The descriptor feature vector. Defaults to np.array([]).
        """
        self.octave = octave
        self.scale_level = scale_level
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.sigma = sigma
        self.x_coord_in_scale_space = x_coord_in_scale_space
        self.y_coord_in_scale_space = y_coord_in_scale_space
        self.omega: float = omega
        self.theta = theta
        self.magnitude = magnitude
        self.descriptor = descriptor

    def __str__(self):
        return (
            "[\n"
            + f"\tx_coord: {self.x_coord},\n"
            + f"\ty_coord: {self.y_coord},\n"
            + f"\toctave: {self.octave},\n"
            + f"\tscale level: {self.scale_level},\n"
            + f"\tx coord in scale space: {self.x_coord_in_scale_space},\n"
            + f"\ty coord in scale space: {self.y_coord_in_scale_space},\n"
            + f"\tsigma: {self.sigma},\n"
            + f"\tomega: {self.omega},\n"
            + f"\torientation angle: {self.theta},\n"
            + f"\tmagnitude: {self.magnitude}\n"
            + "]"
        )

    def __repr__(self):
        return (
            "[\n"
            + f"\tx_coord: {self.x_coord},\n"
            + f"\ty_coord: {self.y_coord},\n"
            + f"\toctave: {self.octave},\n"
            + f"\tscale level: {self.scale_level},\n"
            + f"\tx coord in scale space: {self.x_coord_in_scale_space},\n"
            + f"\ty coord in scale space: {self.y_coord_in_scale_space},\n"
            + f"\tsigma: {self.sigma},\n"
            + f"\tomega: {self.omega},\n"
            + f"\torientation angle: {self.theta},\n"
            + f"\tmagnitude: {self.magnitude}\n"
            + "]"
        )
