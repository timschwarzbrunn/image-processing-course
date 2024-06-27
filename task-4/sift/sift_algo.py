import cv2
import numpy as np
import scipy
from sift.sift_keypoint import SIFT_KeyPoint
from sift.sift_params import SIFT_Params
from typing import Tuple


class SIFT_Algorithm:
    """
    Represents the SIFT algorithm.
    """

    # Methods are / were obfuscated by purpose as they are meant to be used only for comparison.
    @staticmethod
    def create_scale_space(u_in, sift_params: SIFT_Params):
        """
        Creates a scale space for a given image.
        Args:
            u_in (np.ndarray): the image
            sift_Params (SIFT_Params): the sift parameters
        Returns:
            Tuple[list[list[np.ndarray]], list[float], list[list[float]]]:
                - the scale space divide into octave - scales - images, the delta values and the sigma values.
        """
        scale_space = []
        deltas = [sift_params.delta_min]
        sigmas = [[sift_params.sigma_min]]
        R = cv2.resize(
            u_in,
            (0, 0),
            fx=1.0 / sift_params.delta_min,
            fy=1.0 / sift_params.delta_min,
            interpolation=cv2.INTER_LINEAR,
        )
        G = (
            np.sqrt(sift_params.sigma_min**2 - sift_params.sigma_in**2)
            / sift_params.delta_min
        )
        S = cv2.GaussianBlur(R, (0, 0), G)
        M = [S]
        for F in range(1, sift_params.n_scales_per_octave + 3):
            H = sift_params.sigma_min * pow(2.0, F / sift_params.n_scales_per_octave)
            G = np.sqrt(H**2 - sigmas[0][F - 1] ** 2) / sift_params.delta_min
            sigmas[0].append(H)
            M.append(cv2.GaussianBlur(M[F - 1], (0, 0), G))
        scale_space.append(M)
        for idx_octave in range(1, sift_params.n_octaves):
            N = deltas[idx_octave - 1] * 2
            deltas.append(N)
            P = scale_space[idx_octave - 1][sift_params.n_scales_per_octave - 1]
            sigmas.append([sigmas[idx_octave - 1][sift_params.n_scales_per_octave - 1]])
            P = cv2.resize(P, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            Q = [P]
            for F in range(1, sift_params.n_scales_per_octave + 3):
                H = (
                    N
                    / sift_params.delta_min
                    * sift_params.sigma_min
                    * pow(2.0, F / sift_params.n_scales_per_octave)
                )
                G = np.sqrt(H**2 - sigmas[idx_octave][F - 1] ** 2) / N
                sigmas[idx_octave].append(H)
                Q.append(cv2.GaussianBlur(Q[F - 1], (0, 0), G))
            scale_space.append(Q)
        return scale_space, deltas, sigmas

    @staticmethod
    def create_dogs(scale_space, sift_params: SIFT_Params):
        """
        Creates the difference of gaussians for a given scale space.
        Args:
            scale_space (list[list[np.ndarray]]): the scale space
            sift_params (SIFT_Params): the sift parameters
        Returns:
            list[list[np.ndarray]]: the difference of gaussians.
        """
        dogs = []
        for idx_octave in range(0, sift_params.n_octaves):
            C = scale_space[idx_octave]
            F = []
            for G in range(sift_params.n_scales_per_octave + 2):
                F.append(cv2.subtract(C[G + 1], C[G]))
            dogs.append(F)
        return dogs

    @staticmethod
    def find_discrete_extremas(dogs, sift_params: SIFT_Params, sigmas, deltas):
        """
        Finds the discrete extremas for a given difference of gaussians.
        Args:
            dogs (list[list[np.ndarray]]): the difference of gaussians
            sift_params (SIFT_Params): the sift parameters (used for n_oct and n_spo)
        Returns:
            list[KeyPoint]: the discrete extremas.
        """
        keypoints = []
        for idx_octave in range(0, sift_params.n_octaves):
            print(f"Extrema Calculation: Octave {idx_octave}")
            I = dogs[idx_octave]
            for G in range(1, sift_params.n_scales_per_octave + 1):
                H = I[G]
                N = I[G - 1]
                O = I[G + 1]
                for B in range(1, H.shape[0] - 1):
                    for C in range(1, H.shape[1] - 1):
                        E = H[B - 1 : B + 2, C - 1 : C + 2].flatten()
                        E = np.delete(E, 4)
                        E = np.append(E, N[B - 1 : B + 2, C - 1 : C + 2].flatten())
                        E = np.append(E, O[B - 1 : B + 2, C - 1 : C + 2].flatten())
                        if np.max(E) < H[B, C] or np.min(E) > H[B, C]:
                            keypoints.append(
                                SIFT_KeyPoint(
                                    idx_octave,
                                    G,
                                    C,
                                    B,
                                    sigmas[idx_octave][G],
                                    deltas[idx_octave] * C,
                                    deltas[idx_octave] * B,
                                )
                            )
        return keypoints

    # non-minized function from here on
    @staticmethod
    def taylor_expansion(
        extremas: list[SIFT_KeyPoint],
        dog_scales: list[list[np.ndarray]],
        sift_params: SIFT_Params,
        deltas: list[float],
        sigmas: list[list[float]],
    ) -> list[SIFT_KeyPoint]:
        """
        Finetunes locations of extrema using taylor expansion.
        Args:
            extremas (list[KeyPoint]): The extremas to finetune
            dog_scales (list[list[np.ndarray]]): The difference of gaussians images to finetune with
            sift_params (SIFT_Params): The sift parameters
            deltas (list[float]): The deltas for each octave
            sigmas (list[list[float]]): The sigmas for each octave and scale level
        Returns:
            list[KeyPoint]: The new Extremum. Newly created KeyPoint Objects.
        """
        new_extremas: list[SIFT_KeyPoint] = []
        for extremum in extremas:
            # discard low contrast candidate keypoints
            # this step is done separately in the C-Implementation
            # but can be included here for "slightly" better performance
            if (
                abs(
                    dog_scales[extremum.octave][extremum.scale_level][
                        extremum.y_coord, extremum.x_coord
                    ]
                )
                < 0.8 * sift_params.threshold_dog_response
            ):
                continue
            # 0-based index of the current extremum
            scale_level: int = extremum.scale_level
            x_coord: int = extremum.x_coord
            y_coord: int = extremum.y_coord
            # for each adjustment
            # will break if new location is found
            # will be adjusted maximum 5 times
            for _ in range(5):
                # get the current dog image
                # s is initialized from dog images
                # but also represent the scale level
                current: np.ndarray = dog_scales[extremum.octave][scale_level]
                previous: np.ndarray = dog_scales[extremum.octave][scale_level - 1]
                next_scale: np.ndarray = dog_scales[extremum.octave][scale_level + 1]

                # called $\bar{g}^o_{s,m,n}$ in the paper
                # represent the first derivative  in a finite difference scheme
                # is Transposed, as we calculate [a,b,c] values, but want [[a],[b],[c]]
                g_o_smn = np.matrix(
                    [
                        [
                            next_scale[y_coord, x_coord] - previous[y_coord, x_coord],
                            current[y_coord, x_coord + 1]
                            - current[y_coord, x_coord - 1],
                            current[y_coord + 1, x_coord]
                            - current[y_coord - 1, x_coord],
                        ]
                    ]
                ).T

                # calcuation of hessian matrix
                # note that s is the first dimension, then x, then y
                h11: float = (
                    next_scale[y_coord, x_coord]
                    - previous[y_coord, x_coord]
                    - 2 * current[y_coord, x_coord]
                )
                h22: float = (
                    current[y_coord, x_coord + 1]
                    + current[y_coord, x_coord - 1]
                    - 2 * current[y_coord, x_coord]
                )
                h33: float = (
                    current[y_coord + 1, x_coord]
                    - current[y_coord - 1, x_coord]
                    - 2 * current[y_coord, x_coord]
                )

                # h12, h13 and h23 are reused for h21, h31 and h32
                # as they are the same value
                h12: float = (
                    next_scale[y_coord, x_coord, x_coord + 1]
                    - next_scale[y_coord, x_coord - 1]
                    - previous[y_coord, x_coord + 1]
                    - previous[y_coord, x_coord - 1]
                ) / 4
                h13: float = (
                    next_scale[y_coord + 1, x_coord]
                    - next_scale[y_coord - 1, x_coord]
                    - previous[y_coord + 1, x_coord]
                    - previous[y_coord - 1, x_coord]
                ) / 4
                h23: float = (
                    current[y_coord + 1, x_coord + 1]
                    - current[y_coord + 1, x_coord - 1]
                    - current[y_coord - 1, x_coord + 1]
                    - current[y_coord - 1, x_coord - 1]
                ) / 4

                hessian = np.matrix([[h11, h12, h13], [h12, h22, h23], [h13, h23, h33]])
                # inverse of a matrix with det = 0 is not possible
                # therefore we break here
                # this is just safety, should not happen
                if np.linalg.det(hessian) == 0:
                    break
                # calculate offset, this will be of shape (1,3) ([[a],[b],[c]])
                alpha = -np.linalg.inv(hessian) * g_o_smn
                # the every value is below the drop off of 0.6
                # we found the new location
                if np.max(np.abs(alpha)) < 0.6:
                    # this is simplified from 'w+alphaT*g + 0.5*alphaT*H*alpha'
                    # to 'w+0.5*alphaT*g' following the paper
                    # pseudocode in the paper does not simplify here
                    # omega represent the value of the DoG interpolated extremum
                    omega = current[extremum.y_coord, extremum.x_coord] + 0.5 * (
                        g_o_smn[0, 0] * alpha[0, 0]
                        + g_o_smn[1, 0] * alpha[1, 0]
                        + g_o_smn[2, 0] * alpha[2, 0]
                    )
                    # get the current delta and sigma for the corresponding new location
                    delta_oe = deltas[extremum.octave]
                    # sigma is calculated from the scale
                    sigma = sigmas[extremum.octave][scale_level] * pow(
                        sigmas[0][1] - sigmas[0][0], alpha[0, 0]
                    )
                    # and the keypoint coordinates
                    x = delta_oe * (alpha[1, 0] + x_coord)
                    y = delta_oe * (alpha[2, 0] + y_coord)
                    # create new Extremum object with the corresponding values
                    new_extremum = SIFT_KeyPoint(
                        extremum.octave,
                        scale_level,
                        x_coord,
                        y_coord,
                        sigma,
                        x,
                        y,
                        omega,
                    )
                    new_extremas.append(new_extremum)
                    break
                # update coordinates by +1 or -1 if the abs(alpha) is > 0.6
                # but borders are not reached
                # we could optionally exclude extremum that are close to the border
                # but those are also excluded later on aswell
                if (
                    alpha[0, 0] > 0.6
                    and scale_level + 1 < len(dog_scales[extremum.octave]) - 1
                ):
                    scale_level += 1
                elif alpha[0, 0] < -0.6 and scale_level - 1 > 0:
                    scale_level -= 1
                if alpha[1, 0] > 0.6 and x_coord + 1 < current.shape[1] - 1:
                    x_coord += 1
                elif alpha[1, 0] < -0.6 and x_coord - 1 > 0:
                    x_coord -= 1
                if alpha[2, 0] > 0.6 and y_coord + 1 < current.shape[0] - 1:
                    y_coord += 1
                elif alpha[2, 0] < -0.6 and y_coord - 1 > 0:
                    y_coord -= 1
        return new_extremas

    @staticmethod
    def filter_extremas(
        extremas: list[SIFT_KeyPoint],
        dogs: list[list[np.ndarray]],
        sift_params: SIFT_Params,
    ) -> list[SIFT_KeyPoint]:
        """
        Filters Extrema based on contrast and curvature.
        Args:
            extremas (list[KeyPoint]): The extrema to filter
            dogs (list[list[np.ndarray]]): The dog values to calculate curvature from
            sift_params (SIFT_Params): The sift parameters
        Returns:
            list[KeyPoint]: Filtered Extremum. Returns same objects from input.
        """
        filtered_extremas: list[SIFT_KeyPoint] = []
        for extremum in extremas:
            # current location of the extremum
            scale_level: int = extremum.scale_level
            x_coord: int = extremum.x_coord
            y_coord: int = extremum.y_coord
            # current dog image
            current: np.ndarray = dogs[extremum.octave][scale_level]

            # contrast drop off from the calculate omega value of taylor expansion
            if abs(extremum.omega) < sift_params.threshold_dog_response:
                continue
            # filter off extremas at the border
            if (
                x_coord < 1
                or x_coord > current.shape[1] - 2
                or y_coord < 1
                or y_coord > current.shape[0] - 2
            ):
                continue

            # 2d-Hessian matrix over x and y
            h11: float = (
                current[y_coord, x_coord + 1] - current[y_coord, x_coord - 1]
            ) / 2
            h22: float = (
                current[y_coord + 1, x_coord] - current[y_coord - 1, x_coord]
            ) / 2
            # h12 will be resued for h21
            h12 = (
                current[y_coord + 1, x_coord + 1]
                - current[y_coord + 1, x_coord - 1]
                - current[y_coord - 1, x_coord + 1]
                - current[y_coord - 1, x_coord - 1]
            ) / 4

            hessian = np.matrix([[h11, h12], [h12, h22]])

            trace = np.trace(hessian)
            determinant = np.linalg.det(hessian)
            # if we divide by 0, extremum is not valid
            if determinant == 0:
                continue
            edgeness: float = (trace * trace) / determinant

            # curvature drop off
            if (
                abs(edgeness)
                >= ((sift_params.threshold_edge + 1) ** 2) / sift_params.threshold_edge
            ):
                continue

            filtered_extremas.append(extremum)
        return filtered_extremas

    @staticmethod
    def gradient_2d(
        scale_space: list[list[np.ndarray]], sift_params: SIFT_Params
    ) -> dict[Tuple[int, int, int, int], Tuple[float, float]]:
        """
        Calculate the 2d gradient for each pixel in the scale space.
        Args:
            scale_space (list[list[np.ndarray]]): the scale space to calculate the gradient from
            sift_params (SIFT_Params): the sift parameters
        Returns:
            dict[Tuple[int, int, int, int], Tuple[float, float]]: Dictionary containing each x and y gradients.
            Key is (octave, scale_level, x_coord, y_coord) and value is (grad_x_coord, grad_y_coord).
        """
        gradients: dict[Tuple[int, int, int, int], Tuple[float, float]] = {}
        for idx_octave in range(0, sift_params.n_octaves):
            # include all images from 0 to n_scales_per_octave+2
            for idx_scale_level in range(0, sift_params.n_scales_per_octave + 3):
                # get current image
                img: np.ndarray = scale_space[idx_octave][idx_scale_level]
                # for each pixel in the image
                for x_coord in range(0, img.shape[1]):
                    for y_coord in range(0, img.shape[0]):
                        # if pixel is at the border, we use the difference to the other side
                        if x_coord == 0:
                            grad_x_coord = (
                                img[y_coord, img.shape[1] - 1] - img[y_coord, 1]
                            ) / 2
                        elif x_coord == img.shape[1] - 1:
                            grad_x_coord = (
                                img[y_coord, 0] - img[y_coord, img.shape[1] - 2]
                            ) / 2
                        # otherwise difference to the left and right pixel
                        else:
                            grad_x_coord = (
                                img[y_coord, x_coord + 1] - img[y_coord, x_coord - 1]
                            ) / 2
                        # same for y
                        if y_coord == 0:
                            grad_y_coord = (
                                img[img.shape[0] - 1, x_coord] - img[1, x_coord]
                            ) / 2
                        elif y_coord == img.shape[0] - 1:
                            grad_y_coord = (
                                img[0, x_coord] - img[img.shape[0] - 2, x_coord]
                            ) / 2
                        else:
                            grad_y_coord = (
                                img[y_coord + 1, x_coord] - img[y_coord - 1, x_coord]
                            ) / 2
                        gradients[(idx_octave, idx_scale_level, x_coord, y_coord)] = (
                            grad_x_coord,
                            grad_y_coord,
                        )
        return gradients

    @staticmethod
    def _float_modulo(val: float, mod: float) -> float:
        """
        Performs modulo operation with float modulo.
        Args:
            val (float): the value
            mod (float): the modulo
        Returns:
            float: val % mod with float modulo.
        """
        z: float = val
        n: int = 0
        # if value is negative
        if z < 0:
            # get number of mods fit into value + 1 as we want to make it positive
            n = int((-z) / mod) + 1
            # and add them
            z += n * mod
        # get number of mods fit into value
        n = int(z / mod)
        # and subtract them
        z -= n * mod
        return z

    @staticmethod
    def assign_orientations(
        keypoints: list[SIFT_KeyPoint],
        scale_space: list[list[np.ndarray]],
        sift_params: SIFT_Params,
        gradients: dict[Tuple[int, int, int, int], Tuple[float, float]],
        deltas: list[float],
    ) -> list[SIFT_KeyPoint]:
        """
        Assign orientation angle and magnitude to each Keypoint.
        Args:
            keypoints (list[KeyPoint]): List of keypoints to assign orientation to
            scale_space (list[list[np.ndarray]]): The scale space to calculate from
            sift_params (SIFT_Params): The sift parameters
            gradients (dict[Tuple[int, int, int, int], Tuple[float, float]]): The gradients for each pixel
            deltas (list[float]): list of deltas for each octave
        Returns:
            list[KeyPoint]: List of keypoints with assigned orientation. KeyPoint are same objects.
        """
        new_keypoints: list[SIFT_KeyPoint] = []
        for keypoint in keypoints:
            # current image
            image = scale_space[keypoint.octave][keypoint.scale_level]
            # as x and y are calculate in taylor expansion with m*delta or n*delta, we need to divide by delta
            # this follow the C-implementation
            key_x = keypoint.x_coord_in_scale_space / deltas[keypoint.octave]
            key_y = keypoint.y_coord_in_scale_space / deltas[keypoint.octave]
            key_sigma = keypoint.sigma / deltas[keypoint.octave]
            # border limit is half the window size
            # this depends on the sigma of the keypoint
            border_limit = 3 * sift_params.lambda_orientation * key_sigma
            # if keypoint is not at the border
            if (
                border_limit <= key_x
                and key_x <= image.shape[1] - border_limit
                and border_limit <= key_y
                and key_y <= image.shape[0] - border_limit
            ):
                # initialize histogram
                h_k: np.ndarray = np.zeros(sift_params.n_bins)
                # for each pixel in the window
                # this follows the C-implementation
                # and does not represent the paper pseudo code
                for x_coord in range(
                    max(0, int((key_x - border_limit + 0.5))),
                    min(image.shape[1] - 1, int(key_x + border_limit + 0.5)),
                ):
                    for y_coord in range(
                        max(0, int(key_y - border_limit + 0.5)),
                        min(image.shape[0] - 1, int(key_y + border_limit + 0.5)),
                    ):
                        # calculate the normalized positions
                        s_x = (x_coord - key_x) / key_sigma
                        s_y = (y_coord - key_y) / key_sigma
                        # get gradients
                        grad_x_coord, grad_y_coord = gradients[
                            (keypoint.octave, keypoint.scale_level, x_coord, y_coord)
                        ]
                        # calculate magnitude
                        magnitude: float = np.sqrt(grad_x_coord**2 + grad_y_coord**2)
                        # calculate gaussian weight
                        weight: float = np.exp(
                            (-(s_x**2 + s_y**2))
                            / (2 * (sift_params.lambda_orientation**2))
                        )
                        # and angle
                        angle = SIFT_Algorithm._float_modulo(
                            np.arctan2(grad_x_coord, grad_y_coord), 2 * np.pi
                        )
                        # calculate histogram index
                        if angle < 0:
                            angle += 2 * np.pi
                        b_index = (
                            int(angle / (2 * np.pi) * sift_params.n_bins + 0.5)
                            % sift_params.n_bins
                        )
                        # add weight to histogram
                        h_k[b_index] += weight * magnitude
                # smooth histogram
                for _ in range(6):
                    # mode wrap represents a circular convolution
                    h_k = scipy.ndimage.convolve1d(
                        h_k, weights=[1 / 3, 1 / 3, 1 / 3], mode="wrap"
                    )
                # maximum of the histogram
                max_h_k = np.max(h_k)
                # for each histogram bin
                for k in range(sift_params.n_bins):
                    # get previous and next histogram bin
                    prev_hist: np.float32 = h_k[
                        (k - 1 + sift_params.n_bins) % sift_params.n_bins
                    ]
                    next_hist: np.float32 = h_k[(k + 1) % sift_params.n_bins]
                    # if current bin is above threshold and is a local maximum
                    if (
                        h_k[k] > prev_hist
                        and h_k[k] > next_hist
                        and h_k[k] >= sift_params.threshold_local_maxima * max_h_k
                    ):
                        # calculate offset
                        offset: float = (prev_hist - next_hist) / (
                            2 * (prev_hist + next_hist - 2 * h_k[k])
                        )
                        # and new angle
                        angle = (k + offset + 0.5) * 2 * np.pi / sift_params.n_bins
                        if angle > 2 * np.pi:
                            angle -= 2 * np.pi
                        new_keypoint = SIFT_KeyPoint(
                            keypoint.octave,
                            keypoint.scale_level,
                            keypoint.x_coord,
                            keypoint.y_coord,
                            keypoint.sigma,
                            keypoint.x_coord_in_scale_space,
                            keypoint.y_coord_in_scale_space,
                            keypoint.omega,
                            angle,
                            keypoint.magnitude,
                        )
                        new_keypoints.append(new_keypoint)
        return new_keypoints

    @staticmethod
    def create_descriptors(
        keypoints: list[SIFT_KeyPoint],
        scale_space: list[list[np.ndarray]],
        sift_params: SIFT_Params,
        gradients: dict[Tuple[int, int, int, int], Tuple[float, float]],
        deltas: list[float],
    ) -> list[SIFT_KeyPoint]:
        """
        Creates Key Descriptors for each Keypoint.
        Args:
            extremas (list[KeyPoint]): The keypoints to create descriptors for
            scale_space (list[list[np.ndarray]]): The scalespace to calculate from
            sift_params (SIFT_Params): The sift parameters
            gradients (dict[Tuple[int, int, int, int], Tuple[float, float]]): The gradients for each pixel
            deltas (list[float]): list of deltas for each octave
        Returns:
            list[KeyPoint]: The keypoints with descriptors. KeyPoint are same objects.
        """
        new_keypoints: list[SIFT_KeyPoint] = []
        for keypoint in keypoints:
            # current image
            img = scale_space[keypoint.octave][keypoint.scale_level]
            # current location
            key_x = keypoint.x_coord_in_scale_space / deltas[keypoint.octave]
            key_y = keypoint.y_coord_in_scale_space / deltas[keypoint.octave]
            key_sigma = keypoint.sigma / deltas[keypoint.octave]
            # relative patch size
            relative_patch_size = (
                (1 + 1 / sift_params.n_hist) * sift_params.lambda_descriptor * key_sigma
            )
            # and the actual border limit
            border_limit = np.sqrt(2) * relative_patch_size
            if (
                border_limit <= key_x
                and key_x <= img.shape[1] - border_limit
                and border_limit <= key_y
                and key_y <= img.shape[0] - border_limit
            ):
                # initialize histograms
                histograms: list[list[np.ndarray]] = [
                    [np.zeros(sift_params.n_ori) for _ in range(sift_params.n_hist)]
                    for _ in range(sift_params.n_hist)
                ]
                # for each pixel in patch
                # this follows C-implementation
                # and deviates from the paper pseudo code
                for x_coord in range(
                    max(0, int(key_x - border_limit + 0.5)),
                    min(img.shape[1] - 1, int(key_x + border_limit + 0.5)),
                ):
                    for y_coord in range(
                        max(0, int(key_y - border_limit + 0.5)),
                        min(img.shape[0] - 1, int(key_y + border_limit + 0.5)),
                    ):
                        # normalized positions by angle of keypoint
                        x_vedge_mn = np.cos(-keypoint.theta) * (
                            x_coord - key_x
                        ) - np.sin(-keypoint.theta) * (y_coord - key_y)
                        y_vedge_mn = np.sin(-keypoint.theta) * (
                            x_coord - key_x
                        ) + np.cos(-keypoint.theta) * (y_coord - key_y)
                        # if pixel is in patch
                        if max(abs(x_vedge_mn), abs(y_vedge_mn)) < relative_patch_size:
                            # get gradient
                            delta_m, delta_n = gradients[
                                (
                                    keypoint.octave,
                                    keypoint.scale_level,
                                    x_coord,
                                    y_coord,
                                )
                            ]
                            # calculate new angle, subtract the keypoint angle
                            theta_xy = SIFT_Algorithm._float_modulo(
                                (np.arctan2(delta_m, delta_n) - keypoint.theta),
                                2 * np.pi,
                            )
                            magnitude: float = np.sqrt(
                                delta_m * delta_m + delta_n * delta_n
                            )
                            weight: float = np.exp(
                                -(x_vedge_mn * x_vedge_mn + y_vedge_mn * y_vedge_mn)
                                / (2 * (sift_params.lambda_descriptor * key_sigma) ** 2)
                            )
                            # x and y histogram coordinates
                            alpha = (
                                x_vedge_mn
                                / (
                                    2
                                    * sift_params.lambda_descriptor
                                    * key_sigma
                                    / sift_params.n_hist
                                )
                                + (sift_params.n_hist - 1) / 2
                            )
                            beta = (
                                y_vedge_mn
                                / (
                                    2
                                    * sift_params.lambda_descriptor
                                    * key_sigma
                                    / sift_params.n_hist
                                )
                                + (sift_params.n_hist - 1) / 2
                            )
                            # and bin coordinate
                            gamma = theta_xy / (2 * np.pi) * sift_params.n_ori
                            # for each histogram
                            for i in range(
                                max(
                                    0,
                                    int(alpha),
                                    min(int(alpha) + 1, sift_params.n_hist - 1) + 1,
                                )
                            ):
                                for j in range(
                                    max(0, int(beta)),
                                    min(int(beta) + 1, sift_params.n_hist - 1) + 1,
                                ):
                                    # get bin index to the left
                                    k = (
                                        int(gamma) + sift_params.n_ori
                                    ) % sift_params.n_ori
                                    # add weight
                                    temp = 1.0 - (gamma - np.floor(gamma))
                                    temp *= 1.0 - abs(float(i - alpha))
                                    temp *= 1.0 - abs(float(j - beta))
                                    temp *= weight * magnitude
                                    histograms[i][j][k] += temp
                                    # get bin to the right
                                    k = (
                                        int(gamma) + 1 + sift_params.n_ori
                                    ) % sift_params.n_ori
                                    # add weight
                                    temp = 1.0 - (gamma - np.floor(gamma))
                                    temp *= 1.0 - abs(float(i - alpha))
                                    temp *= 1.0 - abs(float(j - beta))
                                    temp *= weight * magnitude
                                    histograms[i][j][k] += temp
                # create descriptor vector
                f = np.array(histograms).flatten()
                # this is L2 normalization (sqrt(x^2+y^2+...))
                f_norm = np.linalg.norm(f, 2)
                # cap at 0.2*norm
                for l in range(0, f.shape[0]):
                    f[l] = min(f[l], 0.2 * f_norm)
                # recalcualte norm
                f_norm = np.linalg.norm(f, 2)
                # quantize to 0-255
                for l in range(0, f.shape[0]):
                    f[l] = min(np.floor(512 * f[l] / f_norm), 255)
                # set descriptor vector
                keypoint.descriptor = f
                new_keypoints.append(keypoint)
        return new_keypoints

    @staticmethod
    def match_keypoints(
        keypoints_a: list[SIFT_KeyPoint],
        keypoints_b: list[SIFT_KeyPoint],
        sift_params: SIFT_Params,
    ) -> dict[SIFT_KeyPoint, list[SIFT_KeyPoint]]:
        """
        Matches two sets of keypoints and returns the matched keypoints of b to a.
        Args:
            keypoints_a (list[KeyPoint]): list of keypoints of image a
            keypoints_b (list[KeyPoint]): list of keypoints of image b
            sift_params (SIFT_Params): the parameters for the SIFT algorithm
        Returns:
            dict[KeyPoint, list[KeyPoint]]: the matched keypoints of b to a.
        """
        matches: dict[SIFT_KeyPoint, list[SIFT_KeyPoint]] = dict[
            SIFT_KeyPoint, list[SIFT_KeyPoint]
        ]()
        for keypoint_a in keypoints_a:
            # calculate distances to all keypoints in b
            distances = []  # : list[float] deletes because of conversion error warning.
            # corresponding keypoints to distances
            key_points: list[SIFT_KeyPoint] = []
            # compute distances
            for keypoint_b in keypoints_b:
                # L2 norm as distance
                distance = np.linalg.norm(
                    keypoint_a.descriptor - keypoint_b.descriptor, 2
                )
                # if distance is smaller than threshold
                if distance < sift_params.threshold_match_absolute:
                    distances.append(distance)
                    key_points.append(keypoint_b)
            # if only one or no keypoint is close enough
            # discard the keypoint
            if len(distances) < 2:
                continue
            # convert to numpy array
            distances = np.array(distances)
            # find minimum distance index
            min_f_b_index = np.argmin(distances)
            # and extract distance and keypoint
            min_f_b = distances[min_f_b_index]
            keypoint_b = key_points[min_f_b_index]
            # remove from distances
            distances = np.delete(distances, min_f_b_index)
            # and find second minimum distance
            min_2_f_b = np.min(distances)
            # if the ratio of the two distances is smaller than threshold
            if min_f_b < sift_params.threshold_match_relative * min_2_f_b:
                # add to matches
                if keypoint_a in matches:
                    matches[keypoint_a].append(keypoint_b)
                else:
                    matches[keypoint_a] = [keypoint_b]
        return matches
