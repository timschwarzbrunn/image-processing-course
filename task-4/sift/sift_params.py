class SIFT_Params:
    """
    Represents all Hyperparameters for SIFT.
    """

    def __init__(
        self,
        n_octaves: int = 4,
        n_scales_per_octave: int = 3,
        sigma_in: float = 0.5,
        sigma_min: float = 0.8,
        delta_min: float = 0.5,
        threshold_dog_response: float = 0.015,
        threshold_edge: float = 10,
        n_bins: int = 36,
        lambda_orientation: float = 1.5,
        threshold_local_maxima: float = 0.8,
        n_hist: int = 4,
        n_ori: int = 8,
        lambda_descriptor: float = 6,
        threshold_match_absolute: float = 300,
        threshold_match_relative: float = 0.6,
    ):
        """
        Represents all Hyperparameters for SIFT.
        Args:
            n_octaves (int, optional): Number of Octaves. Minimum Octave should result in min(12) Pixel Width and Height. Defaults to 4.
            n_scales_per_octave (int, optional): Number of Scales per Octave. Defaults to 3.
            sigma_in (float, optional): assumed blurr level of input image. Defaults to 0.5.
            sigma_min (float, optional): Blurr-Level of Seed Image v_0^1. Defaults to 0.8.
            delta_min (float, optional): Sampling Distance in Seed Image v_0^1. Defaults to 0.5.
            threshold_dog_response (float, optional): Threshold over the DogResponse. Relative to n_spo. Defaults to 0.015.
            threshold_edge (float, optional): Threshold over the ratio of principal curvatures. Defaults to 10.
            n_bins (int, optional): Number of bins in the gradient orientation histogram. Defaults to 36.
            lambda_orientation (float, optional): Sets how local the analysis of gradient distribution around each keypoint is.
                Patch width is 6*lambda_ori*sigma. Defaults to 1.5.
            threshold_local_maxima (float, optional): Threshold for condiering local maxima in the gradient orientation histogram. Defaults to 0.8.
            n_hist (int, optional): Number of Histograms in the normalized patch (n_hist**2). Defaults to 4.
            n_ori (int, optional): Number of bins in the descriptor histogram. Defaults to 8.
            lambda_descriptor (float, optional): Sets how local the descriptor is. Patch width is 2*lambda_descriptor*sigma. Defaults to 6.
            threshold_match_absolute (float, optional): Threshold for absolute distance between two descriptors. Defaults to 300.
            threshold_match_relative (float, optional): Threshold for relative distance between two descriptors. Defaults to 0.6.
        """
        self.n_octaves = n_octaves
        self.n_scales_per_octave = n_scales_per_octave
        self.sigma_in = sigma_in
        self.sigma_min = sigma_min
        self.delta_min = delta_min
        self.threshold_dog_response = threshold_dog_response
        self.threshold_edge = threshold_edge
        self.n_bins = n_bins
        self.lambda_orientation = lambda_orientation
        self.threshold_local_maxima = threshold_local_maxima
        self.n_hist = n_hist
        self.n_ori = n_ori
        self.lambda_descriptor = lambda_descriptor
        self.threshold_match_absolute = threshold_match_absolute
        self.threshold_match_relative = threshold_match_relative
