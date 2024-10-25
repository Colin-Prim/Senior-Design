import numpy as np
from scipy.ndimage import gaussian_filter1d

class KeypointSmoothing:
    def __init__(self, window_size=5, method='moving_average', alpha=0.2, sigma=1):
        """
        Initializes a smoother for 2D keypoints.

        Args:
            window_size (int): The size of the moving average window.
            method (str): Smoothing method ('moving_average', 'exponential', 'gaussian').
            alpha (float): Smoothing factor for exponential smoothing.
            sigma (float): Standard deviation for Gaussian smoothing.
        """
        self.window_size = window_size
        self.method = method
        self.alpha = alpha
        self.sigma = sigma
        self.history = []

    def smooth(self, keypoints):
        """
        Smooths the input keypoints based on the chosen method.

        Args:
            keypoints (list): List of 2D keypoints (x, y, confidence).

        Returns:
            list: Smoothed 2D keypoints.
        """
        if keypoints is not None:
            self.history.append(keypoints)

        if len(self.history) > self.window_size:
            self.history.pop(0)

        if len(self.history) < self.window_size:
            return keypoints  # Return original if not enough history

        keypoints_np = np.array(self.history)

        if self.method == 'moving_average':
            avg_keypoints = np.mean(keypoints_np, axis=0)
            return avg_keypoints.tolist()
        elif self.method == 'exponential':
            exp_keypoints = self._exponential_smoothing(keypoints_np)
            return exp_keypoints.tolist()
        elif self.method == 'gaussian':
            gauss_keypoints = self._gaussian_smoothing(keypoints_np)
            return gauss_keypoints.tolist()
        else:
            raise ValueError(f"Unsupported smoothing method: {self.method}")

    def _exponential_smoothing(self, keypoints_np):
        """
        Applies exponential smoothing to 2D keypoints.

        Args:
            keypoints_np (numpy.ndarray): 2D keypoints history.

        Returns:
            numpy.ndarray: Smoothed keypoints.
        """
        smoothed = np.zeros_like(keypoints_np[-1])
        for i in range(len(keypoints_np[0])):
            smoothed[i] = (1 - self.alpha) * keypoints_np[-1][i] + self.alpha * keypoints_np[-2][i]
        return smoothed

    def _gaussian_smoothing(self, keypoints_np):
        """
        Applies Gaussian smoothing to 2D keypoints.

        Args:
            keypoints_np (numpy.ndarray): 2D keypoints history.

        Returns:
            numpy.ndarray: Smoothed keypoints.
        """
        return gaussian_filter1d(keypoints_np, sigma=self.sigma, axis=0)[-1]

    def reset(self):
        """
        Resets the history of keypoints.
        """
        self.history = []
