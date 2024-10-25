import numpy as np

def normalize_keypoints(keypoints_2d, image_size):
    """
    Normalizes 2D keypoints to a [-1, 1] range based on image dimensions.

    Args:
        keypoints_2d (list): List of 2D keypoints (x, y, confidence).
        image_size (tuple): Width and height of the image.

    Returns:
        np.ndarray: Normalized 2D keypoints.
    """
    width, height = image_size
    normalized_keypoints = []

    for x, y, confidence in keypoints_2d:
        x_norm = (x / width) * 2 - 1
        y_norm = (y / height) * 2 - 1
        normalized_keypoints.append([x_norm, y_norm, confidence])

    return np.array(normalized_keypoints)

def filter_keypoints(keypoints_2d, confidence_threshold=0.5):
    """
    Filters 2D keypoints based on confidence.

    Args:
        keypoints_2d (list): List of 2D keypoints (x, y, confidence).
        confidence_threshold (float): Minimum confidence to retain keypoints.

    Returns:
        list: Filtered 2D keypoints.
    """
    return [kp for kp in keypoints_2d if kp[2] >= confidence_threshold]

def prepare_batch(keypoints_2d_batch, image_size):
    """
    Prepares a batch of 2D keypoints for 3D lifting.

    Args:
        keypoints_2d_batch (list): List of 2D keypoint lists.
        image_size (tuple): Width and height of the image.

    Returns:
        np.ndarray: Batch of normalized 2D keypoints.
    """
    batch = [normalize_keypoints(keypoints, image_size) for keypoints in keypoints_2d_batch]
    return np.array(batch)
