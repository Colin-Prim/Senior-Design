import cv2

def visualize_2d_keypoints(frame, keypoints, connections=None, confidence_threshold=0.5):
    """
    Visualizes 2D keypoints on a given frame.

    Args:
        frame (numpy.ndarray): The video frame (BGR).
        keypoints (list): List of 2D keypoints (x, y, confidence).
        connections (list): List of tuples indicating connections between keypoints.
        confidence_threshold (float): Minimum confidence to display a keypoint.

    Returns:
        numpy.ndarray: Frame with visualized keypoints.
    """
    if keypoints is None:
        return frame

    # Draw keypoints
    for (x, y, confidence) in keypoints:
        if confidence > confidence_threshold:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)  # Green for confident keypoints
        else:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)  # Red for unconfident keypoints

    # Draw connections between keypoints
    if connections:
        for (start, end) in connections:
            if start < len(keypoints) and end < len(keypoints):
                start_point, end_point = keypoints[start], keypoints[end]
                if start_point[2] > confidence_threshold and end_point[2] > confidence_threshold:
                    cv2.line(frame, start_point[:2], end_point[:2], (255, 0, 0), 2)  # Blue line

    return frame


def draw_bbox(frame, keypoints, padding=10):
    """
    Draws a bounding box around the detected keypoints.

    Args:
        frame (numpy.ndarray): The video frame (BGR).
        keypoints (list): List of 2D keypoints (x, y, confidence).
        padding (int): Padding around the bounding box.

    Returns:
        numpy.ndarray: Frame with drawn bounding box.
    """
    if keypoints is None:
        return frame

    # Extract valid x and y coordinates
    x_coords = [x for (x, y, _) in keypoints if x > 0]
    y_coords = [y for (x, y, _) in keypoints if y > 0]

    if x_coords and y_coords:
        x_min, x_max = min(x_coords) - padding, max(x_coords) + padding
        y_min, y_max = min(y_coords) - padding, max(y_coords) + padding
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    return frame


def display_frame(frame, window_name="Pose Estimation"):
    """
    Displays the frame in an OpenCV window.

    Args:
        frame (numpy.ndarray): The video frame (BGR).
        window_name (str): Name of the window.
    """
    cv2.imshow(window_name, frame)
    cv2.waitKey(1)


def save_frame(frame, file_path):
    """
    Saves the frame as an image.

    Args:
        frame (numpy.ndarray): The video frame (BGR).
        file_path (str): Path to save the image file.
    """
    cv2.imwrite(file_path, frame)
