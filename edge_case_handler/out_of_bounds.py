def handle_out_of_bounds(frames_3d, frame_width, frame_height):
    """
    Clips keypoints that are out of bounds to the frame dimensions.
    """
    for frame in frames_3d:
        frame[:, 0] = np.clip(frame[:, 0], 0, frame_width)  # X-axis
        frame[:, 1] = np.clip(frame[:, 1], 0, frame_height)  # Y-axis

    return frames_3d
