def handle_unusual_poses(frames_3d):
    """
    Handles unusual poses like crouching or sitting by adjusting joint offsets.
    """
    for frame in frames_3d:
        # Adjust offsets for crouching/sitting
        if frame[0][1] < frame[4][1]:  # Hips are below knees
            frame[0][1] = (frame[1][1] + frame[4][1]) / 2  # Adjust hip height

    return frames_3d
