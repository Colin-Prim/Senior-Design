import numpy as np

def handle_missing_keypoints(frames_3d, interpolation=True):
    """
    Handles missing keypoints in the 3D frames by interpolation or extrapolation.
    """
    for i in range(1, len(frames_3d) - 1):
        current_frame = frames_3d[i]

        for j, keypoint in enumerate(current_frame):
            if np.any(np.isnan(keypoint)):
                if interpolation:
                    # Interpolate missing keypoint from neighbors
                    prev_frame = frames_3d[i - 1]
                    next_frame = frames_3d[i + 1]
                    frames_3d[i][j] = (prev_frame[j] + next_frame[j]) / 2
                else:
                    # Extrapolate using the previous frame
                    frames_3d[i][j] = frames_3d[i - 1][j]

    return frames_3d
