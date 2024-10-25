import numpy as np

def handle_occlusions(frames_3d):
    """
    Handles occlusions in 3D keypoints by estimating occluded joints
    based on visible landmarks.
    """
    for frame in frames_3d:
        # Example: Estimate hip joint positions based on spine and legs
        if np.any(np.isnan(frame[0])):  # If Hips are occluded
            frame[0] = (frame[1] + frame[4]) / 2  # Average of spine and legs

        # Example: Estimate arms based on shoulders and elbows
        for side in ["Left", "Right"]:
            shoulder_idx = 10 if side == "Left" else 13
            elbow_idx = 11 if side == "Left" else 14
            hand_idx = 12 if side == "Left" else 15

            if np.any(np.isnan(frame[elbow_idx])):  # If elbow is occluded
                frame[elbow_idx] = (frame[shoulder_idx] + frame[hand_idx]) / 2

    return frames_3d
