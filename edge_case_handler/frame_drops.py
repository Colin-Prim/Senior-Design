import numpy as np

def handle_frame_drops(frames_3d, max_drop=2):
    """
    Handles dropped frames by duplicating the last valid frame.
    """
    last_valid_frame = None

    for i, frame in enumerate(frames_3d):
        if frame is not None:
            last_valid_frame = frame
        else:
            # If a frame is missing, duplicate the last valid frame
            if last_valid_frame is not None and i < max_drop:
                frames_3d[i] = last_valid_frame

    return frames_3d
