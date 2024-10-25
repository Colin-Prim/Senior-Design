import cv2
import mediapipe as mp
import numpy as np

class MediapipeEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, 
                                      model_complexity=2, 
                                      min_detection_confidence=0.5, 
                                      min_tracking_confidence=0.5)

    def estimate(self, frame):
        # Convert the frame to RGB as Mediapipe expects RGB input
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            keypoints_2d = np.array([(lm.x, lm.y, lm.visibility) for lm in landmarks])

            # Scale keypoints back to image size
            h, w, _ = frame.shape
            keypoints_2d[:, 0] *= w  # Scale X to image width
            keypoints_2d[:, 1] *= h  # Scale Y to image height
            
            return keypoints_2d
        else:
            return None

    def __del__(self):
        self.pose.close()
