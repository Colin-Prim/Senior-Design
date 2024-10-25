import cv2
import mediapipe as mp
import numpy as np

def extract_2d_keypoints(video_path, output_file, num_joints=33):
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    keypoints_2d = []

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                frame_keypoints = []
                for i in range(num_joints):
                    landmark = results.pose_landmarks.landmark[i]
                    frame_keypoints.append([landmark.x, landmark.y])
                keypoints_2d.append(frame_keypoints)
            else:
                # Handle missing keypoints by using NaN
                keypoints_2d.append([[np.nan, np.nan]] * num_joints)

    cap.release()

    # Convert to NumPy array and save
    keypoints_2d = np.array(keypoints_2d, dtype=np.float32)
    np.save(output_file, keypoints_2d)
    print(f"2D keypoints saved to {output_file}")

if __name__ == "__main__":
    extract_2d_keypoints("TaiChi.mp4", "2d_keypoints.npy")
