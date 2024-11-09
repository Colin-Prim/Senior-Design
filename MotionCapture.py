import cv2
import os
import numpy as np
import mediapipe as mp
from bvh_writer.bvh_writer import BVHWriter
from bvh_writer.bvh_skeleton import BVHSkeleton

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)

def main():
    # Initial setup (loading video, setting up skeleton, etc.)
    video_path = input("Enter the path of the video file (e.g., TaiChi.mp4): ")
    cap = cv2.VideoCapture(video_path)
    
    # Initialize BVH skeleton and writer
    bvh_skeleton = BVHSkeleton().get_default_skeleton()
    bvh_writer = BVHWriter(bvh_skeleton)
    
    # List to store 3D keypoints for each frame
    keypoints_3d_list = []
    initial_root_position = None  # To store the initial X, Y, Z positions of the root joint

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run MediaPipe pose estimation
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Check if 3D keypoints are available
        if not results.pose_landmarks:
            print(f"Skipping frame {frame_idx} due to lack of 3D keypoints.")
            frame_idx += 1
            continue

        # Extract 3D landmarks and convert to numpy array
        landmarks = results.pose_landmarks.landmark
        keypoints_3d = np.array([(lm.x, lm.y, lm.z) for lm in landmarks]) * 100  # Scale to fit BVH

        # Prepare motion data for BVH export
        frame_motion = []

        # Use the first landmark (hips) as the root position
        root_position = keypoints_3d[mp_pose.PoseLandmark.LEFT_HIP.value]

        # Set initial root position at origin for the first frame
        if initial_root_position is None:
            initial_root_position = root_position.copy()
            print(f"Initial Root Position: {initial_root_position}")

        # Adjust root position relative to initial position
        root_x, root_y, root_z = root_position - initial_root_position
        frame_motion.extend([root_x, root_y, root_z])  # Xposition, Yposition, Zposition
        frame_motion.extend([0.0, 0.0, 0.0])  # Initial rotation for the root joint

        # Loop through each joint and add 3D positions
        for joint_idx in range(1, 33):  # assuming 33 keypoints from MediaPipe
            joint_position = keypoints_3d[joint_idx]
            frame_motion.extend(joint_position.tolist())

        keypoints_3d_list.append(frame_motion)
        print(f"Processed frame {frame_idx + 1} with root position {root_x}, {root_y}, {root_z}")
        
        frame_idx += 1

    # Release resources
    cap.release()
    pose.close()

    # Write to BVH file if motion data was generated
    if keypoints_3d_list:
        output_file = "output.bvh"
        try:
            bvh_writer.write_bvh(output_file, keypoints_3d_list)
            print(f"BVH file successfully written to {output_file}.")
        except Exception as e:
            print(f"Error writing BVH file: {e}")
    else:
        print("No 3D keypoints were generated. BVH file not created.")

if __name__ == "__main__":
    main()
