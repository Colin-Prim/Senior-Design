import cv2
import os
from video_handler.video_reader import VideoReader
from estimator_2d.mediapipe_estimator import MediapipeEstimator
from estimator_3d.advanced_lifter import AdvancedLifter
from bvh_writer.bvh_writer import BVHWriter
from bvh_writer.bvh_skeleton import BVHSkeleton
import numpy as np

def main():
    # Initial setup (loading models, video, etc.)
    video_path = input("Enter the path of the video file (e.g., TaiChi.mp4): ")
    video_reader = VideoReader(video_path)
    video_properties = video_reader.get_video_properties()
    print(f"Video Properties: {video_properties}")

    # Load 2D and 3D estimators
    estimator_2d = MediapipeEstimator()
    lifter = AdvancedLifter(model_path="estimator_3d/3d_lifting_model.pth")

    # Initialize BVH skeleton and writer
    bvh_skeleton = BVHSkeleton().get_default_skeleton()
    bvh_writer = BVHWriter(bvh_skeleton)

    # List to store 3D keypoints for each frame
    keypoints_3d_list = []
    initial_root_position = None  # To store the initial X, Y, Z positions of the root joint

    for frame_idx, frame in enumerate(video_reader):
        # 2D pose estimation
        keypoints_2d = estimator_2d.estimate(frame)

        # Check if 2D keypoints are valid
        if keypoints_2d is None or keypoints_2d.shape != (33, 3):
            print(f"Skipping frame {frame_idx} due to invalid 2D keypoints.")
            continue

        # Prepare 2D keypoints for 3D lifting (use only x and y coordinates)
        keypoints_2d_input = keypoints_2d[:, :2].flatten()[None, :]  # Shape: (1, 66)

        # 3D pose lifting
        try:
            keypoints_3d = lifter.lift(keypoints_2d_input)
            
            # Check if 3D keypoints are valid and reshape if needed
            if keypoints_3d is None or keypoints_3d.shape[0] < 22:
                print(f"Unexpected shape for 3D keypoints at frame {frame_idx}: {keypoints_3d.shape if keypoints_3d is not None else 'None'}")
                continue
            
            # Slice to the first 22 joints if there are more than 22
            keypoints_3d = keypoints_3d[:22]

            # Adjust the scaling of 3D keypoints for visualization
            keypoints_3d[:, :3] *= 10.0

            # For the first frame, set the initial root position
            if initial_root_position is None:
                initial_root_position = keypoints_3d[0, :3].copy()
                print(f"Initial Root Position: {initial_root_position}")

            # Construct motion data for BVH
            frame_motion = []

            # Adjust the root joint's initial position to center it at the origin
            root_x, root_y, root_z = keypoints_3d[0, :3] - initial_root_position
            frame_motion.extend([root_x, root_y, root_z])  # Xposition, Yposition, Zposition
            frame_motion.extend([0.0, 0.0, 0.0])  # Initial rotation for the root joint

            # Add rotation data for each joint
            for joint_idx in range(1, len(keypoints_3d)):
                frame_motion.extend(keypoints_3d[joint_idx, :].tolist())

            keypoints_3d_list.append(frame_motion)

        except Exception as e:
            print(f"Error during 3D lifting: {e}")
            print(f"3D lifting failed at frame {frame_idx}.")
            continue

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
