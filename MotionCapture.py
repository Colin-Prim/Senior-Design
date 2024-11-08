import cv2
import os
import torch
import numpy as np
from scipy.ndimage import gaussian_filter1d
from video_handler.video_reader import VideoReader
from estimator_2d.mediapipe_estimator import MediapipeEstimator
from models.advanced_lifter import AdvancedLifter
from bvh_writer.bvh_writer import BVHWriter
from bvh_writer.bvh_skeleton import BVHSkeleton

def smooth_keypoints(keypoints, sigma=2.5):
    """Applies a Gaussian smoothing filter to reduce erratic movements."""
    return gaussian_filter1d(keypoints, sigma=sigma, axis=0)

def smooth_trajectory(trajectory, sigma=4):
    """Applies a Gaussian filter to the root trajectory for less erratic global movement."""
    return gaussian_filter1d(trajectory, sigma=sigma, axis=0)

def main():
    # Initial setup (loading models, video, etc.)
    video_path = input("Enter the path of the video file (e.g., TaiChi.mp4): ")
    video_reader = VideoReader(video_path)
    video_properties = video_reader.get_video_properties()
    print(f"Video Properties: {video_properties}")

    # Load 2D and 3D estimators
    estimator_2d = MediapipeEstimator()
    lifter_model = AdvancedLifter(model_path="3d_lifting_model.pth")
    
    print("Model weights loaded successfully from 3d_lifting_model.pth")

    # Initialize BVH skeleton and writer
    bvh_skeleton = BVHSkeleton().get_default_skeleton()
    bvh_writer = BVHWriter(bvh_skeleton)

    # List to store smoothed 3D keypoints for each frame
    keypoints_3d_list = []
    root_positions = []  # Store root positions for further smoothing
    initial_root_position = None  # Used to determine initial offset for centering

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
            # Convert to torch tensor before passing to model
            keypoints_3d = lifter_model.lift(torch.from_numpy(keypoints_2d_input).float())
            
            # Reshape and select the first 22 joints for a simpler skeleton
            keypoints_3d = np.reshape(keypoints_3d, (-1, 3))[:22]

            # Apply smoothing to reduce jitter for this frame
            keypoints_3d = smooth_keypoints(keypoints_3d, sigma=2.5)

            # Adjust the scaling of 3D keypoints for visualization
            keypoints_3d *= 10.0

            # Determine the initial root position for centering
            if initial_root_position is None:
                initial_root_position = keypoints_3d[0].copy()
            
            # Calculate the offset to keep the character centered
            root_position = keypoints_3d[0] - initial_root_position
            root_positions.append(root_position)

            # Center root joint by subtracting the current offset
            keypoints_3d[0] -= root_position  # Move to origin based on initial root position

            # Construct motion data for BVH
            frame_motion = []

            # Root joint (hips) position
            frame_motion.extend(root_position)  # Xposition, Yposition, Zposition
            frame_motion.extend([0.0, 0.0, 0.0])  # Initial rotation for the root joint

            # Add rotation data for each joint
            for joint_idx in range(1, len(keypoints_3d)):
                frame_motion.extend(keypoints_3d[joint_idx, :].tolist())

            keypoints_3d_list.append(frame_motion)

        except Exception as e:
            print(f"Error during 3D lifting at frame {frame_idx}: {e}")
            continue

    # Smooth the root trajectory over time
    root_positions = np.array(root_positions)
    smoothed_root_positions = smooth_trajectory(root_positions, sigma=4)

    # Apply smoothed root positions back to the keypoints_3d_list
    for idx, root_position in enumerate(smoothed_root_positions):
        keypoints_3d_list[idx][:3] = root_position  # Update Xposition, Yposition, Zposition of root joint

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
