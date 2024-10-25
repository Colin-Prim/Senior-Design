import numpy as np

def generate_synthetic_3d_keypoints(input_2d_file, output_3d_file, depth_scale=100):
    # Load the 2D keypoints
    keypoints_2d = np.load(input_2d_file)

    # Initialize 3D keypoints array
    num_frames, num_joints, _ = keypoints_2d.shape
    keypoints_3d = np.zeros((num_frames, num_joints, 3), dtype=np.float32)

    # Copy 2D keypoints and add synthetic Z-axis
    keypoints_3d[..., :2] = keypoints_2d

    # Add synthetic depth based on Y-coordinate
    for frame_idx in range(num_frames):
        for joint_idx in range(num_joints):
            if not np.isnan(keypoints_2d[frame_idx, joint_idx, 1]):
                z_value = depth_scale * (1 - keypoints_2d[frame_idx, joint_idx, 1])  # Inverse depth scaling
                keypoints_3d[frame_idx, joint_idx, 2] = z_value
            else:
                keypoints_3d[frame_idx, joint_idx, 2] = np.nan

    # Save the 3D keypoints
    np.save(output_3d_file, keypoints_3d)
    print(f"Synthetic 3D keypoints saved to {output_3d_file}")

if __name__ == "__main__":
    generate_synthetic_3d_keypoints("2d_keypoints.npy", "3d_keypoints.npy")
