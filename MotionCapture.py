import cv2
import os
from video_handler.video_reader import VideoReader
from estimator_2d.mediapipe_estimator import MediapipeEstimator
from estimator_3d.advanced_lifter import AdvancedLifter
from bvh_writer.bvh_writer import BVHWriter
from bvh_writer.bvh_skeleton import BVHSkeleton

def main():
    print("=== 2D to 3D Pose Estimation and BVH Conversion ===")
    video_path = input("Enter the path of the video file (e.g., TaiChi.mp4): ")
    video_reader = VideoReader(video_path)
    video_properties = video_reader.get_video_properties()
    print(f"Video Properties: {video_properties}")

    estimator_2d = MediapipeEstimator()
    model_path = 'estimator_3d/3d_lifting_model.pth'
    lifter = AdvancedLifter(model_path=model_path)

    # Set up the BVH skeleton definition
    bvh_skeleton = BVHSkeleton().get_default_skeleton()
    
    # Ensure that BVHWriter receives the correct skeleton format
    bvh_writer = BVHWriter(skeleton_definition=bvh_skeleton)

    keypoints_3d_list = []
    output_file = video_path.replace(".mp4", "_output.bvh")

    for frame_idx, frame in enumerate(video_reader):
        keypoints_2d = estimator_2d.estimate(frame)

        # Reshape the 2D keypoints to exclude confidence values
        keypoints_2d = keypoints_2d[:, :2].flatten().reshape(1, -1)

        try:
            keypoints_3d = lifter.lift(keypoints_2d)
            keypoints_3d_list.append(keypoints_3d)
        except Exception as e:
            print(f"Error during 3D lifting: {e}")
            print(f"3D lifting failed at frame {frame_idx}.")
            continue

    if keypoints_3d_list:
        try:
            bvh_writer.write_bvh(output_file, keypoints_3d_list)
            print(f"BVH file saved as {output_file}")
        except Exception as e:
            print(f"Error writing BVH file: {e}")
    else:
        print("No 3D keypoints were generated. BVH file not created.")

if __name__ == "__main__":
    main()
