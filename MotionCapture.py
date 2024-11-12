import cv2
import numpy as np
import mediapipe as mp
from bvh_writer.bvh_writer import BVHWriter
from bvh_writer.bvh_skeleton import BVHSkeleton
import base64


def process_video(user_file, output_file_path, cancel_signal):
    # Open the video file
    cap = cv2.VideoCapture(user_file)
    if not cap.isOpened():
        print("Error reading video file")
        return

    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1 / fps
    current_frame = 0

    print(f"Frame count: {frame_count}, Frame time: {frame_time}")

    # Initialize MediaPipe Pose and Drawing utils
    mpDraw = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False,
                        min_detection_confidence=0.5)

    # Initialize BVH skeleton and writer
    bvh_skeleton = BVHSkeleton().get_default_skeleton()
    bvh_writer = BVHWriter(bvh_skeleton)
    initial_root_position = None

    # Open output BVH file for writing
    with open(output_file_path, 'w') as bvh_file:
        # Write the BVH hierarchy and motion header
        bvh_file.write("HIERARCHY\n")
        bvh_writer._write_joint_hierarchy(bvh_file, "ROOT", "Hips", bvh_skeleton["Hips"])
        bvh_file.write("\nMOTION\n")
        bvh_file.write(f"Frames: {frame_count}\n")
        bvh_file.write(f"Frame Time: {frame_time}\n")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_frame += 1

            # Convert frame to RGB for processing
            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)

            # Check if 3D keypoints are available
            if results.pose_landmarks:
                # Extract 3D landmarks and convert to numpy array
                landmarks = results.pose_landmarks.landmark
                keypoints_3d = np.array([(lm.x, lm.y, lm.z) for lm in landmarks]) * 100  # Scale to fit BVH

                # Prepare motion data for BVH export
                frame_motion = []
                root_position = keypoints_3d[mp_pose.PoseLandmark.LEFT_HIP.value]

                # Set initial root position
                if initial_root_position is None:
                    initial_root_position = root_position.copy()
                    print(f"Initial Root Position: {initial_root_position}")

                # Adjust root position relative to initial position
                root_x, root_y, root_z = root_position - initial_root_position
                frame_motion.extend([root_x, root_y, root_z])  # Xposition, Yposition, Zposition
                frame_motion.extend([0.0, 0.0, 0.0])  # Initial rotation for the root joint

                # Loop through each joint and add 3D positions
                for joint_idx in range(1, 33):  # Assuming 33 keypoints from MediaPipe
                    joint_position = keypoints_3d[joint_idx]
                    frame_motion.extend(joint_position.tolist())

                # Write the frame motion data directly to the BVH file
                bvh_file.write(" ".join(map(str, frame_motion)) + "\n")
                print(f"Processed frame {current_frame} with root position {root_x}, {root_y}, {root_z}")

                # Draw the pose annotation on the frame
                mpDraw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Convert frame to JPEG format
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_encoded = base64.b64encode(buffer).decode('utf-8')

                # Yield the frame as a base64-encoded string
                yield frame_encoded

            # If cancellation signal is received, stop processing
            if cancel_signal['value']:
                bvh_file.flush()
                break

    # Release resources
    cap.release()
    pose.close()
    cv2.destroyAllWindows()
    print(f"BVH file successfully written to {output_file_path}.")


if __name__ == "__main__":
    # Initial setup (loading video, setting up skeleton, etc.)
    video_path = input("Enter the path of the video file (e.g., TaiChi.mp4): ")

    # Create the generator
    frame_generator = process_video(video_path, "output.bvh", {'value': False})

    # Iterate over the generator to process and display frames
    for frame_encoded in frame_generator:
        # Optionally display the frame using OpenCV (for testing purposes)
        frame = cv2.imdecode(np.frombuffer(base64.b64decode(frame_encoded), np.uint8), cv2.IMREAD_COLOR)

    cv2.destroyAllWindows()
