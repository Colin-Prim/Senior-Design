import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.transform import Rotation as R
from tkinter import Tk, filedialog

# Define the BVH header
bvh_header = """HIERARCHY
ROOT Hips
{{
    OFFSET 0.000000 0.000000 0.000000
    CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
    JOINT Spine
    {{
        OFFSET 0.000000 10.000000 0.000000
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT Spine1
        {{
            OFFSET 0.000000 10.000000 0.000000
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT Spine2
            {{
                OFFSET 0.000000 10.000000 0.000000
                CHANNELS 3 Zrotation Xrotation Yrotation
                JOINT Neck
                {{
                    OFFSET 0.000000 10.000000 0.000000
                    CHANNELS 3 Zrotation Xrotation Yrotation
                    JOINT Head
                    {{
                        OFFSET 0.000000 10.000000 0.000000
                        CHANNELS 3 Zrotation Xrotation Yrotation
                        End Site
                        {{
                            OFFSET 0.000000 5.000000 0.000000
                        }}
                    }}
                }}
            }}
        }}
    }}
    JOINT LeftUpLeg
    {{
        OFFSET 5.000000 0.000000 0.000000
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT LeftLeg
        {{
            OFFSET 0.000000 -10.000000 0.000000
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT LeftFoot
            {{
                OFFSET 0.000000 -10.000000 0.000000
                CHANNELS 3 Zrotation Xrotation Yrotation
                End Site
                {{
                    OFFSET 0.000000 -5.000000 0.000000
                }}
            }}
        }}
    }}
    JOINT RightUpLeg
    {{
        OFFSET -5.000000 0.000000 0.000000
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT RightLeg
        {{
            OFFSET 0.000000 -10.000000 0.000000
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT RightFoot
            {{
                OFFSET 0.000000 -10.000000 0.000000
                CHANNELS 3 Zrotation Xrotation Yrotation
                End Site
                {{
                    OFFSET 0.000000 -5.000000 0.000000
                }}
            }}
        }}
    }}
}}
MOTION
Frames: {frame_count}
Frame Time: {frame_time}
"""


def extract_pose_data(pose_landmarks):
    landmarks = {'Hips': 0, 'LeftUpLeg': 23, 'LeftLeg': 25, 'LeftFoot': 27, 'RightUpLeg': 24, 'RightLeg': 26,
                 'RightFoot': 28, 'Spine': 11, 'Spine1': 12, 'Spine2': 13, 'Neck': 0, 'Head': 0}

    parent_child_pairs = {
        'Hips': None,
        'Spine': 'Hips',
        'Spine1': 'Spine',
        'Spine2': 'Spine1',
        'Neck': 'Spine2',
        'Head': 'Neck',
        'LeftUpLeg': 'Hips',
        'LeftLeg': 'LeftUpLeg',
        'LeftFoot': 'LeftLeg',
        'RightUpLeg': 'Hips',
        'RightLeg': 'RightUpLeg',
        'RightFoot': 'RightLeg'
    }

    def vector_to_quaternion(v1, v2):
        cross_product = np.cross(v1, v2)
        dot_product = np.dot(v1, v2)
        w = np.sqrt((np.linalg.norm(v1) ** 2) * (np.linalg.norm(v2) ** 2)) + dot_product
        q = np.array([w, cross_product[0], cross_product[1], cross_product[2]])
        q = q / np.linalg.norm(q)
        return q

    def quaternion_to_euler(q):
        r = R.from_quat([q[1], q[2], q[3], q[0]])
        euler = r.as_euler('xyz', degrees=True)
        return euler

    def calculate_rotation(parent, child):
        parent_landmark = pose_landmarks.landmark[parent]
        child_landmark = pose_landmarks.landmark[child]
        parent_vector = np.array([parent_landmark.x, parent_landmark.y, parent_landmark.z])
        child_vector = np.array([child_landmark.x, child_landmark.y, child_landmark.z])
        q = vector_to_quaternion(parent_vector, child_vector)
        euler = quaternion_to_euler(q)
        return euler

    pose_data = []
    for joint, index in landmarks.items():
        landmark = pose_landmarks.landmark[index]
        x, y, z = landmark.x, landmark.y, landmark.z
        rotation_x, rotation_y, rotation_z = 0, 0, 0
        if parent_child_pairs[joint]:
            parent_joint = parent_child_pairs[joint]
            parent_index = landmarks[parent_joint]
            euler = calculate_rotation(parent_index, index)
            rotation_x, rotation_y, rotation_z = euler
        pose_data.append(f"{x:.6f} {y:.6f} {z:.6f} {rotation_x:.6f} {rotation_y:.6f} {rotation_z:.6f}")

    return " ".join(pose_data) + "\n"


def process_video(user_file, output_file_path):
    cap = cv2.VideoCapture(user_file)
    if not cap.isOpened():
        print("Error reading video file")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1 / fps
    current_frame = 0

    mpDraw = mp.solutions.drawing_utils
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()

    print(f"Frame count: {frame_count}, Frame time: {frame_time}")

    with open(output_file_path, 'w') as bvh_file:
        bvh_header_formatted = bvh_header.format(frame_count=frame_count, frame_time=frame_time)
        print(f"Formatted BVH header:\n{bvh_header_formatted}")
        bvh_file.write(bvh_header_formatted)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_frame += 1
            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result2 = pose.process(imgRGB)

            if result2.pose_landmarks:
                frame_data = extract_pose_data(result2.pose_landmarks)
                bvh_file.write(frame_data)
                print(f"Frame {current_frame} data: {frame_data.strip()}")

                # Draw the pose annotation on the frame
                mpDraw.draw_landmarks(frame, result2.pose_landmarks, mpPose.POSE_CONNECTIONS)

            # Display the frame
            cv2.imshow('Pose Estimation', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


def choose_file():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select a video file",
                                           filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")])
    root.destroy()
    return file_path


def main():
    file_path = choose_file()
    if file_path:
        output_file_path = "output.bvh"
        process_video(file_path, output_file_path)


if __name__ == "__main__":
    main()
