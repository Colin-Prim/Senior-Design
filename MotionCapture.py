import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import filedialog, Toplevel, Label, Button, messagebox
import os
import numpy as np
from scipy.spatial.transform import Rotation as R

# Global variable to control stopping
stop_processing = False

# Paths for the uploaded and output files
uploaded_file_path = ''
output_file_path = 'output.bvh'

# Define the BVH header
bvh_header = """HIERARCHY
ROOT Hips
{
    OFFSET 0.000000 0.000000 0.000000
    CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
    JOINT Spine
    {
        OFFSET 0.000000 10.000000 0.000000
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT Spine1
        {
            OFFSET 0.000000 10.000000 0.000000
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT Spine2
            {
                OFFSET 0.000000 10.000000 0.000000
                CHANNELS 3 Zrotation Xrotation Yrotation
                JOINT Neck
                {
                    OFFSET 0.000000 10.000000 0.000000
                    CHANNELS 3 Zrotation Xrotation Yrotation
                    JOINT Head
                    {
                        OFFSET 0.000000 10.000000 0.000000
                        CHANNELS 3 Zrotation Xrotation Yrotation
                        End Site
                        {
                            OFFSET 0.000000 5.000000 0.000000
                        }
                    }
                }
            }
        }
    }
    JOINT LeftUpLeg
    {
        OFFSET 5.000000 0.000000 0.000000
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT LeftLeg
        {
            OFFSET 0.000000 -10.000000 0.000000
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT LeftFoot
            {
                OFFSET 0.000000 -10.000000 0.000000
                CHANNELS 3 Zrotation Xrotation Yrotation
                End Site
                {
                    OFFSET 0.000000 -5.000000 0.000000
                }
            }
        }
    }
    JOINT RightUpLeg
    {
        OFFSET -5.000000 0.000000 0.000000
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT RightLeg
        {
            OFFSET 0.000000 -10.000000 0.000000
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT RightFoot
            {
                OFFSET 0.000000 -10.000000 0.000000
                CHANNELS 3 Zrotation Xrotation Yrotation
                End Site
                {
                    OFFSET 0.000000 -5.000000 0.000000
                }
            }
        }
    }
}
MOTION
Frames: {frame_count}
Frame Time: {frame_time}
"""


def select_file():
    global uploaded_file_path
    filetypes = [
        ("Video files", "*.mp4 *.avi *.mov *.mkv"),
        ("All files", "*.*")
    ]
    user_file = filedialog.askopenuser_file(title="Select a video file", filetypes=filetypes)
    if user_file:
        uploaded_file_path = user_file
        show_processing_screen()


def show_processing_screen():
    global processing_screen
    processing_screen = Toplevel(root)
    processing_screen.title("Processing Video")
    processing_screen.geometry("800x450")

    processing_label = Label(processing_screen, text="Displaying frame x of xx", font=("Arial", 10))
    processing_label.pack(side="bottom", anchor="se")

    cancel_button = Button(processing_screen, text="Cancel", command=stop_processing_video, bg="red", fg="white")
    cancel_button.pack(side="bottom", anchor="sw")

    processing_screen.protocol("WM_DELETE_WINDOW", stop_processing_video)
    process_video(uploaded_file_path, processing_label)


def process_video(user_file, processing_label):
    global stop_processing
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

    with open(output_file_path, 'w') as bvh_file:
        bvh_file.write(bvh_header.format(frame_count=frame_count, frame_time=frame_time))
        while cap.isOpened() and not stop_processing:
            ret, frame = cap.read()
            if not ret:
                break

            current_frame += 1
            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for mediapipe

            result2 = pose.process(imgRGB)

            if result2.pose_landmarks:
                # Extract and write pose data to BVH file
                frame_data = extract_pose_data(result2.pose_landmarks)
                bvh_file.write(frame_data)

            processing_label.config(text=f"Displaying frame {current_frame} of {frame_count}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    show_final_screen()


def extract_pose_data(pose_landmarks):
    # Define a mapping from MediaPipe landmarks to BVH hierarchy joints
    landmarks = {
        'Hips': 0,
        'LeftUpLeg': 23,
        'LeftLeg': 25,
        'LeftFoot': 27,
        'RightUpLeg': 24,
        'RightLeg': 26,
        'RightFoot': 28,
        'Spine': 11,
        'Spine1': 12,
        'Spine2': 13,
        'Neck': 0,  # No direct match in MediaPipe, needs adjustment
        'Head': 0  # No direct match in MediaPipe, needs adjustment
    }

    # Adjust the mapping for the neck and head
    landmarks['Neck'] = 0  # Use the nose landmark for neck approximation
    landmarks['Head'] = 0  # Use the nose landmark for head approximation

    # Define parent-child relationships for BVH hierarchy
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
        # Calculate quaternion that rotates vector v1 to vector v2
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
        pose_data.append(f"{x} {y} {z} {rotation_x} {rotation_y} {rotation_z}")

    return " ".join(pose_data) + "\n"


def stop_processing_video():
    global stop_processing
    stop_processing = True


def show_final_screen():
    processing_screen.destroy()
    final_screen = Toplevel(root)
    final_screen.title("Video Processing Complete")
    final_screen.geometry("800x450")

    play_again_button = Button(final_screen, text="Play Again", command=lambda: play_video(output_file_path))
    play_again_button.pack(side="top", pady=10)

    download_button = Button(final_screen, text="Download", command=download_file)
    download_button.pack(side="left", padx=10)

    try_again_button = Button(final_screen, text="Try Again", command=restart_program)
    try_again_button.pack(side="left", padx=10)

    back_to_home_button = Button(final_screen, text="Back to Home", command=final_screen.destroy)
    back_to_home_button.pack(side="right", padx=10)


def play_video(filepath):
    messagebox.showinfo("Play Again", "BVH file playback is not supported in this application.")


def download_file():
    messagebox.showinfo("Download", "Download functionality to be implemented.")


def restart_program():
    global stop_processing, uploaded_file_path
    stop_processing = False
    uploaded_file_path = ''
    if os.path.exists(output_file_path):
        os.remove(output_file_path)
    root.destroy()
    main()


def main():
    global root
    root = tk.Tk()
    root.title("Video File Selector")
    root.geometry("300x150")

    # Add a button to select the file
    select_button = Button(root, text="Select Video File", command=select_file)
    select_button.pack(expand=True)

    # Run the Tkinter event loop
    root.mainloop()


if __name__ == "__main__":
    main()
