import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import filedialog, Toplevel, Label, Button, messagebox
import os

# Global variable to control stopping
stop_processing = False

# Paths for the uploaded and output files
uploaded_file_path = ''
output_file_path = 'output.mp4'
segmentation_file_path = 'segmentation.mp4'

def select_file():
    global uploaded_file_path
    filetypes = [
        ("Video files", "*.mp4 *.avi *.mov *.mkv"),
        ("All files", "*.*")
    ]
    filename = filedialog.askopenfilename(title="Select a video file", filetypes=filetypes)
    if filename:
        uploaded_file_path = filename
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

def process_video(filename, processing_label):
    global stop_processing
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        print("Error reading video file")
        return

    # Output file
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
      
    # VideoWriter objects for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    result = cv2.VideoWriter(output_file_path, fourcc, 10, size)
    resultSeg = cv2.VideoWriter(segmentation_file_path, fourcc, 10, size)

    mpDraw = mp.solutions.drawing_utils
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    while cap.isOpened() and not stop_processing:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame += 1
        frame_canny = cv2.Canny(frame, 100, 200)  # 100 and 200 are thresholds
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for mediapipe

        result2 = pose.process(imgRGB)

        if result2.pose_landmarks:
            mpDraw.draw_landmarks(frame, result2.pose_landmarks, mpPose.POSE_CONNECTIONS)

        cv2.imshow('Pose Estimation', frame)
        result.write(frame)  # Write to output file 
        resultSeg.write(frame_canny)  # Write segmentation output

        processing_label.config(text=f"Displaying frame {current_frame} of {frame_count}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    result.release()
    resultSeg.release()
    cv2.destroyAllWindows()
    show_final_screen()

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
    cap = cv2.VideoCapture(filepath)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Output Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def download_file():
    messagebox.showinfo("Download", "Download functionality to be implemented.")

def restart_program():
    global stop_processing, uploaded_file_path
    stop_processing = False
    uploaded_file_path = ''
    for file in [output_file_path, segmentation_file_path]:
        if os.path.exists(file):
            os.remove(file)
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