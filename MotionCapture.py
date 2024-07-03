import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import filedialog

# Global variable to control stopping
stop_processing = False

def select_file():
    filetypes = [
        ("Video files", "*.mp4 *.avi *.mov *.mkv"),
        ("All files", "*.*")
    ]
    filename = filedialog.askopenfilename(title="Select a video file", filetypes=filetypes)
    if filename:
        process_video(filename)

def process_video(filename):
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
    result = cv2.VideoWriter('output.mp4', fourcc, 10, size)
    resultSeg = cv2.VideoWriter('segmentation.mp4', fourcc, 10, size)

    mpDraw = mp.solutions.drawing_utils
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()

    while cap.isOpened() and not stop_processing:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_canny = cv2.Canny(frame, 100, 200)  # 100 and 200 are thresholds
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for mediapipe

        result2 = pose.process(imgRGB)

        if result2.pose_landmarks:
            mpDraw.draw_landmarks(frame, result2.pose_landmarks, mpPose.POSE_CONNECTIONS)

        cv2.imshow('Pose Estimation', frame)
        result.write(frame)  # Write to output file 
        resultSeg.write(frame_canny)  # Write segmentation output

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    result.release()
    resultSeg.release()
    cv2.destroyAllWindows()

def stop_processing_video():
    global stop_processing
    stop_processing = True

# Create the Tkinter window
root = tk.Tk()
root.title("Video File Selector")
root.geometry("300x150")

# Add a button to select the file
select_button = tk.Button(root, text="Select Video File", command=select_file)
select_button.pack(expand=True)

# Add a button to stop the processing
stop_button = tk.Button(root, text="Cancel", command=stop_processing_video)
stop_button.pack(expand=True)

# Run the Tkinter event loop
root.mainloop()