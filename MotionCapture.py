#Literally just adding this to check my bot

# Motion Capture and Segmentation

import cv2
import mediapipe as mp


filename = 'TaiChi.mp4'
#filename = 'LeftVideoSN001_comp.avi'
cap = cv2.VideoCapture(filename)
if (cap.isOpened() == False): 
        print("Error reading video file")

# output file
# We need to set resolutions. So, convert them from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

#frame_width = frame_width//2
#frame_height = frame_height//2

size = (frame_width, frame_height)
  
# Below VideoWriter object will create a frame of above defined The output 
# is stored in 'output.mp4' file.
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
result = cv2.VideoWriter('output.mp4', fourcc, 10, size)
resultSeg = cv2.VideoWriter('segmentation.mp4', fourcc, 10, size)

"""
result = cv2.VideoWriter('output.mp4', 
                            0x00000021,
                            10, size)
resultSeg = cv2.VideoWriter('segmentation.mp4', 
                            0x00000021,
                            10, size)
"""
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

while(cap.isOpened()):

    ret, frame = cap.read()
    if not ret:
        break
    
    frame_canny = cv2.Canny(frame, 100, 200)  # 100 and 200 are thresholds
    # mediapipe use RGB ordering
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result2 = pose.process(imgRGB)

    if result2.pose_landmarks:
        mpDraw.draw_landmarks(frame, result2.pose_landmarks,
                              mpPose.POSE_CONNECTIONS)

    cv2.imshow('Pose Estimation',frame)
    result.write(frame)  # for output file 
    resultSeg.write(frame_canny)  # segmentation output
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

