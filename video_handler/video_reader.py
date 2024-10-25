import cv2

class VideoReader:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

    def get_video_properties(self):
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = self.cap.get(cv2.CAP_PROP_FPS)

        return {
            'total_frames': total_frames,
            'frame_width': frame_width,
            'frame_height': frame_height,
            'frame_rate': frame_rate
        }

    def __iter__(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame

    def release(self):
        self.cap.release()
