import threading
import queue
import cv2

class ThreadedVideoReader:
    def __init__(self, video_path, buffer_size=64):
        """
        Initializes a threaded video reader.

        Args:
            video_path (str): Path to the video file.
            buffer_size (int): Maximum number of frames to store in the buffer.
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.buffer_size = buffer_size
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.stop_flag = threading.Event()

    def start(self):
        """
        Starts the threaded video reading.
        """
        self.thread = threading.Thread(target=self._read_frames, daemon=True)
        self.thread.start()

    def _read_frames(self):
        """
        Reads frames from the video and stores them in the queue.
        """
        while not self.stop_flag.is_set():
            if self.frame_queue.full():
                continue

            ret, frame = self.cap.read()
            if not ret:
                break

            self.frame_queue.put(frame)

        self.cap.release()

    def read(self):
        """
        Retrieves the next frame from the queue.

        Returns:
            numpy.ndarray: The next video frame.
        """
        if self.frame_queue.empty():
            return None
        return self.frame_queue.get()

    def stop(self):
        """
        Stops the threaded video reading.
        """
        self.stop_flag.set()
        if self.thread.is_alive():
            self.thread.join()

    def is_opened(self):
        """
        Checks if the video capture is still open.

        Returns:
            bool: True if the video is open, False otherwise.
        """
        return self.cap.isOpened()
