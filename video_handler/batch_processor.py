import numpy as np
import cv2

class BatchProcessor:
    def __init__(self, batch_size=16):
        """
        Initializes the batch processor.

        Args:
            batch_size (int): Number of frames to process in a batch.
        """
        self.batch_size = batch_size
        self.frame_buffer = []

    def add_frame(self, frame):
        """
        Adds a video frame to the buffer.

        Args:
            frame (numpy.ndarray): A video frame.
        """
        self.frame_buffer.append(frame)

    def is_ready(self):
        """
        Checks if the frame buffer is ready for batch processing.

        Returns:
            bool: True if the buffer has enough frames, False otherwise.
        """
        return len(self.frame_buffer) >= self.batch_size

    def get_batch(self):
        """
        Returns the current batch of frames.

        Returns:
            list: A list of video frames.
        """
        if not self.is_ready():
            raise ValueError("Insufficient frames in buffer for batch processing.")

        batch = self.frame_buffer[:self.batch_size]
        self.frame_buffer = self.frame_buffer[self.batch_size:]  # Remove processed frames

        return batch

    def clear_buffer(self):
        """
        Clears the frame buffer.
        """
        self.frame_buffer = []

    def process_batch(self, batch, processor_fn):
        """
        Processes a batch of frames using a specified function.

        Args:
            batch (list): List of video frames.
            processor_fn (function): A function to apply to each frame.

        Returns:
            list: List of processed frames.
        """
        processed_batch = [processor_fn(frame) for frame in batch]
        return processed_batch
