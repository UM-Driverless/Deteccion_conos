import os, sys, cv2, time
import multiprocessing
from abc import ABC, abstractmethod

class Camera(ABC):
    def __init__(self, cam_queue):
        self.cam_queue = cam_queue # Use the provided queue to send images to the main process
        self.process = None  # Initialize the process attribute

    def start(self):
        """Start the process for capturing images."""
        if self.process is None:
            self.process = multiprocessing.Process(target=self.start_capture, daemon=True)
            self.process.start()

    def stop(self):
        """Stop the process safely, ensuring all resources are cleaned up."""
        if self.process is not None:
            self.process.terminate()  # Send a signal to terminate the process
            self.process.join()  # Wait for the process to finish
            self.process = None

    @abstractmethod
    def start_capture(self):
        """Method to be implemented by subclasses for capturing images."""
        pass

class ImageFileCamera(Camera):
    def __init__(self, cam_queue, IMG_PATH):
        super().__init__(cam_queue)
        self.IMG_PATH = IMG_PATH

    def start_capture(self):
        """Continuously read an image from a file and put it into the queue."""
        while True:
            image = cv2.imread(self.IMG_PATH)
            if image is not None:
                self.cam_queue.put(image)
            else:
                print(f"Failed to read image from {self.IMG_PATH}")
            time.sleep(0.1)  # Simulate a delay

# Example usage
if __name__ == '__main__':
    cam_queue = multiprocessing.Queue(maxsize=1)
    image_file_camera = ImageFileCamera(cam_queue, 'test_media/cones_image.png')
    image_file_camera.start()
    
    try:
        # Display images continuously
        while True:
            if not cam_queue.empty():
                image = cam_queue.get()
                cv2.imshow('Camera Feed', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        image_file_camera.stop()
        cv2.destroyAllWindows()
