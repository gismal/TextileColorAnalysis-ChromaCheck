import cv2
import os
import logging

class ImageLoader:
    def __init__(self, path):
        self.path = path
        self.image = None

    def load_image(self):
        if self.image is not None:
            return self.image
        try:
            if not os.path.exists(self.path):
                raise FileNotFoundError(f"File '{self.path}' does not exist.")
            self.image = cv2.imread(self.path, cv2.IMREAD_COLOR)  # Ensure color image is loaded
            if self.image is None:
                raise IOError(f"Unable to load image at '{self.path}'. Check file integrity.")
            logging.info(f"Loaded image '{self.path}' successfully.")
            return self.image
        except Exception as e:
            logging.error(f"Error loading image: {e}")
            return None

    def resize_image(self, width, height=None):
        if self.image is None:
            logging.error("Error: No image loaded to resize.")
            return None
        try:
            if height is None:
                height = int(self.image.shape[0] * width / self.image.shape[1])
            return cv2.resize(self.image, (width, height))
        except Exception as e:
            logging.error(f"Error resizing image: {e}")
            return None
