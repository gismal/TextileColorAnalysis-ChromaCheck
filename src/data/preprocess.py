import logging
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def exception_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {str(e)}")
            return None
    return wrapper

class Preprocessor:
    def __init__(self, resize_factor=1.1, kernel_size=(3, 3), bilateral_d=9, bilateral_sigma_color=75, bilateral_sigma_space=75):
        """Initialize the preprocessor with configurable parameters."""
        self.resize_factor = resize_factor
        self.kernel_size = kernel_size
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space

    @exception_handler
    def preprocess(self, img):
        """Preprocess the image with resizing, morphological opening, and bilateral filtering."""
        logging.info("Starting preprocessing")
        # Resize
        img = cv2.resize(img, None, fx=self.resize_factor, fy=self.resize_factor, interpolation=cv2.INTER_LINEAR)
        # Morphological opening
        kernel = np.ones(self.kernel_size, np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        # Bilateral filtering
        img = cv2.bilateralFilter(img, self.bilateral_d, self.bilateral_sigma_color, self.bilateral_sigma_space)
        logging.info("Preprocessing completed")
        return img

    @exception_handler
    def unsharp_mask(self, image, amount=1.0, threshold=0):
        """Apply unsharp masking to enhance image details."""
        blurred = cv2.GaussianBlur(image, (5, 5), 1.0)  # Using default kernel_size=(5, 5), sigma=1.0
        mask = cv2.subtract(image, blurred)
        sharpened = cv2.addWeighted(image, 1.0 + amount, mask, -amount, 0)
        if threshold > 0:
            low_contrast_mask = np.absolute(mask) < threshold
            np.copyto(sharpened, image, where=low_contrast_mask)
        return sharpened

    @exception_handler
    def resize_image(self, img, size=(256, 256)):
        """Resize the image to a specified size."""
        return cv2.resize(img, size)

# Example usage (can be removed if not needed in the module)
if __name__ == "__main__":
    # Test the class
    preprocessor = Preprocessor()
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    processed_img = preprocessor.preprocess(test_img)
    print("Processed image shape:", processed_img.shape if processed_img is not None else "Failed")