import logging
import cv2
import numpy as np

def exception_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {str(e)}")
            return None
    return wrapper


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    mask = cv2.subtract(image, blurred)
    sharpened = cv2.addWeighted(image, 1.0 + amount, mask, -amount, 0)
    if threshold > 0:
        low_contrast_mask = np.absolute(mask) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

@exception_handler
def preprocess(img):
    logging.info("Starting preprocessing")
    img = cv2.resize(img, None, fx=1.1, fy=1.1, interpolation=cv2.INTER_LINEAR)
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = cv2.bilateralFilter(img, 9, 75, 75)
    logging.info("Preprocessing completed")
    return img

def resize_image(img, size=(256, 256)):
    return cv2.resize(img, size)

