import cv2
import numpy as np

class ContourExtractor:
    @staticmethod
    def extract_and_smooth_contours(image, min_area=1000):
        if image is None:
            raise ValueError("Invalid image provided for contour extraction.")
        try:
            # Convert to LAB color space to separate lightness and color information
            lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            # Use the L channel for contour detection
            l_channel = lab_image[:, :, 0]
            # Apply thresholding on the L channel
            _, binary_image = cv2.threshold(l_channel, 127, 255, cv2.THRESH_BINARY_INV)
            contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            filtered_contours = [cnt for i, cnt in enumerate(contours) if cv2.contourArea(cnt) > min_area and hierarchy[0, i, 3] == -1]
            smoothed_contours = [cv2.approxPolyDP(cnt, 5, True) for cnt in filtered_contours]
            return smoothed_contours
        except Exception as e:
            print(f"Error during contour extraction: {e}")
            return []

    @staticmethod
    def apply_mask(image, contours):
        if image is None or contours is None:
            raise ValueError("Invalid inputs for mask application.")
        try:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mask = np.zeros_like(gray_image)
            cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
            masked_image = cv2.bitwise_and(image, image, mask=mask)
            return masked_image
        except Exception as e:
            print(f"Error during mask application: {e}")
            return None
