import cv2
import numpy as np
import logging
from skimage.color import rgb2lab

def preprocess(img):
    try:
        img = cv2.GaussianBlur(img, (7, 7), 0)
        img = cv2.bilateralFilter(img, 9, 75, 75)
        return img
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        return None

def resize_image(img, size=(256, 256)):
    return cv2.resize(img, size)

def align_images(reference_image, test_image):
    try:
        gray_ref = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
        gray_test = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray_ref, None)
        kp2, des2 = sift.detectAndCompute(gray_test, None)

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) < 10:
            logging.warning("Not enough matches found between reference and test image.")
            return test_image

        pts_ref = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts_test = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(pts_test, pts_ref, cv2.RANSAC, 5.0)

        if H is None or H.shape != (3, 3) or np.linalg.det(H) == 0:
            logging.error("Homography computation failed.")
            return test_image

        height, width, _ = reference_image.shape
        aligned_test_image = cv2.warpPerspective(test_image, H, (width, height))

        # Remove black borders by cropping to the common area
        aligned_test_image_gray = cv2.cvtColor(aligned_test_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(aligned_test_image_gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        aligned_test_image = aligned_test_image[y:y+h, x:x+w]

        return aligned_test_image
    except Exception as e:
        logging.error(f"Error aligning images: {str(e)}")
        return test_image

def k_mean_segmentation(image, k):
    try:
        pixels = image.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        segmented_image = centers[labels.flatten()].reshape(image.shape)

        avg_colors = [cv2.mean(image, mask=(labels.reshape(image.shape[:2]) == i).astype(np.uint8))[:3] for i in range(k)]
        return segmented_image, avg_colors, labels
    except Exception as e:
        logging.error(f"Error in K-means segmentation: {str(e)}")
        return None, None, None
