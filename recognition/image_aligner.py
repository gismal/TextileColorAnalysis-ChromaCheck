import cv2
import numpy as np
import logging

class ImageAligner:
    @staticmethod
    def align_images(img1, img2, kp1, kp2, matches):
        if len(matches) < 4:
            logging.warning("Not enough matches to compute homography.")
            return None
        try:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is None or H.shape != (3, 3) or np.linalg.det(H) == 0:
                logging.error("Homography matrix is not valid or degenerate.")
                return None
            height, width = img1.shape[:2]
            aligned_img = cv2.warpPerspective(img2, H, (width, height))
            logging.info(f"Aligned image using homography matrix:\n{H}")
            return aligned_img
        except Exception as e:
            logging.error(f"Error during image alignment: {e}")
            return None
