import cv2
import logging

class FeatureMatcher:
    @staticmethod
    def orb_feature_matching(img1, img2, max_features=500):
        if img1 is None or img2 is None:
            raise ValueError("One or both images are empty or not loaded properly.")
        try:
            orb = cv2.ORB_create(nfeatures=max_features)
            kp1, des1 = orb.detectAndCompute(img1, None)
            kp2, des2 = orb.detectAndCompute(img2, None)
            
            if des1 is None or des2 is None:
                raise ValueError("No descriptors found for one or both images.")
            
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)
            good_matches = [m for m, n in matches if n and m.distance < 0.7 * n.distance]

            logging.info(f"Found {len(good_matches)} good matches between the images.")
            return kp1, des1, kp2, des2, good_matches
        except Exception as e:
            logging.error(f"Error during ORB feature matching: {e}")
            return [], [], [], [], []
