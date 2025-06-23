import cv2
from sklearn.cluster import KMeans
import numpy as np

class PatternDetector:
    @staticmethod
    def cluster_keypoints(keypoints, descriptors, num_clusters=3):
        if not keypoints or not descriptors:
            raise ValueError("Keypoints and descriptors must not be empty.")
        try:
            kmeans = KMeans(n_clusters=num_clusters, random_state=0)
            kmeans.fit(descriptors)
            clustered_keypoints = [[] for _ in range(num_clusters)]
            for idx, label in enumerate(kmeans.labels_):
                clustered_keypoints[label].append(keypoints[idx])
            return clustered_keypoints
        except Exception as e:
            print(f"Error during keypoints clustering: {e}")
            return []

    @staticmethod
    def template_matching(img, template, threshold=0.7):
        try:
            res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= threshold)
            return loc
        except Exception as e:
            print(f"Error during template matching: {e}")
            return np.array([]), np.array([])
