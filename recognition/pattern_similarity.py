from skimage import metrics
import logging

class PatternSimilarity:
    @staticmethod
    def compare_images(img1, img2):
        if img1 is None or img2 is None:
            raise ValueError("One or both images are empty or not loaded properly.")
        try:
            # Ensure the smaller dimension of the images is at least 7
            min_dim = min(img1.shape[:2])
            win_size = min(7, min_dim // 2 * 2 + 1)  # Ensure win_size is an odd value
            ssim = metrics.structural_similarity(img1, img2, win_size=win_size, channel_axis=-1)
            return ssim
        except Exception as e:
            logging.error(f"Error comparing images: {e}")
            return 0
