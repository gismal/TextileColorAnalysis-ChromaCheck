import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from skimage.color import deltaE_ciede2000
import logging

class ColorAnalyzer:
    @staticmethod
    def compute_average_color(image):
        if image is None:
            raise ValueError("Invalid image provided for color analysis.")
        try:
            avg_color = np.mean(image.reshape(-1, image.shape[-1]), axis=0)
            return avg_color
        except Exception as e:
            logging.error(f"Error computing average color: {e}")
            return None

    @staticmethod
    def visualize_colors(ref_colors, aligned_colors, title):
        try:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            for ax, colors, label in zip(axes, [ref_colors, aligned_colors], ['Reference', 'Aligned']):
                color_patches = np.zeros((100, 100 * len(colors), 3), dtype=np.uint8)
                for i, color in enumerate(colors):
                    color_patches[:, i * 100:(i + 1) * 100] = color
                ax.imshow(color_patches)
                ax.set_title(f'{label} Image Colors')
                ax.axis('off')
            plt.suptitle(title)
            plt.show()

            # Calculate color differences using CIEDE2000
            differences = [deltaE_ciede2000(ref, align) for ref, align in zip(ref_colors, aligned_colors)]
            logging.info(f'Color differences (CIEDE2000): {differences}')
        except Exception as e:
            logging.error(f"Error visualizing colors: {e}")
