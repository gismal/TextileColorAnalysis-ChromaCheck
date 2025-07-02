from skimage.color import rgb2lab
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_colors_to_cielab(avg_colors):
    """Convert RGB colors to CIELAB using standard conversion.
    
    Args:
        avg_colors (list): List of RGB colors as (R, G, B) tuples or arrays.
    
    Returns:
        list: CIELAB colors as (L, a, b) tuples.
    """
    logging.info("Converting RGB colors to CIELAB")
    avg_colors_lab = []
    for color in avg_colors:
        color_rgb = np.uint8([[color]])
        color_lab = rgb2lab(color_rgb / 255.0)[0][0]
        avg_colors_lab.append(tuple(color_lab))
    logging.info("Conversion to CIELAB completed")
    return avg_colors_lab

def bgr_to_rgb(color):
    return color[::-1]
