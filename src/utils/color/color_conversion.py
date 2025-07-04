import cv2
import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab

def convert_colors_to_cielab(colors):
    """Convert a list of RGB colors to CIELAB color space.
    
    Args:
        colors (list or np.ndarray): List of RGB colors in [R, G, B] format (0-255).
    
    Returns:
        np.ndarray: Array of LAB colors.
    """
    if not isinstance(colors, np.ndarray):
        colors = np.array(colors, dtype=np.uint8)
    # Ensure shape is (n, 3) and convert to float32 for skimage
    colors = colors.reshape(-1, 3).astype(np.float32) / 255.0
    lab_colors = rgb2lab(colors)
    return lab_colors

def convert_colors_to_cielab_dbn(dbn, scaler_x, scaler_y, scaler_y_ab, colors):
    """Convert RGB colors to CIELAB using a DBN model with scaling.
    
    Args:
        dbn: Trained Deep Belief Network model.
        scaler_x: Scaler for input data.
        scaler_y: Scaler for L channel.
        scaler_y_ab: Scaler for a,b channels.
        colors (list or np.ndarray): List of RGB colors in [R, G, B] format (0-255).
    
    Returns:
        np.ndarray: Array of predicted LAB colors.
    """
    if not isinstance(colors, np.ndarray):
        colors = np.array(colors, dtype=np.float32) / 255.0
    scaled_input = scaler_x.transform(colors.reshape(-1, 3))
    prediction = dbn.predict(scaled_input)
    l_channel = scaler_y.inverse_transform(prediction[:, 0].reshape(-1, 1))
    ab_channels = scaler_y_ab.inverse_transform(prediction[:, 1:])
    return np.hstack((l_channel, ab_channels))

def ciede2000_distance(lab1, lab2):
    """Calculate CIEDE2000 color difference between two LAB colors.
    
    Args:
        lab1 (list or np.ndarray): First LAB color.
        lab2 (list or np.ndarray): Second LAB color.
    
    Returns:
        float: CIEDE2000 distance.
    """
    return deltaE_ciede2000(np.array([lab1]), np.array([lab2]))[0]

def bgr_to_rgb(color):
    """Convert a BGR color (OpenCV format) to RGB.
    
    Args:
        color (list or np.ndarray): BGR color in [B, G, R] format.
    
    Returns:
        np.ndarray: RGB color in [R, G, B] format.
    """
    if not isinstance(color, np.ndarray):
        color = np.array(color, dtype=np.uint8)
    return color[::-1]