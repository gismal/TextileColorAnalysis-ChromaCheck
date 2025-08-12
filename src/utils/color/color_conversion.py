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
    if colors.max() > 1.0:
        colors_normalized = colors / 255.0
    else:
        colors_normalized = colors.copy()

    if hasattr(scaler_x, 'data_max_') and scaler_x.data_max[0] > 1.0:
        input_for_scaling = colors if colors.max() > 1.0 else colors* 255.0
    else:
        input_fr_scaling = colors_normalized
    
    scaled_input = scaler_x.transform(input_for_scaling.reshaper(-1, 3))
    prediction = dbn.predict(scaled_input)

    l_channel = scaler_y.inverse_transform(prediction[:, [0]])
    ab_channels = scaler_y_ab.inverse_transform(prediction[:, 1:])

    result = np.hstack((l_channel, ab_channels))

    result[:, 0] = np.clip(result[:, 0], 0, 100)      # L-Kanal: 0-100
    result[:, 1:] = np.clip(result[:, 1:], -128, 127) # a,b-Kanäle: -128 bis 127
    
    return result

def ciede2000_distance(lab1, lab2):
    """Calculate CIEDE2000 color difference between two LAB colors.
    
    Args:
        lab1 (list or np.ndarray): First LAB color.
        lab2 (list or np.ndarray): Second LAB color.
    
    Returns:
        float: CIEDE2000 distance.
    """
    lab1 = np.array(lab1, dtype= np.float64)
    lab2 = np.array(lab2, dtype= np.float64)

    if lab1.shape != (3,) or lab2.shape != (3,):
         raise ValueError(f"LAB colors must be 3D. Got shapes: {lab1.shape}, {lab2.shape}")
    
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

# ZUSÄTZLICHE HILFSFUNKTION für bessere Debug-Möglichkeiten
def validate_lab_color(lab_color, color_name="Unknown"):
    """Validate that a LAB color is in valid ranges.
    
    Args:
        lab_color (np.ndarray): LAB color to validate.
        color_name (str): Name for logging purposes.
    
    Returns:
        bool: True if valid, False otherwise.
    """
    if not isinstance(lab_color, np.ndarray):
        lab_color = np.array(lab_color)
    
    L, a, b = lab_color
    
    valid = True
    if not (0 <= L <= 100):
        print(f"WARNING: {color_name} L-channel out of range: {L} (should be 0-100)")
        valid = False
    if not (-128 <= a <= 127):
        print(f"WARNING: {color_name} a-channel out of range: {a} (should be -128 to 127)")
        valid = False
    if not (-128 <= b <= 127):
        print(f"WARNING: {color_name} b-channel out of range: {b} (should be -128 to 127)")
        valid = False
    
    return valid

# VERBESSERTE VERSION der DBN-Konvertierung mit Debug-Ausgaben
def convert_colors_to_cielab_dbn_debug(dbn, scaler_x, scaler_y, scaler_y_ab, colors, debug=True):
    """Debug-Version der DBN-Konvertierung mit ausführlichen Logs."""
    
    if debug:
        print(f"Input colors shape: {np.array(colors).shape}")
        print(f"Input colors range: [{np.array(colors).min()}, {np.array(colors).max()}]")
    
    if not isinstance(colors, np.ndarray):
        colors = np.array(colors, dtype=np.float32)
    
    # Bestimmen Sie den Input-Bereich
    if colors.max() > 1.0:
        input_for_scaling = colors
        if debug: print("Using 0-255 range for scaler input")
    else:
        input_for_scaling = colors * 255.0
        if debug: print("Converting 0-1 to 0-255 range for scaler input")
    
    scaled_input = scaler_x.transform(input_for_scaling.reshape(-1, 3))
    
    if debug:
        print(f"Scaled input shape: {scaled_input.shape}")
        print(f"Scaled input range: [{scaled_input.min()}, {scaled_input.max()}]")
    
    prediction = dbn.predict(scaled_input)
    
    if debug:
        print(f"DBN prediction shape: {prediction.shape}")
        print(f"DBN prediction range: [{prediction.min()}, {prediction.max()}]")
    
    l_channel = scaler_y.inverse_transform(prediction[:, [0]])
    ab_channels = scaler_y_ab.inverse_transform(prediction[:, 1:])
    
    result = np.hstack((l_channel, ab_channels))
    
    if debug:
        print(f"Before clipping - L: [{result[:, 0].min()}, {result[:, 0].max()}]")
        print(f"Before clipping - a: [{result[:, 1].min()}, {result[:, 1].max()}]")
        print(f"Before clipping - b: [{result[:, 2].min()}, {result[:, 2].max()}]")
    
    # Clipping
    result[:, 0] = np.clip(result[:, 0], 0, 100)
    result[:, 1:] = np.clip(result[:, 1:], -128, 127)
    
    if debug:
        print(f"After clipping - L: [{result[:, 0].min()}, {result[:, 0].max()}]")
        print(f"After clipping - a: [{result[:, 1].min()}, {result[:, 1].max()}]")
        print(f"After clipping - b: [{result[:, 2].min()}, {result[:, 2].max()}]")
    
    return result