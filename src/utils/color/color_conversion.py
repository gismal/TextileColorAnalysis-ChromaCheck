import cv2
import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab
import logging
# --- FIX 4: Import Any ---
from typing import List, Tuple, Union, Any, TYPE_CHECKING

if TYPE_CHECKING:
    # Try importing the actual classes for runtime checks if possible but primarily for type hinting.
    from src.models.pso_dbn import DBN
    from sklearn.preprocessing import MinMaxScaler
else:
    # Define dummy types if imports fail, necessary for type hints to not break
    DBN = Any
    MinMaxScaler = Any

logger = logging.getLogger(__name__)

def convert_colors_to_cielab(colors: Union[List, np.ndarray]) -> np.ndarray:
    """
    Convert a list/array of RGB colors to CIELAB using skimage.
    Handles input range conversion.

    Args:
        colors (Union[List, np.ndarray]): List or array of RGB colors
                                           in [R, G, B] format (expected range 0-255).

    Returns:
        np.ndarray: Array of LAB colors (standard CIELAB ranges). Returns empty array on error.
    """
    if colors is None:
        logger.warning("Input colors for CIELAB conversion is None")
        return np.array([])

    try:
        if not isinstance(colors, np.ndarray):
            try:
                colors_arr = np.array(colors, dtype = np.float32)
            except ValueError:
                logger.error("Could not convert input list to NumPy array")
                return np.array([])
        else:
            colors_arr = colors.astype(np.float32) # Work with a float copy

        # Ensure shape is (n, 3) using colors_arr
        if colors_arr.ndim == 1 and colors_arr.size % 3 == 0:
            colors_arr = colors_arr.reshape(-1, 3)
        elif colors_arr.ndim != 2 or colors_arr.shape[1] != 3:
            if colors_arr.size == 0:
                 return np.array([])
            logger.error(f"Input colors have unexpected shape: {colors_arr.shape}. Expected (n, 3).")
            return np.array([])

        # Ensure data is in [0, 255] range before normalization
        if np.any(colors_arr < 0) or np.any(colors_arr > 255):
            logger.warning("Input RGB colors are outside [0, 255] range. Clamping values.")
            colors_arr = np.clip(colors_arr, 0, 255)

        # Normalize RGB from [0, 255] to [0, 1] for skimage
        colors_normalized = colors_arr / 255.0

        # Perform conversion
        lab_colors = rgb2lab(colors_normalized)
        return lab_colors

    except Exception as e:
        logger.error(f"Error during skimage RGB to LAB conversion: {e}", exc_info=True)
        return np.array([])

def convert_colors_to_cielab_dbn(dbn: DBN,
                                 scaler_x: MinMaxScaler,
                                 scaler_y: MinMaxScaler,
                                 scaler_y_ab: MinMaxScaler,
                                 colors: Union[List, np.ndarray]) -> np.ndarray:
    """
    Convert RGB colors to CIELAB using a DBN model with scaling.
    
    Args:
        dbn: Trained Deep Belief Network model.
        scaler_x (MinMaxScaler): Scaler fitted on RGB data [0, 255] to range [0, 1].
        scaler_y (MinMaxScaler): Scaler fitted on L data [0, 100] to range [0, 1].
        scaler_y_ab (MinMaxScaler): Scaler fitted on a,b data [-128, 127] to range [0, 1].
        colors (Union[List, np.ndarray]): List or array of RGB colors
                                           in [R, G, B] format (expected range 0-255).
    Returns:
        np.ndarray: Array of predicted LAB colors.
    """
    if colors is None: # Use 'is None'
        logger.warning("Input colors for DBN CIELAB conversion is None")
        return np.array([])

    try:
        # ensure input is a NumPy array
        if not isinstance(colors, np.ndarray):
            try:
                colors_arr = np.array(colors, dtype= np.float32)
            except ValueError:
                logger.error("Could not convert input list to NumPy array for DBN. Check color formats")
                return np.array([])
        else:
            colors_arr = colors.astype(np.float32)
            
        if colors_arr.ndim == 1 and colors_arr.size % 3 == 0:
            colors_arr = colors_arr.reshape(-1,3)
        elif colors_arr.ndim != 2 or colors_arr.shape[1] != 3:
            if colors_arr.size == 0: return np.array([])
            logger.error(f"Input RGB colors for DBN have unexpected shape: {colors_arr.shape}. Expected (n, 3).")
            return np.array([])

        # Ensure data is in [0, 255] range (expected by scaler_x)
        if np.any(colors_arr < 0) or np.any(colors_arr > 255):
            logger.warning("Input RGB colors for DBN are outside [0, 255] range. Clamping values.")
            colors_arr = np.clip(colors_arr, 0, 255)

        # Scaler handles the normalization
        scaled_input = scaler_x.transform(colors_arr)

        # Predict using DBN
        prediction_scaled = dbn.predict(scaled_input)

        # Inverse transform
        if prediction_scaled.shape[1] != 3:
            logger.error(f"DBN prediction has unexpected shape: {prediction_scaled.shape}. Expected (n, 3).")
            return np.array([])

        l_channel_predicted = scaler_y.inverse_transform(prediction_scaled[:, [0]])
        ab_channels_predicted = scaler_y_ab.inverse_transform(prediction_scaled[:, 1:])

        lab_colors_predicted = np.hstack((l_channel_predicted, ab_channels_predicted))

        # Clip to valid CIELAB ranges
        lab_colors_predicted[:, 0] = np.clip(lab_colors_predicted[:, 0], 0, 100)
        lab_colors_predicted[:, 1:] = np.clip(lab_colors_predicted[:, 1:], -128, 127)

        return lab_colors_predicted

    except Exception as e:
        logger.error(f"Error during DBN RGB to LAB conversion: {e}", exc_info=True)
        return np.array([])

def ciede2000_distance(lab1: Union[List, Tuple, np.ndarray],
                       lab2: Union[List, Tuple, np.ndarray]) -> float:
    """Calculate CIEDE2000 color difference between two LAB colors."""
    try:
        lab1_arr = np.array(lab1, dtype=np.float64).reshape(1, 3)
        lab2_arr = np.array(lab2, dtype=np.float64).reshape(1, 3)
        distance = deltaE_ciede2000(lab1_arr, lab2_arr)[0]
        return distance
    except Exception as e:
        logger.error(f"Error calculating CIEDE2000 distance between {lab1} and {lab2}: {e}")
        return float('inf')

def bgr_to_rgb(color: Union[List, Tuple, np.ndarray]) -> np.ndarray:
    """
    Convert a BGR color (OpenCV default) to RGB.
    
    Args:
        color (Union[List, Tuple, np.ndarray]): BGR color in [B, G, R] format.
    
    Returns:
        np.ndarray: RGB color in [R, G, B] format. Returns empty array on error.
    """
    try:
        color_arr = np.array(color)
        if color_arr.shape[-1] != 3:
            raise ValueError(f"Input color must have 3 channels (BGR). Got shape: {color_arr.shape}")
        return color_arr[..., ::-1] # Efficient BGR -> RGB conversion
    except Exception as e:
        logger.error(f"Error converting BGR to RGB for {color}: {e}")
        return np.array([])