# src/data/load_data.py (REVISED)

import logging
import cv2
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# Use getLogger to inherit logging config from main.py
logger = logging.getLogger(__name__)

# --- Configuration Loading  ---
def merge_dicts(base: Dict, overrides: Dict) -> Dict:
    """Recursively merges override dict into base dict."""
    for key, value in overrides.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            # If it's a dictionary in both, merge recursively
            merge_dicts(base[key], value)
        else:
            # Otherwise, overwrite the base value
            base[key] = value
    return base

def load_config(specific_config_path: str) -> Optional[Dict[str, Any]]:
    """
    Loads configuration, merging a specific config (one of the pattern configs) file over defaults.yaml.

    Args:
        specific_config_path (str): Path to the specific config file 

    Returns:
        Optional[Dict[str, Any]]: The merged configuration dict, or None on failure.
    """
    try:
        specific_path_obj = Path(specific_config_path)
        script_dir = Path(__file__).parent.absolute()
        project_root_guess = script_dir.parent.parent
        config_base_dir = project_root_guess / "configurations" # Assuming configurations dir is at src level
        default_config_path = config_base_dir / "defaults.yaml"

        # 1. Load defaults
        config = {}
        if default_config_path.exists():
            with open(default_config_path, 'r') as f:
                config = yaml.safe_load(f) or {} # Handle empty default file
                logger.info(f"Loaded default configuration from {default_config_path}")
        else:
            logger.warning(f"Default configuration file not found at {default_config_path}. Proceeding without defaults.")

        # 2. Load specific config
        if not specific_path_obj.exists():
             raise FileNotFoundError(f"Specific configuration file not found: {specific_path_obj}")
             
        with open(specific_path_obj, 'r') as f:
            specific_config = yaml.safe_load(f)
            logger.info(f"Loading specific configuration from {specific_path_obj}...")

        # 3. Merge specific config onto defaults 
        if specific_config:
             config = merge_dicts(config, specific_config)
             logger.info("Merged specific configuration over defaults.")
        else:
             # If specific file is empty but defaults exist, use defaults
             if config:
                 logger.warning(f"Specific configuration file {specific_path_obj} is empty. Using defaults only.")
             else:
                 raise ValueError(f"Both default and specific configuration files are empty or invalid.")

        logger.info("Configuration loaded successfully.")
        return config

    except FileNotFoundError as e:
        logger.error(f"Configuration file loading error: {e}")
        return None
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        return None
    except ValueError as e: 
        logger.error(f"Configuration error: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}", exc_info=True)
        return None

# --- Image Loading ---
def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load an image using OpenCV with basic error handling.

    Args:
        image_path (str): Path to the image file.

    Returns:
        Optional[np.ndarray]: Loaded image (RGB format), or None if failed.
    """
    try:
        image_path_str = str(image_path) # Ensure it's a string
        if not Path(image_path_str).exists():
             logger.error(f"Image file does not exist: {image_path_str}")
             return None
             
        image = cv2.imread(image_path_str)
        if image is None:
            logger.error(f"OpenCV failed to load image (may be corrupt or invalid format): {image_path_str}")
            return None
        return image
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None

# --- Data Loading for Training ---
def load_data(image_paths: List[str], target_size: Optional[Tuple[int, int]] = (100, 100)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load images, convert to RGB and LAB, resize, and flatten for DBN training.

    Args:
        image_paths (List[str]): List of paths to test images.
        target_size (Optional[Tuple[int, int]]): Target size (width, height) for resizing. 
                                                  Set to None to disable resizing. Defaults to (100, 100).

    Returns:
        Tuple[np.ndarray, np.ndarray]: (rgb_data, lab_data) 
            - rgb_data: shape (n_images, height*width*3), dtype=float32, range [0, 255]
            - lab_data: shape (n_images, height*width*3), dtype=float32, range OpenCV LAB
            Returns empty arrays if no images are loaded.

    Raises:
         ValueError: If image_paths is empty or no images could be loaded.
    """
    if not image_paths:
        raise ValueError("Image paths list cannot be empty.")

    rgb_data_list = []
    lab_data_list = []
    loaded_count = 0
    logger.info(f"Processing {len(image_paths)} images for training data...")

    for image_path in image_paths:
        image_bgr = load_image(image_path)
        if image_bgr is not None:
            try:
                # Convert BGR (OpenCV default) to RGB
                rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

                # Resize if target_size is provided
                if target_size:
                    rgb = cv2.resize(rgb, target_size, interpolation=cv2.INTER_AREA)
                
                rgb_flat = rgb.reshape(-1).astype(np.float32) 
                lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
                lab_flat = lab.reshape(-1).astype(np.float32)

                rgb_data_list.append(rgb_flat)
                lab_data_list.append(lab_flat)
                loaded_count += 1
            except Exception as e:
                 logger.error(f"Error processing image {image_path} after loading: {e}")

    if loaded_count == 0:
        raise ValueError("No valid images could be loaded or processed from the provided paths.")

    # Stack into 2D arrays: (n_images, height*width*3)
    rgb_array = np.vstack(rgb_data_list)
    lab_array = np.vstack(lab_data_list)
    
    logger.info(f"Successfully loaded and processed {loaded_count}/{len(image_paths)} images.")
    logger.info(f"rgb_data shape: {rgb_array.shape}, lab_data shape: {lab_array.shape}")
    
    return rgb_array, lab_array
