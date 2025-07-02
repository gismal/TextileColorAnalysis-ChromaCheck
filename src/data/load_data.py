import logging
import cv2
import yaml
from src.config import BASE_OUTPUT_DIR
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def exception_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {str(e)}")
            return None
    return wrapper 

@exception_handler 
def load_config(config_path):
    """Load config from a YAML file

    Args: 
        config_path: str: Path to the config YAML file
    
    Returns:
        dict: Configuration data, or None if loading fails
    
    """
    logging.info(f"Loading configuration from {config_path}...")
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            logging.info("Configuration loaded successfully.")
        if not config:
            raise ValueError("Config file is empty")
        return config
    except (FileNotFoundError, yaml.YAMLError, ValueError) as e:
          logging.info("Error loading config from {config_path}: {e}")
          return None

def validate_config(config):
    """"Validate the config keys

    Args:
        config (dict): Config data to validate

    Returns:
        bool: True if valid, False otherwise
    
    """
    required_keys = ['reference_image_path', 'test_images', 'distance_threshold', 'kmeans_clusters', 'predefined_k']
    for key in required_keys:
        if key not in config:
            logging.error(f"Missing required config key: {key}")
            return False
    return all(key in config for key in required_keys)

def load_image(image_path):
      """Load an image using OpenCV.
      
      Args:
          image_path (str): Path to the image file.
      
      Returns:
          numpy.ndarray: Loaded image, or None if failed.
      """
      image = cv2.imread(image_path)
      if image is None:
          print(f"Failed to load image: {image_path}")
      return image