import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Attempt to import the custom error from its primary location.
# If it's not there (perhaps due to future refactoring), define a fallback.
try:
    # This assumes InvalidConfigurationError might be defined in load_data
    # or potentially moved to a dedicated errors module later.
    from src.data.load_data import InvalidConfigurationError
except ImportError:
    # Fallback definition if the import fails
    class InvalidConfigurationError(ValueError):
        """Custom exception for invalid configuration values or structure."""
        pass

# Get the root logger instance, which setup_logging will configure.
# Using getLogger() ensures we modify the central logger.
logger = logging.getLogger()

# --- Logging Setup ---

def setup_logging(output_dir: Path, log_level: str = 'INFO'):
    """
    Sets up project-wide logging configured to output to both a file and the console.

    It clears any existing handlers attached to the root logger to prevent
    duplicate log messages, especially when run in interactive environments
    like notebooks. Logs are written to 'processing.log' inside the specified
    output directory at DEBUG level, while console output is filtered based on
    the provided `log_level`. Attempts to set console output encoding to UTF-8.

    Args:
        output_dir: The directory where the 'processing.log' file will be created.
        log_level: The minimum level for messages displayed on the console
                   (e.g., 'INFO', 'DEBUG'). Case-insensitive. Defaults to 'INFO'.

    Raises:
        SystemExit: If the log file handler cannot be created (critical failure).
    """
    # Clear existing handlers to prevent duplicate logging, e.g., in Jupyter notebooks.
    if logger.hasHandlers():
        logger.handlers.clear()
        # logger.debug("Cleared existing logging handlers.") # Uncomment for debugging setup

    # Map string log levels to Python's logging constants.
    log_level_map = {'DEBUG': logging.DEBUG, 'INFO': logging.INFO,
                     'WARNING': logging.WARNING, 'ERROR': logging.ERROR}
    console_log_level = log_level_map.get(log_level.upper(), logging.INFO)

    # Define different formats for file (detailed) and console (simple).
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Set the root logger's level to the lowest possible (DEBUG) so that
    # handlers can filter messages based on their own levels.
    logger.setLevel(logging.DEBUG)

    # 1. File Handler Setup
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = output_dir / 'processing.log'
        # Use 'a' to append to the log file if it already exists.
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8') # Specify encoding
        file_handler.setLevel(logging.DEBUG) # Log everything to the file
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler) # Add handler to the root logger
    except Exception as e:
        # Logging to file is crucial, exit if it fails.
        print(f"FATAL ERROR: Could not create log file handler at {log_file}: {e}")
        sys.exit(1)

    # 2. Console Handler Setup
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_log_level) # Filter console output by user setting
    console_handler.setFormatter(simple_formatter)

    # Attempt to set console encoding to UTF-8 to handle special characters like Delta (Δ).
    # This might fail depending on the terminal environment.
    try:
        # Re-opening stdout with UTF-8 encoding
        console_handler.setStream(open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1))
        logger.debug("Console handler stream encoding set to UTF-8.")
    except Exception as e:
        logger.warning(f"Could not set console encoding to UTF-8: {e}. Using default system encoding.")

    # Add the console handler *before* logging the completion message.
    logger.addHandler(console_handler)

    # Log the completion message - this will now go to both handlers.
    logging.info(f"Logging setup complete. Console level: {log_level}. Log file: {log_file}")


# --- Configuration Validation ---

def validate_processing_config(config: Dict[str, Any], project_root: Path) -> bool:
    """
    Validates the loaded configuration dictionary and resolves relative file paths.

    Checks for the presence of required keys (both base and within
    `segmentation_params`). It supports legacy configurations where segmentation
    parameters might be at the root level. Converts relative paths for reference
    and test images into absolute paths based on the project root and verifies
    their existence. Performs basic checks on numeric parameter values.

    Note:
        This function modifies the input `config` dictionary **in-place** by
        resolving paths and potentially moving legacy segmentation parameters.

    Args:
        config: The configuration dictionary loaded from the YAML file(s).
                This dictionary will be modified directly.
        project_root: The absolute path to the project's root directory, used
                      as the base for resolving relative paths in the config.

    Returns:
        True if the configuration is valid, False otherwise. Logs errors on failure.

    Raises:
        FileNotFoundError: If the reference image or any test image file specified
                           in the config cannot be found after path resolution.
        InvalidConfigurationError: If required configuration keys are missing or lists
                                   like `test_images` or `methods` are empty.
        ValueError: If numeric configuration values are invalid (e.g., non-positive).
    """
    logger.debug("Starting configuration validation...")
    required_base_keys = ['reference_image_path', 'test_images']
    # Ensure all expected segmentation keys are listed for validation
    required_seg_keys = ['distance_threshold', 'predefined_k', 'k_values',
                         'som_values', 'dbscan_eps', 'dbscan_min_samples', 'methods',
                         'strategy_subsample', 'dbscan_eps_range', 'dbscan_min_samples_range',
                         'som_iterations', 'som_sigma', 'som_learning_rate'] # Include keys added earlier

    try:
        # --- Check Base Keys ---
        missing_keys = [key for key in required_base_keys if key not in config]
        if missing_keys:
            raise InvalidConfigurationError(f"Missing required base config keys: {missing_keys}")

        # --- Locate or Create segmentation_params ---
        seg_params = config.get('segmentation_params')
        if not seg_params or not isinstance(seg_params, dict):
             logger.warning("Key 'segmentation_params' not found or invalid. "
                            "Checking root level for segmentation keys (legacy support).")
             # Check if all required keys exist at the root level
             missing_seg_keys = [key for key in required_seg_keys if key not in config]
             if missing_seg_keys:
                  raise InvalidConfigurationError(f"Missing required segmentation keys "
                                                  f"(checked root and 'segmentation_params'): {missing_seg_keys}")
             # Move keys from root to a new 'segmentation_params' dictionary
             config['segmentation_params'] = {key: config[key] for key in required_seg_keys if key in config}
             seg_params = config['segmentation_params']
             logger.info("Moved segmentation keys found at root level into 'segmentation_params' dictionary.")
        else:
             # If 'segmentation_params' exists, check if it contains all required keys
             missing_seg_keys = [key for key in required_seg_keys if key not in seg_params]
             if missing_seg_keys:
                  raise InvalidConfigurationError(f"Missing required keys within 'segmentation_params': {missing_seg_keys}")

        # --- Validate and Resolve File Paths ---
        # Purpose: Convert relative paths from config (e.g., "dataset/...")
        #          into absolute paths (e.g., "C:/.../dataset/...") based on where
        #          the project is located, making the config machine-independent.

        # Reference Image Path
        ref_path_str = config['reference_image_path']
        ref_path = Path(ref_path_str)
        if not ref_path.is_absolute():
            ref_path = (project_root / ref_path).resolve() # resolve() makes path absolute and canonical
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference image not found at resolved path: {ref_path}")
        config['reference_image_path'] = str(ref_path) # Update config with the absolute path
        logger.debug(f"Reference image path validated: {ref_path}")

        # Test Images Paths
        resolved_test_images = []
        missing_images_paths = [] # Store missing paths for a clear error message
        test_image_list = config.get('test_images')
        if not test_image_list: # Check if the list is empty
             raise InvalidConfigurationError("'test_images' list cannot be empty in the configuration.")

        for img_path_str in test_image_list:
            img_path = Path(img_path_str)
            if not img_path.is_absolute():
                img_path = (project_root / img_path).resolve()
            if not img_path.exists():
                missing_images_paths.append(str(img_path)) # Add the resolved (but missing) path
            else:
                resolved_test_images.append(str(img_path)) # Add the resolved, existing path

        if missing_images_paths:
            raise FileNotFoundError(f"Test image(s) not found at resolved paths: {missing_images_paths}")
        config['test_images'] = resolved_test_images # Update config with absolute paths
        logger.debug(f"Validated {len(resolved_test_images)} test image paths.")

        # --- Basic Numeric Value Validation ---
        # Purpose: Catch obviously incorrect parameters early on.
        # if seg_params['distance_threshold'] <= 0:
        #     logger.warning("distance_threshold is non-positive. This might affect calculations if used.")
        if seg_params['predefined_k'] <= 0:
            raise ValueError("segmentation_params.predefined_k must be a positive integer.")
        if seg_params['dbscan_eps'] <= 0:
            raise ValueError("segmentation_params.dbscan_eps must be positive.")
        if seg_params['dbscan_min_samples'] <= 0:
             raise ValueError("segmentation_params.dbscan_min_samples must be positive.")
        if not isinstance(seg_params.get('k_values'), list) or not seg_params['k_values']:
             raise InvalidConfigurationError("segmentation_params.k_values must be a non-empty list.")
        if not isinstance(seg_params.get('methods'), list) or not seg_params['methods']:
            raise InvalidConfigurationError("segmentation_params.methods list cannot be empty.")

        logger.info("Configuration validation passed successfully.")
        return True

    # --- Exception Handling ---
    # Purpose: Catch specific expected errors and provide informative logs.
    #          The final 'except Exception' catches unexpected issues.
    except (InvalidConfigurationError, FileNotFoundError, ValueError) as e:
        logger.error(f"Configuration validation failed: {e}")
        return False
    except Exception as e:
         # Log the full traceback for unexpected errors
         logger.error(f"Unexpected error during configuration validation: {e}", exc_info=True)
         return False

# --- Other Helper Functions (Not included here) ---
# Functions like 'safe_image_load', 'create_lab_converters', 'setup_pipeline_configs'
# were originally in main.py but have been integrated elsewhere (load_data.py, pipeline.py)
# or are no longer needed due to the refactoring (setup_pipeline_configs replaced by direct
# dataclass initialization in ProcessingPipeline.__init__). Keeping them out of this 'setup'
# module maintains its focus on initial setup tasks (logging, config validation).