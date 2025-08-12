import sys
import os
import argparse
import pstats
import logging
import cProfile
import traceback
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from contextlib import contextmanager
import time

# Define absolute paths
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Project imports
from src.data.load_data import load_config, validate_config, load_data
from src.models.pso_dbn import DBN, pso_optimize, convert_colors_to_cielab_dbn
from src.data.preprocess import Preprocessor
from src.models.segmentation.segmentation import (
    Segmenter, SegmentationConfig, ModelConfig, 
    SegmentationError, InvalidConfigurationError
)
from src.utils.color.color_analysis import ColorMetricCalculator
from src.utils.image_utils import process_reference_image, ciede2000_distance
from src.utils.file_utils import save_output
from src.utils.visualization import save_reference_summary_plot
from src.utils.output_manager import OutputManager

# Force TensorFlow to use CPU and reduce logging
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Setup logging
def setup_logging(output_dir: Path, log_level: str = 'INFO'):
    """Setup comprehensive logging configuration."""
    log_level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR
    }
    
    level = log_level_map.get(log_level.upper(), logging.INFO)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    
    # File handler
    log_file = output_dir / 'processing.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(simple_formatter)
    
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logging.info(f"Logging setup complete. Log file: {log_file}")

@contextmanager
def timer(operation_name: str):
    """Context manager for timing operations."""
    start = time.time()
    try:
        logging.info(f"Starting: {operation_name}")
        yield
    finally:
        duration = time.time() - start
        logging.info(f"Completed: {operation_name} in {duration:.2f} seconds")

def create_scalers_dict(scaler_x, scaler_y, scaler_y_ab):
    """Create scalers dictionary for consistent reference."""
    return {
        'scaler_x': scaler_x,
        'scaler_y': scaler_y,
        'scaler_y_ab': scaler_y_ab
    }

def safe_image_load(image_path: str) -> Optional[np.ndarray]:
    """Safely load an image with comprehensive error handling."""
    try:
        image_path = str(image_path)
        if not os.path.exists(image_path):
            logging.error(f"Image file does not exist: {image_path}")
            return None
            
        image = cv2.imread(image_path)
        if image is None:
            logging.error(f"OpenCV failed to load image: {image_path}")
            return None
            
        if image.size == 0:
            logging.error(f"Loaded image is empty: {image_path}")
            return None
            
        logging.debug(f"Successfully loaded image: {image_path}, shape: {image.shape}")
        return image
        
    except Exception as e:
        logging.error(f"Exception loading image {image_path}: {e}")
        return None

def validate_processing_config(config: Dict[str, Any]) -> bool:
    """Enhanced configuration validation."""
    required_keys = [
        'reference_image_path', 'test_images', 'distance_threshold',
        'kmeans_clusters', 'predefined_k', 'k_values', 'som_values'
    ]
    
    try:
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise InvalidConfigurationError(f"Missing config keys: {missing_keys}")
        
        # Validate file paths
        if not Path(config['reference_image_path']).exists():
            raise FileNotFoundError(f"Reference image not found: {config['reference_image_path']}")
        
        missing_images = [img for img in config['test_images'] if not Path(img).exists()]
        if missing_images:
            raise FileNotFoundError(f"Test images not found: {missing_images}")
        
        # Validate numeric parameters
        if config['distance_threshold'] <= 0:
            raise ValueError("distance_threshold must be positive")
        
        if config['predefined_k'] <= 0:
            raise ValueError("predefined_k must be positive")
        
        if not config['k_values'] or min(config['k_values']) <= 0:
            raise ValueError("k_values must contain positive integers")
            
        if not config['som_values'] or min(config['som_values']) <= 0:
            raise ValueError("som_values must contain positive integers")
        
        logging.info("Configuration validation passed")
        return True
        
    except Exception as e:
        logging.error(f"Configuration validation failed: {e}")
        return False

def validate_loaded_data(rgb_data, lab_data, min_samples=100):
    """Validate that loaded data is suitable for training."""
    if rgb_data is None or lab_data is None:
        raise ValueError("Data loading failed - got None values")
    
    if not hasattr(rgb_data, '__len__') or not hasattr(lab_data, '__len__'):
        raise ValueError("Loaded data must be array-like")
    
    if len(rgb_data) == 0 or len(lab_data) == 0:
        raise ValueError("Loaded data arrays are empty")
    
    # Calculate total available samples
    total_samples = 0
    for i, (rgb_img, lab_img) in enumerate(zip(rgb_data, lab_data)):
        if hasattr(rgb_img, 'size'):
            total_samples += rgb_img.size // 3
        else:
            logging.warning(f"RGB image {i} does not have size attribute")
    
    if total_samples < min_samples:
        raise ValueError(f"Insufficient data: {total_samples} < {min_samples}")
    
    logging.info(f"Data validation passed: {total_samples} total RGB-LAB pairs available")
    return True

def efficient_data_sampling(rgb_data, lab_data, n_samples=800, max_per_image=None):
    """Memory-efficient data sampling with comprehensive error handling."""
    rgb_samples = []
    lab_samples = []
    
    # Handle different data structures
    if hasattr(rgb_data, 'shape') and len(rgb_data.shape) == 2:
        # Data is already flattened as (n_images, n_pixels_per_image)
        logging.info(f"Data is pre-flattened: RGB {rgb_data.shape}, LAB {lab_data.shape}")
        
        n_images = rgb_data.shape[0]
        n_pixels_per_image = rgb_data.shape[1] // 3  # Each pixel has 3 channels
        
        if max_per_image is None:
            max_per_image = max(50, n_samples // n_images)
        
        logging.info(f"Sampling {n_samples} total samples, max {max_per_image} per image")
        
        for i in range(n_images):
            try:
                # Extract RGB and LAB data for this image
                rgb_flat = rgb_data[i].reshape(-1, 3)  # Reshape to (n_pixels, 3)
                lab_flat = lab_data[i].reshape(-1, 3)  # Reshape to (n_pixels, 3)
                
                logging.debug(f"Processing image {i+1}/{n_images}: RGB {rgb_flat.shape}, LAB {lab_flat.shape}")
                
                # Calculate available pixels
                n_pixels = rgb_flat.shape[0]
                n_per_image = min(max_per_image, n_pixels)
                
                if n_per_image <= 0:
                    logging.warning(f"Skipping image {i+1}: insufficient pixels")
                    continue
                
                # Random sampling
                indices = np.random.choice(n_pixels, n_per_image, replace=False)
                
                # Process RGB
                rgb_sample = rgb_flat[indices].astype(np.float32)
                
                # Ensure RGB is in 0-255 range
                if rgb_sample.max() <= 1.0:
                    rgb_sample = rgb_sample * 255.0
                
                # Process LAB with proper validation
                lab_sample = lab_flat[indices].astype(np.float32).copy()
                
                # Validate and convert LAB ranges
                if lab_sample[:, 0].max() > 100:  # L channel should be 0-100
                    logging.debug(f"Image {i+1}: Normalizing L channel from 0-255 to 0-100")
                    lab_sample[:, 0] = lab_sample[:, 0] / 255.0 * 100.0
                
                # Convert a,b channels from OpenCV (0-255) to CIELAB (-128,127)
                if lab_sample[:, 1:].min() >= 0:  # Only convert if in 0-255 range
                    lab_sample[:, 1:] = lab_sample[:, 1:] - 128.0
                
                rgb_samples.append(rgb_sample)
                lab_samples.append(lab_sample)
                
                logging.debug(f"Image {i+1}: Sampled {n_per_image} pixels")
                
            except Exception as e:
                logging.error(f"Error sampling from image {i+1}: {e}")
                logging.debug(f"RGB data shape for image {i+1}: {rgb_data[i].shape if hasattr(rgb_data[i], 'shape') else 'No shape attribute'}")
                logging.debug(f"LAB data shape for image {i+1}: {lab_data[i].shape if hasattr(lab_data[i], 'shape') else 'No shape attribute'}")
                continue
    
    else:
        # Data is structured as list of images
        if max_per_image is None:
            max_per_image = max(50, n_samples // len(rgb_data))
        
        logging.info(f"Sampling {n_samples} total samples, max {max_per_image} per image")
        
        for i, (img_rgb, img_lab) in enumerate(zip(rgb_data, lab_data)):
            try:
                if not hasattr(img_rgb, 'shape') or not hasattr(img_lab, 'shape'):
                    logging.warning(f"Skipping image {i+1}: invalid array format")
                    continue
                    
                logging.debug(f"Processing image {i+1}/{len(rgb_data)}: RGB {img_rgb.shape}, LAB {img_lab.shape}")
                
                # Validate shapes match
                if img_rgb.shape[:2] != img_lab.shape[:2]:
                    logging.warning(f"Image {i+1}: RGB and LAB shape mismatch, skipping")
                    continue
                
                # Calculate available pixels
                n_pixels = img_rgb.shape[0] * img_rgb.shape[1]
                n_per_image = min(max_per_image, n_pixels)
                
                if n_per_image <= 0:
                    logging.warning(f"Skipping image {i+1}: insufficient pixels")
                    continue
                
                # Random sampling
                indices = np.random.choice(n_pixels, n_per_image, replace=False)
                
                # Process RGB
                rgb_flat = img_rgb.reshape(-1, 3)
                rgb_sample = rgb_flat[indices].astype(np.float32)
                
                # Ensure RGB is in 0-255 range
                if rgb_sample.max() <= 1.0:
                    rgb_sample = rgb_sample * 255.0
                
                # Process LAB with proper validation
                lab_flat = img_lab.reshape(-1, 3)
                lab_sample = lab_flat[indices].astype(np.float32).copy()
                
                # Validate and convert LAB ranges
                if lab_sample[:, 0].max() > 100:  # L channel should be 0-100
                    logging.debug(f"Image {i+1}: Normalizing L channel from 0-255 to 0-100")
                    lab_sample[:, 0] = lab_sample[:, 0] / 255.0 * 100.0
                
                # Convert a,b channels from OpenCV (0-255) to CIELAB (-128,127)
                if lab_sample[:, 1:].min() >= 0:  # Only convert if in 0-255 range
                    lab_sample[:, 1:] = lab_sample[:, 1:] - 128.0
                
                rgb_samples.append(rgb_sample)
                lab_samples.append(lab_sample)
                
                logging.debug(f"Image {i+1}: Sampled {n_per_image} pixels")
                
            except Exception as e:
                logging.error(f"Error sampling from image {i+1}: {e}")
                continue
    
    if not rgb_samples:
        raise ValueError("No samples could be extracted from any image")
    
    # Efficiently concatenate
    rgb_result = np.vstack(rgb_samples)
    lab_result = np.vstack(lab_samples)
    
    logging.info(f"Sampling completed: {rgb_result.shape[0]} samples from {len(rgb_samples)} images")
    logging.info(f"RGB range: [{rgb_result.min():.2f}, {rgb_result.max():.2f}]")
    logging.info(f"LAB ranges: L[{lab_result[:, 0].min():.2f}, {lab_result[:, 0].max():.2f}], "
                f"a[{lab_result[:, 1].min():.2f}, {lab_result[:, 1].max():.2f}], "
                f"b[{lab_result[:, 2].min():.2f}, {lab_result[:, 2].max():.2f}]")
    
    return rgb_result, lab_result

def safe_pso_optimization(dbn, x_train_scaled, y_train_scaled, initial_weights, max_retries=3):
    """PSO optimization with retry logic and fallback."""
    # Create bounds with some margin
    bounds = []
    for w in initial_weights:
        w_min, w_max = w.min(), w.max()
        margin = 0.1 * max(abs(w_min), abs(w_max))
        bounds.append((w_min - margin, w_max + margin))
    
    for attempt in range(max_retries):
        try:
            logging.info(f"PSO optimization attempt {attempt + 1}/{max_retries}")
            
            with timer("PSO optimization"):
                optimized_weights = pso_optimize(dbn, x_train_scaled, y_train_scaled, bounds)
            
            # Validate optimized weights
            if any(np.isnan(w).any() or np.isinf(w).any() for w in optimized_weights):
                raise ValueError("PSO produced invalid weights (NaN or Inf)")
            
            # Check if weights are reasonable (not too extreme)
            for i, w in enumerate(optimized_weights):
                if abs(w).max() > 100:  # Arbitrary threshold
                    logging.warning(f"PSO weights layer {i} seem extreme: max={abs(w).max():.2f}")
            
            logging.info("PSO optimization completed successfully")
            return optimized_weights
            
        except Exception as e:
            logging.warning(f"PSO attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                logging.info("Retrying with slightly different initialization...")
                # Add small random noise to initial weights for retry
                initial_weights = [w + np.random.normal(0, 0.001, w.shape) for w in initial_weights]
            else:
                logging.error("All PSO attempts failed, using initial weights")
    
    return initial_weights

def create_lab_converters(dbn, scaler_x, scaler_y, scaler_y_ab):
    """Create color converters to avoid lambda scope issues."""
    def lab_traditional_converter(rgb_color):
        """Convert RGB to LAB using OpenCV."""
        try:
            # Ensure RGB is in uint8 format
            rgb_color = np.array(rgb_color, dtype=np.uint8)
            rgb_array = rgb_color.reshape(1, 1, 3)
            lab_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2LAB)
            lab_result = lab_array[0, 0].astype(np.float32)
            
            # Convert OpenCV LAB to standard CIELAB
            lab_result[0] = lab_result[0] * 100.0 / 255.0  # L: 0-255 -> 0-100
            lab_result[1:] = lab_result[1:] - 128.0         # a,b: 0-255 -> -128,127
            
            return lab_result
        except Exception as e:
            logging.error(f"Traditional LAB conversion failed for {rgb_color}: {e}")
            return np.array([50.0, 0.0, 0.0])  # Return neutral gray as fallback
    
    def lab_dbn_converter(rgb_color):
        """Convert RGB to LAB using PSO-DBN."""
        try:
            return convert_colors_to_cielab_dbn(
                dbn, scaler_x, scaler_y, scaler_y_ab, [rgb_color]
            )[0]
        except Exception as e:
            logging.error(f"DBN LAB conversion failed for {rgb_color}: {e}")
            return np.array([50.0, 0.0, 0.0])  # Return neutral gray as fallback
    
    return lab_traditional_converter, lab_dbn_converter

def main(config_path='configurations/block_config.yaml', log_level='INFO'):
    """Main processing function with comprehensive error handling."""
    profiler = cProfile.Profile()
    profiler.enable()
    
    start_time = time.time()
    
    try:
        # Setup
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        setup_logging(OUTPUT_DIR, log_level)
        
        logging.info("=" * 80)
        logging.info("TEXTILE COLOR ANALYSIS SYSTEM - PROCESSING START")
        logging.info("=" * 80)
        logging.info(f"Current working directory: {os.getcwd()}")
        logging.info(f"Output directory: {OUTPUT_DIR}")
        logging.info(f"Config file: {config_path}")
        
        # Load and validate configuration
        with timer("Configuration loading"):
            config = load_config(config_path)
            if config is None:
                raise ValueError("Failed to load configuration")
            
            if not validate_config(config):
                raise ValueError("Configuration validation failed")
            
            if not validate_processing_config(config):
                raise ValueError("Processing configuration validation failed")

        # Extract configuration values
        reference_image_path = config['reference_image_path']
        test_images = config['test_images']
        distance_threshold = config['distance_threshold']
        k = config['kmeans_clusters']
        predefined_k = config['predefined_k']
        k_values = config['k_values']
        som_values = config['som_values']
        
        # Derive dataset name and setup output manager
        dataset_name = os.path.splitext(os.path.basename(config_path))[0].replace('_config', '')
        output_manager = OutputManager(OUTPUT_DIR, dataset_name)
        
        logging.info(f"Processing dataset: {dataset_name}")
        logging.info(f"Reference image: {reference_image_path}")
        logging.info(f"Test images: {len(test_images)} files")

        # Load and save reference image
        with timer("Reference image loading"):
            reference_image = safe_image_load(reference_image_path)
            if reference_image is None:
                raise ValueError(f"Failed to load reference image: {reference_image_path}")
            output_manager.save_reference_image(reference_image_path, reference_image)

        # Load and save test images
        with timer("Test images loading"):
            valid_test_images = []
            for image_path in test_images:
                test_image = safe_image_load(image_path)
                if test_image is not None:
                    output_manager.save_test_image(image_path, test_image)
                    valid_test_images.append(image_path)
                else:
                    logging.warning(f"Skipping invalid test image: {image_path}")
            
            if not valid_test_images:
                raise ValueError("No valid test images found")
            
            logging.info(f"Successfully loaded {len(valid_test_images)} test images")

        # Load and prepare data for DBN training
        with timer("Training data preparation"):
            rgb_data, lab_data = load_data(valid_test_images)
            validate_loaded_data(rgb_data, lab_data)
            
            logging.info(f"Loaded data - RGB: {len(rgb_data)} images, LAB: {len(lab_data)} images")

            # Sample training data
            rgb_samples, lab_samples = efficient_data_sampling(rgb_data, lab_data, n_samples=800)

        # Train-test split
        with timer("Data splitting"):
            x_train, x_test, y_train, y_test = train_test_split(
                rgb_samples, lab_samples, test_size=0.2, random_state=42
            )
            logging.info(f"Training set: {x_train.shape}, Test set: {x_test.shape}")

        # Initialize and train DBN
        with timer("DBN initialization and training"):
            input_size = 3
            output_size = 3
            hidden_layers = [100, 50, 25]
            dbn = DBN(input_size, hidden_layers, output_size)
            logging.info(f"DBN initialized with architecture: {input_size} -> {hidden_layers} -> {output_size}")

            # Scale data
            scaler_x = MinMaxScaler(feature_range=(0, 1))
            scaler_y = MinMaxScaler(feature_range=(0, 100))     # L channel: 0-100
            scaler_y_ab = MinMaxScaler(feature_range=(-128, 127))  # a,b channels: -128 to 127
            
            x_train_scaled = scaler_x.fit_transform(x_train)
            y_l_scaled = scaler_y.fit_transform(y_train[:, [0]])
            y_ab_scaled = scaler_y_ab.fit_transform(y_train[:, 1:])
            y_train_scaled = np.hstack((y_l_scaled, y_ab_scaled))
            
            logging.info(f"Scaled training data - X: {x_train_scaled.shape}, Y: {y_train_scaled.shape}")
            logging.info(f"X range: [{x_train_scaled.min():.3f}, {x_train_scaled.max():.3f}]")
            logging.info(f"Y range: [{y_train_scaled.min():.3f}, {y_train_scaled.max():.3f}]")

            # Build model and optimize with PSO
            sample_input = np.zeros((1, input_size))
            dbn.model(sample_input)  # Initialize model
            initial_weights = dbn.model.get_weights()
            
            logging.info(f"Initial model has {len(initial_weights)} weight layers")
            for i, w in enumerate(initial_weights):
                logging.info(f"Layer {i}: shape {w.shape}, range [{w.min():.3f}, {w.max():.3f}]")

            # PSO optimization
            optimized_weights = safe_pso_optimization(dbn, x_train_scaled, y_train_scaled, initial_weights)
            dbn.model.set_weights(optimized_weights)

        # Create scalers dictionary
        scalers = create_scalers_dict(scaler_x, scaler_y, scaler_y_ab)

        # Process reference image
        with timer("Reference image processing"):
            reference_results = process_reference_image(
                reference_image_path, dbn, scaler_x, scaler_y, scaler_y_ab, k
            )
            
            if len(reference_results) != 4 or reference_results[0] is None:
                raise ValueError("Failed to process reference image")
                
            reference_kmeans_opt, reference_som_opt, original_image, dpc_k = reference_results
            
            target_colors = np.array(reference_kmeans_opt['avg_colors_lab'])
            output_manager.save_reference_summary(original_image)
            logging.info(f"Reference processed - target colors: {len(target_colors)}, DPC k: {dpc_k}")

        # Initialize results collection
        all_delta_e = []

        # Create color converters
        lab_traditional_converter, lab_dbn_converter = create_lab_converters(
            dbn, scaler_x, scaler_y, scaler_y_ab
        )

        # Process test images
        for image_idx, image_path in enumerate(valid_test_images):
            with timer(f"Processing test image {image_idx + 1}/{len(valid_test_images)}"):
                logging.info(f"Processing test image: {image_path}")
                
                image = safe_image_load(image_path)
                if image is None:
                    logging.error(f"Failed to reload test image: {image_path}")
                    continue

                # Preprocess the image
                preprocessor = Preprocessor(
                    initial_resize=config.get('preprocess_initial_resize', 512),
                    target_size=(config.get('preprocess_target_size_width', 128), config.get('preprocess_target_size_height', 128)),
                    denoise_h=config.get('preprocess_denoise_h', 10),
                    max_colors=config.get('preprocess_max_colors', 8),
                    edge_enhance=config.get('preprocess_edge_enhance', False),
                    unsharp_amount=config.get('preprocess_unsharp_amount', 0.0),
                    unsharp_threshold=config.get('preprocess_unsharp_threshold', 0)
                )
                
                try:
                    preprocessed_image = preprocessor.preprocess(image)
                    image_name = os.path.splitext(os.path.basename(image_path))[0]
                    output_manager.save_preprocessed_image(image_name, preprocessed_image)
                    
                    unique_colors = len(np.unique(preprocessed_image.reshape(-1, 3), axis=0))
                    logging.info(f"Preprocessing completed - unique colors: {unique_colors}")
                    
                except Exception as e:
                    logging.error(f"Preprocessing failed for {image_path}: {e}")
                    continue

                # Test both k-determination strategies
                for k_type in ['determined', 'predefined']:
                    with timer(f"Segmentation with k_type: {k_type}"):
                        logging.info(f"Starting segmentation with k_type: {k_type}")
                        
                        try:
                            # Create configuration objects
                            seg_config = SegmentationConfig(
                                target_colors=target_colors,
                                distance_threshold=distance_threshold,
                                predefined_k=predefined_k,
                                k_values=k_values,
                                som_values=som_values,
                                k_type=k_type,
                                methods=config.get('segmentation_methods', 
                                                 ['kmeans_opt', 'kmeans_predef', 'som_opt', 'som_predef', 'dbscan'])
                            )
                            
                            model_config = ModelConfig(
                                dbn=dbn,
                                scalers=list(scalers.values()),
                                reference_kmeans_opt=reference_kmeans_opt,
                                reference_som_opt=reference_som_opt
                            )

                            # Perform segmentation
                            segmenter = Segmenter(preprocessed_image, seg_config, model_config, output_manager)
                            processing_result = segmenter.process()
                            
                            if not processing_result.results:
                                logging.warning(f"No successful segmentation results for {k_type}")
                                continue

                            # Process results for Delta E calculation
                            color_metric_calculator = ColorMetricCalculator(target_colors)
                            
                            for method_name, result in processing_result.results.items():
                                if not result.is_valid():
                                    logging.warning(f"Invalid result for {method_name}")
                                    continue
                                    
                                rgb_colors = result.avg_colors
                                if not rgb_colors:
                                    logging.warning(f"No RGB colors available for {method_name}")
                                    continue

                                try:
                                    # Find best matches
                                    best_matches = color_metric_calculator.find_best_matches(rgb_colors)
                                    
                                    # Calculate Delta E for both conversion methods
                                    delta_e_traditional = color_metric_calculator.compute_delta_e(
                                        rgb_colors, lab_traditional_converter, best_matches
                                    )
                                    delta_e_dbn = color_metric_calculator.compute_delta_e(
                                        rgb_colors, lab_dbn_converter, best_matches
                                    )

                                    # Collect results
                                    all_delta_e.append({
                                        'dataset': dataset_name,
                                        'image': image_name,
                                        'method': method_name.replace('_opt', '').replace('_predef', ''),
                                        'k_type': k_type,
                                        'n_clusters': result.n_clusters,
                                        'traditional': delta_e_traditional,
                                        'pso_dbn': delta_e_dbn,
                                        'processing_time': result.processing_time
                                    })
                                    
                                    logging.info(f"{method_name} ({k_type}): "
                                               f"ΔE_traditional={delta_e_traditional:.2f}, "
                                               f"ΔE_dbn={delta_e_dbn:.2f}, "
                                               f"k={result.n_clusters}")

                                except Exception as e:
                                    logging.error(f"Delta E calculation failed for {method_name}: {e}")
                                    continue

                        except Exception as e:
                            logging.error(f"Segmentation failed for {image_path} with {k_type}: {e}")
                            logging.debug(traceback.format_exc())
                            continue

        # Save results
        with timer("Results saving"):
            if all_delta_e:
                output_manager.save_delta_e_results(dataset_name, all_delta_e)
                
                # Log summary statistics
                traditional_values = [r['traditional'] for r in all_delta_e if not np.isnan(r['traditional'])]
                dbn_values = [r['pso_dbn'] for r in all_delta_e if not np.isnan(r['pso_dbn'])]
                
                if traditional_values and dbn_values:
                    logging.info(f"Results Summary:")
                    logging.info(f"  Traditional ΔE: mean={np.mean(traditional_values):.2f}, "
                               f"std={np.std(traditional_values):.2f}")
                    logging.info(f"  PSO-DBN ΔE: mean={np.mean(dbn_values):.2f}, "
                               f"std={np.std(dbn_values):.2f}")
                    
                    improvement = (np.mean(traditional_values) - np.mean(dbn_values)) / np.mean(traditional_values) * 100
                    logging.info(f"  PSO-DBN improvement: {improvement:.1f}%")
                
                logging.info(f"Saved Delta E results for {len(all_delta_e)} method combinations")
            else:
                logging.warning("No Delta E results to save")

    except Exception as e:
        logging.error(f"Main execution failed: {e}")
        logging.debug(traceback.format_exc())
        raise
    finally:
        profiler.disable()
        
        total_time = time.time() - start_time
        logging.info("=" * 80)
        logging.info(f"PROCESSING COMPLETED IN {total_time:.2f} SECONDS")
        logging.info("=" * 80)
        
        # Save profiling results
        try:
            stats = pstats.Stats(profiler).sort_stats('cumtime')
            profile_path = OUTPUT_DIR / 'profile_stats.txt'
            with open(profile_path, 'w') as f:
                stats.print_stats(20, file=f)
            logging.info(f"Profiling results saved to: {profile_path}")
        except Exception as e:
            logging.warning(f"Failed to save profiling results: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Textile Color Analysis System using PSO-optimized DBN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Use default config
  python main.py --config custom_config.yaml       # Use custom config
  python main.py --log-level DEBUG                 # Enable debug logging
  python main.py --config block_config.yaml --log-level INFO
        """
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='configurations/block_config.yaml',
        help='Path to configuration YAML file (default: configurations/block_config.yaml)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set the logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Override output directory (default: ./output)'
    )
    
    parser.add_argument(
        '--profile',
        action='store_true',
        help='Enable detailed profiling output'
    )
    
    args = parser.parse_args()
    
    # Set up output directory
    output_dir_path = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    
    try:
        # Validate config file exists
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Configuration file not found: {config_path}")
            print(f"Current working directory: {os.getcwd()}")
            print("Available configuration files:")
            config_dir = Path("configurations")
            if config_dir.exists():
                for config_file in config_dir.glob("*.yaml"):
                    print(f"  - {config_file}")
            else:
                print("  - No 'configurations' directory found")
            sys.exit(1)
        
        print(f"Starting Textile Color Analysis System...")
        print(f"Configuration: {config_path}")
        print(f"Log level: {args.log_level}")
        print(f"Output directory: {OUTPUT_DIR}")
        print("-" * 60)
        
        # Run main processing
        main(
            config_path=str(config_path),
            log_level=args.log_level
        )
        
        print("\n" + "=" * 60)
        print("✅ Processing completed successfully!")
        print(f"📁 Results saved to: {OUTPUT_DIR}")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n❌ Processing interrupted by user")
        sys.exit(1)
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        sys.exit(1)
        
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("\nFor detailed error information, check the log file in the output directory.")
        sys.exit(1)