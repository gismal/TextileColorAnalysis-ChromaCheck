# main.py (GÜNCELLENMİŞ VE TEMİZLENMİŞ SON HALİ)

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
# train_test_split artık DBNTrainer içinde kullanılıyor, buradan kaldırılabilir
# from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Project imports
from src.data.load_data import load_config, load_data # validate_config kaldırıldı
from src.models.pso_dbn import (
    DBN, DBNConfig, PSOConfig, convert_colors_to_cielab_dbn # convert_colors_to_cielab_dbn eklendi
)
from src.models.dbn_trainer import DBNTrainer, TrainConfig 
from src.data.preprocess import Preprocessor 
from src.models.segmentation.segmentation import (
    Segmenter, SegmentationConfig, ModelConfig, 
    SegmentationError, InvalidConfigurationError
)
from src.utils.color.color_analysis import ColorMetricCalculator
# ciede2000_distance artık Segmenter içinde kullanılıyor, buradan kaldırılabilir
# from src.utils.image_utils import process_reference_image, ciede2000_distance 
from src.utils.image_utils import process_reference_image # Sadece process_reference_image kaldı
# save_output muhtemelen OutputManager tarafından hallediliyor, kontrol edilmeli
# from src.utils.file_utils import save_output 
# save_reference_summary_plot muhtemelen OutputManager tarafından hallediliyor, kontrol edilmeli
# from src.utils.visualization import save_reference_summary_plot 
from src.utils.output_manager import OutputManager

# Force TensorFlow to use CPU and reduce logging
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Setup logging (Bu fonksiyon değişmedi)
def setup_logging(output_dir: Path, log_level: str = 'INFO'):
    """Setup comprehensive logging configuration."""
    log_level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR
    }
    level = log_level_map.get(log_level.upper(), logging.INFO)
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    log_file = output_dir / 'processing.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    logging.info(f"Logging setup complete. Log file: {log_file}")

# Timer context manager (Bu fonksiyon değişmedi)
@contextmanager
def timer(operation_name: str):
    """Context manager for timing operations."""
    start_time = time.perf_counter() # Daha hassas zamanlama için perf_counter kullanıldı
    try:
        logging.info(f"Starting: {operation_name}")
        yield
    finally:
        duration = time.perf_counter() - start_time
        logging.info(f"Completed: {operation_name} in {duration:.2f} seconds")

# create_scalers_dict fonksiyonu tamamen silindi

# safe_image_load fonksiyonu (Bu fonksiyon değişmedi)
def safe_image_load(image_path: str) -> Optional[np.ndarray]:
    """Safely load an image with comprehensive error handling."""
    try:
        image_path_str = str(image_path)
        if not os.path.exists(image_path_str):
            logging.error(f"Image file does not exist: {image_path_str}")
            return None
        image = cv2.imread(image_path_str)
        if image is None:
            logging.error(f"OpenCV failed to load image: {image_path_str}")
            return None
        if image.size == 0:
            logging.error(f"Loaded image is empty: {image_path_str}")
            return None
        logging.debug(f"Successfully loaded image: {image_path_str}, shape: {image.shape}")
        return image
    except Exception as e:
        logging.error(f"Exception loading image {image_path_str}: {e}")
        return None

# validate_processing_config fonksiyonu (Bu fonksiyon değişmedi)
def validate_processing_config(config: Dict[str, Any]) -> bool:
    """Enhanced configuration validation specific to processing steps."""
    required_keys = [
        'reference_image_path', 'test_images', 'distance_threshold',
        'kmeans_clusters', 'predefined_k', 'k_values', 'som_values'
    ]
    try:
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise InvalidConfigurationError(f"Missing required processing config keys: {missing_keys}")
        
        ref_path = Path(config['reference_image_path'])
        if not ref_path.exists():
             # Try resolving relative to project root if absolute fails
             ref_path_rel = PROJECT_ROOT / config['reference_image_path']
             if not ref_path_rel.exists():
                  raise FileNotFoundError(f"Reference image not found at '{config['reference_image_path']}' or relative to project root.")
             config['reference_image_path'] = str(ref_path_rel.resolve()) # Update config with absolute path
             
        resolved_test_images = []
        missing_images = []
        for img_path_str in config['test_images']:
            img_path = Path(img_path_str)
            if not img_path.exists():
                img_path_rel = PROJECT_ROOT / img_path_str
                if not img_path_rel.exists():
                    missing_images.append(img_path_str)
                else:
                    resolved_test_images.append(str(img_path_rel.resolve()))
            else:
                 resolved_test_images.append(str(img_path.resolve())) # Use absolute if exists
                 
        if missing_images:
            raise FileNotFoundError(f"Test images not found: {missing_images}")
        config['test_images'] = resolved_test_images # Update config with absolute paths

        if config['distance_threshold'] <= 0:
            raise ValueError("distance_threshold must be positive")
        if config['predefined_k'] <= 0:
            raise ValueError("predefined_k must be positive")
        if not config['k_values'] or min(config['k_values']) <= 0:
            raise ValueError("k_values must contain positive integers")
        if not config['som_values'] or min(config['som_values']) <= 0:
            raise ValueError("som_values must contain positive integers")
        
        # Check for segmentation params (needed by Segmenter)
        if 'segmentation_params' not in config:
             # Add default if missing, or raise error if needed
             logging.warning("Missing 'segmentation_params' in config, using defaults.")
             config['segmentation_params'] = {'dbscan_eps': 10.0, 'dbscan_min_samples': 5}
             
        logging.info("Processing configuration validation passed")
        return True
        
    except (InvalidConfigurationError, FileNotFoundError, ValueError) as e:
        logging.error(f"Configuration validation failed: {e}")
        return False
    except Exception as e: # Catch any other unexpected errors
         logging.error(f"Unexpected error during configuration validation: {e}", exc_info=True)
         return False

# validate_loaded_data fonksiyonu (Bu fonksiyon değişmedi)
def validate_loaded_data(rgb_data, lab_data, min_samples=100):
    """Validate that loaded data is suitable for training."""
    if rgb_data is None or lab_data is None or not isinstance(rgb_data, np.ndarray) or not isinstance(lab_data, np.ndarray):
        raise ValueError("Data loading failed or returned invalid type")
    if rgb_data.size == 0 or lab_data.size == 0:
        raise ValueError("Loaded data arrays are empty")
    if rgb_data.shape[0] != lab_data.shape[0]:
         raise ValueError("RGB and LAB data must have the same number of samples (images).")
         
    # Check total number of pixels (samples)
    # Assuming shape is (n_images, height*width*3)
    if len(rgb_data.shape) != 2 or rgb_data.shape[1] % 3 != 0:
         raise ValueError("Loaded RGB data has unexpected shape. Expected (n_images, n_pixels*3).")
         
    total_pixels = rgb_data.shape[0] * (rgb_data.shape[1] // 3)
    
    if total_pixels < min_samples:
        raise ValueError(f"Insufficient total pixels for training: {total_pixels} < {min_samples}")
    
    logging.info(f"Data validation passed: {total_pixels} total pixels available from {rgb_data.shape[0]} images.")
    return True

# create_lab_converters fonksiyonu (convert_colors_to_cielab_dbn import edildiği için çalışmalı)
def create_lab_converters(dbn, scaler_x, scaler_y, scaler_y_ab):
    """Create color converters using the imported function."""
    # Ensure scalers are the correct type (MinMaxScaler expected by convert_colors_to_cielab_dbn)
    if not all(isinstance(s, MinMaxScaler) for s in [scaler_x, scaler_y, scaler_y_ab]):
        logging.warning("One or more scalers passed to create_lab_converters are not MinMaxScaler. "
                        "convert_colors_to_cielab_dbn might expect MinMaxScaler.")

    def lab_traditional_converter(rgb_color):
        # ... (implementation unchanged) ...
        try:
            rgb_color = np.array(rgb_color, dtype=np.uint8).reshape(1, 1, 3)
            lab_array = cv2.cvtColor(rgb_color, cv2.COLOR_RGB2LAB)
            lab_result = lab_array[0, 0].astype(np.float32)
            lab_result[0] = lab_result[0] * 100.0 / 255.0
            lab_result[1:] = lab_result[1:] - 128.0
            return lab_result
        except Exception as e:
            logging.error(f"Traditional LAB conversion failed for {rgb_color}: {e}")
            return np.array([50.0, 0.0, 0.0])

    def lab_dbn_converter(rgb_color):
        # Calls the imported function
        try:
            # convert_colors_to_cielab_dbn expects a list of colors
            return convert_colors_to_cielab_dbn(
                dbn, scaler_x, scaler_y, scaler_y_ab, [rgb_color]
            )[0]
        except Exception as e:
            logging.error(f"DBN LAB conversion failed for {rgb_color}: {e}")
            return np.array([50.0, 0.0, 0.0])
            
    return lab_traditional_converter, lab_dbn_converter

# Ana `main` fonksiyonu
def main(config_path='configurations/pattern_configs/block_config.yaml', log_level='INFO'): # Varsayılan yol güncellendi
    """Main processing function."""
    profiler = cProfile.Profile()
    profiler.enable()
    start_time = time.perf_counter() # perf_counter kullanıldı

    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        setup_logging(OUTPUT_DIR, log_level)
        
        logging.info("=" * 80)
        logging.info("TEXTILE COLOR ANALYSIS SYSTEM - PROCESSING START")
        logging.info("=" * 80)
        logging.info(f"Project Root: {PROJECT_ROOT}")
        logging.info(f"Output directory: {OUTPUT_DIR}")
        logging.info(f"Using configuration file: {config_path}")
        
        # Load configuration using the updated load_config
        with timer("Configuration loading and validation"):
            config = load_config(config_path)
            if config is None:
                # load_config logs the error, just exit
                raise ValueError("Failed to load configuration. Check logs.")
            
            # validate_config kaldırıldı
            # Daha kapsamlı validate_processing_config kullanılıyor
            if not validate_processing_config(config):
                # validate_processing_config logs the error
                raise ValueError("Processing configuration validation failed. Check logs.")

        # Extract main config values (paths should now be absolute from validation)
        reference_image_path = config['reference_image_path']
        test_images = config['test_images']
        
        # Extract segmentation parameters (nested dict expected)
        seg_params = config.get('segmentation_params', {}) # Get nested dict
        distance_threshold = seg_params.get('distance_threshold', 0.7) # Get specific values
        k = seg_params.get('kmeans_clusters', 2)
        predefined_k = seg_params.get('predefined_k', 2)
        k_values = seg_params.get('k_values', [2, 3, 4, 5])
        som_values = seg_params.get('som_values', [2, 3, 4, 5])
        # Add DBSCAN params for SegmentationConfig
        dbscan_eps = seg_params.get('dbscan_eps', 10.0)
        dbscan_min_samples = seg_params.get('dbscan_min_samples', 5)
        segmentation_methods = seg_params.get('methods', ['kmeans_opt', 'kmeans_predef', 'som_opt', 'som_predef', 'dbscan'])

        # Derive dataset name from the specific config file path
        dataset_name = Path(config_path).stem.replace('_config', '')
        output_manager = OutputManager(OUTPUT_DIR, dataset_name)
        
        logging.info(f"Processing dataset: {dataset_name}")
        logging.info(f"Reference image: {reference_image_path}")
        logging.info(f"Test images count: {len(test_images)}")

        # Load reference image
        with timer("Reference image loading"):
            reference_image = safe_image_load(reference_image_path)
            if reference_image is None:
                raise ValueError(f"Failed to load reference image: {reference_image_path}")
            # Output manager might need adaptation if path format changed
            output_manager.save_reference_image(Path(reference_image_path).name, reference_image)

        # Load test images
        with timer("Test images loading"):
            valid_test_images = [] # Store paths of successfully loaded images
            test_image_data = {}   # Store loaded image data to pass to load_data
            for image_path in test_images:
                test_image = safe_image_load(image_path)
                if test_image is not None:
                    # Save copy using output manager
                    output_manager.save_test_image(Path(image_path).name, test_image)
                    valid_test_images.append(image_path)
                    test_image_data[image_path] = test_image # Store loaded data
                else:
                    logging.warning(f"Skipping invalid test image: {image_path}")
            if not valid_test_images:
                raise ValueError("No valid test images could be loaded.")
            logging.info(f"Successfully loaded {len(valid_test_images)} test images.")

        # Load training data (using only valid image paths)
        with timer("Training data loading"):
             # Pass target_size from config, or None to disable resize in load_data
             load_data_target_size = tuple(config.get('load_data_resize', [100, 100]))
             rgb_data, lab_data = load_data(valid_test_images, target_size=load_data_target_size)
             validate_loaded_data(rgb_data, lab_data) # Validate shape and content

        # Initialize and train DBN using DBNTrainer
        with timer("DBN initialization and training"):
            dbn_cfg_dict = config.get('dbn_params', {})
            pso_cfg_dict = config.get('pso_params', {})
            train_cfg_dict = config.get('training_params', {})

            dbn_config = DBNConfig(**dbn_cfg_dict)
            pso_config = PSOConfig(**pso_cfg_dict)
            train_config = TrainConfig(**train_cfg_dict)

            trainer = DBNTrainer(
                dbn_config=dbn_config,
                pso_config=pso_config,
                train_config=train_config
            )
            dbn, scalers = trainer.train(rgb_data, lab_data)
        
        scaler_x = scalers['scaler_x']
        scaler_y = scalers['scaler_y']
        scaler_y_ab = scalers['scaler_y_ab']

        # Process reference image to get target colors etc.
        with timer("Reference image processing"):
            # Pass k from segmentation_params
            reference_results = process_reference_image(
                reference_image_path, dbn, scaler_x, scaler_y, scaler_y_ab, k 
            )
            if not reference_results or len(reference_results) != 4 or reference_results[0] is None:
                raise ValueError("Failed to process reference image or received unexpected result format.")
                
            reference_kmeans_opt, reference_som_opt, original_image, dpc_k = reference_results
            
            # Ensure avg_colors_lab exists and is usable
            if 'avg_colors_lab' not in reference_kmeans_opt or not reference_kmeans_opt['avg_colors_lab']:
                 raise ValueError("Reference K-Means processing did not yield 'avg_colors_lab'.")
                 
            target_colors_lab = np.array(reference_kmeans_opt['avg_colors_lab'])
            output_manager.save_reference_summary(Path(reference_image_path).name, original_image) # Pass filename
            logging.info(f"Reference processed - Target LAB colors shape: {target_colors_lab.shape}, DPC k: {dpc_k}")

        # Initialize results collection
        all_delta_e = []

        # Create color converters (RGB -> LAB)
        lab_traditional_converter, lab_dbn_converter = create_lab_converters(
            dbn, scaler_x, scaler_y, scaler_y_ab
        )

        # Process test images individually
        for image_idx, image_path in enumerate(valid_test_images):
            image_name = Path(image_path).stem # Get filename without extension
            with timer(f"Processing test image {image_idx + 1}/{len(valid_test_images)} ({image_name})"):
                
                # Retrieve already loaded image data
                image = test_image_data.get(image_path)
                if image is None:
                    logging.error(f"Failed to retrieve loaded image data for: {image_path}. Attempting reload.")
                    image = safe_image_load(image_path) # Fallback to reload
                    if image is None:
                         logging.error(f"Fallback reload failed for: {image_path}. Skipping.")
                         continue

                # Preprocess the image
                preprocess_params = config.get('preprocess_params', {}) # Get params from config
                preprocessor = Preprocessor(
                    initial_resize=preprocess_params.get('initial_resize', 512),
                    target_size=tuple(preprocess_params.get('target_size', [128, 128])),
                    denoise_h=preprocess_params.get('denoise_h', 10),
                    max_colors=preprocess_params.get('max_colors', 8),
                    edge_enhance=preprocess_params.get('edge_enhance', False),
                    unsharp_amount=preprocess_params.get('unsharp_amount', 0.0),
                    unsharp_threshold=preprocess_params.get('unsharp_threshold', 0)
                )
                
                try:
                    preprocessed_image = preprocessor.preprocess(image)
                    if preprocessed_image is None:
                        logging.error(f"Preprocessing returned None for {image_name}. Skipping.")
                        continue
                    # Save using output manager
                    output_manager.save_preprocessed_image(image_name, preprocessed_image)
                    unique_colors = len(np.unique(preprocessed_image.reshape(-1, 3), axis=0))
                    logging.info(f"Preprocessing completed - unique colors: {unique_colors}")
                    
                except Exception as e:
                    logging.error(f"Preprocessing failed for {image_name}: {e}", exc_info=True)
                    continue

                # Test both k-determination strategies ('determined', 'predefined')
                for k_type in ['determined', 'predefined']:
                    with timer(f"Segmentation ({image_name}) with k_type: {k_type}"):
                        
                        try:
                            # Create SegmentationConfig dynamically for this k_type
                            seg_config = SegmentationConfig(
                                target_colors=target_colors_lab, # Use LAB colors
                                distance_threshold=distance_threshold,
                                predefined_k=predefined_k,
                                k_values=k_values,
                                som_values=som_values,
                                k_type=k_type,
                                methods=segmentation_methods, # Use methods from config
                                dbscan_eps=dbscan_eps,
                                dbscan_min_samples=dbscan_min_samples
                            )
                            
                            model_config = ModelConfig(
                                dbn=dbn,
                                scalers=[scaler_x, scaler_y, scaler_y_ab], # Pass as list
                                reference_kmeans_opt=reference_kmeans_opt,
                                reference_som_opt=reference_som_opt
                            )

                            # Perform segmentation using the Facade
                            segmenter = Segmenter(preprocessed_image, seg_config, model_config, output_manager)
                            processing_result = segmenter.process() # Returns ProcessingResult
                            
                            if not processing_result.results:
                                logging.warning(f"No segmentation results returned for {image_name} with k_type={k_type}")
                                continue

                            # Process results for Delta E calculation
                            # Target colors for calculator should be LAB
                            color_metric_calculator = ColorMetricCalculator(target_colors_lab) 
                            
                            for method_name, result in processing_result.results.items():
                                if not result.is_valid():
                                    logging.warning(f"Invalid result object for {method_name} on {image_name}")
                                    continue
                                    
                                # The segmentation result avg_colors are BGR/RGB from OpenCV mean
                                segmented_rgb_colors = result.avg_colors 
                                if not segmented_rgb_colors:
                                    logging.warning(f"No average RGB colors found for {method_name} on {image_name}")
                                    continue

                                try:
                                    # Convert segmented RGB colors to LAB using both methods
                                    segmented_lab_traditional = [lab_traditional_converter(rgb) for rgb in segmented_rgb_colors]
                                    segmented_lab_dbn = [lab_dbn_converter(rgb) for rgb in segmented_rgb_colors]

                                    # Find best matches based on LAB distances
                                    # We need to compute Delta E between segmented LAB and target LAB
                                    
                                    # Calculate Delta E for traditional conversion
                                    delta_e_traditional_list = color_metric_calculator.compute_all_delta_e(segmented_lab_traditional)
                                    avg_delta_e_traditional = np.mean(delta_e_traditional_list) if delta_e_traditional_list else float('nan')

                                    # Calculate Delta E for DBN conversion
                                    delta_e_dbn_list = color_metric_calculator.compute_all_delta_e(segmented_lab_dbn)
                                    avg_delta_e_dbn = np.mean(delta_e_dbn_list) if delta_e_dbn_list else float('nan')


                                    # Collect results
                                    all_delta_e.append({
                                        'dataset': dataset_name,
                                        'image': image_name,
                                        'method': method_name.replace('_opt', '').replace('_predef', ''),
                                        'k_type': k_type,
                                        'n_clusters': result.n_clusters,
                                        'traditional_avg_delta_e': avg_delta_e_traditional,
                                        'pso_dbn_avg_delta_e': avg_delta_e_dbn,
                                        'processing_time': result.processing_time
                                    })
                                    
                                    logging.info(f"{method_name} ({k_type}) on {image_name}: "
                                               f"Avg ΔE Trad={avg_delta_e_traditional:.2f}, "
                                               f"Avg ΔE DBN={avg_delta_e_dbn:.2f}, "
                                               f"k={result.n_clusters}")

                                except Exception as e:
                                    logging.error(f"Delta E calculation failed for {method_name} on {image_name}: {e}", exc_info=True)
                                    continue

                        except Exception as e:
                            logging.error(f"Segmentation failed for {image_name} with {k_type}: {e}", exc_info=True)
                            continue

        # Save Delta E results
        with timer("Results saving"):
            if all_delta_e:
                # Adjust save function if needed for new keys
                output_manager.save_delta_e_results(dataset_name, all_delta_e) 
                
                # Log summary statistics (using new keys)
                trad_values = [r['traditional_avg_delta_e'] for r in all_delta_e if not np.isnan(r['traditional_avg_delta_e'])]
                dbn_values = [r['pso_dbn_avg_delta_e'] for r in all_delta_e if not np.isnan(r['pso_dbn_avg_delta_e'])]
                
                if trad_values and dbn_values:
                    mean_trad = np.mean(trad_values)
                    mean_dbn = np.mean(dbn_values)
                    logging.info("--- Results Summary ---")
                    logging.info(f"  Avg Traditional ΔE: mean={mean_trad:.2f}, std={np.std(trad_values):.2f}")
                    logging.info(f"  Avg PSO-DBN ΔE    : mean={mean_dbn:.2f}, std={np.std(dbn_values):.2f}")
                    if mean_trad > 1e-6: # Avoid division by zero
                       improvement = (mean_trad - mean_dbn) / mean_trad * 100
                       logging.info(f"  Avg PSO-DBN Improvement: {improvement:.1f}%")
                    logging.info("-----------------------")
                
                logging.info(f"Saved Delta E results for {len(all_delta_e)} entries.")
            else:
                logging.warning("No Delta E results were generated to save.")

    except (ValueError, FileNotFoundError, InvalidConfigurationError, SegmentationError) as e:
        # Catch specific configuration/processing errors
        logging.error(f"Terminating due to error: {e}")
        # Optionally re-raise or handle differently
        # raise 
    except Exception as e:
        # Catch any other unexpected errors
        logging.error(f"An unexpected error occurred in main execution: {e}", exc_info=True)
        # raise # Re-raise to see full traceback if needed during debugging
    finally:
        profiler.disable()
        total_time = time.perf_counter() - start_time
        logging.info("=" * 80)
        logging.info(f"PROCESSING COMPLETED IN {total_time:.2f} SECONDS")
        logging.info("=" * 80)
        
        # Save profiling results
        try:
            stats = pstats.Stats(profiler).sort_stats('cumtime')
            profile_path = OUTPUT_DIR / 'profile_stats.txt'
            with open(profile_path, 'w') as f:
                stats.stream = f
                stats.print_stats(30) # Show more lines
            logging.info(f"Profiling results saved to: {profile_path}")
        except Exception as e:
            logging.warning(f"Failed to save profiling results: {e}")

# Command-line argument parsing (Bu kısım değişmedi, default path hariç)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Textile Color Analysis System using PSO-optimized DBN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main.py --config configurations/pattern_configs/block_config.yaml
  python src/main.py --config configurations/pattern_configs/flowers_config.yaml --log-level DEBUG
        """
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        # Varsayılan yol yeni yapıya göre güncellendi
        default='configurations/pattern_configs/block_config.yaml', 
        help='Path to the specific pattern configuration YAML file (e.g., configurations/pattern_configs/block_config.yaml)'
    )
    # ... (log-level, output-dir, profile argümanları aynı kaldı) ...
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', help='Set logging level')
    parser.add_argument('--output-dir', type=str, default=None, help='Override output directory')
    parser.add_argument('--profile', action='store_true', help='Enable detailed profiling output')
    
    args = parser.parse_args()
    
    # OUTPUT_DIR'ı args'a göre ayarla (opsiyonel)
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir).resolve()
    
    # Ana fonksiyonu çalıştırmadan önce temel kontroller
    try:
        config_path = Path(args.config)
        # Config dosyasının varlığını kontrol et (load_config da yapıyor ama burada erken fail etmek iyi olabilir)
        if not config_path.exists():
             # Try resolving relative to project root
             config_path_rel = PROJECT_ROOT / args.config
             if not config_path_rel.exists():
                  print(f"❌ Error: Configuration file not found at '{args.config}' or relative to project root '{PROJECT_ROOT}'.")
                  sys.exit(1)
             config_path = config_path_rel # Use resolved path
             
        # Check if default config exists (optional but good practice)
        default_cfg_path = Path("configurations/defaults.yaml")
        if not default_cfg_path.exists():
             print(f"⚠️ Warning: Default configuration '{default_cfg_path}' not found.")
             
        print(f"Starting Textile Color Analysis System...")
        print(f"Using Config: {config_path.relative_to(PROJECT_ROOT)}") # Projeye göre relative path göster
        print(f"Log Level: {args.log_level}")
        print(f"Output Dir: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")
        print("-" * 60)
        
        main(config_path=str(config_path), log_level=args.log_level)
        
        print("\n" + "=" * 60)
        print(f"✅ Processing completed successfully!")
        print(f"📁 Results saved to: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")
        print("=" * 60)
        
    except (ValueError, FileNotFoundError, InvalidConfigurationError, SegmentationError) as e:
        print(f"\n❌ Configuration or Processing Error: {e}")
        print("Check the log file in the output directory for details.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n❌ Processing interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        # Print traceback for unexpected errors during development
        # traceback.print_exc() 
        print("Check the log file in the output directory for detailed error information.")
        sys.exit(1)
        