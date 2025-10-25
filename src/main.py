# main.py (TEMİZLENMİŞ VE TÜM HATALARI DÜZELTİLMİŞ)

import sys
import os
import argparse
import pstats
import logging
import cProfile
import traceback
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from contextlib import contextmanager
import time

# Define absolute paths
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "output" # Varsayılan, __main__ içinde override edilebilir
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use('Agg')
import numpy as np
import cv2
from sklearn.preprocessing import MinMaxScaler

# Project imports
from src.data.load_data import load_config, load_data
from src.models.pso_dbn import DBN, DBNConfig, PSOConfig
from src.models.dbn_trainer import DBNTrainer, TrainConfig
from src.data.preprocess import Preprocessor, PreprocessingConfig
from src.models.segmentation.segmentation import (
    Segmenter, SegmentationConfig, ModelConfig,
    SegmentationError, InvalidConfigurationError
)
from src.utils.color.color_analysis import ColorMetricCalculator
from src.utils.image_utils import process_reference_image
from src.utils.color.color_conversion import convert_colors_to_cielab, convert_colors_to_cielab_dbn
from src.utils.output_manager import OutputManager

# Force TensorFlow to use CPU and reduce logging
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Function Definitions ---

def setup_logging(output_dir: Path, log_level: str = 'INFO'):
    """Setup comprehensive logging configuration."""
    logger_instance = logging.getLogger()
    if logger_instance.hasHandlers():
        logger_instance.handlers.clear()
        
    log_level_map = {'DEBUG': logging.DEBUG, 'INFO': logging.INFO, 'WARNING': logging.WARNING, 'ERROR': logging.ERROR}
    level = log_level_map.get(log_level.upper(), logging.INFO)
    detailed_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
    simple_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    logger_instance.setLevel(level)
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = output_dir / 'processing.log'
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger_instance.addHandler(file_handler)
    except Exception as e:
        print(f"FATAL ERROR: Could not create FileHandler at {log_file}. Error: {e}")
        
    if not any(isinstance(h, logging.StreamHandler) for h in logger_instance.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(simple_formatter)
        logger_instance.addHandler(console_handler)

    logging.info(f"Logging setup complete. Log file: {output_dir / 'processing.log'}")


@contextmanager
def timer(operation_name: str):
    """Context manager for timing operations."""
    start_time = time.perf_counter()
    log_func = logging.info
    log_func(f"Starting: {operation_name}")
    try:
        yield
    finally:
        duration = time.perf_counter() - start_time
        log_func(f"Completed: {operation_name} in {duration:.2f} seconds")


def safe_image_load(image_path: str) -> Optional[np.ndarray]:
    """Safely load an image with comprehensive error handling."""
    try:
        image_path_str = str(image_path)
        if not os.path.exists(image_path_str):
            logging.error(f"Image file does not exist: {image_path_str}")
            return None
        image = cv2.imread(image_path_str)
        if image is None:
            logging.error(f"OpenCV failed to load image (corrupt?): {image_path_str}")
            return None
        if image.size == 0:
            logging.error(f"Loaded image is empty: {image_path_str}")
            return None
        logging.debug(f"Successfully loaded image: {image_path_str}, shape: {image.shape}")
        return image
    except Exception as e:
        logging.error(f"Exception loading image {image_path_str}: {e}")
        return None


def validate_processing_config(config: Dict[str, Any]) -> bool:
    """Enhanced configuration validation specific to processing steps."""
    required_base_keys = ['reference_image_path', 'test_images']
    required_seg_keys = ['distance_threshold', 'predefined_k', 'k_values', 'som_values']
    try:
        missing_keys = [key for key in required_base_keys if key not in config]
        if missing_keys:
            raise InvalidConfigurationError(f"Missing required base config keys: {missing_keys}")
            
        seg_params = config.get('segmentation_params')
        if not seg_params or not isinstance(seg_params, dict):
             logging.warning("'segmentation_params' not found, checking root config...")
             missing_seg_keys = [key for key in required_seg_keys if key not in config]
             if missing_seg_keys:
                  raise InvalidConfigurationError(f"Missing required seg keys: {missing_seg_keys}")
             config['segmentation_params'] = {key: config[key] for key in required_seg_keys if key in config}
             seg_params = config['segmentation_params']
        else:
             missing_seg_keys = [key for key in required_seg_keys if key not in seg_params]
             if missing_seg_keys:
                  raise InvalidConfigurationError(f"Missing keys within 'segmentation_params': {missing_seg_keys}")

        ref_path_str = config['reference_image_path']
        ref_path = Path(ref_path_str)
        if not ref_path.is_absolute(): ref_path = (PROJECT_ROOT / ref_path).resolve()
        if not ref_path.exists(): raise FileNotFoundError(f"Reference image not found: {ref_path}")
        config['reference_image_path'] = str(ref_path)

        resolved_test_images = []
        missing_images = []
        for img_path_str in config.get('test_images', []):
            img_path = Path(img_path_str)
            if not img_path.is_absolute(): img_path = (PROJECT_ROOT / img_path).resolve()
            if not img_path.exists(): missing_images.append(img_path_str)
            else: resolved_test_images.append(str(img_path))
        if missing_images: raise FileNotFoundError(f"Test images not found: {missing_images}")
        config['test_images'] = resolved_test_images

        if seg_params['distance_threshold'] <= 0: raise ValueError("distance_threshold must be positive")
        if seg_params['predefined_k'] <= 0: raise ValueError("predefined_k must be positive")
        if not seg_params['k_values'] or min(seg_params['k_values']) <= 0: raise ValueError("k_values must be positive integers")
        if not seg_params['som_values'] or min(seg_params['som_values']) <= 0: raise ValueError("som_values must be positive integers")

        logging.info("Processing configuration validation passed")
        return True

    except (InvalidConfigurationError, FileNotFoundError, ValueError) as e:
        logging.error(f"Configuration validation failed: {e}")
        return False
    except Exception as e:
         logging.error(f"Unexpected error during configuration validation: {e}", exc_info=True)
         return False


def validate_loaded_data(rgb_data, lab_data, min_samples=100):
    """Validate that loaded data is suitable for training."""
    if rgb_data is None or lab_data is None or not isinstance(rgb_data, np.ndarray) or not isinstance(lab_data, np.ndarray):
        raise ValueError("Data loading failed or returned invalid type")
    if rgb_data.size == 0 or lab_data.size == 0:
        raise ValueError("Loaded data arrays are empty")
    if rgb_data.shape[0] != lab_data.shape[0]:
         raise ValueError(f"RGB ({rgb_data.shape[0]}) and LAB ({lab_data.shape[0]}) data must have the same number of samples.")
    if len(rgb_data.shape) != 2 or rgb_data.shape[1] % 3 != 0:
         raise ValueError(f"Loaded RGB data has unexpected shape {rgb_data.shape}.")
    total_pixels = rgb_data.shape[0] * (rgb_data.shape[1] // 3)
    if total_pixels < min_samples:
        raise ValueError(f"Insufficient total pixels for training: {total_pixels} < {min_samples}")
    logging.info(f"Data validation passed: {total_pixels} total pixels available from {rgb_data.shape[0]} images.")
    return True


def create_lab_converters(dbn, scaler_x, scaler_y, scaler_y_ab):
    """Create color converters using imported functions."""
    logging.debug("Creating LAB converters...")
    
    def lab_traditional_converter(rgb_color):
        try:
            lab_result = convert_colors_to_cielab([rgb_color])
            if lab_result.size == 0: raise ValueError("Conversion returned empty array")
            return lab_result[0]
        except Exception as e:
            logging.error(f"Traditional (skimage) LAB conversion failed for {rgb_color}: {e}")
            return np.array([50.0, 0.0, 0.0])

    def lab_dbn_converter(rgb_color):
        try:
            lab_result = convert_colors_to_cielab_dbn(
                dbn, scaler_x, scaler_y, scaler_y_ab, [rgb_color]
            )
            if lab_result.size == 0:
                 raise ValueError("DBN Conversion returned empty array")
            return lab_result[0]
        except Exception as e:
            logging.error(f"DBN LAB conversion failed for {rgb_color}: {e}")
            return np.array([50.0, 0.0, 0.0])
            
    logging.debug("LAB converters created.")
    return lab_traditional_converter, lab_dbn_converter


# --- Helper function definitions ---

def setup_pipeline_configs(config: Dict[str, Any]) -> Tuple[DBNConfig, PSOConfig, TrainConfig, PreprocessingConfig]:
    """Creates configuration objects from the main config dictionary."""
    logging.debug("Setting up pipeline configs...")
    dbn_cfg_dict = config.get('dbn_params', {})
    pso_cfg_dict = config.get('pso_params', {})
    train_cfg_dict = config.get('training_params', {})
    preproc_cfg_dict = config.get('preprocess_params', {})

    dbn_config = DBNConfig(**dbn_cfg_dict)
    pso_config = PSOConfig(**pso_cfg_dict)
    train_config = TrainConfig(**train_cfg_dict)
    preproc_cfg_dict['target_size'] = tuple(preproc_cfg_dict.get('target_size', [128, 128]))
    preprocess_config = PreprocessingConfig(**preproc_cfg_dict)
    
    logging.debug("Pipeline configs set up.")
    return dbn_config, pso_config, train_config, preprocess_config


def run_reference_processing(config: Dict[str, Any],
                             dbn: DBN,
                             scalers: Dict[str, MinMaxScaler],
                             output_manager: OutputManager,
                             preprocess_config: PreprocessingConfig
                             ) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
    """Loads, processes reference image, and returns formatted dicts."""
    logging.debug("Running reference processing...")
    reference_image_path = config['reference_image_path']
    seg_params = config.get('segmentation_params', {})
    k = seg_params.get('kmeans_clusters', 2)

    reference_kmeans_opt_dict = None
    reference_som_opt_dict = None
    target_colors_lab_final = np.array([]) 

    with timer("Reference image loading and processing"):
        scaler_x = scalers['scaler_x']
        scaler_y = scalers['scaler_y']
        scaler_y_ab = scalers['scaler_y_ab']

        logging.debug("Calling image_utils.process_reference_image...")
        kmeans_result_obj, som_result_obj, original_image, dpc_k = process_reference_image(
            reference_image_path, dbn, scaler_x, scaler_y, scaler_y_ab, k, preprocess_config
        )
        logging.debug(f"process_reference_image returned types: K({type(kmeans_result_obj)}), S({type(som_result_obj)})")

        if kmeans_result_obj and kmeans_result_obj.is_valid():
            logging.debug("[run_ref_proc] Received VALID kmeans_result_obj. Creating dict.")
            temp_kmeans_dict = None 
            temp_target_colors = np.array([]) 
            try:
                avg_colors_rgb = [tuple(c) for c in kmeans_result_obj.avg_colors]
                avg_colors_lab = convert_colors_to_cielab(avg_colors_rgb)
                avg_colors_lab_dbn = convert_colors_to_cielab_dbn(dbn, scaler_x, scaler_y, scaler_y_ab, avg_colors_rgb)
                
                lab_list_for_dict = avg_colors_lab.tolist() if isinstance(avg_colors_lab, np.ndarray) and avg_colors_lab.size > 0 else []
                lab_dbn_list_for_dict = avg_colors_lab_dbn.tolist() if isinstance(avg_colors_lab_dbn, np.ndarray) and avg_colors_lab_dbn.size > 0 else []

                temp_kmeans_dict = {
                    'original_image': kmeans_result_obj.segmented_image,
                    'segmented_image': kmeans_result_obj.segmented_image,
                    'avg_colors': avg_colors_rgb,
                    'avg_colors_lab': lab_list_for_dict,
                    'avg_colors_lab_dbn': lab_dbn_list_for_dict,
                    'labels': kmeans_result_obj.labels
                }
                if lab_list_for_dict:
                     temp_target_colors = np.array(lab_list_for_dict)
                logging.info("Formatted K-Means reference results.")
            except Exception as e:
                 logging.error(f"Error formatting K-Means reference result: {e}", exc_info=True)
            
            reference_kmeans_opt_dict = temp_kmeans_dict
            target_colors_lab_final = temp_target_colors
        else:
             logging.warning("[run_ref_proc] Received INVALID or None kmeans_result_obj.")

        if som_result_obj and som_result_obj.is_valid():
            logging.debug("[run_ref_proc] Received VALID som_result_obj. Creating dict.")
            temp_som_dict = None
            try:
                avg_colors_rgb = [tuple(c) for c in som_result_obj.avg_colors]
                avg_colors_lab = convert_colors_to_cielab(avg_colors_rgb)
                avg_colors_lab_dbn = convert_colors_to_cielab_dbn(dbn, scaler_x, scaler_y, scaler_y_ab, avg_colors_rgb)
                lab_list_for_dict = avg_colors_lab.tolist() if isinstance(avg_colors_lab, np.ndarray) and avg_colors_lab.size > 0 else []
                lab_dbn_list_for_dict = avg_colors_lab_dbn.tolist() if isinstance(avg_colors_lab_dbn, np.ndarray) and avg_colors_lab_dbn.size > 0 else []

                temp_som_dict = {
                     'original_image': som_result_obj.segmented_image,
                    'segmented_image': som_result_obj.segmented_image,
                    'avg_colors': avg_colors_rgb,
                    'avg_colors_lab': lab_list_for_dict,
                    'avg_colors_lab_dbn': lab_dbn_list_for_dict,
                    'labels': som_result_obj.labels
                }
                logging.info("Formatted SOM reference results.")
            except Exception as e:
                 logging.error(f"Error formatting SOM reference result: {e}", exc_info=True)
            reference_som_opt_dict = temp_som_dict
        else:
            logging.warning("[run_ref_proc] Received INVALID or None som_result_obj.")

        if reference_kmeans_opt_dict is None or target_colors_lab_final.size == 0:
             logging.error(f"K-Means dict is None ({reference_kmeans_opt_dict is None}) or target colors empty ({target_colors_lab_final.size == 0}).")
             raise ValueError("Failed to process reference image: K-Means segmentation failed or yielded no valid target colors.")

        if original_image is not None:
             output_manager.save_reference_summary(Path(reference_image_path).name, original_image)
             logging.info(f"Reference processed - Target LAB colors shape: {target_colors_lab_final.shape}, DPC k: {dpc_k}")
        else:
             logging.warning("Could not save reference summary, original image was None.")

    logging.debug("Reference processing finished.")
    return target_colors_lab_final, reference_kmeans_opt_dict, reference_som_opt_dict


def process_single_test_image(
    image_path: str, 
    image_data: np.ndarray, 
    config: Dict[str, Any], 
    preprocess_config: PreprocessingConfig, 
    dbn: DBN, 
    scalers: Dict[str, MinMaxScaler], 
    target_colors_lab: np.ndarray, 
    reference_kmeans_opt: Dict, 
    reference_som_opt: Dict, 
    output_manager: OutputManager,
    lab_traditional_converter,
    lab_dbn_converter
) -> List[Dict[str, Any]]:
    """Handles preprocessing, segmentation (both k-types), and Delta E calculation for one image."""
    
    image_name = Path(image_path).stem
    single_image_delta_e_results = []
    
    with timer(f"Processing test image {image_name}"):
        # Preprocess
        preprocessor = Preprocessor(config=preprocess_config)
        try:
            preprocessed_image = preprocessor.preprocess(image_data)
            if preprocessed_image is None:
                logging.error(f"Preprocessing returned None for {image_name}. Skipping.")
                return single_image_delta_e_results
            output_manager.save_preprocessed_image(image_name, preprocessed_image)
            unique_colors = len(np.unique(preprocessed_image.reshape(-1, 3), axis=0))
            logging.info(f"Preprocessing completed for {image_name} - unique colors: {unique_colors}")
        except Exception as e:
            logging.error(f"Preprocessing failed for {image_name}: {e}", exc_info=True)
            return single_image_delta_e_results

        # Extract common segmentation params
        seg_params = config.get('segmentation_params', {})
        distance_threshold = seg_params.get('distance_threshold', 0.7)
        predefined_k = seg_params.get('predefined_k', 2)
        k_values = seg_params.get('k_values', [2, 3, 4, 5])
        som_values = seg_params.get('som_values', [2, 3, 4, 5])
        dbscan_eps = seg_params.get('dbscan_eps', 10.0)
        dbscan_min_samples = seg_params.get('dbscan_min_samples', 5)
        segmentation_methods = seg_params.get('methods', ['kmeans_opt', 'kmeans_predef', 'som_opt', 'som_predef', 'dbscan'])

        logging.debug(f"Starting segmentation loop for {image_name}")
        for k_type in ['determined', 'predefined']:
            with timer(f"Segmentation ({image_name}) with k_type: {k_type}"):
                try:
                    logging.debug(f"Creating SegmentationConfig for k_type={k_type}")
                    seg_config = SegmentationConfig(
                        target_colors=target_colors_lab,
                        distance_threshold=distance_threshold,
                        predefined_k=predefined_k,
                        k_values=k_values,
                        som_values=som_values,
                        k_type=k_type,
                        methods=segmentation_methods,
                        dbscan_eps=dbscan_eps,
                        dbscan_min_samples=dbscan_min_samples
                    )
                    
                    logging.debug(f"Creating ModelConfig for k_type={k_type}")
                    model_config = ModelConfig(
                        dbn=dbn,
                        # --- 's' HATASI DÜZELTİLDİ ---
                        scalers=[scalers['scaler_x'], scalers['scaler_y'], scalers['scaler_y_ab']],
                        reference_kmeans_opt=reference_kmeans_opt,
                        reference_som_opt=reference_som_opt
                    )

                    logging.debug(f"Creating Segmenter for k_type={k_type}")
                    segmenter = Segmenter(preprocessed_image, seg_config, model_config, output_manager)
                    
                    logging.debug(f"Calling segmenter.process() for k_type={k_type}")
                    processing_result = segmenter.process()

                    if not processing_result.results:
                        logging.warning(f"No segmentation results for {image_name} (k_type={k_type})")
                        continue

                    logging.debug(f"Creating ColorMetricCalculator")
                    color_metric_calculator = ColorMetricCalculator(target_colors_lab)

                    logging.debug(f"Looping through segmentation results for k_type={k_type}")
                    for method_name, result in processing_result.results.items():
                        if not result.is_valid():
                            logging.warning(f"Invalid result for {method_name} on {image_name}")
                            continue
                        segmented_rgb_colors = result.avg_colors
                        if not segmented_rgb_colors:
                            logging.warning(f"No average RGB colors for {method_name} on {image_name}")
                            continue
                        
                        logging.debug(f"Calculating DeltaE for method: {method_name}")
                        try:
                            segmented_lab_traditional = convert_colors_to_cielab(segmented_rgb_colors)
                            segmented_lab_dbn = convert_colors_to_cielab_dbn(
                                dbn, scalers['scaler_x'], scalers['scaler_y'], scalers['scaler_y_ab'], segmented_rgb_colors
                            )

                            # --- .size yerine len() / .size KULLANILDI (Düzeltildi) ---
                            is_trad_empty = (not isinstance(segmented_lab_traditional, np.ndarray)) or (segmented_lab_traditional.size == 0)
                            is_dbn_empty = (not isinstance(segmented_lab_dbn, np.ndarray)) or (segmented_lab_dbn.size == 0)
                            
                            if is_trad_empty or is_dbn_empty:
                                 logging.warning(f"Color conversion failed (empty result) for {method_name} on {image_name}")
                                 continue
                            # --- DÜZELTME SONU ---

                            delta_e_traditional_list = color_metric_calculator.compute_all_delta_e(segmented_lab_traditional)
                            delta_e_dbn_list = color_metric_calculator.compute_all_delta_e(segmented_lab_dbn)

                            avg_delta_e_traditional = np.mean([d for d in delta_e_traditional_list if d != float('inf')]) if any(d != float('inf') for d in delta_e_traditional_list) else float('nan')
                            avg_delta_e_dbn = np.mean([d for d in delta_e_dbn_list if d != float('inf')]) if any(d != float('inf') for d in delta_e_dbn_list) else float('nan')

                            single_image_delta_e_results.append({
                                'dataset': output_manager.dataset_name, 
                                'image': image_name,
                                'method': method_name.replace('_opt', '').replace('_predef', ''),
                                'k_type': k_type,
                                'n_clusters': result.n_clusters,
                                'traditional_avg_delta_e': avg_delta_e_traditional,
                                'pso_dbn_avg_delta_e': avg_delta_e_dbn,
                                'processing_time': result.processing_time
                            })
                            logging.debug(f"DeltaE calculated and appended for: {method_name}")
                            logging.info(f"{method_name} ({k_type}) on {image_name}: "
                                       f"Avg ΔE Trad={avg_delta_e_traditional:.2f}, "
                                       f"Avg ΔE DBN={avg_delta_e_dbn:.2f}, "
                                       f"k={result.n_clusters}")

                        except Exception as e:
                            logging.error(f"Delta E calculation failed for {method_name} on {image_name}: {e}", exc_info=True)
                            continue
                    logging.debug(f"Finished loop through segmentation results for k_type={k_type}")
                
                except Exception as e:
                    logging.error(f"Segmentation failed for {image_name} with {k_type}: {type(e).__name__}: {e}", exc_info=True)
                    traceback.print_exc(file=sys.stdout)
                    continue 
        logging.debug(f"Finished segmentation loop for {image_name}") 
                    
    logging.debug(f"Finished processing single test image: {image_name}")
    return single_image_delta_e_results


def save_and_summarize_results(all_delta_e: List[Dict[str, Any]], output_manager: OutputManager):
    """Saves Delta E results and logs summary statistics."""
    with timer("Results saving and summary"):
        if not all_delta_e:
            logging.warning("No Delta E results were generated to save.")
            return
        try:
            dataset_name = output_manager.dataset_name
            output_manager.save_delta_e_results(dataset_name, all_delta_e)
            
            trad_values = [r['traditional_avg_delta_e'] for r in all_delta_e if 'traditional_avg_delta_e' in r and not np.isnan(r['traditional_avg_delta_e'])]
            dbn_values = [r['pso_dbn_avg_delta_e'] for r in all_delta_e if 'pso_dbn_avg_delta_e' in r and not np.isnan(r['pso_dbn_avg_delta_e'])]
            
            if trad_values and dbn_values:
                mean_trad = np.mean(trad_values)
                mean_dbn = np.mean(dbn_values)
                logging.info("--- Overall Results Summary ---")
                logging.info(f"  Avg Traditional ΔE: mean={mean_trad:.2f}, std={np.std(trad_values):.2f} ({len(trad_values)} results)")
                logging.info(f"  Avg PSO-DBN ΔE    : mean={mean_dbn:.2f}, std={np.std(dbn_values):.2f} ({len(dbn_values)} results)")
                if abs(mean_trad) > 1e-6:
                   improvement = (mean_trad - mean_dbn) / mean_trad * 100
                   logging.info(f"  Avg PSO-DBN Improvement: {improvement:.1f}%")
                logging.info("-----------------------------")
            else:
                 logging.warning("Could not calculate summary statistics (not enough valid Delta E values).")
            logging.info(f"Saved Delta E results for {len(all_delta_e)} entries.")
        except Exception as e:
            logging.error(f"Failed to save and summarize results: {e}", exc_info=True)


# --- Ana `main` fonksiyonu ---
def main(config_path='configurations/pattern_configs/block_config.yaml', log_level='INFO'):
    logging.debug(">>> Entered main() function <<<") 
    profiler = cProfile.Profile()
    profiler.enable()
    start_time = time.perf_counter()

    global OUTPUT_DIR 
    if OUTPUT_DIR is None:
         OUTPUT_DIR = (PROJECT_ROOT / "output").resolve()
         logging.warning(f"OUTPUT_DIR was None, setting default in main(): {OUTPUT_DIR}")

    try:
        logging.debug("Inside main try block, before setup.")
        # setup_logging __main__ bloğunda zaten çağrıldı
        
        logging.info("=" * 80 + "\nTEXTILE COLOR ANALYSIS SYSTEM - PROCESSING START\n" + "=" * 80)
        logging.info(f"Project Root: {PROJECT_ROOT}")
        logging.info(f"Output directory: {OUTPUT_DIR}")
        logging.info(f"Using configuration file: {config_path}")

        with timer("Configuration loading and validation"):
            config = load_config(config_path)
            if config is None: raise ValueError("Failed to load configuration.")
            if not validate_processing_config(config): raise ValueError("Processing configuration validation failed.")

        dbn_config, pso_config, train_config, preprocess_config = setup_pipeline_configs(config)
        dataset_name = Path(config_path).stem.replace('_config', '')
        output_manager = OutputManager(OUTPUT_DIR, dataset_name)
        logging.info(f"Processing dataset: {dataset_name}")

        test_image_paths = config['test_images']
        valid_test_images = []
        test_image_data = {}
        with timer("Loading all test images"):
             for image_path in test_image_paths:
                  image = safe_image_load(image_path)
                  if image is not None:
                       output_manager.save_test_image(Path(image_path).name, image)
                       valid_test_images.append(image_path)
                       test_image_data[image_path] = image
                  else:
                       logging.warning(f"Skipping invalid test image on initial load: {image_path}")
             if not valid_test_images: raise ValueError("No valid test images could be loaded.")
        logging.info(f"Successfully loaded {len(valid_test_images)} test images.")

        with timer("Loading training data"):
             load_data_target_size = tuple(config.get('load_data_resize', [100, 100]))
             rgb_data, lab_data = load_data(valid_test_images, target_size=load_data_target_size)
             validate_loaded_data(rgb_data, lab_data)

        with timer("DBN initialization and training"):
            trainer = DBNTrainer(dbn_config, pso_config, train_config)
            dbn, scalers = trainer.train(rgb_data, lab_data)
        
        target_colors_lab, ref_kmeans, ref_som = run_reference_processing(
            config, dbn, scalers, output_manager, preprocess_config
        )

        lab_traditional_converter, lab_dbn_converter = create_lab_converters(
            dbn, scalers['scaler_x'], scalers['scaler_y'], scalers['scaler_y_ab']
        )

        all_delta_e_results = []
        logging.debug("Starting loop over test images...")
        for image_idx, image_path in enumerate(valid_test_images):
            image_data_for_processing = test_image_data.get(image_path)
            if image_data_for_processing is None: 
                logging.warning(f"Skipping {Path(image_path).name} as its data wasn't loaded.")
                continue

            results_for_image = process_single_test_image(
                image_path=image_path,
                image_data=image_data_for_processing,
                config=config,
                preprocess_config=preprocess_config,
                dbn=dbn,
                scalers=scalers,
                target_colors_lab=target_colors_lab,
                reference_kmeans_opt=ref_kmeans,
                reference_som_opt=ref_som,
                output_manager=output_manager,
                lab_traditional_converter=lab_traditional_converter,
                lab_dbn_converter=lab_dbn_converter
            )
            all_delta_e_results.extend(results_for_image)
        logging.debug("Finished loop over test images.")

        save_and_summarize_results(all_delta_e_results, output_manager)

    except (ValueError, FileNotFoundError, InvalidConfigurationError, SegmentationError) as e:
        logging.error(f"Terminating due to known error: {e}", exc_info=True)
    except Exception as e:
        logging.error(f"An unexpected error occurred in main execution: {e}", exc_info=True)
    finally:
        logging.debug("Entering main finally block.")
        profiler.disable()
        total_time = time.perf_counter() - start_time
        logging.info("=" * 80 + f"\nPROCESSING COMPLETED IN {total_time:.2f} SECONDS\n" + "=" * 80)
        try:
            stats = pstats.Stats(profiler).sort_stats('cumtime')
            profile_path = OUTPUT_DIR / 'profile_stats.txt'
            with open(profile_path, 'w') as f:
                stats.stream = f
                stats.print_stats(30)
            logging.info(f"Profiling results saved to: {profile_path}")
        except Exception as e:
            logging.warning(f"Failed to save profiling results: {e}")
        logging.debug("Exiting main finally block.")


# --- if __name__ == "__main__": block ---
if __name__ == "__main__":
    print("DEBUG: >>> Script execution started (__name__ == '__main__') <<<")

    parser = argparse.ArgumentParser(
        description="Textile Color Analysis System...",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:\n  python src/main.py --config configurations/pattern_configs/block_config.yaml"""
    )
    parser.add_argument('--config', type=str, default='configurations/pattern_configs/block_config.yaml', help='Path to pattern config YAML')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', help='Set logging level')
    parser.add_argument('--output-dir', type=str, default=None, help='Override output directory')
    parser.add_argument('--profile', action='store_true', help='Enable detailed profiling')
    
    print("DEBUG: About to parse arguments...")
    args = parser.parse_args()
    print(f"DEBUG: Arguments parsed: {args}")

    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir).resolve()
        print(f"DEBUG: OUTPUT_DIR overridden by args: {OUTPUT_DIR}")
    else:
        OUTPUT_DIR = (PROJECT_ROOT / "output").resolve()
        print(f"DEBUG: Using default OUTPUT_DIR: {OUTPUT_DIR}")
    
    main_execution_successful = False
    
    try:
        print("DEBUG: Inside __main__ try block, before config path check.")
        config_path = Path(args.config)
        if not config_path.is_absolute(): config_path = (PROJECT_ROOT / config_path).resolve()
        if not config_path.exists():
             print(f"❌ Error: Configuration file not found: {config_path}")
             sys.exit(1)
        print(f"DEBUG: Config path resolved: {config_path}")

        default_cfg_path = PROJECT_ROOT / "configurations/defaults.yaml"
        if not default_cfg_path.exists(): print(f"⚠️ Warning: Default configuration '{default_cfg_path.relative_to(PROJECT_ROOT)}' not found.")
        else: print(f"DEBUG: Default config found: {default_cfg_path}")

        try: display_config_path = config_path.relative_to(PROJECT_ROOT)
        except ValueError: display_config_path = args.config
        try: display_output_dir = OUTPUT_DIR.relative_to(PROJECT_ROOT)
        except ValueError: display_output_dir = OUTPUT_DIR

        print(f"Starting Textile Color Analysis System...")
        print(f"Using Config: {display_config_path}")
        print(f"Log Level: {args.log_level}")
        print(f"Output Dir: {display_output_dir}")
        print("-" * 60)
        
        print("DEBUG: Calling setup_logging from __main__...")
        setup_logging(OUTPUT_DIR, args.log_level)
        print("DEBUG: setup_logging from __main__ finished.")

        print(f"DEBUG: >>> About to call main() function... <<<")
        main(config_path=str(config_path), log_level=args.log_level)
        print("DEBUG: <<< main() function finished >>>")
        
        main_execution_successful = True 

    except (ValueError, FileNotFoundError, InvalidConfigurationError, SegmentationError) as e:
        print(f"\nDEBUG: *** Specific error caught in __main__: {type(e).__name__}: {e} ***")
        logging.error(f"CRITICAL ERROR: {e}", exc_info=True)
        print(f"\n❌ Configuration or Processing Error: {e}\nCheck 'processing.log' in the output directory for details.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nDEBUG: *** KeyboardInterrupt caught in __main__ ***")
        logging.warning("Processing interrupted by user.")
        print("\n❌ Processing interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nDEBUG: *** Exception caught in __main__ try block: {type(e).__name__}: {e} ***")
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        traceback.print_exc(file=sys.stdout)
        print(f"\n❌ An unexpected error occurred: {e}\nCheck 'processing.log' in the output directory for details.")
        sys.exit(1)

    if main_execution_successful:
        try: display_output_dir_final = OUTPUT_DIR.relative_to(PROJECT_ROOT)
        except ValueError: display_output_dir_final = OUTPUT_DIR
        print("\n" + "=" * 60 + f"\n✅ Processing completed successfully!\n📁 Results saved to: {display_output_dir_final}\n" + "=" * 60)