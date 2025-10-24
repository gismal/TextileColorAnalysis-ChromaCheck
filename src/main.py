# main.py (DEBUG PRINT'LER EKLENMİŞ TAM HALİ)

print("DEBUG: Script starting, trying imports...") # <-- EN BAŞA EKLENDİ

import sys
import os
import argparse
import pstats
import logging
import cProfile
import traceback
from pathlib import Path
from dataclasses import dataclass, field # field importunu da kontrol edelim
from typing import List, Optional, Dict, Any, Tuple
from contextlib import contextmanager
import time

print("DEBUG: Basic imports successful.") # <-- EKLENDİ

# Define absolute paths
try:
    SCRIPT_DIR = Path(__file__).parent.absolute()
    PROJECT_ROOT = SCRIPT_DIR.parent
    # OUTPUT_DIR'ı __main__ bloğunda args'a göre ayarlayacağız
    sys.path.insert(0, str(PROJECT_ROOT))
    print(f"DEBUG: PROJECT_ROOT determined: {PROJECT_ROOT}") # <-- EKLENDİ
except Exception as e:
    print(f"FATAL ERROR determining paths: {e}")
    sys.exit(1)

# Import other libraries (potential hang points)
try:
    print("DEBUG: Importing matplotlib...") # <-- EKLENDİ
    import matplotlib
    matplotlib.use('Agg')
    print("DEBUG: Importing numpy...") # <-- EKLENDİ
    import numpy as np
    print("DEBUG: Importing cv2...") # <-- EKLENDİ
    import cv2
    print("DEBUG: Importing sklearn...") # <-- EKLENDİ
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    print("DEBUG: External library imports successful.") # <-- EKLENDİ
except ImportError as e:
    print(f"FATAL ERROR during external library import: {e}")
    sys.exit(1)
except Exception as e:
    print(f"FATAL UNEXPECTED ERROR during external library import: {e}")
    sys.exit(1)


# Project imports (potential hang points or import errors)
try:
    print("DEBUG: Importing project modules...") # <-- EKLENDİ
    from src.data.load_data import load_config, load_data
    from src.models.pso_dbn import DBN, DBNConfig, PSOConfig, convert_colors_to_cielab_dbn
    from src.models.dbn_trainer import DBNTrainer, TrainConfig
    from src.data.preprocess import Preprocessor, PreprocessingConfig # PreprocessingConfig'i buradan import etmeliyiz
    from src.models.segmentation.segmentation import (
        Segmenter, SegmentationConfig, ModelConfig,
        SegmentationError, InvalidConfigurationError
    )
    from src.utils.color.color_analysis import ColorMetricCalculator
    from src.utils.image_utils import process_reference_image
    from src.utils.color.color_conversion import convert_colors_to_cielab
    from src.utils.output_manager import OutputManager
    print("DEBUG: Project module imports successful.") # <-- EKLENDİ
except ImportError as e:
    print(f"FATAL ERROR during project module import: {e}")
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"FATAL UNEXPECTED ERROR during project module import: {e}")
    traceback.print_exc()
    sys.exit(1)

# --- Define Global OUTPUT_DIR later in __main__ ---
OUTPUT_DIR = None # Initialize as None

# Force TensorFlow to use CPU and reduce logging
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print("DEBUG: Environment variables set.") # <-- EKLENDİ

# --- Function Definitions ---

def setup_logging(output_dir: Path, log_level: str = 'INFO'):
    print(f"DEBUG: setup_logging called with output_dir={output_dir}, level={log_level}")
    logger_instance = logging.getLogger()
    if logger_instance.hasHandlers():
        print("DEBUG: Logger already has handlers, clearing them.")
        # Bu satır bazen sorun yaratabilir, özellikle stream handler kapanırsa
        # logger_instance.handlers.clear() # Şimdilik yoruma alalım
        # Alternatif: Sadece file handler'ları temizle? Veya hiç temizleme?
        # Şimdilik temizlemeden devam edelim, tekrar çağrılmazsa sorun olmaz.
        pass

    log_level_map = {'DEBUG': logging.DEBUG, 'INFO': logging.INFO, 'WARNING': logging.WARNING, 'ERROR': logging.ERROR}
    level = log_level_map.get(log_level.upper(), logging.INFO)
    detailed_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
    simple_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    logger_instance.setLevel(level)

    # File handler
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = output_dir / 'processing.log'
        print(f"DEBUG: Attempting to create FileHandler for: {log_file}")
        # Check if file handler already exists
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file) for h in logger_instance.handlers):
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(detailed_formatter)
            logger_instance.addHandler(file_handler)
            print("DEBUG: FileHandler added.")
        else:
            print("DEBUG: FileHandler already exists for this file.")
    except Exception as e:
        print(f"FATAL ERROR creating FileHandler: {e}")
        # sys.exit(1) # Loglama kritikse çıkışı zorla

    # Console handler
    print("DEBUG: Attempting to create StreamHandler.")
    # Check if stream handler already exists
    if not any(isinstance(h, logging.StreamHandler) for h in logger_instance.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(simple_formatter)
        logger_instance.addHandler(console_handler)
        print("DEBUG: StreamHandler added.")
    else:
         print("DEBUG: StreamHandler already exists.")

    try:
         logging.info(f"Logging setup complete. Log file: {log_file}")
         print("DEBUG: Initial log message sent via logger.")
    except NameError: # log_file tanımlı değilse (hata durumunda)
         print("ERROR sending initial log message: Log file path not defined due to earlier error.")
    except Exception as e:
         print(f"ERROR sending initial log message: {e}")


@contextmanager
def timer(operation_name: str):
    start_time = time.perf_counter()
    # Loglama başlamamışsa print kullan
    log_func = logging.info if logging.getLogger().hasHandlers() else print
    log_func(f"Starting: {operation_name}")
    try:
        yield
    finally:
        duration = time.perf_counter() - start_time
        log_func = logging.info if logging.getLogger().hasHandlers() else print
        log_func(f"Completed: {operation_name} in {duration:.2f} seconds")


def safe_image_load(image_path: str) -> Optional[np.ndarray]:
    try:
        image_path_str = str(image_path)
        if not os.path.exists(image_path_str):
            # Print yerine logging kullanmak daha iyi ama loglama başlamamış olabilir
            print(f"ERROR: Image file does not exist: {image_path_str}")
            # logging.error(f"Image file does not exist: {image_path_str}")
            return None
        image = cv2.imread(image_path_str)
        if image is None:
            print(f"ERROR: OpenCV failed to load image: {image_path_str}")
            # logging.error(f"OpenCV failed to load image: {image_path_str}")
            return None
        if image.size == 0:
            print(f"ERROR: Loaded image is empty: {image_path_str}")
            # logging.error(f"Loaded image is empty: {image_path_str}")
            return None
        # logging.debug(f"Successfully loaded image: {image_path_str}, shape: {image.shape}")
        return image
    except Exception as e:
        print(f"ERROR: Exception loading image {image_path_str}: {e}")
        # logging.error(f"Exception loading image {image_path_str}: {e}")
        return None

def validate_processing_config(config: Dict[str, Any]) -> bool:
    print("DEBUG: Entering validate_processing_config...")
    # Anahtarları daha esnek kontrol edelim
    required_base_keys = ['reference_image_path', 'test_images']
    required_seg_keys = ['distance_threshold', 'predefined_k', 'k_values', 'som_values']
    try:
        missing_keys = [key for key in required_base_keys if key not in config]
        if missing_keys:
            raise InvalidConfigurationError(f"Missing required base config keys: {missing_keys}")

        seg_params = config.get('segmentation_params')
        if not seg_params or not isinstance(seg_params, dict):
             # Geriye dönük uyumluluk için ana config'de arayalım
             print("DEBUG: 'segmentation_params' not found or not a dict, checking root config...")
             missing_seg_keys = [key for key in required_seg_keys if key not in config]
             if missing_seg_keys:
                  raise InvalidConfigurationError(f"Missing required segmentation keys (checked root and 'segmentation_params'): {missing_seg_keys}")
             # Ana config'den alınan değerleri seg_params'a taşıyalım (isteğe bağlı)
             config['segmentation_params'] = {key: config[key] for key in required_seg_keys if key in config}
             seg_params = config['segmentation_params']
             print("DEBUG: Found segmentation keys in root, moved to 'segmentation_params'.")
        else:
             missing_seg_keys = [key for key in required_seg_keys if key not in seg_params]
             if missing_seg_keys:
                  raise InvalidConfigurationError(f"Missing required keys within 'segmentation_params': {missing_seg_keys}")

        # Path validation
        ref_path_str = config['reference_image_path']
        ref_path = Path(ref_path_str)
        if not ref_path.is_absolute(): ref_path = (PROJECT_ROOT / ref_path).resolve()
        if not ref_path.exists(): raise FileNotFoundError(f"Reference image not found: {ref_path}")
        config['reference_image_path'] = str(ref_path)
        print(f"DEBUG: Validated reference path: {ref_path}")

        resolved_test_images = []
        missing_images = []
        for img_path_str in config.get('test_images', []):
            img_path = Path(img_path_str)
            if not img_path.is_absolute(): img_path = (PROJECT_ROOT / img_path).resolve()
            if not img_path.exists(): missing_images.append(img_path_str)
            else: resolved_test_images.append(str(img_path))
        if missing_images: raise FileNotFoundError(f"Test images not found: {missing_images}")
        config['test_images'] = resolved_test_images
        print(f"DEBUG: Validated {len(resolved_test_images)} test image paths.")

        # Numeric validation (basic)
        if seg_params['distance_threshold'] <= 0: raise ValueError("distance_threshold must be positive")
        if seg_params['predefined_k'] <= 0: raise ValueError("predefined_k must be positive")

        print("DEBUG: validate_processing_config finished successfully.")
        # Logging başlamış olmalı
        logging.info("Processing configuration validation passed")
        return True

    except (InvalidConfigurationError, FileNotFoundError, ValueError) as e:
        print(f"DEBUG: validate_processing_config FAILED: {e}")
        # Logging başlamış olmalı
        logging.error(f"Configuration validation failed: {e}")
        return False
    except Exception as e:
         print(f"DEBUG: validate_processing_config FAILED unexpectedly: {e}")
         logging.error(f"Unexpected error during configuration validation: {e}", exc_info=True)
         return False


def validate_loaded_data(rgb_data, lab_data, min_samples=100):
    print("DEBUG: Validating loaded data...") # <-- EKLENDİ
    if rgb_data is None or lab_data is None or not isinstance(rgb_data, np.ndarray) or not isinstance(lab_data, np.ndarray):
        raise ValueError("Data loading failed or returned invalid type")
    if rgb_data.size == 0 or lab_data.size == 0:
        raise ValueError("Loaded data arrays are empty")
    if rgb_data.shape[0] != lab_data.shape[0]:
        raise ValueError(f"RGB ({rgb_data.shape[0]}) and LAB ({lab_data.shape[0]}) data must have the same number of samples (images).")

    if len(rgb_data.shape) != 2 or rgb_data.shape[1] % 3 != 0:
        raise ValueError(f"Loaded RGB data has unexpected shape {rgb_data.shape}. Expected (n_images, n_pixels*3).")
    if len(lab_data.shape) != 2 or lab_data.shape[1] % 3 != 0:
         raise ValueError(f"Loaded LAB data has unexpected shape {lab_data.shape}. Expected (n_images, n_pixels*3).")
    if rgb_data.shape[1] != lab_data.shape[1]:
         raise ValueError(f"Flattened RGB ({rgb_data.shape[1]}) and LAB ({lab_data.shape[1]}) data must have the same number of features (pixels*3).")

    total_pixels = rgb_data.shape[0] * (rgb_data.shape[1] // 3)
    if total_pixels < min_samples:
        raise ValueError(f"Insufficient total pixels for training: {total_pixels} < {min_samples}")

    print("DEBUG: Loaded data validation passed.") # <-- EKLENDİ
    logging.info(f"Data validation passed: {total_pixels} total pixels available from {rgb_data.shape[0]} images.")
    return True

# create_lab_converters (içi aynı, print eklendi)
def create_lab_converters(dbn, scaler_x, scaler_y, scaler_y_ab):
    print("DEBUG: Creating LAB converters...")
    if not all(isinstance(s, MinMaxScaler) for s in [scaler_x, scaler_y, scaler_y_ab]):
        logging.warning("...") # Mesaj aynı

    def lab_traditional_converter(rgb_color):
        try:
            # color_conversion'daki fonksiyonu kullanıyoruz
            lab_result = convert_colors_to_cielab([rgb_color])
            if lab_result.size == 0: raise ValueError("Conversion returned empty array")
            return lab_result[0] # Return the single LAB tuple
        except Exception as e:
            logging.error(f"Traditional (skimage) LAB conversion failed for {rgb_color}: {e}")
            return np.array([50.0, 0.0, 0.0])

    def lab_dbn_converter(rgb_color):
        try:
            # color_conversion'daki fonksiyonu kullanıyoruz
            lab_result = convert_colors_to_cielab_dbn(dbn, scaler_x, scaler_y, scaler_y_ab, [rgb_color])
            if lab_result.size == 0: raise ValueError("DBN Conversion returned empty array")
            return lab_result[0] # Return the single LAB tuple
        except Exception as e:
            logging.error(f"DBN LAB conversion failed for {rgb_color}: {e}")
            return np.array([50.0, 0.0, 0.0])

    print("DEBUG: LAB converters created.")
    return lab_traditional_converter, lab_dbn_converter

# --- Helper function definitions ---

def setup_pipeline_configs(config: Dict[str, Any]) -> Tuple[DBNConfig, PSOConfig, TrainConfig, PreprocessingConfig]:
    print("DEBUG: Setting up pipeline configs...")
    dbn_cfg_dict = config.get('dbn_params', {})
    pso_cfg_dict = config.get('pso_params', {})
    train_cfg_dict = config.get('training_params', {})
    preproc_cfg_dict = config.get('preprocess_params', {})

    dbn_config = DBNConfig(**dbn_cfg_dict)
    pso_config = PSOConfig(**pso_cfg_dict)
    train_config = TrainConfig(**train_cfg_dict)
    # Ensure target_size is tuple
    preproc_cfg_dict['target_size'] = tuple(preproc_cfg_dict.get('target_size', [128, 128]))
    preprocess_config = PreprocessingConfig(**preproc_cfg_dict)

    print("DEBUG: Pipeline configs set up.")
    return dbn_config, pso_config, train_config, preprocess_config

# src/main.py İÇİNDE SADECE BU FONKSİYONU DEĞİŞTİR

def run_reference_processing(config: Dict[str, Any],
                             dbn: DBN,
                             scalers: Dict[str, MinMaxScaler],
                             output_manager: OutputManager,
                             preprocess_config: PreprocessingConfig
                             ) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
    print("DEBUG: Running reference processing...")
    reference_image_path = config['reference_image_path']
    seg_params = config.get('segmentation_params', {})
    k = seg_params.get('kmeans_clusters', config.get('kmeans_clusters', 2))

    reference_kmeans_opt_dict = None
    reference_som_opt_dict = None
    target_colors_lab_final = np.array([])

    with timer("Reference image loading and processing"):
        scaler_x = scalers['scaler_x']
        scaler_y = scalers['scaler_y']
        scaler_y_ab = scalers['scaler_y_ab']

        print("DEBUG: Calling image_utils.process_reference_image...")
        kmeans_result_obj, som_result_obj, original_image, dpc_k = process_reference_image(
            reference_image_path, dbn, scaler_x, scaler_y, scaler_y_ab, k, preprocess_config
        )
        print(f"DEBUG: process_reference_image returned types: K({type(kmeans_result_obj)}), S({type(som_result_obj)}), I({type(original_image)}), D({type(dpc_k)})")

        # --- K-Means Sonucunu İşle ---
        if kmeans_result_obj and kmeans_result_obj.is_valid():
            print("DEBUG: [run_ref_proc] Received VALID kmeans_result_obj. Trying to create dict.")
            temp_kmeans_dict = None # Geçici değişken
            temp_target_colors = np.array([]) # Geçici değişken
            try:
                avg_colors_rgb = [tuple(c) for c in kmeans_result_obj.avg_colors]
                print(f"DEBUG: [run_ref_proc] Input avg_colors_rgb for K-Means: {avg_colors_rgb}")
                avg_colors_lab = convert_colors_to_cielab(avg_colors_rgb)
                print(f"DEBUG: [run_ref_proc] Output avg_colors_lab: {type(avg_colors_lab)}, Shape: {getattr(avg_colors_lab, 'shape', 'N/A')}")
                avg_colors_lab_dbn = convert_colors_to_cielab_dbn(dbn, scaler_x, scaler_y, scaler_y_ab, avg_colors_rgb)
                print(f"DEBUG: [run_ref_proc] Output avg_colors_lab_dbn: {type(avg_colors_lab_dbn)}, Shape: {getattr(avg_colors_lab_dbn, 'shape', 'N/A')}")

                if not isinstance(avg_colors_lab, np.ndarray) or avg_colors_lab.size == 0:
                     print("DEBUG: [run_ref_proc] convert_colors_to_cielab returned empty!")
                     lab_list_for_dict = []
                else:
                     lab_list_for_dict = avg_colors_lab.tolist()

                lab_dbn_list_for_dict = avg_colors_lab_dbn.tolist() if isinstance(avg_colors_lab_dbn, np.ndarray) and avg_colors_lab_dbn.size > 0 else []

                # Sözlüğü geçici değişkene ata
                temp_kmeans_dict = {
                    'original_image': kmeans_result_obj.segmented_image,
                    'segmented_image': kmeans_result_obj.segmented_image,
                    'avg_colors': avg_colors_rgb,
                    'avg_colors_lab': lab_list_for_dict,
                    'avg_colors_lab_dbn': lab_dbn_list_for_dict,
                    'labels': kmeans_result_obj.labels
                }
                
                # Hedef renkleri geçici değişkene ata
                if lab_list_for_dict:
                     temp_target_colors = np.array(lab_list_for_dict)

                print("DEBUG: [run_ref_proc] Successfully created temp_kmeans_dict and temp_target_colors.") # <-- YENİ PRINT
                logging.info("Formatted K-Means reference results.")

            except Exception as e:
                 # --- YENİ DEBUG PRINT BURADA ---
                 print(f"DEBUG: [run_ref_proc] *** EXCEPTION caught while formatting K-Means dict: {type(e).__name__}: {e} ***")
                 logging.error(f"Error formatting K-Means reference result: {e}", exc_info=True)
                 # temp değişkenler değişmez (None ve boş array kalır)

            # Try bloğundan sonra atama yap
            reference_kmeans_opt_dict = temp_kmeans_dict
            target_colors_lab_final = temp_target_colors

        else: # kmeans_result_obj geçerli değilse
             print("DEBUG: [run_ref_proc] Received INVALID or None kmeans_result_obj.")
             # reference_kmeans_opt_dict ve target_colors_lab_final başlangıç değerlerinde kalır (None, boş array)

        # --- SOM Sonucunu İşle (Benzer yapı) ---
        if som_result_obj and som_result_obj.is_valid():
            print("DEBUG: [run_ref_proc] Received VALID som_result_obj. Trying to create dict.")
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
                print("DEBUG: [run_ref_proc] Successfully created temp_som_dict.") # <-- YENİ PRINT
                logging.info("Formatted SOM reference results.")
            except Exception as e:
                 print(f"DEBUG: [run_ref_proc] *** EXCEPTION caught while formatting SOM dict: {type(e).__name__}: {e} ***") # <-- YENİ PRINT
                 logging.error(f"Error formatting SOM reference result: {e}", exc_info=True)
                 # temp_som_dict None olarak kalır
                 
            reference_som_opt_dict = temp_som_dict # Try bloğundan sonra ata
        else:
            print("DEBUG: [run_ref_proc] Received INVALID or None som_result_obj.")
            # reference_som_opt_dict None olarak kalır


        # --- Hata Kontrolü ---
        if reference_kmeans_opt_dict is None or target_colors_lab_final.size == 0:
             print(f"DEBUG: [run_ref_proc] FINAL CHECK FAILED: K-Means dict is None ({reference_kmeans_opt_dict is None}) or target colors empty ({target_colors_lab_final.size == 0}). Raising ValueError.")
             raise ValueError("Failed to process reference image: K-Means segmentation failed or yielded no valid target colors.")

        # Orijinal görüntüyü kaydet
        if original_image is not None:
             output_manager.save_reference_summary(Path(reference_image_path).name, original_image)
             logging.info(f"Reference processed - Target LAB colors shape: {target_colors_lab_final.shape}, DPC k: {dpc_k}")
        else:
             logging.warning("Could not save reference summary, original image was None.")

    print("DEBUG: Reference processing finished.")
    # reference_som_opt_dict None olsa bile döndür
    return target_colors_lab_final, reference_kmeans_opt_dict, reference_som_opt_dict

# --- main() fonksiyonu içindeki çağırma aynı ---
# target_colors_lab, ref_kmeans, ref_som = run_reference_processing(...)

def process_single_test_image(
    image_path: str,
    image_data: np.ndarray,
    config: Dict[str, Any],
    preprocess_config: PreprocessingConfig, # Use the config object
    dbn: DBN,
    scalers: Dict[str, MinMaxScaler],
    target_colors_lab: np.ndarray,
    reference_kmeans_opt: Dict,
    reference_som_opt: Dict,
    output_manager: OutputManager,
    lab_traditional_converter,
    lab_dbn_converter
) -> List[Dict[str, Any]]:

    image_name = Path(image_path).stem
    print(f"DEBUG: Processing single test image: {image_name}")
    single_image_delta_e_results = []
    with timer(f"Processing test image {image_name}"):
        print(f"DEBUG: Preprocessing image: {image_name}") # <-- EKLENDİ
        preprocessor = Preprocessor(preprocess_config) # Pass the config object
        try:
            preprocessed_image = preprocessor.preprocess(image_data)
            if preprocessed_image is None:
                logging.error(f"Preprocessing returned None for {image_name}. Skipping.")
                return single_image_delta_e_results
            output_manager.save_preprocessed_image(image_name, preprocessed_image)
            # unique_colors = len(np.unique(preprocessed_image.reshape(-1, 3), axis=0))
            # logging.info(f"Preprocessing completed for {image_name} - unique colors: {unique_colors}")
            print(f"DEBUG: Preprocessing finished for {image_name}") # <-- EKLENDİ
        except Exception as e:
            logging.error(f"Preprocessing failed for {image_name}: {e}", exc_info=True)
            return single_image_delta_e_results

        # Extract segmentation params
        seg_params = config.get('segmentation_params', {})
        distance_threshold = seg_params.get('distance_threshold', 0.7)
        predefined_k = seg_params.get('predefined_k', 2)
        k_values = seg_params.get('k_values', [2, 3, 4, 5])
        som_values = seg_params.get('som_values', [2, 3, 4, 5])
        dbscan_eps = seg_params.get('dbscan_eps', 10.0)
        dbscan_min_samples = seg_params.get('dbscan_min_samples', 5)
        segmentation_methods = seg_params.get('methods', ['kmeans_opt', 'kmeans_predef', 'som_opt', 'som_predef', 'dbscan'])

        print(f"DEBUG: Starting segmentation loop for {image_name}") # <-- EKLENDİ
        for k_type in ['determined', 'predefined']:
            with timer(f"Segmentation ({image_name}) with k_type: {k_type}"):
                try:
                    print(f"DEBUG: Values before SegmentationConfig init: k_type={k_type}, "
                              f"dist_thresh={distance_threshold}, predef_k={predefined_k}, "
                              f"k_vals={k_values}, som_vals={som_values}, methods={segmentation_methods}, "
                              f"db_eps={dbscan_eps}, db_min={dbscan_min_samples}")
                    
                    print(f"DEBUG: Creating SegmentationConfig for k_type={k_type}") # <-- EKLENDİ
                    
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
                        scalers=[scalers['scaler_x'], scalers['scaler_y'], scalers['s']], # Pass as list
                        reference_kmeans_opt=reference_kmeans_opt,
                        reference_som_opt=reference_som_opt
                    )
                    print(f"DEBUG: Creating Segmenter for k_type={k_type}") # <-- EKLENDİ
                    segmenter = Segmenter(preprocessed_image, seg_config, model_config, output_manager)
                    print(f"DEBUG: Calling segmenter.process() for k_type={k_type}") # <-- EKLENDİ
                    processing_result = segmenter.process()
                    print(f"DEBUG: segmenter.process() finished for k_type={k_type}") # <-- EKLENDİ

                    # ... (results check) ...

                    print(f"DEBUG: Creating ColorMetricCalculator") # <-- EKLENDİ
                    color_metric_calculator = ColorMetricCalculator(target_colors_lab)

                    print(f"DEBUG: Looping through segmentation results for k_type={k_type}") # <-- EKLENDİ
                    for method_name, result in processing_result.results.items():
                        # ... (validation) ...
                        print(f"DEBUG: Calculating DeltaE for method: {method_name}") # <-- EKLENDİ
                        try:
                            # ... (color conversion) ...
                            # ... (delta e calculation) ...
                            single_image_delta_e_results.append({...}) # İçerik aynı
                            print(f"DEBUG: DeltaE calculated and appended for method: {method_name}") # <-- EKLENDİ
                        except Exception as e:
                             print(f"DEBUG: *** Exception during DeltaE calc for {method_name}: {e} ***") # <-- EKLENDİ
                             logging.error(...)
                             continue
                    print(f"DEBUG: Finished loop through segmentation results for k_type={k_type}") # <-- EKLENDİ
                except Exception as e:
                     print(f"DEBUG: *** Exception during segmentation for k_type={k_type}: {e} ***") # <-- EKLENDİ
                     logging.error(...)
                     continue
        print(f"DEBUG: Finished segmentation loop for {image_name}") # <-- EKLENDİ
    print(f"DEBUG: Finished processing single test image: {image_name}")
    return single_image_delta_e_results

def save_and_summarize_results(all_delta_e: List[Dict[str, Any]], output_manager: OutputManager):
    print("DEBUG: Saving and summarizing results...")
    # ... (içerik aynı) ...
    print("DEBUG: Results saved and summarized.")


# --- Ana `main` fonksiyonu ---
def main(config_path='configurations/pattern_configs/block_config.yaml', log_level='INFO'):
    print("DEBUG: >>> Entered main() function <<<") # Mevcut

    profiler = cProfile.Profile()
    profiler.enable()
    start_time = time.perf_counter()

    global OUTPUT_DIR
    if OUTPUT_DIR is None:
         print("DEBUG: OUTPUT_DIR was None, setting default in main().")
         OUTPUT_DIR = (PROJECT_ROOT / "output").resolve()

    try:
        print("DEBUG: Inside main try block, before setup.") # Mevcut

        print("DEBUG: About to call setup_logging...") # Mevcut
        if not OUTPUT_DIR.exists():
             print(f"DEBUG: OUTPUT_DIR {OUTPUT_DIR} does not exist, creating...")
             OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        setup_logging(OUTPUT_DIR, log_level)
        print("DEBUG: setup_logging finished.") # Mevcut

        print("DEBUG: Initial logging messages sent via logger.") # Mevcut

        print("DEBUG: Entering Configuration loading timer...") # Mevcut
        with timer("Configuration loading and validation"):
            print("DEBUG: Calling load_config...") # Mevcut
            config = load_config(config_path)
            if config is None: raise ValueError("Failed to load configuration.")
            print("DEBUG: load_config finished.") # Mevcut
            print("DEBUG: Calling validate_processing_config...") # Mevcut
            if not validate_processing_config(config): raise ValueError("Processing configuration validation failed.")
            print("DEBUG: validate_processing_config finished.") # Mevcut

        print("DEBUG: Calling setup_pipeline_configs...") # Mevcut
        dbn_config, pso_config, train_config, preprocess_config = setup_pipeline_configs(config)
        dataset_name = Path(config_path).stem.replace('_config', '')
        print("DEBUG: Creating OutputManager...") # Mevcut
        output_manager = OutputManager(OUTPUT_DIR, dataset_name)
        logging.info(f"Processing dataset: {dataset_name}")
        print("DEBUG: Config objects and OutputManager created.") # Mevcut

        print("DEBUG: Entering Load Test Images timer...") # Mevcut
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
                       logging.warning(f"Skipping invalid test image on initial load: {image_path}") # Loglama başlamış olmalı
             if not valid_test_images: raise ValueError("No valid test images could be loaded.")
        print(f"DEBUG: Loaded {len(valid_test_images)} test images.") # Mevcut

        print("DEBUG: Entering Load Training Data timer...") # Mevcut
        with timer("Loading training data"):
             load_data_target_size = tuple(config.get('load_data_resize', [100, 100]))
             rgb_data, lab_data = load_data(valid_test_images, target_size=load_data_target_size)
             validate_loaded_data(rgb_data, lab_data)
        print("DEBUG: Training data loaded and validated.") # Mevcut

        print("DEBUG: Entering DBN Training timer...") # Mevcut
        with timer("DBN initialization and training"):
            trainer = DBNTrainer(dbn_config, pso_config, train_config)
            dbn, scalers = trainer.train(rgb_data, lab_data)
        print("DEBUG: DBN training finished.") # Mevcut

        print("DEBUG: Calling run_reference_processing...") # Mevcut
        target_colors_lab, ref_kmeans, ref_som = run_reference_processing(
            config, dbn, scalers, output_manager, preprocess_config
        )
        print("DEBUG: Reference processing finished.") # Mevcut

        print("DEBUG: Calling create_lab_converters...") # Mevcut
        lab_traditional_converter, lab_dbn_converter = create_lab_converters(
            dbn, scalers['scaler_x'], scalers['scaler_y'], scalers['scaler_y_ab']
        )
        print("DEBUG: Converters created.") # Mevcut

        all_delta_e_results = []
        print("DEBUG: Starting loop over test images...") # Mevcut
        for image_idx, image_path in enumerate(valid_test_images):
            image_data_for_processing = test_image_data.get(image_path)
            if image_data_for_processing is None:
                logging.warning(f"Skipping {Path(image_path).name} as its data wasn't loaded.")
                continue

            results_for_image = process_single_test_image(
                image_path=image_path,
                image_data=image_data_for_processing,
                config=config,
                preprocess_config=preprocess_config, # Pass object
                dbn=dbn,
                scalers=scalers,
                target_colors_lab=target_colors_lab,
                reference_kmeans_opt=ref_kmeans,
                reference_som_opt=ref_som,
                output_manager=output_manager,
                lab_traditional_converter=lab_traditional_converter,
                lab_dbn_converter=lab_dbn_converter
            )
            all_delta_e_results.extend(results_for_image) # Use extend
        print("DEBUG: Finished loop over test images.") # Mevcut

        print("DEBUG: Calling save_and_summarize_results...") # Mevcut
        # Çağrıyı döngü dışına taşıdık (önceki düzeltme)
        save_and_summarize_results(all_delta_e_results, output_manager)
        print("DEBUG: save_and_summarize_results finished.") # Mevcut

    except Exception as e:
        print(f"DEBUG: *** Exception caught in main try block: {type(e).__name__}: {e} ***") # Hata türünü de yazdır
        logging.error(f"An unexpected error occurred in main execution: {e}", exc_info=True)
    finally:
        print("DEBUG: Entering main finally block.") # Mevcut
        profiler.disable()
        total_time = time.perf_counter() - start_time
        # Loglama çalışıyorsa logla, çalışmıyorsa printle
        log_func = logging.info if logging.getLogger().hasHandlers() else print
        log_func("=" * 80 + f"\nPROCESSING COMPLETED IN {total_time:.2f} SECONDS\n" + "=" * 80)
        try:
            stats = pstats.Stats(profiler).sort_stats('cumtime')
            profile_path = OUTPUT_DIR / 'profile_stats.txt'
            with open(profile_path, 'w') as f:
                stats.stream = f
                stats.print_stats(30)
            log_func = logging.info if logging.getLogger().hasHandlers() else print
            log_func(f"Profiling results saved to: {profile_path}")
        except Exception as e:
            log_func = logging.warning if logging.getLogger().hasHandlers() else print
            log_func(f"Failed to save profiling results: {e}")
        print("DEBUG: Exiting main finally block.") # Mevcut


# --- if __name__ == "__main__": block ---
if __name__ == "__main__":
    print("DEBUG: >>> Script execution started (__name__ == '__main__') <<<") # Mevcut

    parser = argparse.ArgumentParser(
        description="Textile Color Analysis System using PSO-optimized DBN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main.py --config configurations/pattern_configs/block_config.yaml
  python src/main.py --config configurations/pattern_configs/flowers_config.yaml --log-level DEBUG
        """
    )
    parser.add_argument('--config', type=str, default='configurations/pattern_configs/block_config.yaml', help='Path to the specific pattern configuration YAML file')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', help='Set logging level')
    parser.add_argument('--output-dir', type=str, default=None, help='Override output directory')
    parser.add_argument('--profile', action='store_true', help='Enable detailed profiling output')

    print("DEBUG: About to parse arguments...") # Mevcut
    args = parser.parse_args()
    print(f"DEBUG: Arguments parsed: {args}") # Mevcut

    # Set up output directory (Global OUTPUT_DIR)
    print("DEBUG: Determining OUTPUT_DIR...") # Mevcut
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir).resolve()
        print(f"DEBUG: OUTPUT_DIR overridden by args: {OUTPUT_DIR}")
    else:
        OUTPUT_DIR = (PROJECT_ROOT / "output").resolve()
        print(f"DEBUG: Using default OUTPUT_DIR: {OUTPUT_DIR}")

    # --- Main try block for __main__ ---
    try:
        print("DEBUG: Inside __main__ try block, before config path check.") # Mevcut
        config_path = Path(args.config)
        if not config_path.is_absolute(): config_path = (PROJECT_ROOT / config_path).resolve()
        if not config_path.exists():
             print(f"❌ Error: Configuration file not found: {args.config}")
             sys.exit(1)
        print(f"DEBUG: Config path resolved: {config_path}") # Mevcut

        default_cfg_path = PROJECT_ROOT / "configurations/defaults.yaml"
        if not default_cfg_path.exists(): print(f"⚠️ Warning: Default configuration '{default_cfg_path.relative_to(PROJECT_ROOT)}' not found.")
        else: print(f"DEBUG: Default config found: {default_cfg_path}") # Mevcut

        try: display_config_path = config_path.relative_to(PROJECT_ROOT)
        except ValueError: display_config_path = args.config
        try: display_output_dir = OUTPUT_DIR.relative_to(PROJECT_ROOT)
        except ValueError: display_output_dir = OUTPUT_DIR

        print(f"Starting Textile Color Analysis System...")
        print(f"Using Config: {display_config_path}")
        print(f"Log Level: {args.log_level}")
        print(f"Output Dir: {display_output_dir}")
        print("DEBUG: Initial setup messages printed.") # Mevcut
        print("-" * 60)

        print(f"DEBUG: >>> About to call main() function with config: {config_path} <<<") # Mevcut
        main(config_path=str(config_path), log_level=args.log_level)
        print("DEBUG: <<< main() function finished >>>") # Mevcut

        try: display_output_dir_final = OUTPUT_DIR.relative_to(PROJECT_ROOT)
        except ValueError: display_output_dir_final = OUTPUT_DIR
        print("\n" + "=" * 60 + f"\n✅ Processing completed successfully!\n📁 Results saved to: {display_output_dir_final}\n" + "=" * 60)

    except Exception as e:
        print(f"\nDEBUG: *** Exception caught in __main__ try block: {type(e).__name__}: {e} ***") # Hata türünü yazdır
        print(f"\n❌ An unexpected error occurred: {e}")
        # traceback.print_exc() # Uncomment for full traceback
        print("Check the log file in the output directory for detailed error information.")
        sys.exit(1)
    # --- Other except blocks ---
    except (ValueError, FileNotFoundError, InvalidConfigurationError, SegmentationError) as e:
        print(f"\nDEBUG: *** Specific error caught in __main__: {type(e).__name__}: {e} ***") # <-- EKLENDİ
        print(f"\n❌ Configuration or Processing Error: {e}\nCheck the log file in the output directory for details.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nDEBUG: *** KeyboardInterrupt caught in __main__ ***") # <-- EKLENDİ
        print("\n❌ Processing interrupted by user.")
        sys.exit(1)