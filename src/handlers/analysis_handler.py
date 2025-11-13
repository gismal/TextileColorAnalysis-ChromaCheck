import logging
import time
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path
from contextlib import contextmanager

from src.handlers.reference_handler import ReferenceHandler

# Gerekli proje içi importlar
from src.data.load_data import load_image
from src.data.preprocess import Preprocessor, PreprocessingConfig
from src.models.pso_dbn import DBN
from src.models.segmentation import (
    Segmenter,
    SegmentationConfig, ModelConfig, SegmentationResult
)
from src.utils.output_manager import OutputManager
from src.utils.color.color_analysis import ColorMetricCalculator
from src.utils.color.color_conversion import convert_colors_to_cielab, convert_colors_to_cielab_dbn
from src.utils.visualization import plot_segmentation_summary, plot_preprocessing_steps, plot_segment_palette

# Scaler tipi için type hinting
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

# Analiz sınıfı kendi zamanlayıcısını kullanabilir
@contextmanager
def timer(operation_name: str):
    start_time = time.perf_counter()
    logger.info(f"Starting: {operation_name}...")
    try:
        yield
    finally:
        duration = time.perf_counter() - start_time
        logger.info(f"Completed: {operation_name} in {duration:.2f} seconds")

class AnalysisHandler:
    """
    Handles the entire analysis workflow for all test images
    
    This class is responsible for:
    1. Looping through each test image
    2. Preprocessing each test image
    3. Running all configured segmentation methods via the Segmetner Facade
    4. Calculating Delta E metrics for each segmentation result
    5. Triggering the visualization for each result
    """
    def __init__(self,
                 preprocess_config: PreprocessingConfig,
                 seg_params: Dict[str, Any],
                 output_manager: OutputManager,
                 reference_handler: ReferenceHandler):
        """
        Initializes the AnalysisHandler
        
        Args:
            preprocess_config: Configuration object for the Preprocessor
            seg_params: The 'segmentation_params' dictionary from the main config
            output_manager The instance of OutputManager to save outputs
        """
        self.preprocess_config = preprocess_config
        self.seg_params = seg_params
        self.output_manager = output_manager
        self.preprocessor = Preprocessor(config= self.preprocess_config)
        self.reference_handler = reference_handler
        
        # execute() will fill those
        self.dbn: Optional[DBN] = None
        self.scalers: Optional[Dict[str, MinMaxScaler]] = None
        self.target_colors_lab : Optional[np.ndarray] = None
        self.ref_kmeans_result : Optional[SegmentationResult] = None
        self.ref_som_result: Optional[SegmentationResult] = None
        self.color_metric_calculator: Optional[ColorMetricCalculator] = None
        
        logger.debug("AnalysisHandler initialized")
        
    def execute(self,
                test_image_paths: List[str],
                dbn: DBN,
                scalers: Dict[str, MinMaxScaler],
                target_colors_lab: np.ndarray,
                ref_kmeans_result: Optional[SegmentationResult],
                ref_som_result: Optional[SegmentationResult]) -> List[Dict[str, Any]]:
        """
        Executes the analysis loop for all test images
        """
        logger.info("Starting test image analysis loop...")
        all_delta_e_results: List[Dict[str, Any]] = []
        
        # save the dynamic params for analysis
        self.dbn = dbn
        self.scalers = scalers
        self.target_colors_lab = target_colors_lab
        self.ref_kmeans_result = ref_kmeans_result
        self.ref_som_result = ref_som_result
        self.color_metric_calculator = ColorMetricCalculator(target_colors_lab)
        
        if self.dbn is None or self.scalers is None or self.target_colors_lab is None:
            raise ValueError("AnalysisHandler.execute called with None model, scalers, or targets.")

        for image_path_str in test_image_paths:
            image_name_stem = Path(image_path_str).stem
            logger.info(f"--- Processing test image: {image_name_stem} ---")

            image_data = load_image(image_path_str)
            if image_data is None:
                logger.warning(f"Skipping {image_name_stem}, could not be loaded.")
                continue

            self.output_manager.set_current_image_stem(image_name_stem)

            try:
                # Her bir görüntüyü işlemek için özel metodu çağır
                results_for_image = self._process_single_image(image_path_str, image_data)
                all_delta_e_results.extend(results_for_image)
            except Exception as e:
                logger.error(f"Failed to process test image '{image_name_stem}': {e}", exc_info=True)
                continue

        self.output_manager.set_current_image_stem(None)
        logger.info("Test image analysis loop finished.")
        return all_delta_e_results

    def _process_single_image(self,
                              image_path: str,
                              image_data: np.ndarray
                              ) -> List[Dict[str, Any]]:
        """
        Processes a single test image: preprocess, segment, calculate Delta E.
        This method mirrors the logic from `pipeline._process_single_test_image`.
        """
        image_name = Path(image_path).stem
        single_image_delta_e_results: List[Dict[str, Any]] = []
        
        # Gerekli verilerin varlığını kontrol et (execute() tarafından ayarlanmış olmalı)
        if (self.dbn is None or
            self.scalers is None or
            self.target_colors_lab is None or
            self.color_metric_calculator is None):
            logger.error(f"AnalysisHandler state is not set. Cannot process image {image_name}.")
            return []
        
        # Tip kontrolü için (MyPy'a yardımcı olmak)
        dbn, scalers, target_colors_lab = self.dbn, self.scalers, self.target_colors_lab
        color_metric_calculator = self.color_metric_calculator

        with timer(f"Single image processing for {image_name}"):
            # 1. Preprocess
            try:
                preprocessed_image = self.preprocessor.preprocess(image_data)
                if preprocessed_image is None:
                    raise ValueError("Preprocessing returned None")
                
                self.output_manager.save_preprocessed_image(image_name, preprocessed_image)
                
                plot_path = self.output_manager.dataset_dir / "processed" / "preprocessed" / f"{image_name}_preprocessing_steps.png"
                plot_preprocessing_steps(image_data, preprocessed_image, output_path=plot_path)
                
            except Exception as e:
                logger.error(f"Preprocessing failed for test image '{image_name}': {e}", exc_info=True)
                return [] 

            # 2. Perform Segmentation for both k_types
            image_all_results = {}
        
            for k_type in ['determined', 'predefined']:
                with timer(f"Segmentation loop ({image_name}, k_type: {k_type})"):
                    try:
                        seg_config = SegmentationConfig(
                            target_colors=target_colors_lab,
                            **self.seg_params,
                            k_type=k_type 
                        )
                        model_config = ModelConfig(
                            dbn=dbn,
                            scalers=[scalers['scaler_x'], scalers['scaler_y'], scalers['scaler_y_ab']],
                            reference_kmeans_result=self.ref_kmeans_result,
                            reference_som_result=self.ref_som_result
                        )

                        segmenter = Segmenter(preprocessed_image, seg_config, model_config, self.output_manager)
                        processing_result = segmenter.process()

                        if not processing_result or not processing_result.results:
                            logger.warning(f"No segmentation results generated for {image_name} (k_type: {k_type}).")
                            continue 

                        # 3. Calculate Delta E
                        for method_name, result in processing_result.results.items():
                            if not result or not result.is_valid():
                                logger.warning(f"Skipping Delta E for invalid/missing result from {method_name} on {image_name}.")
                                continue

                            # Sonuçları sakla (karşılaştırma için)
                            result_key = f"{method_name}_{k_type}"
                            image_all_results[result_key] = result

                            segmented_rgb_colors = result.avg_colors
                            if not segmented_rgb_colors:
                                logger.warning(f"No average RGB colors found for {method_name} on {image_name}. Skipping Delta E.")
                                continue

                            try:
                                # --- Delta E Calculation ---
                                rgb_array = np.clip(np.array(segmented_rgb_colors, dtype=np.float32), 0, 255)
                                segmented_lab_traditional = convert_colors_to_cielab(rgb_array)
                                segmented_lab_dbn_list = convert_colors_to_cielab_dbn(
                                    dbn, scalers['scaler_x'], scalers['scaler_y'], scalers['scaler_y_ab'],
                                    rgb_array
                                )
                                segmented_lab_dbn = np.array(segmented_lab_dbn_list)

                                if segmented_lab_traditional.size == 0 or segmented_lab_dbn.size == 0:
                                    logger.warning(f"Color conversion to LAB failed for {method_name} on {image_name}. Skipping Delta E.")
                                    continue
                                
                                # Find the best matches for the traditional LAB
                                matches_trad = color_metric_calculator.find_best_matches(segmented_lab_traditional)
                                
                                # now for the DBN LAB
                                matches_dbn = color_metric_calculator.find_best_matches(segmented_lab_dbn)
                                
                                # Calculate the avg scores
                                valid_de_trad = [m[2] for m in matches_trad if m[2] != float('inf')]
                                valid_de_dbn = [m[2] for m in matches_dbn if m[2] != float('inf')]
                                
                                avg_delta_e_traditional = np.mean(valid_de_trad) if valid_de_trad else float('inf')
                                avg_delta_e_dbn = np.mean(valid_de_dbn) if valid_de_dbn else float('inf')
                                
                                # segment stats
                                total_pixels = result.labels.size
                                unique_labels, counts = np.unique(result.labels[result.labels >= 0], return_counts=True)

                                # save all the stats in this dictionary
                                segment_stats = {}
                                for label, count in zip(unique_labels, counts):
                                    if label < len(matches_trad) and label < len(matches_dbn):
                                        segment_stats[label] = {
                                            "pixel_count": int(count),
                                            "percentage": (count / total_pixels) * 100.0,
                                            "best_match_idx_trad": matches_trad[label][1],
                                            "best_match_de_trad": matches_trad[label][2],
                                            "best_match_idx_dbn": matches_dbn[label][1],
                                            "best_match_de_dbn": matches_dbn[label][2]
                                        }
                                
                                single_image_delta_e_results.append({
                                    'dataset': self.output_manager.dataset_name,
                                    'image': image_name,
                                    'method': method_name.replace('_opt', '').replace('_predef', ''),
                                    'k_type': k_type,
                                    'n_clusters': result.n_clusters,
                                    'traditional_avg_delta_e': avg_delta_e_traditional,
                                    'pso_dbn_avg_delta_e': avg_delta_e_dbn,
                                    'processing_time': result.processing_time,
                                    'segment_stats': segment_stats
                                })
                                
                                logger.info(f"-> Result {method_name} ({k_type}) on {image_name}: "
                                            f"Avg Delta E Traditional={avg_delta_e_traditional:.2f}, "
                                            f"Avg Delta E DBN={avg_delta_e_dbn:.2f}, "
                                            f"k={result.n_clusters}")
                                
                                # -- Create Individual Summary Plot (Visual 2) ---
                                try:
                                    plot_filename = f"{image_name}_{method_name}_summary.png" 
                                    plot_output_dir = self.output_manager.dataset_dir / "processed" / "segmented" / method_name
                                    plot_output_path = plot_output_dir / plot_filename
                                    
                                    plot_segmentation_summary(
                                        result = result,
                                        original_preprocessed_image = preprocessed_image, 
                                        target_colors_lab = target_colors_lab,
                                        dbn_model = dbn,
                                        scalers = [scalers['scaler_x'], scalers['scaler_y'], scalers['scaler_y_ab']],
                                        output_path = plot_output_path
                                    )
                                except Exception as plot_err:
                                    logger.error(f"Failed to generate segmentation summary plot for {method_name}: {plot_err}", exc_info=True)
                                
                                try:     
                                    plot_palette_filename = f"{image_name}_{method_name}_palette.png"
                                    plot_palette_path = plot_output_dir / plot_palette_filename
                                    
                                    plot_segment_palette(
                                        result= result,
                                        preprocessed_image= preprocessed_image,
                                        dbn_model= dbn,
                                        scalers= [scalers['scaler_x'], scalers['scaler_y'], scalers['scaler_y_ab']],
                                        output_path = plot_palette_path
                                    )
                                except Exception as palette_err:
                                    logger.error(f"Failed to generate SEGMENT PALETTE plot for {method_name}: {palette_err}", exc_info=True)
                                
                                try:
                                    plot_overlay_filename = f"{image_name}_{method_name}_overlay.png"
                                    plot_overlay_path = plot_output_dir / plot_overlay_filename
                                    
                                    from src.utils.visualization import plot_segmentation_overlay
                                    plot_segmentation_overlay(
                                        result=result,
                                        original_image=preprocessed_image,
                                        output_path=plot_overlay_path
                                    )
                                except Exception as overlay_err:
                                    logger.error(f"Failed to generate overlay plot: {overlay_err}", exc_info=True)

                            except Exception as e:
                                logger.error(f"Error calculating Delta E for {method_name} on {image_name} (k_type: {k_type}): {e}", exc_info=True)
                                continue
                                
                    except Exception as e:
                        logger.error(f"Error in segmentation loop for {image_name} (k_type: {k_type}): {e}", exc_info=True)
                        continue

            # Her iki k_type tamamlandıktan SONRA karşılaştırma grafiğini oluştur
            try:
                if image_all_results:  # Sonuç varsa
                    comparison_filename = f"{image_name}_all_methods_comparison.png"
                    comparison_path = self.output_manager.dataset_dir / "processed" / "comparisons" / comparison_filename
                    
                    from src.utils.visualization import plot_all_methods_comparison
                    plot_all_methods_comparison(
                        image_name, 
                        image_all_results, 
                        preprocessed_image, 
                        comparison_path
                    )
            except Exception as comp_err:
                logger.error(f"Failed to generate comparison plot for {image_name}: {comp_err}", exc_info=True)

            return single_image_delta_e_results