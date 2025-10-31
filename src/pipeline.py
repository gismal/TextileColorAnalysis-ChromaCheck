import logging
import time
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path
import numpy as np
from contextlib import contextmanager # For the timer utility

# --- Project Imports ---

# Configuration and Data Loading
from src.data.load_data import load_config, load_data, load_image
from src.data.preprocess import Preprocessor, PreprocessingConfig
from src.data.sampling import efficient_data_sampling 
from src.config_types import TrainConfig 

# Models
from src.models.dbn_trainer import DBNTrainer, DBNConfig, PSOConfig 
from src.models.pso_dbn import DBN
from src.models.segmentation import ( 
    Segmenter,
    SegmentationConfig, ModelConfig, SegmentationResult
)
# Specific function for reference segmentation
from src.models.segmentation.reference import segment_reference_image

# Utilities
from src.utils.output_manager import OutputManager
from src.utils.color.color_analysis import ColorMetricCalculator
from src.utils.color.color_conversion import convert_colors_to_cielab, convert_colors_to_cielab_dbn
from src.utils.visualization import (
    plot_reference_summary,
    plot_segmentation_summary,
    plot_delta_e_summary_bars,
    plot_preprocessing_steps
)
# Setup utilities (validation, logging)
from src.utils.setup import validate_processing_config # setup_logging is called from main.py

# --- Logger ---
logger = logging.getLogger(__name__) # Get logger for this module

# --- Timer Utility ---
@contextmanager
def timer(operation_name: str):
    """
    A single context manager to log the duration of code blocks.
    Helps in performance monitoring by logging the start and completion time of key ops
    
    Args:
        operation_name (str): A descriptive name for the operation being timed
    """
    start_time = time.perf_counter()
    logger.info(f"Starting: {operation_name}...")
    try:
        yield # Execute the code block inside the 'with' statement
    finally:
        duration = time.perf_counter() - start_time
        logger.info(f"Completed: {operation_name} in {duration:.2f} seconds")

# --- Main Pipeline Class ---
class ProcessingPipeline:
    """
    Orchestrates the entire textile color analysis workflow.

    This class is the "brain" of the application. It's responsible for managing
    the sequence of operations: loading configuration, training the DBN model,
    processing the reference image to establish target colors, analyzing all
    test images, and finally saving and summarizing the results.
    """

    def __init__(self,
                 config_path: str,
                 output_dir: Path,
                 project_root: Path):
        """
        Initializes the pipeline, loads and validates configuration, and sets up configs.

        Args:
            config_path: Path to the main configuration YAML file for the dataset.
            output_dir: Base directory where all outputs will be saved.
            project_root: Absolute path to the project's root directory.

        Raises:
            ValueError: If configuration loading or validation fails.
            IOError: If the OutputManager fails to create necessary directories.
            TypeError: If configuration parsing results in unexpected types.
        """
        self.config_path = config_path
        self.output_dir = output_dir
        self.project_root = project_root

        self.output_manager: Optional[OutputManager] = None # Initialized in validation
        # Load config, validate it, and initialize OutputManager
        self.config: Dict[str, Any] = self._load_and_validate_config()

        # --- Initialize Configuration Objects ---
        try:
            self.dbn_config = DBNConfig(**self.config.get('dbn_params', {}))
            self.pso_config = PSOConfig(**self.config.get('pso_params', {}))
            self.train_config = TrainConfig(**self.config.get('training_params', {}))

            preproc_cfg_dict = self.config.get('preprocess_params', {})
            # Ensure target_size is a tuple, as expected by cv2.resize and PreprocessingConfig
            preproc_cfg_dict['target_size'] = tuple(preproc_cfg_dict.get('target_size', [128, 128]))
            self.preprocess_config = PreprocessingConfig(**preproc_cfg_dict)
            
        except TypeError as e:
             logger.error(f"Error initializing configuration dataclasses: {e}. Check YAML structure and dataclass definitions.")
             raise TypeError(f"Configuration structure error: {e}")

        # Placeholders for the trained model and scalers
        self.dbn: Optional[DBN] = None
        self.scalers: Optional[Dict[str, Any]] = None # Expected keys: 'scaler_x', 'scaler_y', 'scaler_y_ab'

    def _load_and_validate_config(self) -> Dict[str, Any]:
        """Loads, merges, validates the configuration, and initializes OutputManager."""
        logger.info(f"Loading configuration from: {self.config_path}")
        # load_config handles merging defaults.yaml with the specific config file
        config = load_config(self.config_path)
        if config is None:
            # load_config logs the specific error (FileNotFound, YAML error)
            raise ValueError("Failed to load configuration file(s). Check logs.")

        # validate_processing_config checks required keys, resolves paths, checks values
        if not validate_processing_config(config, self.project_root):
            # Validation function logs the specific error
            raise ValueError("Configuration validation failed. Check logs.")

        # --- Initialize OutputManager ---
        try:
            # Extract dataset name from the config filename (e.g., "block_config.yaml" -> "block")
            dataset_name = Path(self.config_path).stem.replace('_config', '')
            self.output_manager = OutputManager(self.output_dir, dataset_name)
            logger.info(f"OutputManager initialized successfully for dataset: '{dataset_name}'")
        except IOError as e:
             # Let IOError from OutputManager propagate up
             raise e
        except Exception as e:
             logger.error(f"Failed to initialize OutputManager: {e}", exc_info=True)
             raise RuntimeError(f"OutputManager initialization failed: {e}")

        return config

    def run(self):
        """Executes the entire processing workflow step-by-step."""
        if self.output_manager is None:
             # This should not happen if __init__ succeeded, but as a safeguard.
             raise RuntimeError("OutputManager was not initialized before running the pipeline.")

        logger.info("="*50)
        logger.info(f"Processing Pipeline RUN starting for dataset: {self.output_manager.dataset_name}")
        logger.info("="*50)
        try:
            with timer("Total Pipeline"):
                # --- Step 1: DBN Model Training ---
                with timer("DBN Training"):
                    self.dbn, self.scalers = self._train_dbn_model()
                    if self.dbn is None or self.scalers is None:
                         raise RuntimeError("DBN training failed to return a valid model or scalers.")

                # --- Step 2: Reference Image Processing ---
                with timer("Reference Image Processing"):
                    target_colors_lab, ref_kmeans_result, ref_som_result = self._run_reference_processing()
                    if target_colors_lab is None or target_colors_lab.size == 0:
                         raise ValueError("Reference processing failed to produce target LAB colors.")

                # --- Step 3: Test Image Analysis ---
                with timer("Test Image Analysis Loop"):
                    all_delta_e_results = self._run_test_image_analysis(
                        target_colors_lab,
                        ref_kmeans_result, 
                        ref_som_result    
                    )

                # --- Step 4: Save & Summarize Results ---
                with timer("Saving Final Results"):
                    self._save_and_summarize_results(all_delta_e_results)

            logger.info("="*50)
            logger.info(f"Processing Pipeline RUN completed successfully for: {self.output_manager.dataset_name}")
            logger.info("="*50)

        except (ValueError, RuntimeError, TypeError, FileNotFoundError) as e:
            # Catch expected errors during the pipeline execution
            logger.critical(f"Pipeline execution failed: {e}", exc_info=True)
            raise # Re-raise to indicate failure to main.py
        except Exception as e:
            # Catch any unexpected errors
            logger.critical(f"An unexpected critical error occurred during pipeline execution: {e}", exc_info=True)
            raise
        finally:
             # Ensure logs are flushed even if errors occurred
             logging.shutdown()


    def _train_dbn_model(self) -> Tuple[DBN, Dict[str, Any]]:
        """Loads data specifically for DBN training and executes the training process."""

        logger.info("Loading images specified in config['test_images'] to generate training data...")
        valid_test_image_paths = []
        # Load images listed in the config to be used for sampling training data
        for image_path_str in self.config['test_images']:
            # Use the robust load_image function
            image = load_image(image_path_str)
            if image is not None:
                valid_test_image_paths.append(image_path_str)
                # Save a copy of the input test image for reference
                self.output_manager.save_test_image(Path(image_path_str).name, image)
            else:
                # load_image logs the error, just warn here
                logging.warning(f"Skipping image listed in test_images (cannot be loaded): {image_path_str}")

        if not valid_test_image_paths:
            raise ValueError("No valid images found in 'test_images' list in config. Cannot generate training data.")

        logger.info(f"Loading and processing {len(valid_test_image_paths)} valid images into flattened training arrays...")
        # Get target size for resizing during data loading (can be different from final preprocess target)
        load_data_target_size = tuple(self.config.get('load_data_resize', [100, 100]))
        # load_data returns flattened (n_images, H*W*3) arrays
        rgb_data, lab_data = load_data(valid_test_image_paths, target_size=load_data_target_size)

        # Basic validation of loaded data (could also be moved into DBNTrainer or sampling)
        if rgb_data.size == 0 or lab_data.size == 0:
            raise ValueError("Loading training data resulted in empty arrays.")
        logger.debug(f"Loaded training data shapes - RGB: {rgb_data.shape}, LAB: {lab_data.shape}")

        logger.info(f"Initializing DBNTrainer (Target samples: {self.train_config.n_samples})...")
        trainer = DBNTrainer(self.dbn_config, self.pso_config, self.train_config)
        # The train method handles sampling, splitting, scaling, init, and PSO
        dbn, scalers = trainer.train(rgb_data, lab_data)

        logger.info("DBN training and scaling completed.")
        return dbn, scalers

    def _run_reference_processing(self) -> Tuple[np.ndarray, Optional[SegmentationResult], Optional[SegmentationResult]]:
        """
        Loads, preprocesses, and segments the reference image to extract target LAB colors.

        Uses the dedicated `segment_reference_image` function from the segmentation package.

        Returns:
            A tuple containing:
            - Target LAB colors (np.ndarray) derived from K-Means result.
            - The raw SegmentationResult from K-Means.
            - The raw SegmentationResult from SOM.

        Raises:
            ValueError: If loading, preprocessing, segmentation, or color extraction fails.
        """
        ref_image_path = self.config['reference_image_path']
        logger.info(f"Processing reference image specified in config: {ref_image_path}")

        # 1. Load the reference image
        ref_image_bgr = load_image(ref_image_path)
        if ref_image_bgr is None:
            raise ValueError(f"Failed to load reference image: {ref_image_path}")
        # Save a copy to the inputs folder
        self.output_manager.save_reference_image(Path(ref_image_path).name, ref_image_bgr)

        # 2. Preprocess the reference image using the same config as test images
        logger.info("Preprocessing reference image...")
        ref_preprocessor = Preprocessor(config=self.preprocess_config)
        try:
            preprocessed_ref_image = ref_preprocessor.preprocess(ref_image_bgr)
            if preprocessed_ref_image is None:
                # Preprocessor logs the error, raise specific error here
                raise ValueError("Preprocessing returned None for the reference image.")
            # Optional: save preprocessed reference for debugging
            # self.output_manager.save_preprocessed_image("reference", preprocessed_ref_image)
        except Exception as e:
            logger.error(f"Preprocessing failed for reference image: {e}", exc_info=True)
            raise ValueError(f"Preprocessing failed for reference: {e}")

        # 3. Segment the preprocessed reference using the dedicated function
        # Get relevant parameters from the main config
        ref_seg_params = self.config.get('segmentation_params', {})
        default_k = ref_seg_params.get('predefined_k', 2)
        # Use 'k_values' from config for the k-determination range
        k_range_ref = ref_seg_params.get('k_values', list(range(2, 9)))

        # Call the function responsible for reference segmentation (KMeans & SOM)
        kmeans_result, som_result, determined_k = segment_reference_image(
            preprocessed_image=preprocessed_ref_image,
            dbn=self.dbn, # Pass the trained model
            scalers=[self.scalers['scaler_x'], self.scalers['scaler_y'], self.scalers['scaler_y_ab']], # Pass scalers
            default_k=default_k,
            k_range=k_range_ref
        )
        
        # 4. Extract Target Colors (using K-Means result as the reference palette)
        if not kmeans_result or not kmeans_result.is_valid():
             # segment_reference_image should ideally handle this, but double-check
             raise ValueError("Reference K-Means segmentation failed or was invalid, cannot extract target colors.")
        
        target_colors_lab = np.array([])
        try:
            # Convert the average RGB colors from the K-Means result to LAB
            avg_colors_rgb = [tuple(c) for c in kmeans_result.avg_colors]
            target_colors_lab = convert_colors_to_cielab(avg_colors_rgb) # Uses skimage
            if not isinstance(target_colors_lab, np.ndarray) or target_colors_lab.size == 0:
                raise ValueError("convert_colors_to_cielab failed to produce valid LAB colors from K-Means result.")
        except Exception as e:
            logger.error(f"Error extracting target LAB colors from reference K-Means result: {e}", exc_info=True)
            raise ValueError(f"Target color extraction failed: {e}")

        # 5.Create and Save Summary Plot (visual 1)
        try:
            summary_filename = "reference_summary.png"
            summary_output_path = self.output_manager.dataset_dir / "summaries" / summary_filename
            
            plot_reference_summary(
                kmeans_result=kmeans_result,
                som_result=som_result,
                original_image=ref_image_bgr, # Pass the original BGR image
                target_colors_lab=target_colors_lab, # Pass the extracted LAB colors
                output_path=summary_output_path
            )
        except Exception as plot_err:
             # A plotting error shouldn't stop the whole pipeline, just log it.
             logger.error(f"Failed to generate reference summary plot: {plot_err}", exc_info=True)
        
        logger.info(f"Reference image processed successfully. Determined k={determined_k}. "
                    f"Extracted {target_colors_lab.shape[0]} target LAB colors.")

        # Return the essential outputs for the next stage
        return target_colors_lab, kmeans_result, som_result

    def _run_test_image_analysis(
        self,
        target_colors_lab: np.ndarray,
        ref_kmeans_result: Optional[SegmentationResult],
        ref_som_result: Optional[SegmentationResult]
    ) -> List[Dict[str, Any]]:
        """
        Loads and processes each test image one by one 

        Args:
            target_colors_lab: The target LAB colors derived from the reference image.
            ref_kmeans_result: The raw SegmentationResult from K-Means on the reference.
            ref_som_result: The raw SegmentationResult from SOM on the reference.

        Returns:
            A list containing dictionaries of Delta E results for all processed
            test images and segmentation methods.
        """
        logger.info("Starting test image analysis loop...")
        all_delta_e_results: List[Dict[str, Any]] = []

        # Iterate through the (validated and resolved) test image paths from config
        for image_path_str in self.config['test_images']:
            image_name_stem = Path(image_path_str).stem
            logger.info(f"--- Processing test image: {image_name_stem} ---")

            # Load the current test image
            image_data = load_image(image_path_str)
            if image_data is None:
                logger.warning(f"Skipping {image_name_stem}, could not be loaded.")
                continue # Skip to the next image

            # Inform the OutputManager about the current image for correct filename generation
            self.output_manager.set_current_image_stem(image_name_stem)

            # Process this single image (preprocessing, segmentation, Delta E)
            try:
                results_for_image = self._process_single_test_image(
                    image_path_str,
                    image_data,
                    target_colors_lab,
                    ref_kmeans_result, 
                    ref_som_result
                )
                all_delta_e_results.extend(results_for_image) # Add results to the main list
            except Exception as e:
                 # Log error but continue with the next image if possible
                 logger.error(f"Failed to process test image '{image_name_stem}': {e}", exc_info=True)
                 continue # Move to the next image in the loop

        # Clear the current image stem in OutputManager after the loop finishes
        self.output_manager.set_current_image_stem(None)
        logger.info("Test image analysis loop finished.")
        return all_delta_e_results

    def _process_single_test_image(
        self,
        image_path: str,
        image_data: np.ndarray,
        target_colors_lab: np.ndarray,
        reference_kmeans_result: Optional[SegmentationResult],
        reference_som_result: Optional[SegmentationResult]
    ) -> List[Dict[str, Any]]:
        """
        Processes a single test image: preprocess, segment, calculate Delta E.

        This method handles the core analysis logic for one image against the
        reference target colors.

        Args:
            image_path: The path of the original test image.
            image_data: The loaded image data (NumPy array).
            target_colors_lab: The target LAB colors from the reference image.
            reference_kmeans_result: Raw K-Means result from reference processing.
            reference_som_result: Raw SOM result from reference processing.

        Returns:
            A list of dictionaries, where each dictionary contains the Delta E
            results for one segmentation method applied to this image. Returns
            an empty list if preprocessing fails.
        """
        image_name = Path(image_path).stem
        single_image_delta_e_results: List[Dict[str, Any]] = []

        with timer(f"Single image processing for {image_name}"):
            # 1. Preprocess the test image
            preprocessor = Preprocessor(config=self.preprocess_config)
            try:
                preprocessed_image = preprocessor.preprocess(image_data)

                if preprocessed_image is None:
                    # Preprocessor logs the error
                    raise ValueError("Preprocessing returned None")
                # Save the preprocessed image (e.g., block1_preprocessed.png)
                self.output_manager.save_preprocessed_image(image_name, preprocessed_image)
                
                # Also save the side-by-side comparison plot (Original vs. Preprocessed)
                plot_path = self.output_manager.dataset_dir / "processed" / "preprocessed" / f"{image_name}_preprocessing_steps.png"
                plot_preprocessing_steps(image_data, preprocessed_image, output_path=plot_path)
                
            except Exception as e:
                logger.error(f"Preprocessing failed for test image '{image_name}': {e}", exc_info=True)
                return [] # Cannot proceed without preprocessed image

            # 2. Perform Segmentation for both k_types ('determined', 'predefined')
            seg_params = self.config.get('segmentation_params', {})

            for k_type in ['determined', 'predefined']:
                with timer(f"Segmentation loop ({image_name}, k_type: {k_type})"):
                    try:
                        # Create config objects for this specific run
                        seg_config = SegmentationConfig(
                            target_colors=target_colors_lab,
                            # Get all parameters from the loaded config
                            **seg_params, # Unpack all seg_params
                            k_type=k_type # Override/set the k_type
                        )
                        model_config = ModelConfig(
                            dbn=self.dbn,
                            scalers=[self.scalers['scaler_x'], self.scalers['scaler_y'], self.scalers['scaler_y_ab']],
                            reference_kmeans_result=reference_kmeans_result,
                            reference_som_result=reference_som_result
                        )

                        # --- Instantiate and Run the Segmenter Facade ---
                        segmenter = Segmenter(preprocessed_image, seg_config, model_config, self.output_manager)
                        # The process() method runs all applicable segmenters for this k_type and saves the output images via OutputManager.
                        processing_result = segmenter.process()

                        if not processing_result or not processing_result.results:
                            logger.warning(f"No segmentation results generated for {image_name} (k_type: {k_type}).")
                            continue # Skip Delta E calculation for this k_type

                        # 3. Calculate Delta E for each successful segmentation result
                        color_metric_calculator = ColorMetricCalculator(target_colors_lab)

                        for method_name, result in processing_result.results.items():
                            if not result or not result.is_valid():
                                logger.warning(f"Skipping Delta E for invalid/missing result from method '{method_name}' on '{image_name}'.")
                                continue

                            segmented_rgb_colors = result.avg_colors # This is List[Tuple[float, float, float]]
                            if not segmented_rgb_colors:
                                logger.warning(f"No average RGB colors found for method '{method_name}' on '{image_name}'. Skipping Delta E.")
                                continue

                            try:
                                # --- Delta E Calculation ---
                                rgb_array = np.clip(np.array(segmented_rgb_colors, dtype=np.float32), 0, 255)
                                segmented_lab_traditional = convert_colors_to_cielab(rgb_array)
                                segmented_lab_dbn_list = convert_colors_to_cielab_dbn(
                                    self.dbn, self.scalers['scaler_x'], self.scalers['scaler_y'], self.scalers['scaler_y_ab'],
                                    rgb_array
                                )
                                segmented_lab_dbn = np.array(segmented_lab_dbn_list)

                                # Check if conversions were successful
                                if segmented_lab_traditional.size == 0 or segmented_lab_dbn.size == 0:
                                     logger.warning(f"Color conversion to LAB failed for method '{method_name}' on '{image_name}'. Skipping Delta E.")
                                     continue

                                # Calculate list of minimum Delta E values for each segmented color
                                delta_e_traditional_list = color_metric_calculator.compute_all_delta_e(segmented_lab_traditional)
                                delta_e_dbn_list = color_metric_calculator.compute_all_delta_e(segmented_lab_dbn)

                                # Calculate the average Delta E, ignoring infinite values (no match found)
                                avg_delta_e_traditional = np.mean([d for d in delta_e_traditional_list if d != float('inf')])
                                avg_delta_e_dbn = np.mean([d for d in delta_e_dbn_list if d != float('inf')])

                                # Handle cases where all distances were infinite -> mean is NaN
                                if np.isnan(avg_delta_e_traditional): avg_delta_e_traditional = float('inf')
                                if np.isnan(avg_delta_e_dbn): avg_delta_e_dbn = float('inf')

                                # Append results for this method/k_type to the list for this image
                                single_image_delta_e_results.append({
                                    'dataset': self.output_manager.dataset_name,
                                    'image': image_name,
                                    'method': method_name.replace('_opt', '').replace('_predef', ''), # Clean method name
                                    'k_type': k_type,
                                    'n_clusters': result.n_clusters,
                                    'traditional_avg_delta_e': avg_delta_e_traditional,
                                    'pso_dbn_avg_delta_e': avg_delta_e_dbn,
                                    'processing_time': result.processing_time # Time for segmentation only
                                })
                                logger.info(f"-> Result {method_name} ({k_type}) on {image_name}: "
                                            f"Avg Delta E Traditional={avg_delta_e_traditional:.2f}, "
                                            f"Avg Delta E DBN={avg_delta_e_dbn:.2f}, "
                                            f"k={result.n_clusters}")
                                
                                # -- Create Individual Summary Plot (Visual 2) ---
                                try:
                                    plot_filename = f"{image_name}_{k_type}_summary.png"
                                    plot_output_dir = self.output_manager.dataset_dir / "processed" / "segmented" / method_name
                                    plot_output_path = plot_output_dir / plot_filename
                                    
                                    plot_segmentation_summary(
                                        result = result,
                                        original_preprocessed_image = preprocessed_image,
                                        target_colors_lab = target_colors_lab,
                                        dbn_model = self.dbn,
                                        scalers = [self.scalers['scaler_x'], self.scalers['scaler_y'], self.scalers['scaler_y_ab']],
                                        output_path = plot_output_path
                                    )
                                    
                                except Exception as plot_err:
                                    logger.error(f"Failed to generate segmentation summary plot for {method_name}: {plot_err}", exc_info=True)
                                    
                            except Exception as e:
                                logger.error(f"Delta E calculation failed unexpectedly for method '{method_name}' on '{image_name}': {e}", exc_info=True)
                                continue # Skip to next method result

                    except Exception as e:
                        logger.error(f"Error during segmentation loop for {image_name} (k_type: {k_type}): {e}", exc_info=True)
                        continue # Skip to next k_type

        # Return all collected Delta E results for this single image
        return single_image_delta_e_results

    def _save_and_summarize_results(self, all_delta_e: List[Dict[str, Any]]):
        """
        Saves all collected Delta E results to a CSV file and prints summaries and generates a summary bar chart

        Args:
            all_delta_e: A list containing all Delta E result dictionaries
        """
        if not all_delta_e:
            logger.warning("No Delta E results were generated to save or summarize.")
            return

        # 1. Save to CSV using OutputManager
        logger.info(f"Saving {len(all_delta_e)} total Delta E results entries to CSV...")
        self.output_manager.save_delta_e_results(all_delta_e)

        # 2. Print Console Summaries using Pandas
        try:
            import pandas as pd
            # Prevent potential SettingWithCopyWarning
            pd.options.mode.chained_assignment = None 
            df = pd.DataFrame(all_delta_e)
            
            if df.empty:
                logger.warning("Delta E results list was not empty, but DataFrame is. Cannot summarize.")
                return

            # --- Overall Summary (Grouped only by method name) ---
            logger.info("--- Overall Results Summary (Averaged across images and k_types) ---")
            # Calculate mean Delta E and processing time for each clean method name
            summary = df.groupby('method').agg(
                avg_traditional_delta_e=('traditional_avg_delta_e', 'mean'),
                avg_pso_dbn_delta_e=('pso_dbn_avg_delta_e', 'mean'),
                avg_processing_time=('processing_time', 'mean') # Note: includes k-determination time for _opt methods
            ).reset_index()
            # Print the summary table nicely formatted
            logger.info("\n" + summary.to_string(float_format="%.3f"))

            # --- Detailed Summary (Grouped by method and k_type) ---
            logger.info("--- Detailed Results by Method and k_type (Averaged across images) ---")
            detailed_summary = df.groupby(['method', 'k_type']).agg(
                avg_traditional_delta_e=('traditional_avg_delta_e', 'mean'),
                avg_pso_dbn_delta_e=('pso_dbn_avg_delta_e', 'mean'),
                avg_processing_time=('processing_time', 'mean'),
                avg_n_clusters=('n_clusters', 'mean') # Average cluster count found
            ).reset_index()
            # Print the detailed summary table
            logger.info("\n" + detailed_summary.to_string(float_format="%.3f"))
            logger.info("--- End of Summary ---")
            
            try:
                plot_filename = f"{self.output_manager.dataset_name}_delta_e_summary.png"
                plot_output_path = self.output_manager.dataset_dir / "summaries" / plot_filename

                plot_delta_e_summary_bars(
                    results_df=df,
                    output_path=plot_output_path
                )
            except Exception as plot_err:
                logger.error(f"Failed to generate final Delta E summary plot: {plot_err}", exc_info=True)

        except ImportError:
            logger.warning("Pandas library is not installed. Skipping console summary generation. "
                           "Install pandas (`pip install pandas`) to see summaries.")
        except Exception as e:
            logger.error(f"Failed to generate console summary from results: {e}", exc_info=True)