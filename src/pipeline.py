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

# Handler
from src.handlers.training_handler import TrainingHandler
from src.handlers.reference_handler import ReferenceHandler
from src.handlers.analysis_handler import AnalysisHandler
from src.handlers.summary_handler import SummaryHandler

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
            dbn_config = DBNConfig(**self.config.get('dbn_params', {}))
            pso_config = PSOConfig(**self.config.get('pso_params', {}))
            train_config = TrainConfig(**self.config.get('training_params', {}))
            preprocess_config = PreprocessingConfig(
                **self.config.get('preprocess_params', {})
            )
            # get segmetnation_params as dictionary
            self.segmentation_params = self.config.get('segmentation_params', {})
            
        except TypeError as e:
             logger.error(f"Error initializing configuration dataclasses: {e}. Check YAML structure and dataclass definitions.")
             raise TypeError(f"Configuration structure error: {e}")

        if self.output_manager is None:
            raise RuntimeError("OutputManager was not initialized in _load_and_validate_config")
        
        self.training_handler = TrainingHandler(
            dbn_config= dbn_config,
            pso_config= pso_config,
            train_config= train_config,
            output_manager= self.output_manager
        )
        
        self.reference_handler = ReferenceHandler(
            preprocess_config= self.preprocess_config,
            output_manager= self.output_manager
        )
        
        self.analysis_handler = AnalysisHandler(
            preprocess_config= preprocess_config,
            seg_params= self.segmentation_params,
            output_manager= self.output_manager
        )
        
        self.summary_handler = SummaryHandler(
            output_manager= self.output_manager
        )
        
        # Placeholders for the trained model and scalers
        self.dbn: Optional[DBN] = None
        self.scalers: Optional[Dict[str, Any]] = None # Expected keys: 'scaler_x', 'scaler_y', 'scaler_y_ab'
        self.target_colors_lab: Optional[np.ndarray] = None
        self.ref_kmeans_result: Optional[SegmentationResult] = None
        self.ref_som_result: Optional[SegmentationResult] = None
        self.all_delta_e_results: List[Dict[str, Any]] = [] 
        
        
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
                    train_image_paths = self.config['test_images']
                    load_data_size = tuple(self.config.get('load_data_resize', [100, 100]))
                    
                    self.dbn, self.scalers = self.training_handler.execute(
                        image_paths=train_image_paths,
                        load_data_target_size=load_data_size
                    )
                    
                # --- Step 2: Reference Image Processing ---
                with timer("Reference Image Processing"):
                    if self.dbn is None or self.scalers is None:
                        raise RuntimeError/"DBN model or scalers not available"
                    
                    target_colors_lab, ref_kmeans_result, ref_som_result = self.reference_handler.execute(
                        ref_image_path= self.config['reference_image_path'],
                        dbn = self.dbn,
                        scalers= self.scalers,
                        seg_params= self.segmentation_params
                    )
                    
                # --- Step 3: Test Image Analysis ---
                with timer("Test Image Analysis Loop"):
                    if self.dbn is None or self.scalers is None or self.target_colors_lab is None:
                         raise RuntimeError("Missing DBN, scalers, or target colors for analysis.")
                         
                    self.all_delta_e_results = self.analysis_handler.execute(
                        test_image_paths=self.config['test_images'],
                        dbn=self.dbn,
                        scalers=self.scalers,
                        target_colors_lab=self.target_colors_lab,
                        ref_kmeans_result=self.ref_kmeans_result, 
                        ref_som_result=self.ref_som_result    
                    )

                # --- Step 4: Save & Summarize Results ---
                with timer("Saving Final Results"):
                    self._save_and_summarize_results(self.all_delta_e_results)

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
