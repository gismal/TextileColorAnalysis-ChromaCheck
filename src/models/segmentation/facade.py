import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from .base import (
    SegmenterBase, SegmentationConfig, ModelConfig,
    SegmentationResult, SegmentationError
)
# import the strategy
from .strategy import ClusterStrategy, MetricBasedStrategy
#import the concrete segmenters
from .kmeans import KMeansSegmenter
from .dbscan import DBSCANSegmenter
from .som import SOMSegmenter

from src.utils.output_manager import OutputManager

logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """Collects all results from the Segmenter (facade)."""
    preprocessed_path: str
    results: Dict[str, SegmentationResult] = field(default_factory=dict)

# ====================================================================
# Main Segmenter (Facade Pattern)
# ====================================================================

class Segmenter:
    """Facade class managing the segmentation workflow.
    
    this facade class hides all the complexity of the KMeans DBSCAN, SOM, strategies
    
    'ProcessingPipeline' class should create this 'Segmenter' class and call .process()
    """
    def __init__(self,
                 preprocessed_image: np.ndarray,
                 seg_config: SegmentationConfig,
                 model_config: ModelConfig,
                 output_manager: OutputManager, 
                 cluster_strategy: Optional[ClusterStrategy] = None):
        """
        initialize the Segmenter Facade
        
        it creates instances of the specific segmentation algorithms that are requested in the SegmentationConfig
        
        Args:
            preprocessed_image: The (H, W, 3) image to segment
            seg_config: the config object to define what to run
            model_config: the trained models (DBN, scalers) container
            output_manager: the manager for saving output files
            cluster_strategy: the strategy to use for 'k' determination if None, defaults to MetricBasedStrategy
        """
        if preprocessed_image is None or preprocessed_image.size == 0:
             raise ValueError("Segmenter init: empty/None preprocessed image.")
        
        self.preprocessed_image = preprocessed_image
        self.config = seg_config
        self.models = model_config
        self.output_manager = output_manager
        self.cluster_strategy = cluster_strategy or MetricBasedStrategy()
        self.segmenters: Dict[str, SegmenterBase] = {}
        
        try:
             self._initialize_segmenters()
             logger.info(f"Segmenter Facade initialized for k_type='{self.config.k_type}' with methods: {list(self.segmenters.keys())}")
        except Exception as init_e:
             logger.error(f"CRITICAL ERROR during Segmenter._initialize_segmenters: {init_e}", exc_info=True)
             raise SegmentationError(f"Failed to initialize segmenters: {init_e}")

    def _initialize_segmenters(self):
        """Dynamically create segmenter objects based on config."""
        common_args = {
            "preprocessed_image": self.preprocessed_image,
            "config": self.config,
            "models": self.models,
            "cluster_strategy": self.cluster_strategy
        }
        requested_methods = self.config.methods or []

        method_map = {
             'kmeans_opt': (KMeansSegmenter, 'determined'),
             'kmeans_predef': (KMeansSegmenter, 'predefined'),
             'som_opt': (SOMSegmenter, 'determined'),
             'som_predef': (SOMSegmenter, 'predefined'),
             'dbscan': (DBSCANSegmenter, None)
        }
        
        logging.debug(f"Segmenter._init_seg: Initializing for k_type='{self.config.k_type}'...")
        for method_key, (SegmenterClass, required_k_type) in method_map.items():
            logging.debug(f"Segmenter._init_seg: Checking method '{method_key}'...")
             
            if method_key in requested_methods and (required_k_type is None or required_k_type == self.config.k_type):
                logging.debug(f"Segmenter._init_seg: Method '{method_key}' matches. Trying to init {SegmenterClass.__name__}...")
                try:
                    instance = SegmenterClass(**common_args)
                    self.segmenters[method_key] = instance
                    logging.debug(f"Segmenter._init_seg: Successfully initialized {SegmenterClass.__name__} for '{method_key}'")
                except Exception as e:
                    logging.error(f"Segmenter._init_seg: EXCEPTION during {SegmenterClass.__name__} init for '{method_key}': {type(e).__name__}: {e}", exc_info=True)
                    raise
            else:
                  logging.debug(f"Segmenter._init_seg: Skipping method '{method_key}' (Not requested or k_type mismatch).")


    def process(self) -> ProcessingResult:
        """
        Run all initialized segmentation methods and collect results.
        this is the main entry point called by the pipeline. it iterates over all initialized segmetners and calls their .segment()
        
        Returns:
            ProcessingResult: An object containing the path to the preprocessed image and a dictionary of all SegmentationResult objects
        """
        image_name_stem = "unknown_image"
        preprocessed_path = "unknown_preprocessed_image.png"
        try:
             # output_manager'dan (eğer ayarlandıysa) mevcut resim adını al
            if hasattr(self.output_manager, 'current_image_stem') and self.output_manager.current_image_stem:
                image_name_stem = self.output_manager.current_image_stem
                preprocessed_path = self.output_manager.get_preprocessed_image_path(image_name_stem)
            else:
                logger.warning("OutputManager has no current_image_stem. Using placeholders.")
        except Exception as e:
            logger.warning(f"Could not get image name/path from output_manager: {e}. Using placeholders.")

        results_dict: Dict[str, SegmentationResult] = {}

        if not self.segmenters:
             logger.warning(f"No segmenters initialized for k_type='{self.config.k_type}'...")
             return ProcessingResult(preprocessed_path=preprocessed_path, results=results_dict)

        for method_name, segmenter_instance in self.segmenters.items():
            try:
                logger.info(f"Running segmentation method: {method_name}")
                result = segmenter_instance.segment()

                if result and result.is_valid():
                    logger.info(f"Method {method_name} completed in {result.processing_time:.2f}s with {result.n_clusters} clusters.")
                    # Kaydetme işlemi OutputManager tarafından yönetiliyor
                    self.output_manager.save_segmentation_result(
                        result.segmented_image,
                        method_name,
                        self.config.k_type
                    )
                    results_dict[method_name] = result
                else:
                    logger.warning(f"Method {method_name} did not produce a valid result.")
                    results_dict[method_name] = result or SegmentationResult(method_name=method_name)

            except Exception as e:
                logger.error(f"Critical error processing method {method_name}: {e}", exc_info=True)
                results_dict[method_name] = SegmentationResult(method_name=method_name)

        logger.info(f"Segmentation processing completed for k_type='{self.config.k_type}'.")
        return ProcessingResult(
            preprocessed_path=preprocessed_path,
            results=results_dict
        )