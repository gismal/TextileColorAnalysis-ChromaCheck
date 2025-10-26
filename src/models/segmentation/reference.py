# src/models/segmentation/reference.py
# CORRECTED AND COMPLETED

import logging
import time
import numpy as np
from typing import Tuple, Optional, List # Added List

# Necessary base classes and data structures
from .base import (
    SegmentationConfig, ModelConfig, SegmentationResult, SegmentationError, SegmenterBase
)
# Strategy class (for determining k)
from .strategy import MetricBasedStrategy, ClusterStrategy 
# Concrete segmenter classes (can use directly within the same package)
from .kmeans import KMeansSegmenter
from .som import SOMSegmenter

# Type hinting
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.models.pso_dbn import DBN
    from sklearn.preprocessing import MinMaxScaler
    # PreprocessingConfig is NOT needed here anymore

# Context manager for timing
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# --- Timer (Copied from pipeline.py for self-containment if needed) ---
@contextmanager
def timer(operation_name: str):
    """Logs the duration of a code block."""
    start_time = time.perf_counter()
    logger.info(f"Starting: {operation_name}...")
    try:
        yield
    finally:
        duration = time.perf_counter() - start_time
        logger.info(f"Completed: {operation_name} in {duration:.2f} seconds")

# --- Main Function ---

def segment_reference_image(
    preprocessed_image: np.ndarray,
    dbn: 'DBN',
    scalers: List['MinMaxScaler'], # Use List type hint
    default_k: int,
    k_range: Optional[List[int]] = None # Use List type hint
) -> Tuple[Optional[SegmentationResult], Optional[SegmentationResult], int]:
    """
    Performs K-Means and SOM segmentation specifically for the reference image.

    This function encapsulates the logic previously found in image_utils,
    keeping segmentation logic within the segmentation package. It determines
    the optimal 'k' using MetricBasedStrategy and runs both K-Means and SOM
    with that 'k'.

    Args:
        preprocessed_image: The preprocessed reference image (already quantized).
        dbn: The trained DBN model.
        scalers: List containing [scaler_x, scaler_y, scaler_y_ab].
        default_k: The default 'k' value to use if determination fails.
        k_range: The list of k values to test (e.g., [2, 3, 4, 5]).
                 Defaults to range(2, 9).

    Returns:
        Tuple[Optional[SegmentationResult], Optional[SegmentationResult], int]:
            - The SegmentationResult from K-Means.
            - The SegmentationResult from SOM.
            - The determined optimal 'k' value used.
            Returns (None, None, default_k) on critical failure.
            
    Raises:
        SegmentationError: If the input image is invalid or segmentation fails critically.
    """
    logger.info("Starting reference image segmentation (K-Means & SOM)...")
    start_time_total = time.perf_counter()

    if preprocessed_image is None or preprocessed_image.size == 0:
        raise SegmentationError("Cannot segment None or empty preprocessed image.")

    kmeans_result: Optional[SegmentationResult] = None
    som_result: Optional[SegmentationResult] = None
    determined_k: int = default_k

    try:
        pixels_flat = preprocessed_image.reshape(-1, 3).astype(np.float32)
        num_pixels = pixels_flat.shape[0]
        if num_pixels == 0:
            raise SegmentationError("Image has zero pixels after preprocessing.")

        # --- 1. Determine Optimal K ---
        logger.info("Determining optimal number of clusters for reference image...")
        k_values_to_test = k_range if k_range else list(range(2, 9))
        
        # Temporary config just for k-determination
        temp_seg_config = SegmentationConfig(
            target_colors=np.array([]), distance_threshold=0, predefined_k=default_k,
            k_values=k_values_to_test, som_values=k_values_to_test, 
            k_type='determined', methods=['kmeans_opt'] # Method list doesn't matter much here
        )
        cluster_strategy: ClusterStrategy = MetricBasedStrategy()

        # Use the (already quantized) pixels for k determination
        pixels_for_k_determination = pixels_flat
        
        # Subsample if necessary (logic copied from MetricBasedStrategy for clarity here)
        # We could also get the subsample size from temp_seg_config if needed
        subsample_threshold_for_k = 10000 
        if pixels_for_k_determination.shape[0] > subsample_threshold_for_k:
             logger.info(f"Subsampling {pixels_for_k_determination.shape[0]} pixels to {subsample_threshold_for_k} for reference k-determination")
             indices = np.random.choice(pixels_for_k_determination.shape[0], subsample_threshold_for_k, replace=False)
             pixels_for_k_determination = pixels_for_k_determination[indices]
             
        # Actually determine k
        try:
            determined_k = cluster_strategy.determine_k(pixels_for_k_determination, temp_seg_config) # *** THIS WAS MISSING ***
            logger.info(f"Optimal clusters determined for reference: {determined_k}")
        except Exception as k_err:
            logger.warning(f"Failed to determine optimal clusters: {k_err}. Falling back to default_k={default_k}", exc_info=True)
            determined_k = default_k

        # --- 2. Run K-Means and SOM Segmentation ---
        # Config for running segmentation with the *determined* k
        ref_seg_config = SegmentationConfig(
            target_colors=np.array([]), distance_threshold=0, 
            predefined_k=determined_k, # Use the k we just found
            k_values=[determined_k], som_values=[determined_k], 
            k_type='predefined', # Run in predefined mode now
            methods=['kmeans_predef', 'som_predef'], 
            dbscan_eps=temp_seg_config.dbscan_eps, # Copy from temp or defaults
            dbscan_min_samples=temp_seg_config.dbscan_min_samples 
        )
        
        # Model config needed by segmenters
        ref_model_config = ModelConfig(
            dbn=dbn, 
            scalers=scalers, 
            reference_kmeans_result=None, 
            reference_som_result=None 
        )

        # Run K-Means
        try:
            with timer(f"Reference K-Means (k={determined_k})"):
                 kmeans_segmenter = KMeansSegmenter(
                     preprocessed_image, ref_seg_config, ref_model_config, cluster_strategy 
                 )
                 kmeans_result = kmeans_segmenter.segment()
                 if not kmeans_result or not kmeans_result.is_valid():
                     logger.error("Reference K-Means segmentation failed or produced invalid result.")
                     kmeans_result = None
        except Exception as km_err:
            logger.error(f"Error during reference K-Means execution: {km_err}", exc_info=True)
            kmeans_result = None

        # Run SOM
        try:
            with timer(f"Reference SOM (k={determined_k})"):
                 som_segmenter = SOMSegmenter(
                     preprocessed_image, ref_seg_config, ref_model_config, cluster_strategy 
                 )
                 som_result = som_segmenter.segment()
                 if not som_result or not som_result.is_valid():
                      logger.error("Reference SOM segmentation failed or produced invalid result.")
                      som_result = None
        except Exception as som_err:
            logger.error(f"Error during reference SOM execution: {som_err}", exc_info=True)
            som_result = None

    except Exception as outer_err:
        logger.error(f"Critical error during reference segmentation setup or k-determination: {outer_err}", exc_info=True)
        # Return Nones and default k on outer failure
        return None, None, default_k
    finally:
        duration_total = time.perf_counter() - start_time_total
        logger.info(f"Reference segmentation (KMeans, SOM, k-determination) finished in {duration_total:.2f} seconds.")

    # Return the results (even if one is None) and the k value used
    return kmeans_result, som_result, determined_k