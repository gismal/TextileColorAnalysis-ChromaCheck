import logging
import cv2
import numpy as np
import time 
from typing import Tuple

from .base import (
    SegmenterBase, SegmentationConfig, ModelConfig,
    SegmentationResult, SegmentationError
)
from .strategy import ClusterStrategy

logger = logging.getLogger(__name__)

class KMeansSegmenter(SegmenterBase):
    """
    implements the segmetnation contract using OpenCV's KMeans algo 
    """
    def __init__(self, 
                 preprocessed_image: np.ndarray,
                 config: SegmentationConfig, 
                 models: ModelConfig,
                 cluster_strategy: ClusterStrategy):
        """
        initializes the KMeansSegmenter
        
        Args: 
            preprocessed_image: The (H, W, 3) image to segment
            config: the segmentation configuration object
            models: the trained models (DBN, scalers) container
            cluster_strategy: the strategy object for 'k' determination
        """
        
        try:
            super().__init__(preprocessed_image, config, models, cluster_strategy)
            logger.info(f"KMeansSegmenter initialized with k_type: {self.config.k_type}")
        except Exception as e_init:
             logger.error(f"KMeansSegmenter __init__ failed: {e_init}", exc_info=True)
             raise

    def segment(self) -> SegmentationResult:
        """
        performs segmentation using cv2.kmeans
        it determines the 'k' values based on the k_type ('determined' or 'predefined') and then applies KMeans clustering
        to the original preprocessed image pixels
        
        Returns
            SegmentationResult: A data onject containing and segmented image, average colors, labels and cluster count
            
        Raises:
            SegmentationError: if 'k' is invalid or cv2.kmeans fails
        """
        start_time = time.perf_counter()
        method_name = "kmeans_opt" if self.config.k_type == 'determined' else "kmeans_predef"
        optimal_k = -1
        try:
            pixels_for_k_determination = self.pixels_flat
           
            # set the k value
            if self.config.k_type == 'determined':
                logger.debug("KMeans: Determining optimal k...")
                optimal_k = self.cluster_strategy.determine_k(pixels_for_k_determination, self.config)
            else:
                optimal_k = self.config.predefined_k
            logger.info(f"KMeans: Using k = {optimal_k}")
            
            if not isinstance(optimal_k, int) or optimal_k <= 0: 
                raise SegmentationError(f"Invalid clusters: {optimal_k}")
            
            # initiates segmentation
            # to find the k, we use quantized image but we use original image for segmentation
            pixels_for_segmentation = self.preprocessed_image.reshape(-1, 3).astype(np.float32)
            if pixels_for_segmentation.shape[0] < optimal_k:
                 logger.warning(f"Pixels ({pixels_for_segmentation.shape[0]}) < k ({optimal_k}). Adjusting k.")
                 optimal_k = max(1, pixels_for_segmentation.shape[0])
            if optimal_k < 1: 
                raise SegmentationError("Less than 1 cluster.")
            
            # OpenCV KMeans
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            retval, labels_flat, centers = cv2.kmeans(pixels_for_segmentation, optimal_k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            centers = np.uint8(centers)
            if labels_flat is None or len(labels_flat) != pixels_for_segmentation.shape[0]: raise SegmentationError(f"Label length mismatch or None")
            
            # the results
            segmented_image = centers[labels_flat.flatten()].reshape(self.preprocessed_image.shape)
            labels_2d = labels_flat.reshape(self.preprocessed_image.shape[:2])
            
            avg_colors = []
            if optimal_k > 0:
                 for i in range(optimal_k):
                     mask = (labels_2d == i).astype(np.uint8)
                     # some clusters migth have 0 pixel
                     if np.sum(mask) > 0: 
                         avg_colors.append(cv2.mean(self.preprocessed_image, mask=mask)[:3])
                     else: 
                         logger.warning(f"KMeans empty mask cluster {i} (k={optimal_k}).")
            
            duration = time.perf_counter() - start_time
            
            return SegmentationResult(
                method_name=method_name,
                segmented_image=segmented_image,
                avg_colors=[tuple(c) for c in avg_colors],
                labels=labels_flat,
                n_clusters=optimal_k,
                processing_time=duration
                )
            
        except Exception as e:
             logger.error(f"Error during KMeans segmentation ({method_name}): {e}", exc_info=True)
             duration = time.perf_counter() - start_time
             return SegmentationResult(
                 method_name=method_name,
                 processing_time=duration,
                 n_clusters=optimal_k if optimal_k > 0 else 0)
