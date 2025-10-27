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

from minisom import MiniSom

logger = logging.getLogger(__name__)

class SOMSegmenter(SegmenterBase):
    """
    Implements the segmentation contracts using Self-Organizing Maps (MiniSom).
    """
    def __init__(self, 
                 preprocessed_image,
                 config: SegmentationConfig,
                 models: ModelConfig,
                 cluster_strategy: ClusterStrategy):
        """
        Initializes the SOMSegmenter
        
        Args: 
            preprocessed_image: the (H, W, 3) image to segment
            config: the segmentation config object
            models: the trained models (DBN, scalers) container
            cluster_strategy: the strategy onject for 'k' determination
        """
        try:
            super().__init__(preprocessed_image, config, models, cluster_strategy)
            logger.info(f"SOMSegmenter initialized with k_type: {self.config.k_type}")
        except Exception as e_init:
            logger.error(f"SOMSegmenter __init__ failed: {e_init}", exc_info=True)
            raise

    def segment(self) -> SegmentationResult:
        """
        Performs segmentation using MiniSom
        
        It determines the 'k' value based on the k type and then trains the SOM. it reconstructs the image
        using the SOM's learned weight as the new colour palette
        
        Returns:
            SegmentationResult: A data object containing the segmented image, average colours, labels and cluster count
        """
        start_time = time.perf_counter()
        method_name = "som_opt" if self.config.k_type == 'determined' else "som_predef"
        optimal_k = -1
        try:
            # Normalize it before cuz MiniSom works with the normalized [0,1] values
            pixels_normalized = self.pixels_flat.reshape(-1, 3).astype(np.float32) / 255.0
            if pixels_normalized.shape[0] == 0:
                raise SegmentationError("Zero pixels.")
            
            if self.config.k_type == 'determined':
                optimal_k = self.cluster_strategy.determine_k(pixels_normalized, self.config)
            else:
                optimal_k = self.config.predefined_k
            
            if not isinstance(optimal_k, int) or optimal_k <= 0: 
                raise SegmentationError(f"Invalid clusters for SOM: {optimal_k}")
            logger.info(f"Running SOM segmentation with k={optimal_k}")
            
            # Train SOM, do the segmentation
            # x = 1 y= optimal_k values create a 1D vertical colour map similar to KMeans
            som = MiniSom(x=1, y=optimal_k, input_len=3, 
                                sigma=self.config.som_sigma,
                                learning_rate=self.config.som_learning_rate,
                                random_seed=42)            
            som.random_weights_init(pixels_normalized)
            som.train_random(pixels_normalized, self.config.som_iterations)
            
            # Form the results
            # winner() finds the closest colur for each pixel
            labels_flat = np.array([som.winner(pixel)[1] for pixel in pixels_normalized])
            centers_normalized = np.array([som.get_weights()[0, i] for i in range(optimal_k)])
            centers = np.uint8(np.clip(centers_normalized * 255.0, 0, 255))
            
            original_pixels_shape = self.preprocessed_image.shape
            segmented_image = centers[labels_flat.flatten()].reshape(original_pixels_shape)
            labels_2d = labels_flat.reshape(original_pixels_shape[:2])
            
            # Use SOM centers to calculate avg_colors but masking it to be relevant with KMeans 
            avg_colors = []
            for i in range(optimal_k):
                mask = (labels_2d == i).astype(np.uint8)
                if np.sum(mask) > 0: 
                    avg_colors.append(cv2.mean(self.preprocessed_image, mask=mask)[:3])
                else: 
                    logger.warning(f"SOM empty mask cluster {i} (k={optimal_k}).")
            
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
             logger.error(f"Error during SOM segmentation ({method_name}): {e}", exc_info=True)
             duration = time.perf_counter() - start_time
             return SegmentationResult(
                 method_name=method_name,
                 processing_time=duration,
                 n_clusters=optimal_k if optimal_k > 0 else 0
                 )

