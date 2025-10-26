import logging 
import numpy as np
import time
from typing import Tuple

from .base import(
    SegmenterBase, SegmentationConfig, ModelConfig,
    SegmentationResult, SegmentationError
)
from .strategy import ClusterStrategy

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)

class DBSCANSegmenter(SegmenterBase):
    """
    implements the segmentation contract using sklearn's DBSCAN algorithm
    """
    def __init__(self, 
                 preprocessed_image,
                 config: SegmentationConfig,
                 models: ModelConfig,
                 cluster_strategy: ClusterStrategy):
        """
        initialisez the DBNSCANSegmenter
        
        Args:
            preprocessed_image: the (H, W, 3) image to segment
            config: the segmentation configuration object
            models: the trained models (DBN, scalers) container
            cluster_strategy: the strategy object (mostly unused by DBSCAN, but required by the base class)
        """
        try:
            super().__init__(preprocessed_image, config, models, cluster_strategy)
            logger.info(f"DBSCANSegmenter initialized with k_type: {self.config.k_type}")
        except Exception as e_init:
             logger.error(f"DBSCANSegmenter __init__ failed: {e_init}", exc_info=True)
             raise

    def _run_dbscan(self, pixels: np.ndarray, eps: float, min_samples: int) -> Tuple[np.ndarray, int]:
        # helper function to run the core DBSCAN algorithm
        try:
            if pixels.shape[0] == 0: return np.array([]), 0
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(pixels)
            labels = db.labels_
            # noise increases -1 label, keep it out while counting the clusters
            n_clusters = len(np.unique(labels[labels >= 0]))
            return labels, n_clusters
        except Exception as e:
            logger.error(f"Error during DBSCAN clustering: {e}", exc_info=True)
            return np.full(pixels.shape[0], -1), 0

    def _find_optimal_dbscan_params(self, pixels: np.ndarray) -> Tuple[float, int]:
        """
        a simple grid search to fin reasonable DBSCAN parameters ('eps' and 'min_samples' )
        Note:
            this a very basic heuristic. true DBSCAN parameter tuning is complex (like k-distance graph)
            
        Args:
            pixels (np.ndarray): the pixel data (N,3) to analyze
            
        Returns:
            Tuple[float, int]: the best (eps, min_samples) found
        """
        logger.info("Finding optimal DBSCAN parameters (heuristic)...")
        eps_values = self.config.dbscan_eps_range # <-- Config'den al
        min_samples_values = self.config.dbscan_min_samples_range
        best_silhouette = -1.1
        best_params = (self.config.dbscan_eps, self.config.dbscan_min_samples)
        
        if pixels.shape[0] < 2:
            logger.warning("Too few pixels.")
            return best_params
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                labels, n_clusters = self._run_dbscan(pixels, eps, min_samples)
                if n_clusters > 1:
                    try:
                        silhouette_avg = silhouette_score(pixels, labels)
                        if silhouette_avg > best_silhouette: 
                            best_silhouette = silhouette_avg
                            best_params = (eps, min_samples)
                    except Exception as e: 
                        logger.warning(f"Sil calc failed: {e}")
        
        logger.info(f"Optimal DBSCAN parameters: eps={best_params[0]}, min={best_params[1]}")
        return best_params

    def segment(self) -> SegmentationResult:
        """
        performs segmentation using DBSCAN
        
        it determines parameters based on k_type ('determined' or 'predefined') and then applies DBSCAN.
        it reconstructs the image using the mean colour of each found cluster
        
        Returns: 
            SegmentationResult: a data object containing the segmented image, average colours, labels, cluster counts
        
        """
        start_time = time.perf_counter()
        method_name = "dbscan"
        try:
            pixels = self.pixels_flat
            if pixels.shape[0] == 0: 
                raise SegmentationError("Zero pixels.")
            
            # define the parameters
            if self.config.k_type == 'determined': 
                eps, min_samples = self._find_optimal_dbscan_params(pixels)
            else: 
                eps = self.config.dbscan_eps
                min_samples = self.config.dbscan_min_samples
                logger.info(f"Using predefined DBSCAN.")
            
            # perform the segmentation 
            # we get the labels from the quantized pixels (faster), but the colour averages from originals (more accurate)
            labels, n_clusters = self._run_dbscan(pixels, eps, min_samples)
            if n_clusters == 0: 
                logger.warning("No clusters.")
                return SegmentationResult(method_name=method_name,
                                          processing_time=time.perf_counter()-start_time)
            # construct the image 
            centers = []
            valid_labels = []
            unique_cluster_labels = np.unique(labels[labels >= 0]) # keep the noise off
            original_pixels_for_color = self.pixels_flat
            
            for label_id in unique_cluster_labels:
                
                mask = (labels == label_id)
                if np.sum(mask) > 0: 
                    centers.append(np.mean(original_pixels_for_color[mask], axis=0))
                    valid_labels.append(label_id)
                else: 
                    logger.warning(f"DBSCAN cluster {label_id} empty?")
                    
            if not centers: 
                logger.error("DBSCAN failed to find any valid cluster centers.")
                return SegmentationResult(method_name=method_name, processing_time=time.perf_counter()-start_time)
            
            centers = np.uint8(centers)
            actual_n_clusters = len(centers)
            
            # fill the new centeral colours based on the labels
            segmented_flat = np.zeros_like(original_pixels_for_color, dtype=np.uint8)
            label_to_center_idx = {label_id: idx for idx, label_id in enumerate(valid_labels)}
            
            for i in range(len(labels)):
                if labels[i] in label_to_center_idx: 
                    segmented_flat[i] = centers[label_to_center_idx[labels[i]]]
            
            segmented_image = segmented_flat.reshape(self.preprocessed_image.shape)
            avg_colors = [tuple(c) for c in centers]
            duration = time.perf_counter() - start_time
            
            return SegmentationResult(
                method_name=method_name,
                segmented_image=segmented_image,
                avg_colors=avg_colors,
                labels=labels,
                n_clusters=actual_n_clusters,
                processing_time=duration)
            
        except Exception as e:
             logger.error(f"Error during DBSCAN segmentation: {e}", exc_info=True)
             duration = time.perf_counter() - start_time
             return SegmentationResult(method_name=method_name, processing_time=duration)

