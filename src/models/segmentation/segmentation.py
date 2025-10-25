# src/models/segmentation/segmentation.py (REFACTORED FINAL + DEBUG PRINTS)

print("DEBUG: >>> LOADING segmentation.py (VERSION: 2) <<<")
import logging
import os
import cv2
import numpy as np
import time # Use time instead of datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass, field # field import confirmed
from typing import List, Dict, Any, Optional, Tuple

# --- Imports moved from segmentation_utils.py ---
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from minisom import MiniSom
# --- End of moved imports ---

# Gerekli importlar (pso_dbn'den)
from src.models.pso_dbn import DBN
from sklearn.preprocessing import MinMaxScaler
# Correct import for ciede2000_distance
from src.utils.color.color_conversion import ciede2000_distance

logger = logging.getLogger(__name__)

# ====================================================================
# Data Classes & Custom Exceptions
# ====================================================================

class SegmentationError(Exception):
    """Custom exception for segmentation failures."""
    pass

class InvalidConfigurationError(ValueError):
    """Custom exception for configuration errors."""
    pass

@dataclass
class SegmentationConfig:
    """Configuration settings for the segmentation process."""
    target_colors: np.ndarray
    distance_threshold: float
    predefined_k: int
    k_values: List[int]
    som_values: List[int]
    k_type: str = 'determined'
    # Corrected typo: som_predef
    methods: List[str] = field(default_factory=lambda: ['kmeans_opt', 'kmeans_predef', 'som_opt', 'som_predef', 'dbscan'])
    dbscan_eps: float = 10.0
    dbscan_min_samples: int = 5

@dataclass
class ModelConfig:
    """Holds the trained models and scalers needed for segmentation."""
    dbn: DBN
    scalers: List[MinMaxScaler]
    reference_kmeans_opt: Dict[str, Any]
    reference_som_opt: Dict[str, Any]

@dataclass
class SegmentationResult:
    """Holds the output of a single segmentation method."""
    method_name: str
    segmented_image: Optional[np.ndarray] = None
    avg_colors: List[Tuple[float, float, float]] = field(default_factory=list) # Should store tuples/lists, not ColorArray type
    labels: Optional[np.ndarray] = None
    n_clusters: int = 0
    processing_time: float = 0.0

    def is_valid(self) -> bool:
        """Check if the segmentation result is valid and usable."""
        # Corrected typos: segmented_image (singular), avg_colors (plural)
        return (self.segmented_image is not None and
                self.avg_colors is not None and
                len(self.avg_colors) > 0 and
                self.n_clusters > 0)

@dataclass
class ProcessingResult:
    """Collects all results from the Segmenter (facade)."""
    preprocessed_path: str
    results: Dict[str, SegmentationResult] = field(default_factory=dict)

# ====================================================================
# Strategy Pattern for Cluster Determination
# ====================================================================

class ClusterStrategy(ABC):
    """Abstract base class for k-determination strategies."""
    @abstractmethod
    def determine_k(self, pixels: np.ndarray, config: SegmentationConfig) -> int:
        pass

class MetricBasedStrategy(ClusterStrategy):
    """Finds optimal 'k' using clustering metrics."""
    def determine_k(self, pixels: np.ndarray, config: SegmentationConfig, n_runs=3) -> int:
        """Find the optimal k based on combined normalized metrics."""
        # Determine k range (Simplified logic)
        k_range_list = config.k_values if 'kmeans' in config.methods or 'som' in config.methods else list(range(2,9)) # Fallback range
        min_k = min(k_range_list) if k_range_list else 2
        max_k = max(k_range_list) if k_range_list else 8

        unique_colors = np.unique(pixels, axis=0)
        n_unique = len(unique_colors)
        dynamic_max_k = max(min_k, min(max_k, n_unique)) # Ensure max_k >= min_k and <= n_unique

        logger.info(f"Unique colors: {n_unique}. Adjusted k-range for metric search: [{min_k}, {dynamic_max_k}]")

        if pixels.shape[0] > 10000:
            logger.info("Subsampling pixels for cluster analysis efficiency")
            pixels = pixels[np.random.choice(pixels.shape[0], 10000, replace=False)]
        
        if pixels.shape[0] < min_k: # Handle insufficient points after subsampling
             logger.warning(f"Number of pixels ({pixels.shape[0]}) < min_k ({min_k}). Cannot determine k. Falling back to predefined_k={config.predefined_k}")
             return config.predefined_k

        scores = {'silhouette': [], 'ch': []}
        k_range = list(range(min_k, dynamic_max_k + 1))
        if not k_range:
            logger.warning(f"K-range is empty. Defaulting to predefined_k={config.predefined_k}")
            return config.predefined_k

        for k in k_range:
            # Skip if k is greater than number of samples
            if k > pixels.shape[0]:
                logger.warning(f"Skipping k={k} because it's > number of samples ({pixels.shape[0]})")
                scores['silhouette'].append(-1.0) # Penalize impossible k
                scores['ch'].append(0.0)
                continue
                
            logger.info(f"Testing k={k}")
            try:
                # Use 'auto' for n_init in newer scikit-learn
                kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42).fit(pixels)
                labels = kmeans.labels_

                if len(np.unique(labels)) < 2:
                    logger.warning(f"Only 1 cluster found for k={k}. Assigning worst score.")
                    scores['silhouette'].append(-1.0)
                    scores['ch'].append(0.0)
                    continue

                scores['silhouette'].append(silhouette_score(pixels, labels))
                scores['ch'].append(calinski_harabasz_score(pixels, labels))
                logger.info(f"Metrics for k={k}: Silhouette={scores['silhouette'][-1]:.3f}, CH={scores['ch'][-1]:.1f}")

            except Exception as e:
                logger.error(f"Error calculating metrics for k={k}: {e}", exc_info=True)
                scores['silhouette'].append(-1.0)
                scores['ch'].append(0.0)

        # Normalize scores safely
        sil_scores = np.array(scores['silhouette'])
        ch_scores = np.array(scores['ch'])
        norm_sil = (sil_scores - np.min(sil_scores)) / (np.max(sil_scores) - np.min(sil_scores) + 1e-10)
        norm_ch = (ch_scores - np.min(ch_scores)) / (np.max(ch_scores) - np.min(ch_scores) + 1e-10)

        avg_scores = (norm_sil + norm_ch) / 2
        optimal_k_index = np.argmax(avg_scores)
        optimal_k = k_range[optimal_k_index]
        logger.info(f"Optimal k determined: {optimal_k} (based on score: {avg_scores[optimal_k_index]:.3f})")

        return optimal_k

# ====================================================================
# Abstract Base Class for Segmentation Methods
# ====================================================================

class SegmenterBase(ABC):
    """Abstract base class for segmentation algorithms."""
    def __init__(self,
                 preprocessed_image: np.ndarray,
                 config: SegmentationConfig,
                 models: ModelConfig, # models might not be needed by all segmenters
                 cluster_strategy: ClusterStrategy):
        print(f"DEBUG: [SegmenterBase.__init__] Started for {self.__class__.__name__}") 
        self.preprocessed_image = preprocessed_image
        self.config = config
        self.models = models
        self.cluster_strategy = cluster_strategy
        print(f"DEBUG: [SegmenterBase.__init__] Finished for {self.__class__.__name__}") 
    
    @abstractmethod
    def segment(self) -> SegmentationResult:
        """Performs segmentation and returns a populated SegmentationResult."""
        pass

    def quantize_image(self, n_colors=50) -> Optional[np.ndarray]:
        """Reduces colors using K-means for faster processing."""
        if self.preprocessed_image is None or self.preprocessed_image.size == 0:
            logger.warning("Cannot quantize None or empty image.")
            return None
            
        logger.info(f"Quantizing image (shape: {self.preprocessed_image.shape}) to approx {n_colors} colors")
        pixels = self.preprocessed_image.reshape(-1, 3).astype(np.float32)
        n_pixels_total = pixels.shape[0]
        
        # Ensure n_colors is feasible
        actual_n_colors = max(1, min(n_colors, n_pixels_total))
        if actual_n_colors != n_colors:
             logger.warning(f"Adjusted quantization n_colors from {n_colors} to {actual_n_colors} based on pixel count.")
             
        if actual_n_colors == 1 and n_pixels_total > 0:
             # Handle edge case of 1 color
             center = np.mean(pixels, axis=0)
             quantized = np.tile(center, (self.preprocessed_image.shape[0], self.preprocessed_image.shape[1], 1)).astype(np.uint8)
             return quantized
        elif actual_n_colors < 1:
             logger.error("Cannot quantize to less than 1 color.")
             return None

        # Subsample if necessary
        if n_pixels_total > 20000:
            indices = np.random.choice(n_pixels_total, 20000, replace=False)
            pixels_sample = pixels[indices]
            logger.debug(f"Subsampled pixels for K-means fit: {pixels_sample.shape}")
        else:
            pixels_sample = pixels
        
        # Ensure sample size is adequate for n_colors
        if pixels_sample.shape[0] < actual_n_colors:
             logger.warning(f"Sample size ({pixels_sample.shape[0]}) < n_colors ({actual_n_colors}). Using sample size as n_colors.")
             actual_n_colors = pixels_sample.shape[0]
             if actual_n_colors < 1:
                  logger.error("Cannot quantize with zero samples.")
                  return None

        try:
            kmeans = KMeans(n_clusters=actual_n_colors, n_init='auto', random_state=42).fit(pixels_sample)
            labels = kmeans.predict(pixels) # Predict on full set
            quantized_pixels = kmeans.cluster_centers_[labels]
            return quantized_pixels.reshape(self.preprocessed_image.shape).astype(np.uint8)
        except Exception as e:
             logger.error(f"Error during quantization K-Means: {e}", exc_info=True)
             return None


# ====================================================================
# Concrete Segmentation Method Implementations
# ====================================================================

class KMeansSegmenter(SegmenterBase):
    """Segments image using K-Means."""
    def __init__(self, preprocessed_image, config: SegmentationConfig, models: ModelConfig, cluster_strategy: ClusterStrategy):
        print(f"DEBUG: [KMSegmenter.__init__] STARTING INIT...")
        try:
            print(f"DEBUG: [KMSegmenter.__init__] Args before super(): preprocessed_image shape: {preprocessed_image.shape if preprocessed_image is not None else 'None'}, config type: {type(config)}, models type: {type(models)}, strategy type: {type(cluster_strategy)}")
            try:
                print(f"DEBUG: [KMSegmenter.__init__] Calling super().__init__...")
                super().__init__(preprocessed_image, config, models, cluster_strategy)
                print(f"DEBUG: [KMSegmenter.__init__] super().__init__() finished.")
            except Exception as super_e:
                print(f"DEBUG: [KMSegmenter.__init__] *** EXCEPTION DURING super().__init__(): {type(super_e).__name__}: {super_e} ***")
                raise
            logger.info(f"KMeansSegmenter initialized with k_type: {self.config.k_type}")
            print(f"DEBUG: [KMSegmenter.__init__] FINISHED INIT SUCCESSFULLY.")
        except Exception as e_init:
             print(f"DEBUG: [KMSegmenter.__init__] *** EXCEPTION IN INIT (after super check): {type(e_init).__name__}: {e_init} ***")
             raise
         
    def segment(self) -> SegmentationResult:
        # --- ADDED DEBUG PRINTS ---
        print("DEBUG: [KMSegmenter] Entered segment() method.")
        start_time = time.perf_counter()
        method_name = "kmeans_opt" if self.config.k_type == 'determined' else "kmeans_predef" # Determine name early
        try:
            print("DEBUG: [KMSegmenter] Calling quantize_image()...")
            quantized_img = self.quantize_image()
            if quantized_img is None:
                 print("DEBUG: [KMSegmenter] quantize_image() returned None.")
                 raise SegmentationError("Quantization failed during K-Means segmentation.")
            quantized_pixels = quantized_img.reshape(-1, 3).astype(np.float32)
            print(f"DEBUG: [KMSegmenter] quantize_image() finished, shape: {quantized_pixels.shape}")

            optimal_k = -1
            if self.config.k_type == 'determined':
                print("DEBUG: [KMSegmenter] Determining optimal k...")
                optimal_k = self.cluster_strategy.determine_k(quantized_pixels, self.config)
                print(f"DEBUG: [KMSegmenter] Optimal k = {optimal_k}")
            else:
                optimal_k = self.config.predefined_k
                print(f"DEBUG: [KMSegmenter] Using predefined k = {optimal_k}")

            if not isinstance(optimal_k, int) or optimal_k <= 0:
                 print(f"DEBUG: [KMSegmenter] Invalid k value: {optimal_k}. Raising error.")
                 raise SegmentationError(f"Invalid number of clusters determined or provided: {optimal_k}")

            print("DEBUG: [KMSegmenter] Preparing pixels for cv2.kmeans...")
            pixels_for_segmentation = self.preprocessed_image.reshape(-1, 3).astype(np.float32)
            
            # Check if number of pixels is less than k
            if pixels_for_segmentation.shape[0] < optimal_k:
                 print(f"DEBUG: [KMSegmenter] Number of pixels ({pixels_for_segmentation.shape[0]}) < k ({optimal_k}). Adjusting k.")
                 optimal_k = max(1, pixels_for_segmentation.shape[0]) # Adjust k down
                 if optimal_k == 1:
                      print("DEBUG: [KMSegmenter] Only 1 cluster possible after adjustment.")

            if optimal_k < 1: # Should not happen, but safety check
                 raise SegmentationError("Cannot perform K-Means with less than 1 cluster.")

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            print(f"DEBUG: [KMSegmenter] Calling cv2.kmeans with k={optimal_k} on pixels shape {pixels_for_segmentation.shape}...")

            # --- Potential Failure Point ---
            retval, labels_flat, centers = cv2.kmeans(
                pixels_for_segmentation, optimal_k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
            )
            print("DEBUG: [KMSegmenter] cv2.kmeans finished.")

            centers = np.uint8(centers)
            # Ensure labels_flat length matches pixels
            if len(labels_flat) != pixels_for_segmentation.shape[0]:
                 raise SegmentationError(f"cv2.kmeans returned labels array with unexpected length {len(labels_flat)} vs pixel count {pixels_for_segmentation.shape[0]}")
                 
            segmented_image = centers[labels_flat.flatten()].reshape(self.preprocessed_image.shape)
            labels_2d = labels_flat.reshape(self.preprocessed_image.shape[:2])

            print("DEBUG: [KMSegmenter] Calculating average colors...")
            avg_colors = []
            if optimal_k > 0: # Check if k is positive before looping
                 for i in range(optimal_k):
                     mask = (labels_2d == i).astype(np.uint8)
                     if np.sum(mask) > 0:
                         avg_color_bgr = cv2.mean(self.preprocessed_image, mask=mask)[:3]
                         avg_colors.append(avg_color_bgr)
                     else:
                         logger.warning(f"KMeans generated an empty mask for cluster {i} (k={optimal_k}). Skipping avg color.")
            print(f"DEBUG: [KMSegmenter] Calculated {len(avg_colors)} average colors.")

            duration = time.perf_counter() - start_time
            print("DEBUG: [KMSegmenter] Creating SegmentationResult object.")

            return SegmentationResult(
                method_name=method_name,
                segmented_image=segmented_image,
                avg_colors=[tuple(c) for c in avg_colors], # Convert to tuples
                labels=labels_flat,
                n_clusters=optimal_k,
                processing_time=duration
            )

        except Exception as e:
             print(f"DEBUG: [KMSegmenter] *** EXCEPTION caught in segment(): {type(e).__name__}: {e} ***")
             logger.error(f"Error during KMeans segmentation ({method_name}): {e}", exc_info=True)
             duration = time.perf_counter() - start_time
             return SegmentationResult(method_name=method_name, processing_time=duration, n_clusters=optimal_k if 'optimal_k' in locals() else 0) # Return invalid result
        # --- END DEBUG PRINTS & TRY BLOCK ---

class DBSCANSegmenter(SegmenterBase):
    """Segments image using DBSCAN."""
    """Segments image using DBSCAN."""
    def __init__(self, preprocessed_image, config: SegmentationConfig, models: ModelConfig, cluster_strategy: ClusterStrategy):
        # --- ADDED DEBUG PRINTS and try...except around super() ---
        print(f"DEBUG: [DBSCANSegmenter.__init__] STARTING INIT...")
        try:
            print(f"DEBUG: [DBSCANSegmenter.__init__] Args before super(): preprocessed_image shape: {preprocessed_image.shape if preprocessed_image is not None else 'None'}, ...") # Shortened
            try:
                print(f"DEBUG: [DBSCANSegmenter.__init__] Calling super().__init__...")
                super().__init__(preprocessed_image, config, models, cluster_strategy)
                print(f"DEBUG: [DBSCANSegmenter.__init__] super().__init__() finished.")
            except Exception as super_e:
                print(f"DEBUG: [DBSCANSegmenter.__init__] *** EXCEPTION DURING super().__init__(): {type(super_e).__name__}: {super_e} ***")
                raise
            logger.info(f"DBSCANSegmenter initialized with k_type: {self.config.k_type}")
            print(f"DEBUG: [DBSCANSegmenter.__init__] FINISHED INIT SUCCESSFULLY.")
        except Exception as e_init:
             print(f"DEBUG: [DBSCANSegmenter.__init__] *** EXCEPTION IN INIT (after super check): {type(e_init).__name__}: {e_init} ***")
             raise
         
    # ... (_run_dbscan, _find_optimal_dbscan_params methods - same as before) ...
    def _run_dbscan(self, pixels: np.ndarray, eps: float, min_samples: int) -> Tuple[np.ndarray, int]:
        """Private helper to run DBSCAN and return labels and cluster count."""
        try:
            if pixels.shape[0] == 0: return np.array([]), 0 # Handle empty
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(pixels)
            labels = db.labels_
            n_clusters = len(np.unique(labels[labels >= 0]))
            logger.info(f"DBSCAN(eps={eps}, min={min_samples}) found {n_clusters} clusters.")
            return labels, n_clusters
        except Exception as e:
            logger.error(f"Error during DBSCAN clustering: {e}", exc_info=True)
            return np.full(pixels.shape[0], -1), 0

    def _find_optimal_dbscan_params(self, pixels: np.ndarray) -> Tuple[float, int]:
        """Private helper to find best DBSCAN params."""
        logger.info("Finding optimal DBSCAN parameters...")
        eps_values = [10, 15, 20] # Could come from config
        min_samples_values = [5, 10, 20] # Could come from config
        best_silhouette = -1.1
        best_params = (self.config.dbscan_eps, self.config.dbscan_min_samples)

        if pixels.shape[0] < 2: # Cannot calculate silhouette
            logger.warning("Too few pixels to find optimal DBSCAN params. Using defaults.")
            return best_params

        for eps in eps_values:
            for min_samples in min_samples_values:
                labels, n_clusters = self._run_dbscan(pixels, eps, min_samples)
                if n_clusters > 1:
                    try:
                        silhouette_avg = silhouette_score(pixels, labels)
                        logger.info(f"DBSCAN params (eps={eps}, min={min_samples}) -> clusters={n_clusters}, silhouette: {silhouette_avg:.3f}")
                        if silhouette_avg > best_silhouette:
                            best_silhouette = silhouette_avg
                            best_params = (eps, min_samples)
                    except Exception as e:
                         logger.warning(f"Could not calculate silhouette for DBSCAN(eps={eps}, min={min_samples}): {e}")
                else:
                     logger.info(f"DBSCAN params (eps={eps}, min={min_samples}) -> found <= 1 cluster. Skipping score.")

        logger.info(f"Optimal DBSCAN parameters found: eps={best_params[0]}, min_samples={best_params[1]}")
        return best_params


    def segment(self) -> SegmentationResult:
        start_time = time.perf_counter()
        method_name = "dbscan"
        try:
            quantized_img = self.quantize_image()
            if quantized_img is None: raise SegmentationError("Quantization failed.")
            pixels = quantized_img.reshape(-1, 3).astype(np.float32)
            
            if pixels.shape[0] == 0: raise SegmentationError("Image has zero pixels after quantization.")

            if self.config.k_type == 'determined':
                eps, min_samples = self._find_optimal_dbscan_params(pixels)
            else:
                eps = self.config.dbscan_eps
                min_samples = self.config.dbscan_min_samples
                logger.info(f"Using predefined DBSCAN params: eps={eps}, min_samples={min_samples}")

            labels, n_clusters = self._run_dbscan(pixels, eps, min_samples)

            if n_clusters == 0:
                logger.warning("No clusters found in DBSCAN. Returning empty result.")
                return SegmentationResult(method_name=method_name, processing_time=time.perf_counter() - start_time)

            original_pixels = self.preprocessed_image.reshape(-1, 3).astype(np.float32)
            
            # Ensure labels length matches original pixels if quantization subsampled
            if len(labels) != original_pixels.shape[0]:
                 # Re-run DBSCAN on original pixels if needed, or handle mismatch
                 # For now, let's assume quantization didn't change pixel count drastically
                 # and try to apply labels back. This might be incorrect if quantization was lossy.
                 # A safer approach might be to re-run DBSCAN or map quantized labels back carefully.
                 # Let's try re-running DBSCAN on original preprocessed pixels.
                 logger.warning("Label length mismatch after quantization. Re-running DBSCAN on original preprocessed pixels.")
                 pixels_orig = self.preprocessed_image.reshape(-1,3).astype(np.float32)
                 labels, n_clusters = self._run_dbscan(pixels_orig, eps, min_samples)
                 if n_clusters == 0:
                      logger.error("DBSCAN failed on original pixels after succeeding on quantized.")
                      return SegmentationResult(method_name=method_name, processing_time=time.perf_counter()-start_time)
                 original_pixels = pixels_orig # Use the pixels DBSCAN actually ran on


            centers = []
            valid_labels = []
            # Calculate centers only for valid clusters (labels >= 0)
            for label_id in range(n_clusters):
                 mask = (labels == label_id)
                 if np.sum(mask) > 0:
                      centers.append(np.mean(original_pixels[mask], axis=0))
                      valid_labels.append(label_id)
                 else:
                      logger.warning(f"DBSCAN cluster {label_id} has no points.")

            if not centers: # No valid clusters found
                logger.error("DBSCAN found clusters but failed to calculate centers.")
                return SegmentationResult(method_name=method_name, processing_time=time.perf_counter()-start_time)

            centers = np.uint8(centers)
            actual_n_clusters = len(centers) # Update n_clusters based on valid centers

            segmented_flat = np.zeros_like(original_pixels, dtype=np.uint8)
            # Map valid labels (0, 1, ... actual_n_clusters-1) to centers index
            label_to_center_idx = {label_id: idx for idx, label_id in enumerate(valid_labels)}

            for i in range(len(labels)):
                if labels[i] in label_to_center_idx: # Check if label is a valid cluster ID
                    center_idx = label_to_center_idx[labels[i]]
                    segmented_flat[i] = centers[center_idx]
                # else: leave as black (noise or unassigned)

            segmented_image = segmented_flat.reshape(self.preprocessed_image.shape)
            avg_colors = [tuple(c) for c in centers]

            duration = time.perf_counter() - start_time
            return SegmentationResult(
                method_name=method_name,
                segmented_image=segmented_image,
                avg_colors=avg_colors,
                labels=labels,
                n_clusters=actual_n_clusters, # Use actual count
                processing_time=duration
            )
        except Exception as e:
             logger.error(f"Error during DBSCAN segmentation: {e}", exc_info=True)
             duration = time.perf_counter() - start_time
             return SegmentationResult(method_name=method_name, processing_time=duration)


class SOMSegmenter(SegmenterBase):
    """Segments image using Self-Organizing Maps."""
    def __init__(self, preprocessed_image, config: SegmentationConfig, models: ModelConfig, cluster_strategy: ClusterStrategy):
        print(f"DEBUG: [SOMSegmenter.__init__] STARTING INIT...")
        try:
            print(f"DEBUG: [SOMSegmenter.__init__] Args before super(): preprocessed_image shape: {preprocessed_image.shape if preprocessed_image is not None else 'None'}, ...") # Shortened
            try:
                print(f"DEBUG: [SOMSegmenter.__init__] Calling super().__init__...")
                super().__init__(preprocessed_image, config, models, cluster_strategy)
                print(f"DEBUG: [SOMSegmenter.__init__] super().__init__() finished.")
            except Exception as super_e:
                print(f"DEBUG: [SOMSegmenter.__init__] *** EXCEPTION DURING super().__init__(): {type(super_e).__name__}: {super_e} ***")
                raise
            logger.info(f"SOMSegmenter initialized with k_type: {self.config.k_type}")
            print(f"DEBUG: [SOMSegmenter.__init__] FINISHED INIT SUCCESSFULLY.")
        except Exception as e_init:
             print(f"DEBUG: [SOMSegmenter.__init__] *** EXCEPTION IN INIT (after super check): {type(e_init).__name__}: {e_init} ***")
             raise
         
    def segment(self) -> SegmentationResult:
        start_time = time.perf_counter()
        method_name = "som_opt" if self.config.k_type == 'determined' else "som_predef"
        try:
            quantized_img = self.quantize_image()
            if quantized_img is None: raise SegmentationError("Quantization failed.")
            # SOM works best on normalized pixels [0, 1]
            pixels_normalized = quantized_img.reshape(-1, 3).astype(np.float32) / 255.0
            
            if pixels_normalized.shape[0] == 0: raise SegmentationError("Image has zero pixels after quantization.")

            optimal_k = -1
            if self.config.k_type == 'determined':
                optimal_k = self.cluster_strategy.determine_k(pixels_normalized, self.config)
            else:
                optimal_k = self.config.predefined_k
            
            if not isinstance(optimal_k, int) or optimal_k <= 0:
                 raise SegmentationError(f"Invalid number of clusters for SOM: {optimal_k}")
                 
            logger.info(f"Running SOM segmentation with k={optimal_k}")

            # --- Logic from som_segmentation() ---
            som = MiniSom(x=1, y=optimal_k, input_len=3, sigma=0.5, learning_rate=0.25, random_seed=42)
            som.random_weights_init(pixels_normalized)
            som.train_random(pixels_normalized, 100)

            labels_flat = np.array([som.winner(pixel)[1] for pixel in pixels_normalized])
            centers_normalized = np.array([som.get_weights()[0, i] for i in range(optimal_k)])
            centers = np.uint8(np.clip(centers_normalized * 255.0, 0, 255)) # Clip for safety

            # Apply labels back to original preprocessed image shape/pixels
            original_pixels_shape = self.preprocessed_image.shape
            # Need labels corresponding to original pixels, not normalized/quantized ones
            # Re-predict labels on original normalized pixels? Or map quantized pixels back?
            # Easiest might be to apply centers to labels derived from quantized pixels
            # This assumes quantization didn't drastically change pixel assignments.
            if len(labels_flat) != (original_pixels_shape[0] * original_pixels_shape[1]):
                 # If quantization changed shape or pixel count significantly, this mapping is tricky.
                 # Let's try predicting on the original normalized pixels instead.
                 logger.warning("Label length mismatch after quantization. Re-predicting SOM labels on original preprocessed pixels.")
                 pixels_orig_norm = self.preprocessed_image.reshape(-1, 3).astype(np.float32) / 255.0
                 if pixels_orig_norm.shape[0] > 0:
                      labels_flat = np.array([som.winner(pixel)[1] for pixel in pixels_orig_norm])
                 else:
                      raise SegmentationError("Original preprocessed image has zero pixels.")
                 
            if len(labels_flat) != (original_pixels_shape[0] * original_pixels_shape[1]):
                 raise SegmentationError("Label length still mismatch after re-prediction.")

            segmented_image = centers[labels_flat.flatten()].reshape(original_pixels_shape)
            labels_2d = labels_flat.reshape(original_pixels_shape[:2])

            avg_colors = []
            for i in range(optimal_k):
                mask = (labels_2d == i).astype(np.uint8)
                if np.sum(mask) > 0:
                    avg_color_bgr = cv2.mean(self.preprocessed_image, mask=mask)[:3]
                    avg_colors.append(avg_color_bgr)
                else:
                    logger.warning(f"SOM generated empty mask for cluster {i} (k={optimal_k}). Skipping avg color.")
            # --- End of merged logic ---

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
             return SegmentationResult(method_name=method_name, processing_time=duration, n_clusters=optimal_k if 'optimal_k' in locals() and isinstance(optimal_k, int) else 0)


# ====================================================================
# Main Segmenter (Facade Pattern)
# ====================================================================

class Segmenter:
    """Facade class managing the segmentation workflow."""
    def __init__(self,
                 preprocessed_image: np.ndarray,
                 seg_config: SegmentationConfig,
                 model_config: ModelConfig,
                 output_manager: Any, # Use specific type if available: from src.utils.output_manager import OutputManager
                 cluster_strategy: Optional[ClusterStrategy] = None):

        if preprocessed_image is None or preprocessed_image.size == 0:
             raise ValueError("Segmenter cannot be initialized with an empty or None preprocessed image.")

        self.preprocessed_image = preprocessed_image
        self.config = seg_config
        self.models = model_config
        self.output_manager = output_manager
        self.cluster_strategy = cluster_strategy or MetricBasedStrategy()
        self.segmenters: Dict[str, SegmenterBase] = {}
        self._initialize_segmenters()
        logger.info(f"Segmenter (Facade) initialized for k_type='{self.config.k_type}' with methods: {list(self.segmenters.keys())}")

    def _initialize_segmenters(self):
        """Dynamically create segmenter objects based on config."""
        common_args = {
            "preprocessed_image": self.preprocessed_image,
            "config": self.config,
            "models": self.models, # Pass models down
            "cluster_strategy": self.cluster_strategy
        }
        requested_methods = self.config.methods or [] # Ensure it's a list

        # Map method names from config to classes and expected k_type
        method_map = {
             'kmeans_opt': (KMeansSegmenter, 'determined'),
             'kmeans_predef': (KMeansSegmenter, 'predefined'),
             'som_opt': (SOMSegmenter, 'determined'),
             'som_predef': (SOMSegmenter, 'predefined'),
             'dbscan': (DBSCANSegmenter, None) # DBSCAN handles k_type internally
        }
        print(f"DEBUG: [Segmenter._init_seg] Initializing for k_type='{self.config.k_type}'...") 
        for method_key, (SegmenterClass, required_k_type) in method_map.items():
             print(f"DEBUG: [Segmenter._init_seg] Checking method '{method_key}'...") # <-- YENİ
             # Check if method is requested AND k_type matches (or DBSCAN)
             if method_key in requested_methods and (required_k_type is None or required_k_type == self.config.k_type):
                  print(f"DEBUG: [Segmenter._init_seg] Method '{method_key}' matches. Trying to init {SegmenterClass.__name__}...") 
                  try:
                       print(f"DEBUG: [Segmenter._init_seg]   Args check: preprocessed_image is None: {common_args['preprocessed_image'] is None}")
                       print(f"DEBUG: [Segmenter._init_seg]   Args check: config type: {type(common_args['config'])}")
                       print(f"DEBUG: [Segmenter._init_seg]   Args check: models type: {type(common_args['models'])}")
                       print(f"DEBUG: [Segmenter._init_seg]   Args check: cluster_strategy type: {type(common_args['cluster_strategy'])}")
                       
                       instance = SegmenterClass(**common_args) 
                       self.segmenters[method_key] = instance
                       print(f"DEBUG: [Segmenter._init_seg]   Successfully initialized {SegmenterClass.__name__} for '{method_key}'") 
                       logger.debug(f"Initialized segmenter for {method_key}")
                  except Exception as e:
                       print(f"DEBUG: [Segmenter._init_seg] *** EXCEPTION during {SegmenterClass.__name__} init for '{method_key}': {type(e).__name__}: {e} ***") # Hata mesajını daha detaylı yazdır                       logger.error(f"Failed to initialize segmenter for {method_key}: {e}", exc_info=True)
                       # Hatayı tekrar fırlat ki main.py yakalasın ama burada ne olduğunu bilelim
                       raise
             else:
                 print(f"DEBUG: [Segmenter._init_seg] Skipping method '{method_key}' (Not requested or k_type mismatch).") 
               

    def process(self) -> ProcessingResult:
        """Run all initialized segmentation methods and collect results."""

        # Get image name stem for saving path (might need adjustment)
        # Assuming output_manager has a way to provide this, or we derive it
        try:
             # This is a potential dependency, might need rework
             image_name_stem = self.output_manager.get_current_image_name()
             preprocessed_path = self.output_manager.get_preprocessed_image_path(image_name_stem)
        except AttributeError:
             logger.warning("OutputManager does not have expected methods for image name/path. Using placeholder.")
             preprocessed_path = "unknown_preprocessed_image.png"


        results_dict: Dict[str, SegmentationResult] = {}

        if not self.segmenters:
             logger.warning(f"No segmentation methods initialized for k_type='{self.config.k_type}' and requested methods='{self.config.methods}'. Returning empty results.")
             return ProcessingResult(preprocessed_path=preprocessed_path, results=results_dict)

        for method_name, segmenter_instance in self.segmenters.items():
            try:
                logger.info(f"Running segmentation method: {method_name}")
                result = segmenter_instance.segment()

                if result and result.is_valid():
                    logger.info(f"Method {method_name} completed in {result.processing_time:.2f}s with {result.n_clusters} clusters.")
                    # Saving is handled here
                    self.output_manager.save_segmentation_result(
                        result.segmented_image,
                        method_name,
                        self.config.k_type # Pass k_type for filename differentiation
                    )
                    results_dict[method_name] = result
                else:
                    logger.warning(f"Method {method_name} did not produce a valid result.")
                    # Store the invalid result object if needed for debugging
                    results_dict[method_name] = result or SegmentationResult(method_name=method_name) # Ensure a result object exists

            except Exception as e:
                logger.error(f"Critical error processing method {method_name}: {e}", exc_info=True)
                results_dict[method_name] = SegmentationResult(method_name=method_name) # Store empty result

        logger.info(f"Segmentation processing completed for k_type='{self.config.k_type}'.")
        return ProcessingResult(
            preprocessed_path=preprocessed_path,
            results=results_dict
        )