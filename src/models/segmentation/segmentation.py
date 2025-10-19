import logging
import os
import cv2
import numpy as np
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from minisom import MiniSom
from src.utils.color.color_analysis import ColorMetricCalculator
from src.utils.image_utils import ciede2000_distance
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from src.models.pso_dbn import DBN
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


#
#  Data Classes and Custom Exceptions
#

class SegmentationError(Exception):
    """error exception for general segmentation errors"""
    pass

class InvalidConfigurationError(ValueError):
    """invalid configuration errors"""
    pass

@dataclass
class SegmentationConfig:
    """configs for segmentation process"""
    target_colors: np.ndarray
    distance_threshold: float
    predefined_k: int
    k_values: List[int]
    som_values: List[int]
    k_type: str = 'determined'
    methods: List[str] = field(default_factory = lambda: ['kmeans_opt', 'kmeans_predef', 'som_opt', 'som_predefe', 'dbscan'])
    
    dbscan_eps: float = 10.0
    dbscan_min_samples: int = 3
    
@dataclass
class ModelConfig:
    """configs and references for model"""
    dbn: DBN
    scalers: List[MinMaxScaler]
    reference_kmeans_opt: Dict[str, Any]
    reference_som_opt: Dict[str, Any]
    
@dataclass 
class SegmentationResult:
    """keeps the result of one segmentation method"""
    method_name: str
    segmented_image: Optional[np.ndarray] = None
    avg_colors: List[Tuple[float, float, float]] = field(default_factory = list)
    labels: Optional[np.ndarray] = None
    n_clusters: int = 0
    processing_time: float = 0.0   
    
    def is_valid(self) -> bool:
        """checks if result is valid"""
        return (self.segmented_images is not None and 
                self.avg_color is not None and
                len(self.avg_colors) > 0 and
                self.n_clusters > 0)
    
@dataclass
class ProcessingResult:
    """collects all the segmentation methods"""
    preprocessed_path: str
    results: Dict[str, SegmentationResult] = field(default_factory = dict)
    
# Strategy Pattern for CLuster Determination
# Abstract base class for cluster determination strategies
class ClusterStrategy(ABC):
    @abstractmethod
    def determine_k(self, pixels: np.ndarray, config: SegmentationConfig) -> int:
        pass

# Concrete strategy to find the optimal k using clustering metrics
class MetricBasedStrategy(ClusterStrategy):
    def determine_k(self, pixels : np.ndarray, config: SegmentationConfig, n_runs = 3) -> int:
        """optimal k based on combined normalized metrics"""
        # determine the k range based on the method
        k_range_list = config.k_values if 'kmeans' in config.methods else config.som_values
        min_k = min(k_range_list)
        max_k = max(k_range_list)
        
        unique_colors = np.unique(pixels, axis = 0)
        # adjust max_k to be sensible based on unique colors
        dynamic_max_k = min(max_k, max(min_k + 2, len(unique_colors) // 20))
        # ensure min_k is not greater than dynamic_max_k
        min_k = min(min_k, dynamic_max_k)
        
        logger.info(f"Unique colors: {len(unique_colors)}. Adjusted k-range: [{min_k}, {dynamic_max_k}]")

        if len(pixels) > 10000:
            logger.info("Subsampling pixels for cluster analysis efficiency")
            pixels = pixels[np.random.choice(len(pixels), 10000, replace=False)]

        scores = {'silhouette': [], 'ch': []}
        k_range = list(range(min_k, dynamic_max_k + 1))

        if not k_range:
            logger.warning(f"K-range is empty (min_k={min_k}, max_k={dynamic_max_k}). Defaulting to predefined_k={config.predefined_k}")
            return config.predefined_k

        for k in k_range:
            logger.info(f"Testing k={k}")
            try:
                kmeans = KMeans(n_clusters=k, n_init=n_runs, random_state=42).fit(pixels)
                labels = kmeans.labels_
                
                # Check for single cluster case
                if len(np.unique(labels)) < 2:
                    logger.warning(f"Only 1 cluster found for k={k}. Assigning worst score.")
                    scores['silhouette'].append(-1.0)
                    scores['ch'].append(0.0)
                    continue

                scores['silhouette'].append(silhouette_score(pixels, labels))
                scores['ch'].append(calinski_harabasz_score(pixels, labels))
                logger.info(f"Metrics for k={k}: Silhouette={scores['silhouette'][-1]:.3f}, CH={scores['ch'][-1]:.1f}")
            
            except Exception as e:
                logger.error(f"Error calculating metrics for k={k}: {e}")
                scores['silhouette'].append(-1.0) # Penalize failure
                scores['ch'].append(0.0)

        # Normalize scores to [0, 1]
        sil_scores = np.array(scores['silhouette'])
        ch_scores = np.array(scores['ch'])
        
        # Check for division by zero if all scores are identical
        if (sil_scores.max() - sil_scores.min()) < 1e-10:
            norm_sil = np.zeros_like(sil_scores)
        else:
            norm_sil = (sil_scores - sil_scores.min()) / (sil_scores.max() - sil_scores.min())
            
        if (ch_scores.max() - ch_scores.min()) < 1e-10:
            norm_ch = np.zeros_like(ch_scores)
        else:
            norm_ch = (ch_scores - ch_scores.min()) / (ch_scores.max() - ch_scores.min())
        
        avg_scores = (norm_sil + norm_ch) / 2
        optimal_k = k_range[np.argmax(avg_scores)]
        logger.info(f"Optimal k determined: {optimal_k} (scores: {avg_scores})")
        
        return optimal_k

# Abstract base class for segmentation
class SegmenterBase(ABC):
    def __init__(self,
                 preprocessed_image: np.ndarray,
                 config: SegmentationConfig,
                 models: ModelConfig,
                 cluster_strategy: ClusterStrategy):
        
        self.preprocessed_image = preprocessed_image
        self. config = config
        self.models = models
        self.cluster_strategy = cluster_strategy

    @abstractmethod
    def segment(self):
        """Segment the image and return results.
        
        Returns:
            tuple: (segmented_image, avg_colors, labels)
        """
        pass

    def quantize_image(self, n_colors=50):
        logger.info(f"Quantizing image with {len(np.unique(self.preprocessed_image.reshape(-1, 3), axis=0))} unique colors to {n_colors}")
        pixels = self.preprocessed_image.reshape(-1, 3).astype(np.float32)
        
        # Subsample if the image is very large
        if pixels.shape[0] > 20000:
            indices = np.random.choice(pixels.shape[0], 20000, replace=False)
            pixels_sample = pixels[indices]
        else:
            pixels_sample = pixels
            
        kmeans = KMeans(n_clusters=n_colors, n_init=3, random_state=42).fit(pixels_sample)
        
        # Predict on the full pixel set
        labels = kmeans.predict(pixels)
        quantized_pixels = kmeans.cluster_centers_[labels]
        return quantized_pixels.reshape(self.preprocessed_image.shape).astype(np.uint8)

## Concrete Segmentation Implementations
class KMeansSegmenter(SegmenterBase):
    """Segments an fabric using KMeans clustering"""
    def __init__(self, preprocessed_image, config: SegmentationConfig, models: ModelConfig, cluster_strategy = ClusterStrategy):
        super().__init__(preprocessed_image, config, models, cluster_strategy)
        logger.info(f"KMeansSegmenter initialized with k_type: {self.config.k_type}")
    
    def segment(self) -> SegmentationResult:
        start_time = time.time()
        
        # quantize the images to get stable pixels for clustering
        quantized_pixels = self.quantize_image().reshape(-1,3).astype(np.float32)
        
        if self.config.k_type == 'determined':
            optimal_k = self.cluster_strategy.determine_k(quantized_pixels, self.config)
            method_name = "kmeans_opt"
            logger.info(f"Optimal k determined: {optimal_k}")
        else:
            optimal_k = self.config.predefined_k
            method_name = "kmeans_predef"
            logging.info(f"Using predefined k: {optimal_k}")
            
        pixels_for_segmentation = self.preprocessed_image.reshape(-1,3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        
        _, labels_flat, centers = cv2.kmeans(
            pixels_for_segmentation, optimal_k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        
        centers = np.uint8(centers)
        segmented_image = centers[labels_flat.flatten()].reshape(self.preprocessed_image.shape)
        labels_2d = labels_flat.reshape(self.preprocessed_image.shape[:2])
        
        avg_colors = []
        for i in range(optimal_k):
            mask = (labels_2d == i).astype(np.uint8)
            # calculate avg color from the original preprocessed image for accuracy
            avg_color_bgr = cv2.mean(self.preprocessed_image, mask = mask)[:3]
            avg_colors.append(avg_color_bgr)
            
        duration = time.time() - start_time
        
        return SegmentationResult(
            method_name=method_name,
            segmented_image=segmented_image,
            avg_colors=avg_colors,
            labels=labels_flat,
            n_clusters=optimal_k,
            processing_time=duration
        )
        
class DBSCANSegmenter(SegmenterBase):
    #Segmentation with DBSCAN
    def __init__(self, preprocessed_image, config: SegmentationConfig, models: ModelConfig, cluster_strategy: ClusterStrategy):
        super().__init__(preprocessed_image, config, models, cluster_strategy)
        logger.info(f"DBSCANSegmenter initialized with k_type: {self.config.k_type}")

    def _run_dbscan(self, pixels: np.ndarray, eps: float, min_samples: int) -> Tuple[np.ndarray, int]:
        """Private helper to run DBSCAN and return labels and cluster count."""
        try:
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(pixels)
            labels = db.labels_
            n_clusters = len(np.unique(labels[labels >= 0])) # Count clusters, ignore noise (-1)
            logger.info(f"DBSCAN(eps={eps}, min={min_samples}) found {n_clusters} clusters.")
            return labels, n_clusters
        except Exception as e:
            logger.error(f"Error during DBSCAN clustering: {e}")
            return np.full(len(pixels), -1), 0

    def _find_optimal_dbscan_params(self, pixels: np.ndarray) -> Tuple[float, int]:
        """Private helper to find best DBSCAN params (from optimal_clusters_dbscan)."""
        logger.info("Finding optimal DBSCAN parameters...")
        # These values could also be moved to config
        eps_values = [10, 15, 20]
        min_samples_values = [5, 10, 20] 
        
        best_silhouette = -1.1
        best_params = (self.config.dbscan_eps, self.config.dbscan_min_samples) # Default

        for eps in eps_values:
            for min_samples in min_samples_values:
                labels, n_clusters = self._run_dbscan(pixels, eps, min_samples)
                
                # Need at least 2 clusters to calculate silhouette
                if n_clusters > 1:
                    silhouette_avg = silhouette_score(pixels, labels)
                    logger.info(f"DBSCAN params (eps={eps}, min={min_samples}) -> silhouette: {silhouette_avg:.3f}")
                    if silhouette_avg > best_silhouette:
                        best_silhouette = silhouette_avg
                        best_params = (eps, min_samples)
                else:
                     logger.info(f"DBSCAN params (eps={eps}, min={min_samples}) -> found < 2 clusters. Skipping score.")
        
        logger.info(f"Optimal DBSCAN parameters found: eps={best_params[0]}, min_samples={best_params[1]}")
        return best_params

    def segment(self) -> SegmentationResult:
        start_time = time.time()
        pixels = self.quantize_image().reshape(-1, 3).astype(np.float32)
        
        if self.config.k_type == 'determined':
            eps, min_samples = self._find_optimal_dbscan_params(pixels)
        else:
            eps = self.config.dbscan_eps
            min_samples = self.config.dbscan_min_samples
            logger.info(f"Using predefined DBSCAN params: eps={eps}, min_samples={min_samples}")
            
        labels, n_clusters = self._run_dbscan(pixels, eps, min_samples)

        if n_clusters == 0:
            logger.warning("No clusters found in DBSCAN. Returning empty result.")
            return SegmentationResult(method_name="dbscan", processing_time=time.time() - start_time)

        # Use original preprocessed pixels for color averaging
        original_pixels = self.preprocessed_image.reshape(-1, 3).astype(np.float32)
        
        centers = np.array([np.mean(original_pixels[labels == label], axis=0) 
                            for label in range(n_clusters)])
        centers = np.uint8(centers)
        
        mask = labels >= 0
        segmented_flat = np.zeros_like(original_pixels, dtype=np.uint8)
        
        # Map labels (0, 1, 2...) to the centers array
        segmented_flat[mask] = centers[labels[mask]]
        segmented_image = segmented_flat.reshape(self.preprocessed_image.shape)
        
        avg_colors = [tuple(c) for c in centers] # The centers are the average colors
        
        duration = time.time() - start_time
        return SegmentationResult(
            method_name="dbscan",
            segmented_image=segmented_image,
            avg_colors=avg_colors,
            labels=labels,
            n_clusters=n_clusters,
            processing_time=duration
        )

class SOMSegmenter(SegmenterBase):
    """Segmentation with Self-Organizing Maps (SOM)"""
    def __init__(self, preprocessed_image, config: SegmentationConfig, models: ModelConfig, cluster_strategy: ClusterStrategy):
        super().__init__(preprocessed_image, config, models, cluster_strategy)
        logger.info(f"SOMSegmenter initialized with k_type: {self.config.k_type}")

    def segment(self) -> SegmentationResult:
        start_time = time.time()
        # SOM works best on normalized pixels [0, 1]
        pixels_normalized = self.quantize_image().reshape(-1, 3).astype(np.float32) / 255.0
        
        if self.config.k_type == 'determined':
            optimal_k = self.cluster_strategy.determine_k(pixels_normalized, self.config)
            method_name = "som_opt"
            logger.info(f"Optimal k determined: {optimal_k}")
        else:
            optimal_k = self.config.predefined_k
            method_name = "som_predef"
            logger.info(f"Using predefined k: {optimal_k}")
        
        # --- Logic from som_segmentation() is now INSIDE the class ---
        som = MiniSom(x=1, y=optimal_k, input_len=3, sigma=0.5, learning_rate=0.25)
        som.random_weights_init(pixels_normalized)
        som.train_random(pixels_normalized, 100) # 100 iterations
        
        # Get the winning neuron (cluster index) for each pixel
        labels_flat = np.array([som.winner(pixel)[1] for pixel in pixels_normalized])
        
        # Get the cluster centers (weights) from the SOM, scaling back to [0, 255]
        centers_normalized = np.array([som.get_weights()[0, i] for i in range(optimal_k)])
        centers = np.uint8(centers_normalized * 255.0)

        segmented_image = centers[labels_flat].reshape(self.preprocessed_image.shape)
        labels_2d = labels_flat.reshape(self.preprocessed_image.shape[:2])
        
        avg_colors = []
        for i in range(optimal_k):
            mask = (labels_2d == i).astype(np.uint8)
            avg_color_bgr = cv2.mean(self.preprocessed_image, mask=mask)[:3]
            avg_colors.append(avg_color_bgr)
        # --- End of merged logic ---

        duration = time.time() - start_time
        return SegmentationResult(
            method_name=method_name,
            segmented_image=segmented_image,
            avg_colors=avg_colors,
            labels=labels_flat,
            n_clusters=optimal_k,
            processing_time=duration
        )

"""
Main Segmenter: Facade Patters 
"""
class Segmenter:
    """
    Facade class that manages the entire segmentation workflow
    It initializes the correct segmetnation strategies based on the config and runs them,
    handles the result and saves.
    """
    def __init__(self, 
                 preprocessed_image: np.ndarray,
                 seg_config: SegmentationConfig,
                 model_config: ModelConfig,
                 output_manager: Any
                 cluster_strategy: Optional[ClusterStrategy] = None):
        
        self.preprocessed_image = preprocessed_image
        self.config = seg_config
        self.models = model_config
        
        # store output manager here
        self.output_manager = output_manager
        
        self.cluster_strategy = cluster_strategy or MetricBasedStrategy()
        
        # this dictionary will hold the specific segmenter instances
        self.segmenters: Dict[str, SegmenterBase] = {}
        self._initialize_segmenters()
        logger.info(f"Segmenter (Facade) initialized for k_type='{self.config.k_type}' with methods: {list(self.segmenters.keys())}")
    
    def _initialize_segmenters(self):
        """
        Dynamically create the segmenter objects based on the config
        """
        common_args = {
            "preprocessed_image": self.preprocessed_image,
            "config": self.config,
            "models": self.models,
            "cluster_strategy": self.cluster_strategy
        }
        
        # create instances based on k_type and allowed methods
        if self.config.k_type == 'determined':
            if 'kmeans_opt' in self.config.methods:
                self.segmenters['kmeans_opt'] = KMeansSegmenter(**common_args)
            if 'som_opt' in self.config.methods:
                self.segmenters['som_opt'] = SOMSegmenter(**common_args)
                
        elif self.config.k_type == 'predefined':
            if 'kmeans_predef' in self.config.methods:
                self.segmenters['kmeans_predef'] = KMeansSegmenter(**common_args)
            if 'som_predef' in self.config.methods:
                self.segmenters['som_predef'] = SOMSegmenter(**common_args)
                
        if 'dbscan' in self.config.methods:
            self.segmenters['dbscan'] = DBSCANSegmenter(**common_args)
        
    def process(self):
        """
        Run all initialized segmentation methods and collect their results.
        This method is responsible for saving the output images.
        """
        if self.preprocessed_image is None:
            raise SegmentationError("Preprocessed image is None, cannot segment.")
            
        # The preprocessed image is saved by main.py. We'll just create a
        # placeholder path for the ProcessingResult object.
        preprocessed_path = self.output_manager.get_preprocessed_image_path()
        
        results_dict: Dict[str, SegmentationResult] = {}
        
        for method_name, segmenter_instance in self.segmenters.items():
            try:
                logger.info(f"Running segmentation method: {method_name}")
                result = segmenter_instance.segment()
                
                # REFACTOR 2: Saving is handled here, in the Facade.
                if result.is_valid():
                    self.output_manager.save_segmentation_result(
                        result.segmented_image, 
                        method_name, 
                        self.config.k_type
                    )
                    results_dict[method_name] = result
                    logger.info(f"Method {method_name} completed in {result.processing_time:.2f}s")
                else:
                    logger.warning(f"Method {method_name} did not produce a valid result.")
                    results_dict[method_name] = result # Store the invalid result anyway
                    
            except Exception as e:
                logger.error(f"Error processing method {method_name}: {e}", exc_info=True)
                results_dict[method_name] = SegmentationResult(method_name=method_name) # Store empty result on failure

        logging.info("Segmentation process completed for all methods.")
        return ProcessingResult(
            preprocessed_path=preprocessed_path,
            results=results_dict
        )