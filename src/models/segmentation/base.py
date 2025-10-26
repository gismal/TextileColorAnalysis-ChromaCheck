import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans 

# Type hinting için (gerçek import değil)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.models.pso_dbn import DBN
    from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

# --- Custom Exceptions ---

class SegmentationError(Exception):
    """Custom exception for failures during the segmentation process."""
    pass

class InvalidConfigurationError(ValueError):
    """Custom exception for invalid segmentation configuration."""
    pass

# --- Data Structures (Dataclasses) ---

@dataclass
class SegmentationConfig:
    """
    Holds all configuration settings for a segmentation run.

    This object defines *what* to run (methods, k_type) and 
    *how* to run it (parameters like predefined_k, dbscan_eps).

    Args:
        target_colors (np.ndarray): The reference LAB colors to compare against.
        distance_threshold (float): (Currently unused?) Threshold for distance.
        predefined_k (int): The fixed number of clusters (k) to use 
                            when k_type is 'predefined'.
        k_values (List[int]): The range of k values (e.g., [2, 3, 4, 5]) 
                              to test when k_type is 'determined'.
        som_values (List[int]): (Currently unused?) Specific k range for SOM.
        k_type (str): Either 'determined' (find best k) or 'predefined' (use predefined_k).
        methods (List[str]): List of method keys to run 
                             (e.g., 'kmeans_opt', 'kmeans_predef').
        dbscan_eps (float): The 'eps' parameter for DBSCAN.
        dbscan_min_samples (int): The 'min_samples' parameter for DBSCAN.
    """
    target_colors: np.ndarray
    distance_threshold: float
    predefined_k: int
    k_values: List[int]
    som_values: List[int]
    # --- DÜZELTME: 'k_types' -> 'k_type' ---
    k_type: str = 'determined' 
    methods: List[str] = field(default_factory=lambda: ['kmeans_opt', 'kmeans_predef', 'som_opt', 'som_predef', 'dbscan'])
    dbscan_eps: float = 10.0
    dbscan_min_samples: int = 5
    strategy_subsample: int = 10000 # MetricBasedStrategy için alt örneklem
    dbscan_eps_range: List[float] = field(default_factory=lambda: [10.0, 15.0, 20.0]) # DBSCAN 'determined' için
    dbscan_min_samples_range: List[int] = field(default_factory=lambda: [5, 10, 20]) # DBSCAN 'determined' için
    som_iterations: int = 100 # SOM eğitim iterasyonları
    som_sigma: float = 0.5   # SOM sigma
    som_learning_rate: float = 0.25 # SOM öğrenme oranı
    
@dataclass
class SegmentationResult:
    """
    Holds the output data from a single segmentation method.

    This is a standardized container, ensuring that KMeans, SOM, and DBSCAN
    all return data in the same, predictable format for analysis.

    Args:
        method_name (str): The name of the method that produced this result 
                           (e.g., 'kmeans_opt').
        segmented_image (Optional[np.ndarray]): The resulting segmented image (H, W, 3).
        avg_colors (List[Tuple[float, float, float]]): List of average RGB colors 
                                                       for each found cluster.
        labels (Optional[np.ndarray]): Flattened array of labels (size H*W) 
                                       assigning each pixel to a cluster.
        n_clusters (int): The number of clusters found.
        processing_time (float): Time taken (in seconds) for the segment() method.
    """
    method_name: str
    segmented_image: Optional[np.ndarray] = None
    avg_colors: List[Tuple[float, float, float]] = field(default_factory=list)
    labels: Optional[np.ndarray] = None
    n_clusters: int = 0
    processing_time: float = 0.0

    def is_valid(self) -> bool:
        """
        Checks if the segmentation result is valid and usable for analysis.

        Returns:
            bool: True if the result contains an image, colors, and clusters.
        """
        return (self.segmented_image is not None and
                self.avg_colors is not None and
                len(self.avg_colors) > 0 and
                self.n_clusters > 0)
        
@dataclass
class ModelConfig:
    """
    Holds the trained models and scalers needed by segmentation methods.

    This object is created by the pipeline *after* training and is passed
    to the Segmenter facade. It bundles all ML-related dependencies.
    
    Args:
        dbn (DBN): The trained PSO-DBN model.
        scalers (List[MinMaxScaler]): A list of fitted scalers [scaler_x, scaler_y, scaler_y_ab].
        reference_kmeans_result (Optional[SegmentationResult]): The raw segmentation
            result from processing the K-Means on the reference image.
        reference_som_result (Optional[SegmentationResult]): The raw segmentation
            result from processing the SOM on the reference image.
    """
    dbn: 'DBN'
    scalers: List['MinMaxScaler'] # 'MinMaxScaler' tırnak içinde çünkü TYPE_CHECKING bloğunda
    # --- DÜZELTME: 'referece_' -> 'reference_' ---
    reference_kmeans_result: Optional[SegmentationResult]
    reference_som_result: Optional[SegmentationResult]
    
    
# --- Abstract Base Class (The "Contract") ---

class SegmenterBase(ABC):
    """
    Abstract Base Class (ABC) for all segmentation methods.

    This class defines the "contract" that all segmenters (KMeans, SOM, etc.)
    must follow. It guarantees that every segmenter will have a `segment()`
    method, allowing the main pipeline to treat them all identically
    (this is called Polymorphism).

    Args:
        preprocessed_image (np.ndarray): The (H, W, 3) image ready for segmentation.
        config (SegmentationConfig): The configuration object for this run.
        models (ModelConfig): The container for trained models (DBN, scalers).
        cluster_strategy (ClusterStrategy): The strategy object to use for
                                            determining 'k' if k_type is 'determined'.
    """
    def __init__(self,
                 preprocessed_image: np.ndarray,
                 config: SegmentationConfig,
                 models: ModelConfig,
                 cluster_strategy: 'ClusterStrategy'): # 'ClusterStrategy' tırnak içinde (doğru)
        if preprocessed_image is None:
             raise ValueError(f"{self.__class__.__name__} received a None preprocessed_image.")
        self.preprocessed_image = preprocessed_image
        self.config = config
        self.models = models
        self.cluster_strategy = cluster_strategy
        self.original_shape = preprocessed_image.shape
        self.pixels_flat = preprocessed_image.reshape(-1, 3).astype(np.float32)
        
    @abstractmethod
    def segment(self) -> SegmentationResult:
        """
        Performs the segmentation on the preprocessed image.

        This method MUST be implemented by all child classes.

        Returns:
            SegmentationResult: A data object containing all results.
        """
        pass
    
    # --- Common Helper Method ---
    # Neden: Quantize, tüm segmenter'lar tarafından (DBSCAN, KMeans, SOM)
    # kullanıldığı için, onu temel sınıfa (base class) taşımak
    # kod tekrarını (duplication) engeller.
    def quantize_image(self, n_colors=50) -> Optional[np.ndarray]:
        """
        Reduces the number of unique colors in the image using K-Means.

        This is a common preprocessing step for clustering algorithms
        to make them run faster and be less sensitive to noise.

        Args:
            n_colors (int): The target number of colors to quantize down to.

        Returns:
            Optional[np.ndarray]: The quantized image (H, W, 3) or None on failure.
        
        Raises:
            ValueError: If K-Means fails unexpectedly.
        """
        n_colors = self.config.quantization_colors # <-- Config'den al
        subsample_threshold = self.config.quantization_subsample # <-- Config'den al
        
        if self.preprocessed_image is None or self.preprocessed_image.size == 0:
            logger.warning("Cannot quantize None or empty image.")
            return None
        
        logger.info(f"Quantizing image (shape: {self.preprocessed_image.shape}) to approx {n_colors} colors")
        pixels = self.preprocessed_image.reshape(-1, 3).astype(np.float32)
        n_pixels_total = pixels.shape[0]
        
        # Ensure n_colors is valid
        actual_n_colors = max(1, min(n_colors, n_pixels_total))
        if actual_n_colors != n_colors:
            logger.warning(f"Adjusted quantization n_colors from {n_colors} to {actual_n_colors} (due to pixel count)")
            
        if actual_n_colors < 1:
             logger.error("Cannot quantize to less than 1 color.")
             return None
             
        # Handle edge case of a single-color image
        if actual_n_colors == 1 and n_pixels_total > 0:
             center = np.mean(pixels, axis=0)
             quantized = np.tile(center, (self.original_shape[0], self.original_shape[1], 1)).astype(np.uint8)
             return quantized
        
        # Subsample for efficiency if image is large
        if n_pixels_total > subsample_threshold: 
            indices = np.random.choice(n_pixels_total, subsample_threshold, replace=False)
            pixels_sample = pixels[indices]
        else:
            pixels_sample = pixels
            
        # Ensure we have enough samples to fit the clusters
        if pixels_sample.shape[0] < actual_n_colors:
             logger.warning(f"Sample size ({pixels_sample.shape[0]}) < n_colors ({actual_n_colors}). Using sample size as n_colors.")
             actual_n_colors = pixels_sample.shape[0]
             if actual_n_colors < 1:
                 logger.error("Cannot quantize with zero samples.")
                 return None
                 
        try:
            # --- KMeans'in kullanıldığı yer ---
            kmeans = KMeans(n_clusters=actual_n_colors, n_init='auto', random_state=42).fit(pixels_sample)
            # Predict on all original pixels
            labels = kmeans.predict(pixels)
            quantized_pixels = kmeans.cluster_centers_[labels]
            return quantized_pixels.reshape(self.original_shape).astype(np.uint8)
        
        except Exception as e:
             logger.error(f"Error during quantization K-Means: {e}", exc_info=True)
             raise ValueError(f"K-Means quantization failed: {e}")