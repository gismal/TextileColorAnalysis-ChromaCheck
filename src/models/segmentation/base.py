import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans 

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.models.pso_dbn import DBN
    from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

# --- Custom Exceptions ---
class SegmentationError(Exception):
    """Custom exception for failures during the segmentation process"""
    pass

class InvalidConfigurationError(ValueError):
    """Custom exception for invalid segmentation configuration"""
    pass

# --- Data Structures (Dataclasses) ---
@dataclass
class SegmentationConfig:
    """
    Holds all configuration settings for a segmentation run.
    This object defines what to run (methods, k_type) and how to run it (parameters like predefined_k, dbscan_eps).

    Args:
        target_colors (np.ndarray): The reference LAB colors to compare against.
        distance_threshold (float): Threshold for distance.
        predefined_k (int): The fixed number of clusters (k) to use when k_type is 'predefined'.
        k_values (List[int]): The range of k values (e.g., [2, 3, 4, 5]) to test when k_type is 'determined'.
        som_values (List[int]): Specific k range for SOM.
        k_type (str): Either 'determined' (find best k) or 'predefined' (use predefined_k).
        methods (List[str]): List of method keys to run (e.g., 'kmeans_opt', 'kmeans_predef').
        dbscan_eps (float): The 'eps' parameter for DBSCAN.
        dbscan_min_samples (int): The 'min_samples' parameter for DBSCAN.
    """
    target_colors: np.ndarray
    distance_threshold: float
    predefined_k: int
    k_values: List[int]
    som_values: List[int]
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
    This is a standardized container, ensuring that KMeans, SOM, and DBSCAN all return data in the same, 
    predictable format for analysis.

    Args:
        method_name (str): The name of the method that produced this result (e.g., 'kmeans_opt').
        segmented_image (Optional[np.ndarray]): The resulting segmented image (H, W, 3).
        avg_colors (List[Tuple[float, float, float]]): List of average RGB colors for each found cluster.
        labels (Optional[np.ndarray]): Flattened array of labels (size H*W) assigning each pixel to a cluster.
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

    This object is created by the pipeline *after* training and is passed to the Segmenter facade. It bundles all ML-related dependencies.
    
    Args:
        dbn (DBN): The trained PSO-DBN model.
        scalers (List[MinMaxScaler]): A list of fitted scalers [scaler_x, scaler_y, scaler_y_ab].
        reference_kmeans_result (Optional[SegmentationResult]): The raw segmentation result from processing the K-Means on the reference image.
        reference_som_result (Optional[SegmentationResult]): The raw segmentation result from processing the SOM on the reference image.
    """
    dbn: 'DBN'
    scalers: List['MinMaxScaler'] 
    reference_kmeans_result: Optional[SegmentationResult]
    reference_som_result: Optional[SegmentationResult]
    
    
# --- Abstract Base Class (The "Contract") ---
class SegmenterBase(ABC):
    """
    Abstract Base Class (ABC) for all segmentation methods.

    This class defines the "contract" that all segmenters (KMeans, SOM, etc.) must follow. 
    It guarantees that every segmenter will have a `segment()` method, allowing the main pipeline to treat them all identically

    Args:
        preprocessed_image (np.ndarray): The (H, W, 3) image ready for segmentation.
        config (SegmentationConfig): The configuration object for this run.
        models (ModelConfig): The container for trained models (DBN, scalers).
        cluster_strategy (ClusterStrategy): The strategy object to use for determining 'k' if k_type is 'determined'.
    """
    def __init__(self,
                 preprocessed_image: np.ndarray,
                 config: SegmentationConfig,
                 models: ModelConfig,
                 cluster_strategy: 'ClusterStrategy'): 
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
    