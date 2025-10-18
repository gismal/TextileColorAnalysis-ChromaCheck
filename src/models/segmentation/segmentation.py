import logging
import os
import cv2
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union, Tuple, Callable
from contextlib import contextmanager
import time
from functools import lru_cache, wraps
import threading
from concurrent.futures import ThreadPoolExecutor

from sklearn.cluster import KMeans
from src.utils.color.color_analysis import ColorMetricCalculator
from src.utils.segmentation_utils import k_mean_segmentation, optimal_clusters, optimal_dbscan, optimal_som
from src.utils.image_utils import ciede2000_distance

# ==============================================================================
# CONFIGURATION CLASSES
# ==============================================================================

@dataclass
class SegmentationConfig:
    """Configuration for segmentation parameters."""
    target_colors: np.ndarray
    distance_threshold: float
    predefined_k: int
    k_values: List[int]
    som_values: List[int]
    k_type: str  # 'determined' or 'predefined'
    methods: List[str] = field(default_factory=lambda: ['kmeans_opt', 'som_opt', 'dbscan'])
    quantization_colors: int = 50
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        errors = SegmentationValidator.validate_config(self)
        if errors:
            raise InvalidConfigurationError(f"Configuration errors: {', '.join(errors)}")

@dataclass
class ClusterStrategyConfig:
    """Configuration for cluster determination strategies."""
    min_k: int = 3
    max_k: int = 10
    subsample_size: int = 2000
    random_state: int = 42

@dataclass
class ModelConfig:
    """Configuration for ML models and references."""
    dbn: Optional[Any] = None
    scalers: Optional[List[Any]] = None
    reference_kmeans_opt: Optional[Dict] = None
    reference_som_opt: Optional[Dict] = None

@dataclass
class SegmentationResult:
    """Result of a segmentation operation."""
    segmented_image: np.ndarray
    avg_colors: List[np.ndarray]
    labels: np.ndarray
    similarity: List[float]
    n_clusters: int
    method: str
    processing_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None

    def is_valid(self) -> bool:
        """Check if segmentation result is valid."""
        return (self.success and 
                self.segmented_image is not None and 
                len(self.avg_colors) > 0 and
                self.labels is not None)

@dataclass
class ProcessingResult:
    """Result of processing an image with multiple segmentation methods."""
    results: Dict[str, SegmentationResult]
    errors: Dict[str, str]
    preprocessed_path: str
    total_processing_time: float = 0.0
    
    def get_successful_results(self) -> Dict[str, SegmentationResult]:
        """Get only successful segmentation results."""
        return {k: v for k, v in self.results.items() if v.is_valid()}
    
    def has_errors(self) -> bool:
        """Check if there were any processing errors."""
        return len(self.errors) > 0

# ==============================================================================
# CUSTOM EXCEPTIONS
# ==============================================================================

class SegmentationError(Exception):
    """Base exception for segmentation errors."""
    pass

class InsufficientDataError(SegmentationError):
    """Raised when there's insufficient data for segmentation."""
    pass

class InvalidConfigurationError(SegmentationError):
    """Raised when configuration is invalid."""
    pass

class SegmentationMethodError(SegmentationError):
    """Raised when a segmentation method fails."""
    pass

# ==============================================================================
# VALIDATION
# ==============================================================================

class SegmentationValidator:
    """Validator for segmentation inputs and configurations."""
    
    @staticmethod
    def validate_image(image: np.ndarray) -> bool:
        """Validate input image."""
        return (image is not None and 
                len(image.shape) == 3 and 
                image.shape[2] == 3 and
                image.size > 0)
    
    @staticmethod
    def validate_config(config: SegmentationConfig) -> List[str]:
        """Validate segmentation configuration."""
        errors = []
        
        if config.predefined_k <= 0:
            errors.append("predefined_k must be positive")
        
        if not config.k_values or min(config.k_values) <= 0:
            errors.append("k_values must contain positive integers")
        
        if not config.som_values or min(config.som_values) <= 0:
            errors.append("som_values must contain positive integers")
        
        if config.k_type not in ['determined', 'predefined']:
            errors.append("k_type must be 'determined' or 'predefined'")
        
        if config.distance_threshold <= 0:
            errors.append("distance_threshold must be positive")
        
        if config.quantization_colors <= 0:
            errors.append("quantization_colors must be positive")
        
        return errors

# ==============================================================================
# UTILITY CONTEXT MANAGERS
# ==============================================================================

@contextmanager
def timer(operation_name: str):
    """Context manager for timing operations."""
    start = time.time()
    try:
        yield
    finally:
        duration = time.time() - start
        logging.debug(f"{operation_name} took {duration:.2f} seconds")

# ==============================================================================
# SEGMENTATION METHOD ENUM
# ==============================================================================

class SegmentationMethod(Enum):
    KMEANS_OPT = "kmeans_opt"
    KMEANS_PREDEF = "kmeans_predef"
    SOM_OPT = "som_opt"
    SOM_PREDEF = "som_predef"
    DBSCAN = "dbscan"

# ==============================================================================
# CLUSTER STRATEGIES
# ==============================================================================

class ClusterStrategy(ABC):
    """Abstract base class for cluster determination strategies."""
    
    @abstractmethod
    def determine_k(self, pixels: np.ndarray, default_k: int) -> int:
        """Determine optimal number of clusters."""
        pass

class MetricBasedStrategy(ClusterStrategy):
    """Strategy using metric-based cluster determination."""
    
    def __init__(self, config: ClusterStrategyConfig):
        self.config = config
    
    def determine_k(self, pixels: np.ndarray, default_k: int) -> int:
        """Determine k using metric-based approach."""
        try:
            logging.info(f"Determining optimal k for pixels with shape {pixels.shape}")
            return optimal_clusters(
                pixels, 
                default_k, 
                min_k=self.config.min_k, 
                max_k=self.config.max_k
            )
        except Exception as e:
            logging.warning(f"Cluster determination failed: {e}. Using default_k={default_k}")
            return default_k

class ElbowMethodStrategy(ClusterStrategy):
    """Strategy using elbow method for cluster determination."""
    
    def __init__(self, config: ClusterStrategyConfig):
        self.config = config
    
    def determine_k(self, pixels: np.ndarray, default_k: int) -> int:
        """Determine k using elbow method."""
        try:
            # Subsample if too many pixels
            if len(pixels) > self.config.subsample_size:
                indices = np.random.choice(len(pixels), self.config.subsample_size, replace=False)
                pixels_sample = pixels[indices]
            else:
                pixels_sample = pixels
            
            # Calculate within-cluster sum of squares for different k values
            wcss = []
            k_range = range(self.config.min_k, min(self.config.max_k + 1, len(np.unique(pixels_sample, axis=0)) + 1))
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=self.config.random_state, n_init=3)
                kmeans.fit(pixels_sample)
                wcss.append(kmeans.inertia_)
            
            # Find elbow point
            if len(wcss) < 2:
                return default_k
            
            rates = [wcss[i-1] - wcss[i] for i in range(1, len(wcss))]
            optimal_idx = 0
            for i in range(1, len(rates)):
                if rates[i] < rates[i-1] * 0.5:  # 50% decrease threshold
                    optimal_idx = i
                    break
            
            optimal_k = list(k_range)[optimal_idx]
            logging.info(f"Elbow method determined optimal k: {optimal_k}")
            return optimal_k
            
        except Exception as e:
            logging.warning(f"Elbow method failed: {e}. Using default_k={default_k}")
            return default_k

# ==============================================================================
# BASE SEGMENTER CLASS
# ==============================================================================

class SegmenterBase(ABC):
    """Abstract base class for image segmentation."""
    
    def __init__(self, image: np.ndarray, config: SegmentationConfig, 
                 models: ModelConfig, output_manager: Any, 
                 cluster_strategy: Optional[ClusterStrategy] = None):
        """Initialize base segmenter.
        
        Args:
            image: Input image for segmentation
            config: Segmentation configuration
            models: Model configuration (DBN, scalers, etc.)
            output_manager: Manager for saving outputs
            cluster_strategy: Strategy for determining optimal clusters
        """
        if not SegmentationValidator.validate_image(image):
            raise InvalidConfigurationError("Invalid input image")
        
        self.image = image
        self.config = config
        self.models = models
        self.output_manager = output_manager
        self.cluster_strategy = cluster_strategy or MetricBasedStrategy(ClusterStrategyConfig())
        self.preprocessed_image = None
    
    @abstractmethod
    def segment(self) -> SegmentationResult:
        """Segment the image and return results."""
        pass
    
    def quantize_image(self, n_colors: Optional[int] = None) -> np.ndarray:
        """Quantize the image to reduce the number of unique colors."""
        if n_colors is None:
            n_colors = self.config.quantization_colors
        
        image_to_quantize = self.preprocessed_image if self.preprocessed_image is not None else self.image
        
        if image_to_quantize.size == 0:
            logging.warning("Empty image detected, skipping quantization")
            return image_to_quantize
        
        unique_colors_before = len(np.unique(image_to_quantize.reshape(-1, 3), axis=0))
        logging.debug(f"Quantizing image with {unique_colors_before} unique colors to {n_colors}")
        
        if unique_colors_before <= n_colors:
            logging.debug("Image already has fewer colors than target, skipping quantization")
            return image_to_quantize
        
        try:
            pixels = image_to_quantize.reshape(-1, 3)
            kmeans = KMeans(n_clusters=n_colors, n_init=3, random_state=42)
            kmeans.fit(pixels)
            quantized_pixels = kmeans.cluster_centers_[kmeans.predict(pixels)]
            quantized_image = quantized_pixels.reshape(image_to_quantize.shape).astype(np.uint8)
            
            unique_colors_after = len(np.unique(quantized_image.reshape(-1, 3), axis=0))
            logging.debug(f"Quantization complete: {unique_colors_before} -> {unique_colors_after} colors")
            
            return quantized_image
        except Exception as e:
            logging.warning(f"Quantization failed: {e}. Using original image.")
            return image_to_quantize
    
    def compute_similarity(self, segmentation_data: Tuple[np.ndarray, List[np.ndarray], np.ndarray]) -> List[float]:
        """Compute similarity scores between segmented colors and target colors.
        
        Args:
            segmentation_data: Tuple containing (segmented_image, avg_colors, labels).
        
        Returns:
            List of similarity scores.
        """
        segmented_colors = segmentation_data[1]  # avg_colors
        similarities = []
        
        if not self.config.target_colors or len(self.config.target_colors) == 0:
            logging.warning("No target colors provided for similarity computation")
            return [0.0] * len(segmented_colors)
        
        for seg_color in segmented_colors:
            try:
                if not isinstance(seg_color, (list, np.ndarray)) or len(seg_color) != 3:
                    logging.warning(f"Invalid segmented color: {seg_color}. Skipping.")
                    similarities.append(0.0)
                    continue
                
                min_distance = float('inf')
                for target_color in self.config.target_colors:
                    if not isinstance(target_color, (list, np.ndarray)) or len(target_color) != 3:
                        logging.warning(f"Invalid target color: {target_color}. Skipping.")
                        continue
                    distance = ciede2000_distance(seg_color, target_color)
                    min_distance = min(min_distance, distance)
                
                # Avoid division by zero
                similarity_score = 1.0 / (1.0 + max(min_distance, 1e-6)) if min_distance < 100 else 0.0
                similarities.append(similarity_score)
            except Exception as e:
                logging.error(f"Error computing similarity for color {seg_color}: {e}")
                similarities.append(0.0)
        
        return similarities
    
    def _calculate_average_colors(self, labels: np.ndarray, n_clusters: int) -> List[np.ndarray]:
        """Calculate average colors for each cluster."""
        avg_colors = []
        
        for i in range(n_clusters):
            cluster_mask = (labels == i).reshape(self.image.shape[:2])
            if np.any(cluster_mask):
                cluster_pixels = self.image[cluster_mask]
                if len(cluster_pixels) > 0:
                    avg_color = np.mean(cluster_pixels, axis=0)
                    avg_colors.append(avg_color)
                else:
                    avg_colors.append(np.array([0, 0, 0], dtype=np.float32))
            else:
                avg_colors.append(np.array([0, 0, 0], dtype=np.float32))
        
        return avg_colors
    
    def _create_fallback_result(self, error_message: str) -> SegmentationResult:
        """Create a fallback result when segmentation fails."""
        return SegmentationResult(
            segmented_image=self.image.copy(),
            avg_colors=[],
            labels=np.zeros(self.image.shape[:2], dtype=int),
            similarity=[],
            n_clusters=0,
            method=self.__class__.__name__.replace('Segmenter', '').lower(),
            success=False,
            error_message=error_message
        )

# ==============================================================================
# CONCRETE SEGMENTER CLASSES
# ==============================================================================

class KMeansSegmenter(SegmenterBase):
    """K-means based image segmentation."""
    
    def segment(self) -> SegmentationResult:
        """Perform K-means segmentation."""
        method_name = f"kmeans_{self.config.k_type}"
        
        with timer(f"K-means segmentation ({self.config.k_type})"):
            try:
                # Determine number of clusters
                pixels = self.quantize_image().reshape(-1, 3).astype(np.float32)
                if self.config.k_type == 'determined':
                    optimal_k = self.cluster_strategy.determine_k(pixels, self.config.predefined_k)
                    logging.info(f"Optimal k determined: {optimal_k}")
                else:
                    optimal_k = self.config.predefined_k
                    logging.info(f"Using predefined k: {optimal_k}")
                
                # Perform segmentation
                segmentation_result = k_mean_segmentation(self.image, optimal_k)
                
                if len(segmentation_result) < 2:
                    raise SegmentationMethodError("k_mean_segmentation returned insufficient data")
                
                segmented_image, labels = segmentation_result[0], segmentation_result[1]
                
                # Calculate average colors
                avg_colors = self._calculate_average_colors(labels, optimal_k)
                
                # Compute similarity
                similarity = self.compute_similarity((segmented_image, avg_colors, labels))
                
                result = SegmentationResult(
                    segmented_image=segmented_image,
                    avg_colors=avg_colors,
                    labels=labels,
                    similarity=similarity,
                    n_clusters=optimal_k,
                    method=method_name
                )
                
                logging.info(f"K-means segmentation completed successfully with k={optimal_k}")
                return result
            except Exception as e:
                error_msg = f"K-means segmentation failed: {str(e)}"
                logging.error(error_msg)
                return self._create_fallback_result(error_msg)

class DBSCANSegmenter(SegmenterBase):
    """DBSCAN based image segmentation."""
    
    def segment(self) -> SegmentationResult:
        """Perform DBSCAN segmentation."""
        with timer("DBSCAN segmentation"):
            try:
                pixels = self.quantize_image().reshape(-1, 3).astype(np.float32)
                labels = optimal_dbscan(pixels, self.config.distance_threshold)
                unique_labels = np.unique(labels[labels >= 0])
                
                if len(unique_labels) == 0:
                    logging.warning("No clusters found in DBSCAN")
                    return self._create_fallback_result("No clusters found in DBSCAN")
                
                # Calculate cluster centers
                centers = np.array([
                    np.mean(pixels[labels == label], axis=0) 
                    for label in unique_labels
                ])
                centers = np.clip(centers, 0, 255).astype(np.uint8)
                
                # Create segmented image
                mask = labels >= 0
                segmented_flat = np.zeros_like(pixels)
                segmented_flat[mask] = centers[labels[mask]]
                segmented_image = segmented_flat.reshape(self.image.shape)
                
                # Calculate average colors
                avg_colors = []
                for label in unique_labels:
                    cluster_mask = (labels.reshape(self.image.shape[:2]) == label).astype(np.uint8)
                    mean_color = cv2.mean(self.image, mask=cluster_mask)[:3]
                    avg_colors.append(np.array(mean_color))
                
                # Compute similarity
                similarity = self.compute_similarity((segmented_image, avg_colors, labels))
                
                result = SegmentationResult(
                    segmented_image=segmented_image,
                    avg_colors=avg_colors,
                    labels=labels,
                    similarity=similarity,
                    n_clusters=len(unique_labels),
                    method="dbscan"
                )
                
                logging.info(f"DBSCAN segmentation completed with {len(unique_labels)} clusters")
                return result
            except Exception as e:
                error_msg = f"DBSCAN segmentation failed: {str(e)}"
                logging.error(error_msg)
                return self._create_fallback_result(error_msg)

class SOMSegmenter(SegmenterBase):
    """Self-Organizing Map based image segmentation."""
    
    def segment(self) -> SegmentationResult:
        """Perform SOM segmentation."""
        method_name = f"som_{self.config.k_type}"
        
        with timer(f"SOM segmentation ({self.config.k_type})"):
            try:
                pixels = self.quantize_image().reshape(-1, 3).astype(np.float32) / 255.0
                if self.config.k_type == 'determined':
                    optimal_k = self.cluster_strategy.determine_k(pixels, self.config.predefined_k)
                    logging.info(f"Optimal k determined: {optimal_k}")
                else:
                    optimal_k = self.config.predefined_k
                    logging.info(f"Using predefined k: {optimal_k}")
                
                labels = optimal_som(pixels, optimal_k)
                
                if len(np.unique(labels)) == 0:
                    raise SegmentationMethodError("SOM produced no valid clusters")
                
                # Calculate cluster centers
                max_label = max(labels) if len(labels) > 0 else 0
                centers = []
                for i in range(max_label + 1):
                    cluster_pixels = pixels[labels == i]
                    if len(cluster_pixels) > 0:
                        center = np.mean(cluster_pixels, axis=0) * 255.0
                        centers.append(center)
                    else:
                        centers.append(np.array([0, 0, 0], dtype=np.float32))
                
                centers = np.array(centers).astype(np.uint8)
                
                # Create segmented image
                segmented_flat = np.zeros_like(pixels)
                for i, center in enumerate(centers):
                    mask = labels == i
                    segmented_flat[mask] = center / 255.0
                segmented_image = (segmented_flat.reshape(self.image.shape) * 255).astype(np.uint8)
                
                # Calculate average colors
                avg_colors = []
                for i in range(max_label + 1):
                    cluster_mask = (labels.reshape(self.image.shape[:2]) == i).astype(np.uint8)
                    if np.any(cluster_mask):
                        mean_color = cv2.mean(self.image, mask=cluster_mask)[:3]
                        avg_colors.append(np.array(mean_color))
                    else:
                        avg_colors.append(np.array([0, 0, 0], dtype=np.float32))
                
                # Compute similarity
                similarity = self.compute_similarity((segmented_image, avg_colors, labels))
                
                result = SegmentationResult(
                    segmented_image=segmented_image,
                    avg_colors=avg_colors,
                    labels=labels,
                    similarity=similarity,
                    n_clusters=optimal_k,
                    method=method_name
                )
                
                logging.info(f"SOM segmentation completed successfully with k={optimal_k}")
                return result
            except Exception as e:
                error_msg = f"SOM segmentation failed: {str(e)}"
                logging.error(error_msg)
                return self._create_fallback_result(error_msg)

# ==============================================================================
# MAIN SEGMENTER CLASS (FACTORY)
# ==============================================================================

class Segmenter:
    """Main segmentation class that orchestrates different segmentation methods."""
    
    def __init__(self, image: np.ndarray, config: SegmentationConfig, 
                 models: ModelConfig, output_manager: Any):
        """Initialize the main segmenter.
        
        Args:
            image: Preprocessed image for segmentation
            config: Segmentation configuration
            models: Model configuration
            output_manager: Output management system
        """
        if not SegmentationValidator.validate_image(image):
            raise InvalidConfigurationError("Invalid input image provided")
        
        self.image = image
        self.config = config
        self.models = models
        self.output_manager = output_manager
        self._results: Dict[str, SegmentationResult] = {}
    
    def create_segmenter(self, method: SegmentationMethod) -> SegmenterBase:
        """Factory method to create specific segmenter instances.
        
        Args:
            method: Segmentation method to create
            
        Returns:
            Configured segmenter instance
            
        Raises:
            ValueError: If unknown segmentation method is requested
        """
        segmenter_map = {
            SegmentationMethod.KMEANS_OPT: self._create_kmeans_optimal,
            SegmentationMethod.KMEANS_PREDEF: self._create_kmeans_predefined,
            SegmentationMethod.SOM_OPT: self._create_som_optimal,
            SegmentationMethod.SOM_PREDEF: self._create_som_predefined,
            SegmentationMethod.DBSCAN: self._create_dbscan_segmenter,
        }
        
        creator = segmenter_map.get(method)
        if creator is None:
            raise ValueError(f"Unknown segmentation method: {method.value}")
        
        return creator()
    
    def _create_kmeans_optimal(self) -> KMeansSegmenter:
        """Create K-means segmenter with optimal k determination."""
        config = self._create_config_copy('determined')
        return KMeansSegmenter(
            self.image, config, self.models, self.output_manager,
            cluster_strategy=MetricBasedStrategy(ClusterStrategyConfig())
        )
    
    def _create_kmeans_predefined(self) -> KMeansSegmenter:
        """Create K-means segmenter with predefined k."""
        config = self._create_config_copy('predefined')
        return KMeansSegmenter(
            self.image, config, self.models, self.output_manager
        )
    
    def _create_som_optimal(self) -> SOMSegmenter:
        """Create SOM segmenter with optimal k determination."""
        config = self._create_config_copy('determined')
        return SOMSegmenter(
            self.image, config, self.models, self.output_manager,
            cluster_strategy=MetricBasedStrategy(ClusterStrategyConfig())
        )
    
    def _create_som_predefined(self) -> SOMSegmenter:
        """Create SOM segmenter with predefined k."""
        config = self._create_config_copy('predefined')
        return SOMSegmenter(
            self.image, config, self.models, self.output_manager
        )
    
    def _create_dbscan_segmenter(self) -> DBSCANSegmenter:
        """Create DBSCAN segmenter."""
        return DBSCANSegmenter(
            self.image, self.config, self.models, self.output_manager
        )
    
    def _create_config_copy(self, k_type: str) -> SegmentationConfig:
        """Create a copy of configuration with modified k_type."""
        return SegmentationConfig(
            target_colors=self.config.target_colors,
            distance_threshold=self.config.distance_threshold,
            predefined_k=self.config.predefined_k,
            k_values=self.config.k_values,
            som_values=self.config.som_values,
            k_type=k_type,
            methods=self.config.methods,
            quantization_colors=self.config.quantization_colors
        )
    
    def _save_preprocessed_image(self) -> str:
        """Save the preprocessed image and return its path."""
        try:
            image_name = self.output_manager.get_current_image_name()
            preprocessed_path = os.path.join(
                self.output_manager.processed_dir, 
                f"{image_name}_preprocessed.jpg"
            )
            cv2.imwrite(preprocessed_path, cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR))
            logging.debug(f"Preprocessed image saved to: {preprocessed_path}")
            return preprocessed_path
        except Exception as e:
            logging.error(f"Failed to save preprocessed image: {e}")
            return ""
    
    def process(self) -> ProcessingResult:
        """Process the image with all configured segmentation methods.
        
        Returns:
            ProcessingResult containing results and errors
        """
        start_time = time.time()
        results = {}
        errors = {}
        
        logging.info(f"Starting segmentation processing with methods: {self.config.methods}")
        
        for method_name in self.config.methods:
            try:
                # Convert string to enum
                method = next((m for m in SegmentationMethod if m.value == method_name), None)
                if method is None:
                    error_msg = f"Unknown segmentation method: {method_name}"
                    logging.warning(error_msg)
                    errors[method_name] = error_msg
                    continue
                
                # Create and run segmenter
                logging.info(f"Processing method: {method_name}")
                segmenter = self.create_segmenter(method)
                result = segmenter.segment()
                
                if result.is_valid():
                    results[method_name] = result
                    
                    # Save segmentation output
                    image_name = self.output_manager.get_current_image_name()
                    self.output_manager.save_segmentation_image(
                        image_name, method_name, cv2.cvtColor(result.segmented_image, cv2.COLOR_RGB2BGR)
                    )
                    logging.info(f"Method {method_name} completed successfully")
                else:
                    errors[method_name] = result.error_message or "Unknown segmentation error"
                    logging.error(f"Method {method_name} failed: {errors[method_name]}")
                
            except Exception as e:
                error_msg = f"Unexpected error in {method_name}: {str(e)}"
                logging.error(error_msg)
                errors[method_name] = error_msg
        
        total_time = time.time() - start_time
        
        # Save preprocessed image
        preprocessed_path = self._save_preprocessed_image()
        
        result = ProcessingResult(
            results=results,
            errors=errors,
            preprocessed_path=preprocessed_path,
            total_processing_time=total_time
        )
        
        logging.info(f"Processing completed in {total_time:.2f}s. "
                    f"Successful: {len(results)}, Errors: {len(errors)}")
        
        return result

# ==============================================================================
# BACKWARD COMPATIBILITY LAYER
# ==============================================================================

class BackwardCompatibilityMixin:
    """Mixin to provide backward compatibility with old tuple-based returns."""
    
    @staticmethod
    def result_to_tuple(result: SegmentationResult) -> tuple:
        """Convert SegmentationResult to old tuple format.
        
        Returns:
            tuple: (segmented_image, avg_colors, labels, similarity, n_clusters)
        """
        return (
            result.segmented_image,
            result.avg_colors,
            result.labels,
            result.similarity,
            result.n_clusters
        )
    
    @staticmethod
    def processing_result_to_dict(processing_result: ProcessingResult) -> tuple:
        """Convert ProcessingResult to old format.
        
        Returns:
            tuple: (preprocessed_path, results_dict)
            where results_dict contains tuples instead of SegmentationResult objects
        """
        results_dict = {}
        for method_name, result in processing_result.results.items():
            results_dict[method_name] = BackwardCompatibilityMixin.result_to_tuple(result)
        
        return processing_result.preprocessed_path, results_dict

# ==============================================================================
# FACTORY FUNCTIONS FOR EASY MIGRATION
# ==============================================================================

def create_segmentation_config_from_old_params(target_colors, distance_threshold, 
                                              predefined_k, k_values, som_values, 
                                              k_type, methods=None) -> SegmentationConfig:
    """Create SegmentationConfig from old-style parameters for easy migration."""
    if methods is None:
        methods = ['kmeans_opt', 'kmeans_predef', 'som_opt', 'som_predef', 'dbscan']
    
    return SegmentationConfig(
        target_colors=target_colors,
        distance_threshold=distance_threshold,
        predefined_k=predefined_k,
        k_values=k_values,
        som_values=som_values,
        k_type=k_type,
        methods=methods
    )

def create_model_config_from_old_params(dbn=None, scalers=None, 
                                       reference_kmeans_opt=None, 
                                       reference_som_opt=None) -> ModelConfig:
    """Create ModelConfig from old-style parameters for easy migration."""
    return ModelConfig(
        dbn=dbn,
        scalers=scalers,
        reference_kmeans_opt=reference_kmeans_opt,
        reference_som_opt=reference_som_opt
    )

# ==============================================================================
# LEGACY WRAPPER CLASS (for smooth transition)
# ==============================================================================

class LegacySegmenter(BackwardCompatibilityMixin):
    """Legacy wrapper that maintains the old interface while using new implementation."""
    
    def __init__(self, preprocessed_image, target_colors, distance_threshold, 
                 reference_kmeans_opt, reference_som_opt, dbn, scalers, 
                 predefined_k, k_values, som_values, output_dir, k_type, 
                 methods=['kmeans_opt', 'kmeans_predef', 'som_opt', 'som_predef', 'dbscan'], 
                 output_manager=None):
        """Initialize with old-style parameters."""
        
        # Create new-style configuration objects
        self.seg_config = create_segmentation_config_from_old_params(
            target_colors, distance_threshold, predefined_k, 
            k_values, som_values, k_type, methods
        )
        
        self.model_config = create_model_config_from_old_params(
            dbn, scalers, reference_kmeans_opt, reference_som_opt
        )
        
        # Create new segmenter
        self.new_segmenter = Segmenter(
            preprocessed_image, self.seg_config, self.model_config, output_manager or None
        )
    
    def process(self) -> tuple:
        """Process using new implementation but return old tuple format."""
        processing_result = self.new_segmenter.process()
        return self.processing_result_to_dict(processing_result)
    
    def create_segmenter(self, method):
        """Legacy method mapping."""
        if isinstance(method, str):
            method = SegmentationMethod(method)
        return self.new_segmenter.create_segmenter(method)