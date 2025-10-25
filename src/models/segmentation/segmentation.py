# src/models/segmentation/segmentation.py (TEMİZLENMİŞ SON HALİ)

import logging
import os
import cv2
import numpy as np
import time 
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from minisom import MiniSom

from src.models.pso_dbn import DBN
from sklearn.preprocessing import MinMaxScaler
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
    avg_colors: List[Tuple[float, float, float]] = field(default_factory=list)
    labels: Optional[np.ndarray] = None
    n_clusters: int = 0
    processing_time: float = 0.0

    def is_valid(self) -> bool:
        """Check if the segmentation result is valid and usable."""
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
    @abstractmethod
    def determine_k(self, pixels: np.ndarray, config: SegmentationConfig) -> int:
        pass

class MetricBasedStrategy(ClusterStrategy):
    def determine_k(self, pixels: np.ndarray, config: SegmentationConfig, n_runs=3) -> int:
        k_range_list = []
        if config.k_type == 'determined' and ('kmeans_opt' in config.methods or 'som_opt' in config.methods):
             if 'kmeans_opt' in config.methods and config.k_values:
                  k_range_list = config.k_values
             elif 'som_opt' in config.methods and config.som_values:
                  k_range_list = config.som_values
        if not k_range_list:
             k_range_list = list(range(2, 9))
             logger.debug(f"Using default k-range {k_range_list} for k-determination.")

        min_k = min(k_range_list) if k_range_list else 2
        max_k = max(k_range_list) if k_range_list else 8
        
        unique_colors = np.unique(pixels, axis=0)
        n_unique = len(unique_colors)
        dynamic_max_k = max(min_k, min(max_k, n_unique))
        logger.info(f"Unique colors: {n_unique}. Adjusted k-range: [{min_k}, {dynamic_max_k}]")
        
        if pixels.shape[0] == 0:
             logger.warning("Cannot determine k: empty pixel array. Falling back...")
             return config.predefined_k
        if pixels.shape[0] > 10000:
            logger.info("Subsampling pixels for cluster analysis efficiency")
            indices = np.random.choice(pixels.shape[0], 10000, replace=False)
            pixels = pixels[indices]
        if pixels.shape[0] < min_k:
             logger.warning(f"Pixel count ({pixels.shape[0]}) < min_k ({min_k}). Falling back...")
             return config.predefined_k
        
        scores = {'silhouette': [], 'ch': []}
        k_range = list(range(min_k, dynamic_max_k + 1))
        if not k_range:
            logger.warning(f"K-range is empty. Defaulting...")
            return config.predefined_k
        
        valid_k_scores = {}
        for k in k_range:
            if k > pixels.shape[0]:
                logger.warning(f"Skipping k={k} > samples ({pixels.shape[0]})")
                continue
            
            logger.info(f"Testing k={k}")
            try:
                kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42).fit(pixels)
                labels = kmeans.labels_
                if len(np.unique(labels)) < 2:
                    logger.warning(f"Only 1 cluster found for k={k}. Skipping metrics.")
                    continue
                silhouette_val = silhouette_score(pixels, labels)
                ch_val = calinski_harabasz_score(pixels, labels)
                valid_k_scores[k] = {'silhouette': silhouette_val, 'ch': ch_val}
                logger.info(f"Metrics for k={k}: Sil={silhouette_val:.3f}, CH={ch_val:.1f}")
            except Exception as e:
                logger.error(f"Error metrics for k={k}: {e}", exc_info=True)
        
        if not valid_k_scores:
             logger.error("Could not calculate metrics for any k value. Falling back...")
             return config.predefined_k
        
        k_values_list = list(valid_k_scores.keys())
        sil_scores = np.array([valid_k_scores[k]['silhouette'] for k in k_values_list])
        ch_scores = np.array([valid_k_scores[k]['ch'] for k in k_values_list])
        norm_sil = (sil_scores - np.min(sil_scores)) / (np.max(sil_scores) - np.min(sil_scores) + 1e-10)
        norm_ch = (ch_scores - np.min(ch_scores)) / (np.max(ch_scores) - np.min(ch_scores) + 1e-10)
        avg_scores = (norm_sil + norm_ch) / 2
        optimal_k_index = np.argmax(avg_scores)
        optimal_k = k_values_list[optimal_k_index]
        logger.info(f"Optimal k determined: {optimal_k} (score: {avg_scores[optimal_k_index]:.3f})")
        return optimal_k

# ====================================================================
# Abstract Base Class for Segmentation Methods
# ====================================================================

class SegmenterBase(ABC):
    def __init__(self,
                 preprocessed_image: np.ndarray,
                 config: SegmentationConfig,
                 models: ModelConfig,
                 cluster_strategy: ClusterStrategy):
        if preprocessed_image is None:
             raise ValueError(f"{self.__class__.__name__} received a None preprocessed_image.")
        self.preprocessed_image = preprocessed_image
        self.config = config
        self.models = models
        self.cluster_strategy = cluster_strategy
    
    @abstractmethod
    def segment(self) -> SegmentationResult:
        pass

    def quantize_image(self, n_colors=50) -> Optional[np.ndarray]:
        if self.preprocessed_image is None or self.preprocessed_image.size == 0:
            logger.warning("Cannot quantize None or empty image.")
            return None
        logger.info(f"Quantizing image (shape: {self.preprocessed_image.shape}) to approx {n_colors} colors")
        pixels = self.preprocessed_image.reshape(-1, 3).astype(np.float32)
        n_pixels_total = pixels.shape[0]
        actual_n_colors = max(1, min(n_colors, n_pixels_total))
        if actual_n_colors != n_colors: logger.warning(f"Adjusted quantization n_colors to {actual_n_colors}")
        if actual_n_colors == 1 and n_pixels_total > 0:
             center = np.mean(pixels, axis=0)
             quantized = np.tile(center, (self.preprocessed_image.shape[0], self.preprocessed_image.shape[1], 1)).astype(np.uint8)
             return quantized
        elif actual_n_colors < 1:
             logger.error("Cannot quantize to less than 1 color.")
             return None
        if n_pixels_total > 20000:
            indices = np.random.choice(n_pixels_total, 20000, replace=False)
            pixels_sample = pixels[indices]
        else:
            pixels_sample = pixels
        if pixels_sample.shape[0] < actual_n_colors:
             logger.warning(f"Sample size ({pixels_sample.shape[0]}) < n_colors ({actual_n_colors}). Using sample size.")
             actual_n_colors = pixels_sample.shape[0]
             if actual_n_colors < 1: logger.error("Cannot quantize with zero samples."); return None
        try:
            kmeans = KMeans(n_clusters=actual_n_colors, n_init='auto', random_state=42).fit(pixels_sample)
            labels = kmeans.predict(pixels)
            quantized_pixels = kmeans.cluster_centers_[labels]
            return quantized_pixels.reshape(self.preprocessed_image.shape).astype(np.uint8)
        except Exception as e:
             logger.error(f"Error during quantization K-Means: {e}", exc_info=True)
             return None

# ====================================================================
# Concrete Segmentation Method Implementations
# ====================================================================

class KMeansSegmenter(SegmenterBase):
    def __init__(self, preprocessed_image, config: SegmentationConfig, models: ModelConfig, cluster_strategy: ClusterStrategy):
        try:
            super().__init__(preprocessed_image, config, models, cluster_strategy)
            logger.info(f"KMeansSegmenter initialized with k_type: {self.config.k_type}")
        except Exception as e_init:
             logger.error(f"KMeansSegmenter __init__ failed: {e_init}", exc_info=True)
             raise

    def segment(self) -> SegmentationResult:
        start_time = time.perf_counter()
        method_name = "kmeans_opt" if self.config.k_type == 'determined' else "kmeans_predef"
        optimal_k = -1
        try:
            quantized_img = self.quantize_image()
            if quantized_img is None: raise SegmentationError("Quantization failed.")
            quantized_pixels = quantized_img.reshape(-1, 3).astype(np.float32)
            
            if self.config.k_type == 'determined':
                logger.debug("KMeans: Determining optimal k...")
                optimal_k = self.cluster_strategy.determine_k(quantized_pixels, self.config)
            else:
                optimal_k = self.config.predefined_k
            logger.info(f"KMeans: Using k = {optimal_k}")
            
            if not isinstance(optimal_k, int) or optimal_k <= 0: raise SegmentationError(f"Invalid clusters: {optimal_k}")
            
            pixels_for_segmentation = self.preprocessed_image.reshape(-1, 3).astype(np.float32)
            if pixels_for_segmentation.shape[0] < optimal_k:
                 logger.warning(f"Pixels ({pixels_for_segmentation.shape[0]}) < k ({optimal_k}). Adjusting k.")
                 optimal_k = max(1, pixels_for_segmentation.shape[0])
            if optimal_k < 1: raise SegmentationError("Less than 1 cluster.")
            
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            retval, labels_flat, centers = cv2.kmeans(pixels_for_segmentation, optimal_k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            centers = np.uint8(centers)
            if labels_flat is None or len(labels_flat) != pixels_for_segmentation.shape[0]: raise SegmentationError(f"Label length mismatch or None")
            
            segmented_image = centers[labels_flat.flatten()].reshape(self.preprocessed_image.shape)
            labels_2d = labels_flat.reshape(self.preprocessed_image.shape[:2])
            
            avg_colors = []
            if optimal_k > 0:
                 for i in range(optimal_k):
                     mask = (labels_2d == i).astype(np.uint8)
                     if np.sum(mask) > 0: avg_colors.append(cv2.mean(self.preprocessed_image, mask=mask)[:3])
                     else: logger.warning(f"KMeans empty mask cluster {i} (k={optimal_k}).")
            
            duration = time.perf_counter() - start_time
            return SegmentationResult(method_name=method_name, segmented_image=segmented_image, avg_colors=[tuple(c) for c in avg_colors], labels=labels_flat, n_clusters=optimal_k, processing_time=duration)
        except Exception as e:
             logger.error(f"Error during KMeans segmentation ({method_name}): {e}", exc_info=True)
             duration = time.perf_counter() - start_time
             return SegmentationResult(method_name=method_name, processing_time=duration, n_clusters=optimal_k if optimal_k > 0 else 0)

class DBSCANSegmenter(SegmenterBase):
    def __init__(self, preprocessed_image, config: SegmentationConfig, models: ModelConfig, cluster_strategy: ClusterStrategy):
        try:
            super().__init__(preprocessed_image, config, models, cluster_strategy)
            logger.info(f"DBSCANSegmenter initialized with k_type: {self.config.k_type}")
        except Exception as e_init:
             logger.error(f"DBSCANSegmenter __init__ failed: {e_init}", exc_info=True)
             raise

    def _run_dbscan(self, pixels: np.ndarray, eps: float, min_samples: int) -> Tuple[np.ndarray, int]:
        try:
            if pixels.shape[0] == 0: return np.array([]), 0
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(pixels)
            labels = db.labels_
            n_clusters = len(np.unique(labels[labels >= 0]))
            return labels, n_clusters
        except Exception as e:
            logger.error(f"Error during DBSCAN clustering: {e}", exc_info=True)
            return np.full(pixels.shape[0], -1), 0

    def _find_optimal_dbscan_params(self, pixels: np.ndarray) -> Tuple[float, int]:
        logger.info("Finding optimal DBSCAN parameters...")
        eps_values = [10, 15, 20] 
        min_samples_values = [5, 10, 20] 
        best_silhouette = -1.1
        best_params = (self.config.dbscan_eps, self.config.dbscan_min_samples)
        if pixels.shape[0] < 2: logger.warning("Too few pixels."); return best_params
        for eps in eps_values:
            for min_samples in min_samples_values:
                labels, n_clusters = self._run_dbscan(pixels, eps, min_samples)
                if n_clusters > 1:
                    try:
                        silhouette_avg = silhouette_score(pixels, labels)
                        if silhouette_avg > best_silhouette: best_silhouette = silhouette_avg; best_params = (eps, min_samples)
                    except Exception as e: logger.warning(f"Sil calc failed: {e}")
        logger.info(f"Optimal DBSCAN parameters: eps={best_params[0]}, min={best_params[1]}")
        return best_params

    def segment(self) -> SegmentationResult:
        start_time = time.perf_counter()
        method_name = "dbscan"
        try:
            quantized_img = self.quantize_image();
            if quantized_img is None: raise SegmentationError("Quantization failed.")
            pixels = quantized_img.reshape(-1, 3).astype(np.float32)
            if pixels.shape[0] == 0: raise SegmentationError("Zero pixels.")
            if self.config.k_type == 'determined': eps, min_samples = self._find_optimal_dbscan_params(pixels)
            else: eps = self.config.dbscan_eps; min_samples = self.config.dbscan_min_samples; logger.info(f"Using predefined DBSCAN.")
            labels, n_clusters = self._run_dbscan(pixels, eps, min_samples)
            if n_clusters == 0: logger.warning("No clusters."); return SegmentationResult(method_name=method_name, processing_time=time.perf_counter()-start_time)
            original_pixels = self.preprocessed_image.reshape(-1, 3).astype(np.float32)
            if len(labels) != original_pixels.shape[0]:
                 logger.warning("Label mismatch. Re-run DBSCAN.")
                 pixels_orig = self.preprocessed_image.reshape(-1,3).astype(np.float32)
                 labels, n_clusters = self._run_dbscan(pixels_orig, eps, min_samples)
                 if n_clusters == 0: logger.error("DBSCAN failed on original."); return SegmentationResult(method_name=method_name, processing_time=time.perf_counter()-start_time)
                 original_pixels = pixels_orig
            centers = []; valid_labels = []
            unique_cluster_labels = np.unique(labels[labels >= 0])
            for label_id in unique_cluster_labels:
                 mask = (labels == label_id)
                 if np.sum(mask) > 0: centers.append(np.mean(original_pixels[mask], axis=0)); valid_labels.append(label_id)
                 else: logger.warning(f"DBSCAN cluster {label_id} empty?")
            if not centers: logger.error("DBSCAN failed centers."); return SegmentationResult(method_name=method_name, processing_time=time.perf_counter()-start_time)
            centers = np.uint8(centers); actual_n_clusters = len(centers)
            segmented_flat = np.zeros_like(original_pixels, dtype=np.uint8)
            label_to_center_idx = {label_id: idx for idx, label_id in enumerate(valid_labels)}
            for i in range(len(labels)):
                if labels[i] in label_to_center_idx: segmented_flat[i] = centers[label_to_center_idx[labels[i]]]
            segmented_image = segmented_flat.reshape(self.preprocessed_image.shape)
            avg_colors = [tuple(c) for c in centers]
            duration = time.perf_counter() - start_time
            return SegmentationResult(method_name=method_name, segmented_image=segmented_image, avg_colors=avg_colors, labels=labels, n_clusters=actual_n_clusters, processing_time=duration)
        except Exception as e:
             logger.error(f"Error during DBSCAN segmentation: {e}", exc_info=True)
             duration = time.perf_counter() - start_time
             return SegmentationResult(method_name=method_name, processing_time=duration)


class SOMSegmenter(SegmenterBase):
    """Segments image using Self-Organizing Maps."""
    def __init__(self, preprocessed_image, config: SegmentationConfig, models: ModelConfig, cluster_strategy: ClusterStrategy):
        try:
            super().__init__(preprocessed_image, config, models, cluster_strategy)
            logger.info(f"SOMSegmenter initialized with k_type: {self.config.k_type}")
        except Exception as e_init:
             logger.error(f"SOMSegmenter __init__ failed: {e_init}", exc_info=True)
             raise

    def segment(self) -> SegmentationResult:
        start_time = time.perf_counter()
        method_name = "som_opt" if self.config.k_type == 'determined' else "som_predef"
        optimal_k = -1
        try:
            quantized_img = self.quantize_image()
            if quantized_img is None: raise SegmentationError("Quantization failed.")
            pixels_normalized = quantized_img.reshape(-1, 3).astype(np.float32) / 255.0
            if pixels_normalized.shape[0] == 0: raise SegmentationError("Zero pixels.")
            if self.config.k_type == 'determined': optimal_k = self.cluster_strategy.determine_k(pixels_normalized, self.config)
            else: optimal_k = self.config.predefined_k
            if not isinstance(optimal_k, int) or optimal_k <= 0: raise SegmentationError(f"Invalid clusters for SOM: {optimal_k}")
            logger.info(f"Running SOM segmentation with k={optimal_k}")
            som = MiniSom(x=1, y=optimal_k, input_len=3, sigma=0.5, learning_rate=0.25, random_seed=42)
            som.random_weights_init(pixels_normalized)
            som.train_random(pixels_normalized, 100)
            labels_flat = np.array([som.winner(pixel)[1] for pixel in pixels_normalized])
            centers_normalized = np.array([som.get_weights()[0, i] for i in range(optimal_k)])
            centers = np.uint8(np.clip(centers_normalized * 255.0, 0, 255))
            original_pixels_shape = self.preprocessed_image.shape
            if len(labels_flat) != (original_pixels_shape[0] * original_pixels_shape[1]):
                 logger.warning("Label mismatch. Re-predicting SOM on original.")
                 pixels_orig_norm = self.preprocessed_image.reshape(-1, 3).astype(np.float32) / 255.0
                 if pixels_orig_norm.shape[0] > 0: labels_flat = np.array([som.winner(pixel)[1] for pixel in pixels_orig_norm])
                 else: raise SegmentationError("Original image zero pixels.")
            if len(labels_flat) != (original_pixels_shape[0] * original_pixels_shape[1]): raise SegmentationError("Label length mismatch after re-prediction.")
            segmented_image = centers[labels_flat.flatten()].reshape(original_pixels_shape)
            labels_2d = labels_flat.reshape(original_pixels_shape[:2])
            avg_colors = []
            for i in range(optimal_k):
                mask = (labels_2d == i).astype(np.uint8)
                if np.sum(mask) > 0: avg_colors.append(cv2.mean(self.preprocessed_image, mask=mask)[:3])
                else: logger.warning(f"SOM empty mask cluster {i} (k={optimal_k}).")
            duration = time.perf_counter() - start_time
            return SegmentationResult(method_name=method_name, segmented_image=segmented_image, avg_colors=[tuple(c) for c in avg_colors], labels=labels_flat, n_clusters=optimal_k, processing_time=duration)
        except Exception as e:
             logger.error(f"Error during SOM segmentation ({method_name}): {e}", exc_info=True)
             duration = time.perf_counter() - start_time
             return SegmentationResult(method_name=method_name, processing_time=duration, n_clusters=optimal_k if optimal_k > 0 else 0)


# ====================================================================
# Main Segmenter (Facade Pattern)
# ====================================================================

class Segmenter:
    """Facade class managing the segmentation workflow."""
    def __init__(self,
                 preprocessed_image: np.ndarray,
                 seg_config: SegmentationConfig,
                 model_config: ModelConfig,
                 output_manager: Any,
                 cluster_strategy: Optional[ClusterStrategy] = None):

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
        """Run all initialized segmentation methods and collect results."""
        image_name_stem = "unknown_image" # Varsayılan
        preprocessed_path = "unknown_preprocessed_image.png"
        try:
             # output_manager'dan görüntü adını daha güvenli almayı dene
             # Bu, main.py'deki 'image_name'e dayanır, ki bu da OutputManager'a bağlı
             # Belki de image_name'i Segmenter'a parametre olarak geçmek daha iyi olur
             if hasattr(self.output_manager, 'dataset_name'): # Basit bir varsayım
                 pass # image_name_stem'i burada ayarlamak zor
             
             # main.py'de 'process_single_test_image' içindeki 'image_name' kullanılır
             # Bu yüzden OutputManager'ın onu bilmesine gerek yok, ama 'process'in bilmesi iyi olurdu
             # Şimdilik placeholder'da kalalım, OutputManager kaydederken doğru adı kullanır
             pass 
        except Exception as e:
             logger.warning(f"Could not get image name/path: {e}. Using placeholders.")

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
                    # Kaydetme işlemi OutputManager'a devredildi (main.py'den çağrılan)
                    # ANCAK segmenter.process() içinden kaydetme daha mantıklı
                    # main.py'deki process_single_test_image'e bakalım
                    # Evet, main.py'de segmenter.process() ÇAĞRILDIKTAN SONRA kaydetme yapılmıyor.
                    # Kaydetme mantığı _Segmenter.process_ içinde OLMALI.
                    # OutputManager'ı _init_'te aldığımıza göre, burada kullanmalıyız.
                    
                    # image_name_stem'i almamız lazım. Şimdilik config'den almayı deneyelim?
                    # VEYA en iyisi: 'process' metoduna 'image_name_stem' parametresi ekleyelim.
                    # Şimdilik (main.py'yi bozmamak için) 'unknown' kullanalım
                    
                    # --- DÜZELTME: 'image_name_stem'i 'process' metoduna ekleyelim ---
                    # BU KISIM ŞİMDİLİK ÇALIŞMAYACAK çünkü image_name_stem'i bilmiyoruz
                    # main.py'de `segmenter.process()` çağrısını `segmenter.process(image_name)` olarak değiştirmeliyiz
                    
                    # ŞİMDİLİK KAYDETMEYİ BURADAN ÇAĞIRIYORUZ (main.py'ye güvenmek yerine)
                    # 'image_name'i bir şekilde almamız lazım
                    # 'output_manager' zaten 'dataset_name'i biliyor
                    # 'image_name'i `process_single_test_image`'den almalıyız
                    # Şimdilik bu kaydetmeyi `main.py`'nin `process_single_test_image` içinde
                    # `segmenter.process()`'ten sonra yapacağını varsayalım.
                    # HAYIR, `main.py`'deki kod `processing_result.results`'i alıyor.
                    # Bu, `segmenter.process()`'in içinde kaydetme *olmaması* gerektiğini gösterir.
                    
                    # --- KODU ESKİ HALİNE DÖNDÜR (KAYDETME YOK) ---
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