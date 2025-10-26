import logging 
from abc import ABC, abstractmethod
import numpy as np
from typing import List

from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score

from .base import SegmentationConfig

logger = logging.getLogger(__name__)

class ClusterStrategy(ABC):
    """
    Abstract Base Class for defining a 'k' determination strategy
    
    this ensures that any new strategy we create (e.g. ElbowMethodStrategy) will have
    the same 'determine_k' method, making them interchangeable   
    """
    @abstractmethod
    def determine_k(self, pixels: np.ndarray, config: SegmentationConfig) -> int:
        """
        determines the optimal number of clusters (k) for the given pixels
        
        Args:
            pixels (np.ndarray): the pixel data (N,3) to analuze
            config (SegmentationConfig): the segmentation config, providing k_values range and predefined_k as fallback

        Returns:
            int: the determined optimal k_value
        """
        pass
    
class MetricBasedStrategy(ClusterStrategy):
    """
    determines 'k' by testing multiple k-values and finding the best combined Silhouette and CH score
    """
    
    def determine_k(self, pixels: np.ndarray, config: SegmentationConfig) -> int:
        k_range_list = []
        if config.k_type == 'determined' and ('kmeans_opt' in config.methods or 'som_opt' in config.methods):
            if 'kmeans_opt' in config.methods and config.k_values:
                k_range_list = config.k_values
            elif 'som_opt' in config.methods and config.som_values:
                k_range_list = config.som_values
                
        if not k_range_list:
            k_range_list = list(range(2,9))
            logger.debug(f"using default k-range {k_range_list} for k-determination")
            
        min_k = min(k_range_list) if k_range_list else 2
        max_k = max(k_range_list) if k_range_list else 8
        
        unique_colors = np.unique(pixels, axis=0)
        n_unique = len(unique_colors)
        # arranges the range of k dynamically
        dynamic_max_k = max(min_k, min(max_k, n_unique-1))
        
        logger.info(f"Unique colors: {n_unique}. Adjusted k-range: [{min_k}, {dynamic_max_k}]")
        
        # --- Veri Kontrolleri ---
        if pixels.shape[0] == 0:
             logger.warning("Cannot determine k: empty pixel array. Falling back...")
             return config.predefined_k
        # Neden: Çok fazla piksel üzerinde metrik hesaplamak yavaştır.
        # Rastgele bir alt örneklem (subsample) kullanmak yeterince iyi bir sonuç verir.
        if pixels.shape[0] > 10000:
            logger.info("Subsampling pixels for cluster analysis efficiency")
            indices = np.random.choice(pixels.shape[0], 10000, replace=False)
            pixels = pixels[indices]
            
        if pixels.shape[0] < min_k:
             logger.warning(f"Pixel count ({pixels.shape[0]}) < min_k ({min_k}). Falling back...")
             return config.predefined_k
        
        k_range = list(range(min_k, dynamic_max_k + 1))
        if not k_range or k_range[-1] < min_k:
            logger.warning(f"K-range is empty or invalid ({k_range}). Defaulting...")
            return config.predefined_k
        
        # --- Metrik Hesaplama ---
        valid_k_scores = {}
        for k in k_range:
            if k > pixels.shape[0]:
                logger.warning(f"Skipping k={k} (greater than sample size {pixels.shape[0]})")
                continue
            
            logger.info(f"Testing k={k} for cluster metrics...")
            try:
                kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42).fit(pixels)
                labels = kmeans.labels_
                
                # Neden: Metrikler (Silhouette, CH) en az 2 küme gerektirir.
                if len(np.unique(labels)) < 2:
                    logger.warning(f"Only 1 cluster found for k={k}. Skipping metrics.")
                    continue
                    
                silhouette_val = silhouette_score(pixels, labels)
                ch_val = calinski_harabasz_score(pixels, labels)
                valid_k_scores[k] = {'silhouette': silhouette_val, 'ch': ch_val}
                
            except Exception as e:
                logger.error(f"Error calculating metrics for k={k}: {e}", exc_info=True)
        
        if not valid_k_scores:
             logger.error("Could not calculate metrics for any k value. Falling back...")
             return config.predefined_k
        
        # --- En İyi K'yı Bulma ---
        # Neden: Silhouette ve CH skorları farklı ölçeklerdedir (biri [0,1], diğeri [0, +inf]).
        # Onları 0-1 aralığına normalize edip ortalamasını almak,
        # her iki metriği de hesaba katan adil bir "genel skor" verir.
        k_values_list = list(valid_k_scores.keys())
        sil_scores = np.array([valid_k_scores[k]['silhouette'] for k in k_values_list])
        ch_scores = np.array([valid_k_scores[k]['ch'] for k in k_values_list])
        
        # Normalizasyon (Min-Max Scaling)
        norm_sil = (sil_scores - np.min(sil_scores)) / (np.max(sil_scores) - np.min(sil_scores) + 1e-10)
        norm_ch = (ch_scores - np.min(ch_scores)) / (np.max(ch_scores) - np.min(ch_scores) + 1e-10)
        
        avg_scores = (norm_sil + norm_ch) / 2
        optimal_k_index = np.argmax(avg_scores)
        optimal_k = k_values_list[optimal_k_index]
        
        logger.info(f"Optimal k determined: {optimal_k} (score: {avg_scores[optimal_k_index]:.3f})")
        return int(optimal_k) 