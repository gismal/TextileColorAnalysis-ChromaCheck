import logging
import time
import numpy as np
from typing import Tuple, Optional

# Gerekli temel sınıflar ve veri yapıları
from .base import (
    SegmentationConfig, ModelConfig, SegmentationResult, SegmentationError, SegmenterBase
)
# Strateji sınıfı (k belirlemek için)
from .strategy import MetricBasedStrategy, ClusterStrategy 
# Somut segmenter sınıfları (artık kendi paketimiz içindeyiz, doğrudan kullanabiliriz)
from .kmeans import KMeansSegmenter
from .som import SOMSegmenter

# Type hinting için
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.models.pso_dbn import DBN
    from sklearn.preprocessing import MinMaxScaler
    from src.data.preprocess import PreprocessingConfig

logger = logging.getLogger(__name__)

def segment_reference_image(
    preprocessed_image: np.ndarray,
    dbn: 'DBN',
    scalers: list, # List['MinMaxScaler']
    default_k: int,
    k_range: Optional[list] = None # K-Means/SOM için test edilecek k aralığı
) -> Tuple[Optional[SegmentationResult], Optional[SegmentationResult], int]:
    """
    Performs K-Means and SOM segmentation specifically for the reference image.

    This function encapsulates the logic previously found in image_utils,
    keeping segmentation logic within the segmentation package. It determines
    the optimal 'k' using MetricBasedStrategy and runs both K-Means and SOM
    with that 'k'.

    Args:
        preprocessed_image (np.ndarray): The preprocessed reference image.
        dbn (DBN): The trained DBN model.
        scalers (list): List containing [scaler_x, scaler_y, scaler_y_ab].
        default_k (int): The default 'k' value to use if determination fails.
        k_range (Optional[list]): The list of k values to test (e.g., [2, 3, 4, 5]).
                                  Defaults to range(2, 9).

    Returns:
        Tuple[Optional[SegmentationResult], Optional[SegmentationResult], int]:
            - The SegmentationResult from K-Means.
            - The SegmentationResult from SOM.
            - The determined optimal 'k' value used.
            Returns (None, None, default_k) on critical failure.
            
    Raises:
        SegmentationError: If preprocessing yields an unusable image or segmentation fails badly.
    """
    logger.info("Starting reference image segmentation (K-Means & SOM)...")
    start_time_total = time.perf_counter()

    if preprocessed_image is None or preprocessed_image.size == 0:
        raise SegmentationError("Cannot segment None or empty preprocessed image.")

    kmeans_result: Optional[SegmentationResult] = None
    som_result: Optional[SegmentationResult] = None
    determined_k: int = default_k

    try:
        pixels_flat = preprocessed_image.reshape(-1, 3).astype(np.float32)
        num_pixels = pixels_flat.shape[0]
        if num_pixels == 0:
            raise SegmentationError("Image has zero pixels after preprocessing.")

        # --- 1. Optimal K Belirleme ---
        # Neden: Hem K-Means hem de SOM için aynı 'k' değerini kullanacağız.
        # Strateji deseni burada devreye giriyor.
        logger.info("Determining optimal number of clusters for reference image...")
        k_values_to_test = k_range if k_range else list(range(2, 9))
        
        # Geçici bir config (sadece k belirlemek için)
        temp_seg_config = SegmentationConfig(
            target_colors=np.array([]), distance_threshold=0, predefined_k=default_k,
            k_values=k_values_to_test, som_values=k_values_to_test, # Kullanılmayacak ama gerekli
            k_type='determined', methods=['kmeans_opt'] # Sadece k belirlemek için
        )
        cluster_strategy: ClusterStrategy = MetricBasedStrategy() 

        # (quantize_image fonksiyonunu buraya da taşıyabiliriz veya base'den kullanabiliriz)
        # Şimdilik base'den kullanalım (geçici bir SegmenterBase örneği ile)
        try:
             # quantize_image'ı çağırmak için geçici base class örneği
             class TempBase(SegmenterBase): 
                def segment(self): 
                    pass # dummy segment
             temp_instance = TempBase(preprocessed_image, temp_seg_config, None, None) # models/strategy None olabilir
             quantized_img_for_k = temp_instance.quantize_image()
             if quantized_img_for_k is None: raise ValueError("Quantization for k failed.")
             quantized_pixels_for_k = quantized_img_for_k.reshape(-1, 3).astype(np.float32)
        except Exception as q_err:
             logger.warning(f"Could not quantize image for k-determination: {q_err}. Using original pixels.")
             quantized_pixels_for_k = pixels_flat # fallback

        try:
            determined_k = cluster_strategy.determine_k(quantized_pixels_for_k, temp_seg_config)
            logger.info(f"Optimal clusters determined for reference: {determined_k}")
        except Exception as k_err:
            logger.warning(f"Failed to determine optimal clusters: {k_err}. Falling back to default_k={default_k}", exc_info=True)
            determined_k = default_k

        # --- 2. K-Means ve SOM Segmentasyonu ---
        # Neden: Artık 'k' değerini biliyoruz. Bu 'k' ile hem K-Means hem de SOM'u
        # 'predefined' modda çalıştıracağız.
        
        # Her iki algoritma için ortak konfigürasyon
        ref_seg_config = SegmentationConfig(
            target_colors=np.array([]), distance_threshold=0, 
            predefined_k=determined_k, # Belirlenen k'yı kullan
            k_values=[determined_k], som_values=[determined_k], # Artık aralık değil, tek değer
            k_type='predefined', # 'determined' bitti, şimdi 'predefined'
            methods=['kmeans_predef', 'som_predef'], # Çalışacak yöntemler
            # DBSCAN parametreleri önemli değil
            dbscan_eps=10.0, dbscan_min_samples=5 
        )
        
        # Model konfigürasyonu (DBN ve Scaler'ları içerir)
        # Neden: Segmenter sınıflarının DBN modeline ve scaler'lara ihtiyacı yok
        # (renk dönüşümü bu fonksiyonun dışında yapılacak), ama ModelConfig bekliyorlar.
        # Bu yüzden None/dummy değerler geçebiliriz VEYA ModelConfig'i opsiyonel yapabiliriz.
        # Şimdilik None geçelim. -> HATA: ModelConfig None kabul etmez. Scalers'ı geçelim.
        ref_model_config = ModelConfig(
            dbn=dbn, # DBN hala gerekebilir (eğer segmenter içinde kullanılıyorsa - kontrol et)
            scalers=scalers, # Scalers listesini geçelim
            reference_kmeans_result=None, # Kendini referans alamaz
            reference_som_result=None 
        )

        # K-Means'i Çalıştır
        try:
            with timer(f"Reference K-Means (k={determined_k})"):
                 kmeans_segmenter = KMeansSegmenter(
                     preprocessed_image, ref_seg_config, ref_model_config, cluster_strategy # Strateji aslında kullanılmayacak
                 )
                 kmeans_result = kmeans_segmenter.segment()
                 if not kmeans_result or not kmeans_result.is_valid():
                     logger.error("Reference K-Means segmentation failed or produced invalid result.")
                     kmeans_result = None # Hata durumunda None ata
        except Exception as km_err:
            logger.error(f"Error during reference K-Means: {km_err}", exc_info=True)
            kmeans_result = None

        # SOM'u Çalıştır
        try:
            with timer(f"Reference SOM (k={determined_k})"):
                 som_segmenter = SOMSegmenter(
                     preprocessed_image, ref_seg_config, ref_model_config, cluster_strategy # Strateji aslında kullanılmayacak
                 )
                 som_result = som_segmenter.segment()
                 if not som_result or not som_result.is_valid():
                      logger.error("Reference SOM segmentation failed or produced invalid result.")
                      som_result = None # Hata durumunda None ata
        except Exception as som_err:
            logger.error(f"Error during reference SOM: {som_err}", exc_info=True)
            som_result = None

    except Exception as outer_err:
        logger.error(f"Critical error during reference segmentation: {outer_err}", exc_info=True)
        # Hata durumunda None, None, default_k döndür
        return None, None, default_k
    finally:
        duration_total = time.perf_counter() - start_time_total
        logger.info(f"Reference segmentation finished in {duration_total:.2f} seconds.")

    return kmeans_result, som_result, determined_k


# --- Zamanlayıcı (pipeline.py'den kopyalandı) ---
from contextlib import contextmanager
@contextmanager
def timer(operation_name: str):
    """Logs the duration of a code block."""
    start_time = time.perf_counter()
    logger.info(f"Starting: {operation_name}...")
    try:
        yield
    finally:
        duration = time.perf_counter() - start_time
        logger.info(f"Completed: {operation_name} in {duration:.2f} seconds")