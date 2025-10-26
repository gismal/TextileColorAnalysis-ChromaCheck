# src/data/preprocess.py
# SON VE TEMİZ HALİ

import logging
import cv2
import numpy as np
from sklearn.cluster import KMeans # quantize_image için gerekli
from dataclasses import dataclass, field
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# Exception handler decorator (isteğe bağlı)
def exception_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}", exc_info=True) 
            return None
    return wrapper

@dataclass
class PreprocessingConfig:
    """Configuration settings for the Preprocessor."""
    initial_resize: int = 512
    target_size: Tuple[int, int] = (128, 128) 
    denoise_h: int = 10
    # max_colors: int = 8 # <-- Bu artık gereksiz, quantization_colors kullanılıyor. Silebiliriz.
    # edge_enhance: bool = False # Kullanılmıyorsa silebiliriz.
    unsharp_amount: float = 0.0
    unsharp_threshold: int = 0
    quantization_colors: int = 50 # Hedef max renk sayısı (dinamik tahmin için üst limit)
    quantization_subsample: int = 20000 
    unsharp_blur_kernel_size: Tuple[int, int] = (5, 5)
    unsharp_blur_sigma: float = 1.0

class Preprocessor:
    """
    Applies a series of preprocessing steps to an input image based on configuration.

    Steps include: initial resizing, denoising, optional unsharp masking,
    color quantization, and final resizing.
    """
    
    def __init__(self, config: PreprocessingConfig):
        """Initializes the preprocessor with a configuration object."""
        if not isinstance(config, PreprocessingConfig):
            raise TypeError(f"Preprocessor must be initialized with a PreprocessingConfig object, got {type(config)} instead.")
        self.config = config 
        logger.debug(f"Preprocessor initialized with config: {self.config}")

    def _estimate_n_colors(self, image: np.ndarray) -> int:
        """
        Estimates a reasonable number of colors for quantization based on unique colors.

        Args:
            image (np.ndarray): The image (H, W, 3) to analyze.

        Returns:
            int: The estimated number of colors, capped by config.quantization_colors.
        """
        # Config'den max hedef renk sayısını al
        max_target_colors = self.config.quantization_colors 
        
        pixels_est = image.reshape(-1, 3)
        if pixels_est.size == 0: 
            logger.warning("Cannot estimate colors for empty image. Defaulting to 2.")
            return 2 
            
        try:
            # Optimizasyon: Çok fazla piksel varsa alt örneklem al
            subsample_estimate_threshold = 50000 # Bu da config'e eklenebilir
            if pixels_est.shape[0] > subsample_estimate_threshold: 
                 logger.debug(f"Subsampling pixels ({pixels_est.shape[0]}) for unique color estimation.")
                 indices = np.random.choice(pixels_est.shape[0], subsample_estimate_threshold, replace=False)
                 pixels_est = pixels_est[indices]
                 
            unique_colors_est = np.unique(pixels_est, axis=0)
            n_unique_est = len(unique_colors_est)
            
            # Heuristic: Benzersiz renklerin ~1.5 katı kadar, ama config'deki max sınırı geçmeyecek şekilde. En az 2 renk.
            estimated_colors = max(2, min(int(n_unique_est * 1.5), max_target_colors)) 
            logger.info(f"Estimated {n_unique_est} unique colors. Target quantization colors: {estimated_colors} (Config limit: {max_target_colors})")
            return estimated_colors
            
        except Exception as e:
            logger.warning(f"Error during unique color estimation: {e}. Falling back to config limit ({max_target_colors}).")
            return max_target_colors

    # @exception_handler 
    # quantize_image artık dışarıdan n_colors almayacak
    def quantize_image(self, image: np.ndarray) -> Optional[np.ndarray]: 
        """
        Reduces the number of unique colors in the input image using K-Means.
        Determines the target number of colors dynamically using _estimate_n_colors.
        Uses subsampling size from PreprocessingConfig.
        
        Args:
            image (np.ndarray): The input image (H, W, 3) to quantize.

        Returns:
            Optional[np.ndarray]: The quantized image (H, W, 3) or None on failure.
        """
        subsample_threshold = self.config.quantization_subsample # Config'den al
        
        if image is None or image.size == 0:
            logger.warning("Cannot quantize None or empty image.")
            return None
            
        original_shape = image.shape # Orijinal şekli sakla

        # 1. Hedef renk sayısını tahmin et
        n_colors = self._estimate_n_colors(image) 
        
        logger.info(f"Quantizing image (shape: {original_shape}) to approx {n_colors} colors")
        pixels = image.reshape(-1, 3).astype(np.float32)
        n_pixels_total = pixels.shape[0]
        
        # Hedef renk sayısını piksel sayısına göre ayarla
        actual_n_colors = max(1, min(n_colors, n_pixels_total))
        if actual_n_colors != n_colors:
            logger.warning(f"Adjusted quantization target colors to {actual_n_colors} (due to pixel count)")
            
        if actual_n_colors < 1:
             logger.error("Cannot quantize to less than 1 color.")
             return None
             
        if actual_n_colors == 1 and n_pixels_total > 0:
             center = np.mean(pixels, axis=0)
             quantized = np.tile(center, (original_shape[0], original_shape[1], 1)).astype(np.uint8)
             logger.info("Quantized image to a single average color.")
             return quantized
        
        # K-Means için alt örneklem al
        if n_pixels_total > subsample_threshold:
            logger.debug(f"Subsampling {n_pixels_total} pixels to {subsample_threshold} for K-Means fitting.")
            indices = np.random.choice(n_pixels_total, subsample_threshold, replace=False)
            pixels_sample = pixels[indices]
        else:
            pixels_sample = pixels
            
        if pixels_sample.shape[0] < actual_n_colors:
             logger.warning(f"Sample size ({pixels_sample.shape[0]}) < target colors ({actual_n_colors}). Using sample size as n_colors.")
             actual_n_colors = pixels_sample.shape[0]
             if actual_n_colors < 1: 
                 logger.error("Cannot quantize with zero samples.")
                 return None
                 
        # K-Means'i çalıştır
        try:
            kmeans = KMeans(n_clusters=actual_n_colors, n_init='auto', random_state=42).fit(pixels_sample)
            labels = kmeans.predict(pixels) # Tüm piksellere uygula
            quantized_pixels = kmeans.cluster_centers_[labels]
            quantized_image = quantized_pixels.reshape(original_shape).astype(np.uint8) # Orijinal şekle döndür
            n_final_colors = len(np.unique(quantized_image.reshape(-1, 3), axis=0))
            logger.info(f"Quantization complete. Final unique colors: {n_final_colors} (target was {actual_n_colors})")
            return quantized_image
        except Exception as e:
             logger.error(f"Error during quantization K-Means: {e}", exc_info=True)
             return None # Hata durumunda None döndür

    # @exception_handler
    # image argümanını ekle
    def unsharp_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Applies unsharp masking to enhance image details, using config parameters.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Sharpened image, or original if amount is zero.
        """
        # Config değerlerini self.config üzerinden al
        amount = self.config.unsharp_amount
        threshold = self.config.unsharp_threshold
        # Tuple olduğundan emin ol ve config'den al
        kernel_size = tuple(self.config.unsharp_blur_kernel_size) 
        sigma = self.config.unsharp_blur_sigma
        
        if amount > 0: 
            logger.debug(f"Applying unsharp mask: amount={amount}, threshold={threshold}, kernel={kernel_size}, sigma={sigma}")
            # --- GÜNCELLENMİŞ KISIM ---
            blurred = cv2.GaussianBlur(image, kernel_size, sigma) 
            # --- BİTTİ ---
            
            mask = cv2.subtract(image.astype(np.int16), blurred.astype(np.int16))
            sharpened_float = cv2.addWeighted(image.astype(np.float32), 1.0 + amount, mask.astype(np.float32), -amount, 0)
            sharpened = np.clip(sharpened_float, 0, 255).astype(np.uint8)
            
            if threshold > 0:
                low_contrast_mask = np.absolute(mask) < threshold
                if low_contrast_mask.ndim == 2:
                     low_contrast_mask = np.repeat(low_contrast_mask[:, :, np.newaxis], 3, axis=2)
                np.copyto(sharpened, image, where=low_contrast_mask)
                
            logger.info("Applied unsharp masking.")
            return sharpened
        else:
            logger.debug("Unsharp mask amount is zero or less, skipping.")
            return image 

    # @exception_handler # Buradaki decorator, içindeki hataları yakalar ve None döndürür
    def preprocess(self, img: np.ndarray) -> Optional[np.ndarray]:
        """
        Applies the full preprocessing pipeline: resize, denoise, unsharp, quantize, final resize.

        Args:
            img (np.ndarray): The raw input image (BGR or RGB, assumed uint8).

        Returns:
            Optional[np.ndarray]: The fully preprocessed image, or None if any step fails.
        """
        if img is None or img.size == 0:
             logger.error("Input image to preprocess is None or empty.")
             return None
             
        logger.info(f"Starting preprocessing pipeline for image with shape: {img.shape}")
        
        # --- Adım 1: İlk Yeniden Boyutlandırma ---
        h, w = img.shape[:2]
        if min(h, w) == 0:
             logger.error("Input image has zero height or width.")
             return None
        scale_factor = self.config.initial_resize / min(h, w) 
        new_w = max(1, int(w * scale_factor))
        new_h = max(1, int(h * scale_factor))
        try:
             resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
             logger.info(f"Initial resize from {w}x{h} to {new_w}x{new_h}")
        except cv2.error as e:
             logger.error(f"Initial resize failed: {e}")
             return None # Kritik hata, devam etme

        # --- Adım 2: Gürültü Azaltma ---
        try:
             denoised = cv2.fastNlMeansDenoisingColored(resized, None, h=self.config.denoise_h, templateWindowSize=7, searchWindowSize=21) 
             logger.info(f"Applied non-local means denoising with h={self.config.denoise_h}")
        except cv2.error as e:
             logger.warning(f"Denoising failed: {e}. Continuing without denoising.")
             denoised = resized # Hata olursa orijinal resized ile devam et

        # --- Adım 3: Keskinleştirme (Unsharp Mask) ---
        sharpened_image = self.unsharp_mask(denoised) 
        
        # --- Adım 4: Renk Niceleme (Quantization) ---
        quantized_image = self.quantize_image(sharpened_image) 
        if quantized_image is None: 
             logger.error("Color quantization step failed during preprocessing.")
             return None # Kritik hata, devam etme
        
        # --- Adım 5: Son Yeniden Boyutlandırma ---
        try:
             target_w, target_h = self.config.target_size 
             if target_w <= 0 or target_h <= 0:
                  logger.error(f"Invalid target_size in config: {self.config.target_size}. Cannot perform final resize.")
                  final_image = quantized_image 
             else:
                  # Quantize edilmiş görüntüyü son boyuta getir
                  final_image = cv2.resize(quantized_image, self.config.target_size, interpolation=cv2.INTER_AREA) 
                  logger.info(f"Final resize to target size {self.config.target_size}")
        except cv2.error as e:
             logger.error(f"Final resize failed: {e}")
             return None # Kritik hata, devam etme

        logger.info("Preprocessing pipeline completed successfully.")
        return final_image