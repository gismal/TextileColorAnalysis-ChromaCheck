import logging
import cv2
import numpy as np
from sklearn.cluster import KMeans

# Düzeltme: basicConfig yerine getLogger kullanıyoruz.
# main.py zaten loglamayı başlatıyor.
logger = logging.getLogger(__name__)

def exception_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            return None
    return wrapper

class Preprocessor:
    def __init__(self, initial_resize=512, target_size=(128, 128), denoise_h=10, max_colors=8, edge_enhance=False, unsharp_amount=0.0, unsharp_threshold=0):
        """Initialize the preprocessor with configurable parameters."""
        self.initial_resize = initial_resize
        self.target_size = target_size
        self.denoise_h = denoise_h  # Parameter for non-local means denoising
        self.max_colors = max_colors  # Maximum number of colors to allow
        self.edge_enhance = edge_enhance
        self.unsharp_amount = unsharp_amount  # Disabled by default
        self.unsharp_threshold = unsharp_threshold

    def estimate_n_colors(self, image):
        """Estimate the number of colors based on unique pixel values."""
        unique_colors = len(np.unique(image.reshape(-1, 3), axis=0))
        # Set n_colors to 1.5x unique colors, capped at max_colors
        n_colors = min(int(unique_colors * 1.5), self.max_colors)
        logger.info(f"Estimated {unique_colors} unique colors, setting n_colors to {n_colors}")
        return max(n_colors, 2)  # Ensure at least 2 colors

    @exception_handler
    def quantize_image(self, image):
        """Quantize the image to a dynamically estimated number of colors using K-means."""
        n_colors = self.estimate_n_colors(image)
        pixels = image.reshape(-1, 3)  # Flatten to (n_pixels, 3) for clustering
        if pixels.shape[0] > 10000:  # Subsample for efficiency
            indices = np.random.choice(pixels.shape[0], 10000, replace=False)
            pixels = pixels[indices]
        kmeans = KMeans(n_clusters=n_colors, n_init=3, random_state=42).fit(pixels)
        labels = kmeans.predict(image.reshape(-1, 3))
        quantized = kmeans.cluster_centers_[labels].reshape(image.shape).astype(np.uint8)
        logger.info(f"Quantized to {n_colors} colors, unique colors: {len(np.unique(quantized.reshape(-1, 3), axis=0))}")
        return quantized

    @exception_handler
    def unsharp_mask(self, image):
        """Apply unsharp masking to enhance image details (disabled by default)."""
        if self.unsharp_amount > 0:
            blurred = cv2.GaussianBlur(image, (5, 5), 1.0)
            mask = cv2.subtract(image, blurred)
            sharpened = cv2.addWeighted(image, 1.0 + self.unsharp_amount, mask, -self.unsharp_amount, 0)
            if self.unsharp_threshold > 0:
                low_contrast_mask = np.absolute(mask) < self.unsharp_threshold
                np.copyto(sharpened, image, where=low_contrast_mask)
            logger.info("Applied unsharp masking")
            return sharpened
        return image

    @exception_handler
    def preprocess(self, img):
        """Preprocess the image with denoising, quantization, and resizing."""
        logger.info("Starting preprocessing")
        
        # Initial resize to reduce computation
        scale_factor = self.initial_resize / min(img.shape[:2])
        resized = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        logger.info(f"Initial resize to approx {self.initial_resize}x{self.initial_resize}")

        # Denoise with non-local means (replaces bilateral filter)
        denoised = cv2.fastNlMeansDenoisingColored(resized, None, h=self.denoise_h, templateWindowSize=7, searchWindowSize=21)
        logger.info(f"Applied non-local means denoising with h={self.denoise_h}")

        # Apply unsharp masking (optional)
        sharpened_image = self.unsharp_mask(denoised)
        
        # Quantize colors dynamically
        quantized_image = self.quantize_image(sharpened_image)
        
        # Final resize to target size
        final_image = cv2.resize(quantized_image, self.target_size, interpolation=cv2.INTER_AREA)
        logger.info(f"Resized to {self.target_size[0]}x{self.target_size[1]}")

        logger.info("Preprocessing completed")
        return final_image

# ---------------------------------------------------------------------------
# YENİ EKLENEN FONKSİYON (main.py'den taşındı)
# Bu fonksiyon, DBN eğitim verisi için veri seti düzeyinde örnekleme yapar.
# ---------------------------------------------------------------------------

def efficient_data_sampling(rgb_data, lab_data, n_samples=800, max_per_image=None):
    """Memory-efficient data sampling with comprehensive error handling."""
    rgb_samples = []
    lab_samples = []
    
    # Handle different data structures
    if hasattr(rgb_data, 'shape') and len(rgb_data.shape) == 2:
        # Data is already flattened as (n_images, n_pixels_per_image)
        logger.info(f"Data is pre-flattened: RGB {rgb_data.shape}, LAB {lab_data.shape}")
        
        n_images = rgb_data.shape[0]
        n_pixels_per_image = rgb_data.shape[1] // 3  # Each pixel has 3 channels
        
        if max_per_image is None:
            # n_images 0 olmamalı
            if n_images > 0:
                max_per_image = max(50, n_samples // n_images)
            else:
                max_per_image = max(50, n_samples)
        
        logger.info(f"Sampling {n_samples} total samples, max {max_per_image} per image")
        
        for i in range(n_images):
            try:
                # Extract RGB and LAB data for this image
                rgb_flat = rgb_data[i].reshape(-1, 3)  # Reshape to (n_pixels, 3)
                lab_flat = lab_data[i].reshape(-1, 3)  # Reshape to (n_pixels, 3)
                
                logger.debug(f"Processing image {i+1}/{n_images}: RGB {rgb_flat.shape}, LAB {lab_flat.shape}")
                
                # Calculate available pixels
                n_pixels = rgb_flat.shape[0]
                n_per_image = min(max_per_image, n_pixels)
                
                if n_per_image <= 0:
                    logger.warning(f"Skipping image {i+1}: insufficient pixels")
                    continue
                
                # Random sampling
                indices = np.random.choice(n_pixels, n_per_image, replace=False)
                
                # Process RGB
                rgb_sample = rgb_flat[indices].astype(np.float32)
                
                # Ensure RGB is in 0-255 range
                if rgb_sample.max() <= 1.0:
                    rgb_sample = rgb_sample * 255.0
                
                # Process LAB with proper validation
                lab_sample = lab_flat[indices].astype(np.float32).copy()
                
                # Validate and convert LAB ranges
                if lab_sample[:, 0].max() > 100:  # L channel should be 0-100
                    logger.debug(f"Image {i+1}: Normalizing L channel from 0-255 to 0-100")
                    lab_sample[:, 0] = lab_sample[:, 0] / 255.0 * 100.0
                
                # Convert a,b channels from OpenCV (0-255) to CIELAB (-128,127)
                if lab_sample[:, 1:].min() >= 0:  # Only convert if in 0-255 range
                    lab_sample[:, 1:] = lab_sample[:, 1:] - 128.0
                
                rgb_samples.append(rgb_sample)
                lab_samples.append(lab_sample)
                
                logger.debug(f"Image {i+1}: Sampled {n_per_image} pixels")
                
            except Exception as e:
                logger.error(f"Error sampling from image {i+1}: {e}")
                logger.debug(f"RGB data shape for image {i+1}: {rgb_data[i].shape if hasattr(rgb_data[i], 'shape') else 'No shape attribute'}")
                logger.debug(f"LAB data shape for image {i+1}: {lab_data[i].shape if hasattr(lab_data[i], 'shape') else 'No shape attribute'}")
                continue
    
    else:
        # Data is structured as list of images
        if not rgb_data:
             raise ValueError("RGB data list is empty")
             
        if max_per_image is None:
            max_per_image = max(50, n_samples // len(rgb_data))
        
        logger.info(f"Sampling {n_samples} total samples, max {max_per_image} per image")
        
        for i, (img_rgb, img_lab) in enumerate(zip(rgb_data, lab_data)):
            try:
                if not hasattr(img_rgb, 'shape') or not hasattr(img_lab, 'shape'):
                    logger.warning(f"Skipping image {i+1}: invalid array format")
                    continue
                    
                logger.debug(f"Processing image {i+1}/{len(rgb_data)}: RGB {img_rgb.shape}, LAB {img_lab.shape}")
                
                # Validate shapes match
                if img_rgb.shape[:2] != img_lab.shape[:2]:
                    logger.warning(f"Image {i+1}: RGB and LAB shape mismatch, skipping")
                    continue
                
                # Calculate available pixels
                n_pixels = img_rgb.shape[0] * img_rgb.shape[1]
                n_per_image = min(max_per_image, n_pixels)
                
                if n_per_image <= 0:
                    logger.warning(f"Skipping image {i+1}: insufficient pixels")
                    continue
                
                # Random sampling
                indices = np.random.choice(n_pixels, n_per_image, replace=False)
                
                # Process RGB
                rgb_flat = img_rgb.reshape(-1, 3)
                rgb_sample = rgb_flat[indices].astype(np.float32)
                
                # Ensure RGB is in 0-255 range
                if rgb_sample.max() <= 1.0:
                    rgb_sample = rgb_sample * 255.0
                
                # Process LAB with proper validation
                lab_flat = img_lab.reshape(-1, 3)
                lab_sample = lab_flat[indices].astype(np.float32).copy()
                
                # Validate and convert LAB ranges
                if lab_sample[:, 0].max() > 100:  # L channel should be 0-100
                    logger.debug(f"Image {i+1}: Normalizing L channel from 0-255 to 0-100")
                    lab_sample[:, 0] = lab_sample[:, 0] / 255.0 * 100.0
                
                # Convert a,b channels from OpenCV (0-255) to CIELAB (-128,127)
                if lab_sample[:, 1:].min() >= 0:  # Only convert if in 0-255 range
                    lab_sample[:, 1:] = lab_sample[:, 1:] - 128.0
                
                rgb_samples.append(rgb_sample)
                lab_samples.append(lab_sample)
                
                logger.debug(f"Image {i+1}: Sampled {n_per_image} pixels")
                
            except Exception as e:
                logger.error(f"Error sampling from image {i+1}: {e}")
                continue
    
    if not rgb_samples:
        raise ValueError("No samples could be extracted from any image")
    
    # Efficiently concatenate
    rgb_result = np.vstack(rgb_samples)
    lab_result = np.vstack(lab_samples)
    
    logger.info(f"Sampling completed: {rgb_result.shape[0]} samples from {len(rgb_samples)} images")
    logger.info(f"RGB range: [{rgb_result.min():.2f}, {rgb_result.max():.2f}]")
    logger.info(f"LAB ranges: L[{lab_result[:, 0].min():.2f}, {lab_result[:, 0].max():.2f}], "
                f"a[{lab_result[:, 1].min():.2f}, {lab_result[:, 1].max():.2f}], "
                f"b[{lab_result[:, 2].min():.2f}, {lab_result[:, 2].max():.2f}]")
    
    return rgb_result, lab_result