# src/data/preprocess.py (CORRECTED)

import logging
import cv2
import numpy as np
from sklearn.cluster import KMeans
# --- FIX 1: Import dataclass ---
from dataclasses import dataclass, field
from typing import Tuple,Optional

logger = logging.getLogger(__name__)

def exception_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Log full traceback for better debugging
            logger.error(f"Error in {func.__name__}: {e}", exc_info=True) 
            return None
    return wrapper

@dataclass
class PreprocessingConfig:
    """Configuration settings for the Preprocessor."""
    initial_resize: int = 512
    # Use field for default factory for mutable types like tuples/lists if needed
    # but simple tuple default is fine here.
    target_size: Tuple[int, int] = (128, 128) 
    denoise_h: int = 10
    max_colors: int = 8
    edge_enhance: bool = False
    unsharp_amount: float = 0.0
    unsharp_threshold: int = 0

class Preprocessor:
    # --- FIX 2: Correct __init__ implementation ---
    def __init__(self, config: PreprocessingConfig):
        """Initialize the preprocessor with a config object."""
        # Assign attributes from the config object to the instance
        self.initial_resize = config.initial_resize
        self.target_size = config.target_size
        self.denoise_h = config.denoise_h
        self.max_colors = config.max_colors
        self.edge_enhance = config.edge_enhance # Although not used in preprocess() currently
        self.unsharp_amount = config.unsharp_amount
        self.unsharp_threshold = config.unsharp_threshold
        logger.debug(f"Preprocessor initialized with config: {config}")

    def estimate_n_colors(self, image):
        """Estimate the number of colors based on unique pixel values."""
        # Reshape defensively, handle potential empty image
        pixels = image.reshape(-1, 3)
        if pixels.size == 0:
             logger.warning("Cannot estimate colors for empty image.")
             return 2 # Default to minimum
             
        unique_colors = np.unique(pixels, axis=0)
        n_unique = len(unique_colors)
        
        # Calculate target number of colors, ensure it's at least 2
        # Use max_colors directly from the instance attribute
        n_colors = max(2, min(int(n_unique * 1.5), self.max_colors)) 
        logger.info(f"Estimated {n_unique} unique colors, setting n_colors to {n_colors} (max_colors={self.max_colors})")
        return n_colors

    @exception_handler
    def quantize_image(self, image):
        """Quantize the image to a dynamically estimated number of colors using K-means."""
        n_colors = self.estimate_n_colors(image)
        pixels = image.reshape(-1, 3) 
        if pixels.shape[0] == 0:
             logger.warning("Cannot quantize empty image.")
             return image # Return original empty image
             
        # Subsample if necessary
        if pixels.shape[0] > 10000: 
            indices = np.random.choice(pixels.shape[0], 10000, replace=False)
            pixels_for_fit = pixels[indices]
            logger.debug(f"Subsampled pixels for K-means fit: {pixels_for_fit.shape}")
        else:
            pixels_for_fit = pixels
            
        # Ensure pixels_for_fit is not empty before fitting
        if pixels_for_fit.shape[0] < n_colors:
             logger.warning(f"Number of pixels ({pixels_for_fit.shape[0]}) is less than n_colors ({n_colors}). Adjusting n_colors.")
             n_colors = max(1, pixels_for_fit.shape[0]) # At least 1 cluster needed
             if n_colors == 1:
                  logger.warning("Only one cluster possible. Quantization might not be effective.")

        if n_colors < 2 : # Kmeans requires at least 2 if possible, handle edge case n_colors=1
             if n_colors == 1 and pixels_for_fit.shape[0]>0:
                 center = np.mean(pixels_for_fit, axis=0)
                 quantized = np.tile(center, (image.shape[0], image.shape[1], 1)).astype(np.uint8)
                 labels = np.zeros(pixels.shape[0], dtype=int)
             else:
                 logger.error("Cannot perform K-means with less than 1 cluster or empty pixels.")
                 return None
        else:
             kmeans = KMeans(n_clusters=n_colors, n_init='auto', random_state=42).fit(pixels_for_fit) # Use 'auto' for n_init in newer sklearn
             # Predict on the original full set of pixels
             labels = kmeans.predict(pixels) 
             quantized = kmeans.cluster_centers_[labels].reshape(image.shape).astype(np.uint8)
        
        n_final_colors = len(np.unique(quantized.reshape(-1, 3), axis=0))
        logger.info(f"Quantized to {n_colors} target colors. Final unique colors: {n_final_colors}")
        return quantized

    @exception_handler
    def unsharp_mask(self, image):
        """Apply unsharp masking to enhance image details."""
        # Use instance attributes
        if self.unsharp_amount > 0: 
            blurred = cv2.GaussianBlur(image, (5, 5), 1.0)
            # Ensure subtraction doesn't underflow (convert to signed int temporarily if needed)
            mask = cv2.subtract(image.astype(np.int16), blurred.astype(np.int16))
            
            sharpened_float = cv2.addWeighted(image.astype(np.float32), 1.0 + self.unsharp_amount, mask.astype(np.float32), -self.unsharp_amount, 0)
            # Clip back to uint8 range
            sharpened = np.clip(sharpened_float, 0, 255).astype(np.uint8)
            
            if self.unsharp_threshold > 0:
                low_contrast_mask = np.absolute(mask) < self.unsharp_threshold
                # Apply mask across all color channels if needed
                if low_contrast_mask.ndim == 2: # If mask is grayscale, repeat for 3 channels
                     low_contrast_mask = np.repeat(low_contrast_mask[:, :, np.newaxis], 3, axis=2)
                np.copyto(sharpened, image, where=low_contrast_mask)
                
            logger.info("Applied unsharp masking")
            return sharpened
        return image # Return original if amount is zero

    @exception_handler
    def preprocess(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Preprocess the image with resizing, denoising, sharpening, and quantization."""
        if img is None or img.size == 0:
             logger.error("Input image to preprocess is None or empty.")
             return None
             
        logger.info(f"Starting preprocessing for image with shape: {img.shape}")
        
        # --- Initial resize ---
        # Handle potential division by zero if image dimensions are 0
        h, w = img.shape[:2]
        if min(h, w) == 0:
             logger.error("Input image has zero height or width.")
             return None
        scale_factor = self.initial_resize / min(h, w)
        # Ensure dimensions are integers and positive
        new_w = max(1, int(w * scale_factor))
        new_h = max(1, int(h * scale_factor))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.info(f"Initial resize from {w}x{h} to {new_w}x{new_h}")

        # --- Denoise ---
        denoised = cv2.fastNlMeansDenoisingColored(resized, None, h=self.denoise_h, templateWindowSize=7, searchWindowSize=21)
        logger.info(f"Applied non-local means denoising with h={self.denoise_h}")

        # --- Apply unsharp masking (optional) ---
        sharpened_image = self.unsharp_mask(denoised)
        
        # --- Quantize colors ---
        quantized_image = self.quantize_image(sharpened_image)
        if quantized_image is None: # Handle potential failure in quantization
             logger.error("Color quantization failed.")
             return None
        
        # --- Final resize to target size ---
        # Ensure target_size has positive integer values
        target_w, target_h = self.target_size
        if target_w <= 0 or target_h <= 0:
             logger.error(f"Invalid target_size: {self.target_size}. Using original quantized size.")
             final_image = quantized_image
        else:
             final_image = cv2.resize(quantized_image, self.target_size, interpolation=cv2.INTER_AREA)
             logger.info(f"Resized to target size {self.target_size}")

        logger.info("Preprocessing completed")
        return final_image

# --- efficient_data_sampling function (Unchanged from previous version) ---
# (Make sure List and Tuple are imported from typing if needed)
from typing import List, Tuple 

def efficient_data_sampling(rgb_data: np.ndarray, 
                            lab_data: np.ndarray, 
                            n_samples: int = 800, 
                            max_per_image: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    # ... (function implementation remains the same as your previous version) ...
    # ... (Make sure it handles the expected input shapes from load_data correctly) ...
    rgb_samples = []
    lab_samples = []
    
    # Handle flattened data: shape (n_images, n_pixels*3)
    if rgb_data.ndim == 2 and lab_data.ndim == 2 and rgb_data.shape[0] == lab_data.shape[0]:
        n_images = rgb_data.shape[0]
        if n_images == 0: raise ValueError("Input data arrays are empty.")
        
        # Check if n_pixels can be inferred
        if rgb_data.shape[1] % 3 != 0:
            raise ValueError(f"Flattened RGB data shape {rgb_data.shape} not divisible by 3.")
        n_pixels_per_image = rgb_data.shape[1] // 3
        
        if max_per_image is None:
            max_per_image = max(50, n_samples // n_images)
        
        logger.info(f"Sampling {n_samples} total samples from flattened data, max {max_per_image} per image")
        
        total_sampled_count = 0
        indices_list = [] # Collect all indices first

        for i in range(n_images):
            # Calculate how many to sample from this image
            remaining_samples = n_samples - total_sampled_count
            n_this_image = min(max_per_image, n_pixels_per_image, remaining_samples)

            if n_this_image <= 0 and total_sampled_count >= n_samples:
                break # Stop if we have enough samples
            if n_this_image <= 0:
                 logger.debug(f"Skipping image {i+1}, no samples needed or possible.")
                 continue

            # Generate indices for sampling within this image's flattened pixel range
            img_indices = np.random.choice(n_pixels_per_image, n_this_image, replace=False)
            indices_list.append((i, img_indices)) # Store image index and pixel indices
            total_sampled_count += n_this_image
            logger.debug(f"Planning to sample {n_this_image} pixels from image {i+1}")

        # Now extract samples efficiently using collected indices
        if not indices_list:
             raise ValueError("Could not plan any samples.")
             
        # Preallocate arrays for efficiency
        rgb_result = np.zeros((total_sampled_count, 3), dtype=np.float32)
        lab_result = np.zeros((total_sampled_count, 3), dtype=np.float32)
        current_idx = 0

        for img_idx, pixel_indices in indices_list:
            try:
                num_to_extract = len(pixel_indices)
                # Reshape data for this image
                rgb_flat_img = rgb_data[img_idx].reshape(-1, 3)
                lab_flat_img = lab_data[img_idx].reshape(-1, 3)

                # Extract RGB
                rgb_sample = rgb_flat_img[pixel_indices].astype(np.float32)
                # Ensure RGB is in 0-255 range (redundant if load_data guarantees it)
                # if np.any(rgb_sample > 1.0): # Quick check if needed
                #      rgb_sample = np.clip(rgb_sample, 0, 255)
                # else:
                #      rgb_sample *= 255.0 # Assume 0-1 if max is <= 1
                # rgb_sample = np.clip(rgb_sample, 0, 255) # Final clip

                # Extract LAB and convert ranges
                lab_sample = lab_flat_img[pixel_indices].astype(np.float32).copy()
                if np.any(lab_sample[:, 0] > 100): # Check if L needs conversion
                    lab_sample[:, 0] = lab_sample[:, 0] / 255.0 * 100.0
                if np.any(lab_sample[:, 1:] >= 0): # Check if a,b need conversion
                    lab_sample[:, 1:] = lab_sample[:, 1:] - 128.0
                    
                # Place extracted samples into preallocated arrays
                rgb_result[current_idx : current_idx + num_to_extract] = rgb_sample
                lab_result[current_idx : current_idx + num_to_extract] = lab_sample
                current_idx += num_to_extract
            
            except IndexError:
                 logger.error(f"Index error sampling image {img_idx}. Indices: {pixel_indices.max()} vs shape {rgb_data[img_idx].shape}")
                 continue # Skip this image
            except Exception as e:
                logger.error(f"Error extracting samples from image {img_idx}: {e}")
                continue # Skip this image

        # If errors occurred, the result arrays might be larger than needed
        if current_idx < total_sampled_count:
             logger.warning(f"Only extracted {current_idx}/{total_sampled_count} samples due to errors.")
             rgb_result = rgb_result[:current_idx]
             lab_result = lab_result[:current_idx]

        if rgb_result.shape[0] == 0:
            raise ValueError("No samples could be successfully extracted.")

        logger.info(f"Sampling completed: {rgb_result.shape[0]} samples extracted.")
        logger.info(f"RGB range: [{rgb_result.min():.2f}, {rgb_result.max():.2f}]")
        logger.info(f"LAB ranges: L[{lab_result[:, 0].min():.2f}, {lab_result[:, 0].max():.2f}], "
                    f"a[{lab_result[:, 1].min():.2f}, {lab_result[:, 1].max():.2f}], "
                    f"b[{lab_result[:, 2].min():.2f}, {lab_result[:, 2].max():.2f}]")
        
        return rgb_result, lab_result
        
    else:
         # Handle case where data might be a list of images (less efficient)
         # This part needs adjustment if load_data always returns flattened arrays
         logger.warning("efficient_data_sampling received data not in expected flattened format. Attempting list processing.")
         # ... (Your previous list processing logic could go here as a fallback, 
         #      but ensure load_data format is consistent first) ...
         raise NotImplementedError("List-based sampling needs review based on load_data output format.")