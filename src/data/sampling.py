import logging
import numpy as np
from typing import Tuple,Optional
from src.config_types import TrainConfig

logger = logging.getLogger(__name__)

def efficient_data_sampling(rgb_data: np.ndarray, 
                            lab_data: np.ndarray, 
                            train_config: TrainConfig, 
                            max_per_image: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Efficiently samples a respresentative subset of pixel data for DBN training
    Takes flattenedd RGB and LAB data output from load_data, determines how many samples to take per image based on the total target sample size (n_samples)
    and a minimum per image, randomly selects pixel indices and extracts the corresponding RGB and LAB values. Also ensures LAB values are converted to the 
    standard CIELAB ranges (L: 0-100, a/b: -128-127)  if they are detected in OpenCV's range
    
    Args: 
        rgb_data (np.ndarray): Flattened RGB data (n_images, n_pixels*3), range [0, 255].
        lab_data (np.ndarray): Flattened LAB data (n_images, n_pixels*3), OpenCV range.
        train_config (TrainConfig): Configuration object containing 'n_samples'
                                    and 'min_samples_per_image'.
        max_per_image (Optional[int]): Maximum samples to draw from a single image.
                                       If None, calculated automatically based on n_samples,
                                       n_images, and min_samples_per_image.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - rgb_samples (np.ndarray): Sampled RGB pixels (n_total_samples, 3), range [0, 255].
            - lab_samples (np.ndarray): Sampled LAB pixels (n_total_samples, 3), standard CIELAB range.

    Raises:
        ValueError: If input arrays are empty, shapes are incompatible,
                    or no samples could be extracted.
    
    """
    n_samples = train_config.n_samples
    
    # Handle flattened data: shape (n_images, n_pixels*3)
    if rgb_data.ndim == 2 and lab_data.ndim == 2 and rgb_data.shape[0] == lab_data.shape[0]:
        n_images = rgb_data.shape[0]
        if n_images == 0: raise ValueError("Input data arrays are empty.")
        
        # Check if n_pixels can be inferred
        if rgb_data.shape[1] % 3 != 0:
            raise ValueError(f"Flattened RGB data shape {rgb_data.shape} not divisible by 3.")
        n_pixels_per_image = rgb_data.shape[1] // 3
        
        if max_per_image is None:
            min_per_img = train_config.min_samples_per_image
            max_per_image = max(min_per_img, n_samples // n_images)
        
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
            # replace = False -> prevents to select the same pixel
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
         logger.warning("efficient_data_sampling received data not in expected flattened format. Attempting list processing.")
         raise NotImplementedError("List-based sampling needs review based on load_data output format.")