import os
import numpy as np
import cv2 # Keep cv2 import if needed by future additions, though not directly used by DPC.
import logging
from sklearn.metrics import pairwise_distances_chunked
from pathlib import Path
from typing import Tuple, Optional, List, Union

logger = logging.getLogger(__name__)

# --- Density Peak Clustering (DPC) Helper Functions ---
# Note: These functions implement the DPC algorithm, originally intended
#       for estimating the optimal number of clusters ('k').
#       They are currently NOT directly used by the main segmentation pipeline
#       (which uses MetricBasedStrategy), but are kept for potential future use.
# TODO: Move these DPC-related functions (gaussian_kernel, dpc_clustering,
#       optimal_clusters_dpc) to a more specific module, potentially
#       `src/utils/clustering_utils.py` or similar, as they are not generic
#       image utilities.

def gaussian_kernel(distance: Union[float, np.ndarray], bandwidth: float) -> Union[float, np.ndarray]:
    """
    Computes the Gaussian kernel value for a given distance and bandwidth.

    Used in DPC to estimate local density based on distances to neighboring points.

    Args:
        distance: The Euclidean distance(s) between points.
        bandwidth: The bandwidth (sigma) of the Gaussian kernel. Controls the
                   'reach' of the density estimation.

    Returns:
        The kernel value(s), ranging from 0 to 1.
    """
    # Ensure bandwidth is positive to prevent division by zero or complex numbers
    bandwidth = max(bandwidth, 1e-9)
    # Gaussian function: exp(-0.5 * (distance / bandwidth)^2)
    return np.exp(-0.5 * (distance / bandwidth) ** 2)

def dpc_clustering(pixels: np.ndarray, num_clusters: int, bandwidth: float = 1.0) -> np.ndarray:
    """
    Performs Density Peak Clustering (DPC) on pixel data.

    Calculates local density (rho) and distance to nearest higher density point (delta)
    for each pixel. Identifies cluster centers as points with high rho and high delta.
    Assigns other points to the cluster of their nearest higher-density neighbor.

    Args:
        pixels: The input pixel data, shape (n_samples, n_features).
        num_clusters: The target number of clusters to identify.
        bandwidth: The bandwidth for the Gaussian kernel in density calculation.

    Returns:
        An array of cluster labels assigned to each input pixel. Returns an
        array of -1s (noise) on failure or empty array if input is empty.
    """
    logging.info(f"Starting Density Peak Clustering: target_k={num_clusters}, bandwidth={bandwidth}")
    n_samples = pixels.shape[0]
    if n_samples == 0:
        logging.warning("DPC Clustering received an empty pixel array.")
        return np.array([], dtype=int)
    if pixels.ndim != 2:
         logging.error(f"DPC Clustering expects 2D array (samples, features), got shape {pixels.shape}")
         return np.full(n_samples, -1, dtype=int)

    # --- 1. Calculate Pairwise Distances ---
    distances: np.ndarray
    try:
        distances_gen = pairwise_distances_chunked(pixels, working_memory=256)
        distances = np.vstack(list(distances_gen))
    except MemoryError:
         logging.error("DPC Clustering: MemoryError calculating pairwise distances.")
         return np.full(n_samples, -1, dtype=int)
    except Exception as e:
        logging.error(f"DPC Clustering: Unexpected error during distance calculation: {e}", exc_info=True)
        return np.full(n_samples, -1, dtype=int)

    # --- 2. Calculate Rho (Local Density) ---
    rho = np.zeros(n_samples)
    try:
        rho = np.sum(gaussian_kernel(distances, bandwidth), axis=1) - 1.0 # Subtract self-distance
        rho = np.maximum(rho, 1e-9) # Ensure positive rho
    except Exception as e:
        logging.error(f"DPC Clustering: Error computing rho (density): {e}", exc_info=True)
        return np.full(n_samples, -1, dtype=int)

    # --- 3. Calculate Delta (Distance to Nearest Higher Density Point) ---
    delta = np.zeros(n_samples)
    nearest_higher_neighbor = np.full(n_samples, -1, dtype=int)
    try:
        rho_sorted_indices = np.argsort(-rho) # Indices sorted by density (desc)

        max_dist = np.max(distances) if distances.size > 0 else 0.0
        delta[rho_sorted_indices[0]] = max_dist
        nearest_higher_neighbor[rho_sorted_indices[0]] = rho_sorted_indices[0] # Highest density points to itself

        for i in range(1, n_samples):
            current_point_idx = rho_sorted_indices[i]
            higher_density_indices = rho_sorted_indices[:i] # Indices of points denser than current

            if len(higher_density_indices) == 0:
                 delta[current_point_idx] = max_dist
                 nearest_higher_neighbor[current_point_idx] = current_point_idx
                 continue

            dist_to_higher_density = distances[current_point_idx, higher_density_indices]
            min_dist_idx_in_subset = np.argmin(dist_to_higher_density)
            delta[current_point_idx] = dist_to_higher_density[min_dist_idx_in_subset]
            nearest_higher_neighbor[current_point_idx] = higher_density_indices[min_dist_idx_in_subset]

    except Exception as e:
         logging.error(f"DPC Clustering: Error computing delta (distance): {e}", exc_info=True)
         return np.full(n_samples, -1, dtype=int)

    # --- 4. Identify Cluster Centers ---
    gamma = rho * delta # Decision value: high for potential centers
    actual_num_clusters = min(num_clusters, n_samples)
    if actual_num_clusters <= 0 :
         logging.warning(f"DPC Clustering: Requested num_clusters ({num_clusters}) results in <= 0 actual clusters.")
         return np.full(n_samples, -1, dtype=int)

    center_indices = np.argsort(-gamma)[:actual_num_clusters] # Indices of top gamma points

    # --- 5. Assign Labels ---
    labels = np.full(n_samples, -1, dtype=int) # Initialize as noise

    for k, center_idx in enumerate(center_indices):
        labels[center_idx] = k # Assign labels 0, 1, ... to centers

    # Assign labels to non-centers by following neighbor path
    for i in range(n_samples):
        point_idx = rho_sorted_indices[i] # Iterate in density order
        if labels[point_idx] == -1: # If not a center
            neighbor_idx = nearest_higher_neighbor[point_idx]
            if neighbor_idx != -1 and labels[neighbor_idx] != -1:
                 labels[point_idx] = labels[neighbor_idx] # Assign neighbor's label
            # else: remain -1 (noise)

    found_clusters = len(np.unique(labels[labels >= 0]))
    noise_points = np.sum(labels == -1)
    logging.info(f"Density Peak Clustering completed. Assigned {found_clusters} clusters. Found {noise_points} noise points.")
    return labels


def optimal_clusters_dpc(pixels: np.ndarray,
                         min_k: int = 2,
                         max_k: int = 10,
                         subsample_threshold: int = 1000,
                         bandwidth: Optional[float] = None) -> Optional[int]:
    """
    Estimates the optimal number of clusters ('k') using DPC properties (gamma heuristic).

    Calculates rho and delta for a subsample, then counts points where
    gamma (rho * delta) exceeds a threshold (mean + std_dev) as potential centers.

    Args:
        pixels: Input pixel data (n_samples, n_features).
        min_k: Minimum allowable number of clusters.
        max_k: Maximum allowable number of clusters.
        subsample_threshold: Subsample size for faster calculation if unique pixels exceed this.
        bandwidth: Bandwidth for Gaussian kernel. Estimated if None.

    Returns:
        Estimated optimal number of clusters within [min_k, max_k],
        None if input is empty, or min_k on failure.
    """
    n_samples = pixels.shape[0]
    if n_samples == 0:
        logging.warning("Cannot determine DPC clusters: Input pixel array is empty.")
        return None

    if pixels.ndim != 2:
        logging.error(f"optimal_clusters_dpc expects 2D array (samples, features), got shape {pixels.shape}")
        return min_k # Fallback

    # --- Subsampling based on UNIQUE colors ---
    try:
        unique_colors = np.unique(pixels, axis=0)
        n_unique = len(unique_colors)
    except Exception as e:
         logging.warning(f"Could not find unique colors efficiently: {e}. Using original pixels.")
         unique_colors = pixels
         n_unique = n_samples

    dynamic_max_k = max(min_k, min(max_k, n_unique -1 )) # k must be less than n_samples
    if dynamic_max_k < min_k: # Handle case where unique colors < min_k
         logging.warning(f"Number of unique colors ({n_unique}) is less than min_k ({min_k}). Falling back.")
         return min_k
    logging.info(f"Number of unique colors: {n_unique}. Using dynamic max_k: {dynamic_max_k}")

    if n_unique > subsample_threshold:
        logging.info(f"Subsampling unique colors from {n_unique} down to {subsample_threshold} for DPC gamma analysis.")
        try:
            subsample_indices = np.random.choice(n_unique, subsample_threshold, replace=False)
            subsampled_colors = unique_colors[subsample_indices]
        except ValueError as e:
             logging.warning(f"Subsampling failed ({e}). Using all unique colors.")
             subsampled_colors = unique_colors
    else:
        subsampled_colors = unique_colors

    n_subsample = subsampled_colors.shape[0]
    if n_subsample < min_k:
         logging.warning(f"Number of unique points after subsampling ({n_subsample}) < min_k ({min_k}). Falling back.")
         return min_k

    # --- Calculate DPC properties on the subsample ---
    try:
        distances_gen = pairwise_distances_chunked(subsampled_colors, working_memory=256)
        distances = np.vstack(list(distances_gen))

        if bandwidth is None:
            if distances.size > 0:
                 non_zero_dists = distances[distances > 1e-9]
                 bandwidth = np.median(non_zero_dists) if non_zero_dists.size > 0 else 1.0
            else:
                 bandwidth = 1.0
            logging.info(f"Estimated DPC bandwidth: {bandwidth:.3f}")
        bandwidth = max(bandwidth, 1e-9)

        rho = np.sum(gaussian_kernel(distances, bandwidth), axis=1) - 1.0
        rho = np.maximum(rho, 1e-9)
        delta = np.zeros(n_subsample)
        rho_sorted_indices = np.argsort(-rho)
        max_dist = np.max(distances) if distances.size > 0 else 0.0
        delta[rho_sorted_indices[0]] = max_dist

        for i in range(1, n_subsample):
            current_point_idx = rho_sorted_indices[i]
            higher_density_indices = rho_sorted_indices[:i]
            if len(higher_density_indices) == 0:
                 delta[current_point_idx] = max_dist
                 continue
            dist_to_higher_density = distances[current_point_idx, higher_density_indices]
            delta[current_point_idx] = np.min(dist_to_higher_density)

        gamma = rho * delta

        # --- Heuristic: Count points with high gamma ---
        n_clusters = 0
        if gamma.size > 1:
             gamma_mean = np.mean(gamma)
             gamma_std = np.std(gamma)
             threshold = gamma_mean + 1.0 * gamma_std # Threshold = mean + 1 std dev
             potential_centers_mask = gamma > threshold
             n_clusters = int(np.sum(potential_centers_mask))
             logger.info(f"DPC gamma analysis: Mean={gamma_mean:.3f}, Std={gamma_std:.3f}, Threshold={threshold:.3f} -> Found {n_clusters} potential centers.")
        elif gamma.size == 1:
             n_clusters = 1
             logger.info("DPC gamma analysis: Only one point found.")
        else: # gamma.size == 0
             n_clusters = 0
             logger.warning("DPC gamma analysis: No gamma values calculated.")

        # Ensure result is within bounds [min_k, dynamic_max_k]
        n_clusters_final = max(min_k, min(n_clusters, dynamic_max_k))
        if n_clusters < min_k:
             logger.warning(f"DPC heuristic found {n_clusters} < min_k ({min_k}). Adjusting to {min_k}.")
             n_clusters_final = min_k # Force to min_k if heuristic finds too few

        logger.info(f"Optimal number of clusters estimated by DPC gamma heuristic: {n_clusters_final}")
        return n_clusters_final

    except MemoryError:
        logging.error("MemoryError during optimal_clusters_dpc. Falling back to min_k.")
        return min_k
    except Exception as e:
        logging.error(f"Error calculating optimal clusters using DPC: {e}. Falling back to min_k={min_k}", exc_info=True)
        return min_k