import os
import numpy as np
import cv2
import logging
from sklearn.metrics import pairwise_distances_chunked
import time # Zamanlama için eklendi
from pathlib import Path # Path objeleriyle çalışmak için eklendi
from typing import Tuple, Optional, List # Type hinting için eklendi

# Logging instance
logger = logging.getLogger(__name__)

# --- Helper Functions (gaussian_kernel, dpc_clustering, optimal_clusters_dpc) ---
# (Bu fonksiyonlar aynı kalıyor)

def gaussian_kernel(distance, bandwidth):
    """Compute Gaussian kernel for density estimation."""
    # Add a small epsilon to prevent division by zero if bandwidth is tiny
    bandwidth = max(bandwidth, 1e-9)
    return np.exp(-0.5 * (distance / bandwidth) ** 2)

def dpc_clustering(pixels, num_clusters, bandwidth=1.0):
    """Perform Density Peak Clustering on pixel data."""
    logging.info(f"Starting Density Peak Clustering with num_clusters={num_clusters}, bandwidth={bandwidth}")
    n_samples = len(pixels)
    if n_samples == 0:
        logging.warning("DPC Clustering: Input pixels array is empty.")
        return np.array([], dtype=int)

    # Calculate distances more robustly
    try:
        # Using generator with vstack might be memory intensive for large datasets
        # Consider alternatives if memory becomes an issue
        distances_gen = pairwise_distances_chunked(pixels, working_memory=256) # Smaller memory chunk
        distances = np.vstack(list(distances_gen))
    except MemoryError:
         logging.error("DPC Clustering: MemoryError calculating pairwise distances.")
         # Fallback or simplified approach might be needed here
         return np.full(n_samples, -1, dtype=int) # Return noise labels
    except Exception as e:
        logging.error(f"DPC Clustering: Error calculating pairwise distances: {e}", exc_info=True)
        return np.full(n_samples, -1, dtype=int)

    rho = np.zeros(n_samples)
    delta = np.zeros(n_samples)

    # Compute rho (local density)
    try:
        # Avoid iterating again if distances matrix fits in memory
        rho = np.sum(gaussian_kernel(distances, bandwidth), axis=1)
        # Handle cases where rho might be zero or very small if bandwidth is inappropriate
        rho[rho < 1e-9] = 1e-9 # Prevent division by zero later if used
    except Exception as e:
        logging.error(f"DPC Clustering: Error computing rho: {e}", exc_info=True)
        return np.full(n_samples, -1, dtype=int)

    # Compute delta (distance to nearest higher density point)
    try:
        # Sort points by density in descending order
        rho_sorted_indices = np.argsort(-rho)

        delta[rho_sorted_indices[0]] = -1.0 # Highest density point has no delta initially
        max_dist = np.max(distances) if distances.size > 0 else 0 # Find max distance for later
        
        for i in range(1, n_samples):
            current_point_idx = rho_sorted_indices[i]
            # Consider only points with higher density
            higher_density_indices = rho_sorted_indices[:i]
            if len(higher_density_indices) == 0:
                 # Should not happen if i > 0, but as safety
                 delta[current_point_idx] = max_dist
                 continue
                 
            # Find distance to the *nearest* point with higher density
            delta[current_point_idx] = np.min(distances[current_point_idx, higher_density_indices])

        # Set delta for the highest density point
        delta[rho_sorted_indices[0]] = max_dist # Often set to max distance among all points
        
    except Exception as e:
         logging.error(f"DPC Clustering: Error computing delta: {e}", exc_info=True)
         return np.full(n_samples, -1, dtype=int)

    # Identify cluster centers (high rho and high delta)
    gamma = rho * delta
    # Ensure num_clusters is not more than available samples
    actual_num_clusters = min(num_clusters, n_samples)
    if actual_num_clusters <=0 :
         logging.warning("DPC Clustering: num_clusters is zero or negative. Cannot find centers.")
         return np.full(n_samples, -1, dtype=int)
         
    # Indices of cluster centers (highest gamma values)
    center_indices = np.argsort(-gamma)[:actual_num_clusters]

    # Assign labels
    labels = np.full(n_samples, -1, dtype=int) # Initialize all as noise/unassigned

    # Assign centers their own label (using their index as label ID for now)
    for k, center_idx in enumerate(center_indices):
        labels[center_idx] = k # Assign 0, 1, 2... as labels

    # Assign remaining points to the same cluster as their nearest higher density point
    # Iterate in order of density (highest first)
    for i in range(n_samples):
        point_idx = rho_sorted_indices[i]
        if labels[point_idx] == -1: # If not already a center or assigned
             # Find the nearest point with *higher* density
             higher_density_indices = rho_sorted_indices[:i]
             if len(higher_density_indices) == 0:
                  # This point has the highest density, should already be a center
                  # If not, assign it noise or handle as error? Assign noise for now.
                  labels[point_idx] = -1 
                  continue
                  
             nearest_higher_density_neighbor_idx = higher_density_indices[np.argmin(distances[point_idx, higher_density_indices])]
             
             # Assign the label of the nearest higher density neighbor
             if labels[nearest_higher_density_neighbor_idx] != -1:
                 labels[point_idx] = labels[nearest_higher_density_neighbor_idx]
             else:
                 # Should not happen if logic is correct, indicates an issue
                 # Assign noise for safety
                 labels[point_idx] = -1
                 logging.warning(f"DPC: Point {point_idx} could not be assigned label from neighbor {nearest_higher_density_neighbor_idx}")


    logging.info(f"Density Peak Clustering completed. Found {len(np.unique(labels[labels>=0]))} clusters.")
    return labels


def optimal_clusters_dpc(pixels, min_k=2, max_k=10, subsample_threshold=1000, bandwidth=None) -> Optional[int]:
    """Determine optimal number of clusters using DPC based on gamma thresholding (heuristic)."""
    
    n_samples = pixels.shape[0]
    if n_samples == 0:
        logging.warning("Cannot determine DPC clusters: Input pixels array is empty.")
        return None # Return None to indicate failure

    unique_colors = np.unique(pixels, axis=0)
    n_unique = len(unique_colors)
    # Adjust max_k dynamically but ensure it's at least min_k
    dynamic_max_k = max(min_k, min(max_k, n_unique)) 

    logging.info(f"Number of unique colors: {n_unique}. Using dynamic_max_k: {dynamic_max_k}")

    # Subsample if necessary
    if n_unique > subsample_threshold:
        logging.info(f"Subsampling unique colors from {n_unique} to {subsample_threshold} for DPC analysis")
        subsample_indices = np.random.choice(n_unique, subsample_threshold, replace=False)
        subsampled_colors = unique_colors[subsample_indices]
    else:
        subsampled_colors = unique_colors
        
    n_subsample = subsampled_colors.shape[0]
    if n_subsample < min_k: # Not enough points even after subsampling
         logging.warning(f"Number of unique points ({n_subsample}) is less than min_k ({min_k}). Cannot reliably determine DPC clusters. Falling back to min_k.")
         return min_k

    try:
        # Calculate distances on subsampled points
        distances_gen = pairwise_distances_chunked(subsampled_colors, working_memory=256)
        distances = np.vstack(list(distances_gen))

        # Estimate bandwidth if not provided (e.g., using median distance)
        if bandwidth is None:
            if distances.size > 0:
                 # Calculate median of non-zero distances
                 non_zero_dists = distances[distances > 1e-9]
                 bandwidth = np.median(non_zero_dists) if non_zero_dists.size > 0 else 1.0
            else:
                 bandwidth = 1.0 # Default if no distances calculated
            logging.info(f"Calculated DPC bandwidth: {bandwidth:.3f}")
            
        bandwidth = max(bandwidth, 1e-9) # Ensure positive bandwidth

        # Calculate rho and delta on subsampled points
        rho = np.sum(gaussian_kernel(distances, bandwidth), axis=1)
        rho[rho < 1e-9] = 1e-9
        delta = np.zeros(n_subsample)
        rho_sorted_indices = np.argsort(-rho)
        delta[rho_sorted_indices[0]] = -1.0
        max_dist = np.max(distances) if distances.size > 0 else 0

        for i in range(1, n_subsample):
            current_point_idx = rho_sorted_indices[i]
            higher_density_indices = rho_sorted_indices[:i]
            if len(higher_density_indices) == 0:
                 delta[current_point_idx] = max_dist
                 continue
            delta[current_point_idx] = np.min(distances[current_point_idx, higher_density_indices])
        delta[rho_sorted_indices[0]] = max_dist

        # Calculate gamma
        gamma = rho * delta

        # --- Heuristic to determine number of clusters ---
        # Look for points with significantly high gamma (potential centers)
        # A simple heuristic: points with gamma > mean(gamma) + std(gamma)
        if gamma.size > 1: # Need at least 2 points to calculate mean/std
             gamma_mean = np.mean(gamma)
             gamma_std = np.std(gamma)
             threshold = gamma_mean + gamma_std 
             potential_centers_mask = gamma > threshold
             n_clusters = int(np.sum(potential_centers_mask))
             logging.info(f"DPC gamma analysis: mean={gamma_mean:.3f}, std={gamma_std:.3f}, threshold={threshold:.3f}")
        elif gamma.size == 1:
             n_clusters = 1
        else: # gamma.size == 0
             n_clusters = 0
             
        # Ensure n_clusters is within the desired range [min_k, dynamic_max_k]
        n_clusters = max(min_k, min(n_clusters, dynamic_max_k))
        # Handle case where threshold finds 0 clusters but we need at least min_k
        if n_clusters < min_k: n_clusters = min_k 
        
        logging.info(f"Optimal number of clusters estimated by DPC gamma heuristic: {n_clusters}")
        return n_clusters

    except MemoryError:
        logging.error("MemoryError during optimal_clusters_dpc. Falling back to min_k.")
        return min_k
    except Exception as e:
        logging.error(f"Error in calculating optimal clusters using DPC: {e}. Falling back to min_k={min_k}", exc_info=True)
        return min_k
