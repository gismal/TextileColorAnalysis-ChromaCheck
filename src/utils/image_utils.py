import os
import numpy as np
import cv2
import logging
import pandas as pd
from skimage.color import deltaE_ciede2000
from sklearn.metrics import pairwise_distances_chunked

from src.data.preprocess import Preprocessor
from src.models.pso_dbn import convert_colors_to_cielab_dbn
from src.utils.segmentation_utils import k_mean_segmentation, optimal_clusters, som_segmentation
from src.utils.color.color_conversion import convert_colors_to_cielab

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ciede2000_distance(color1, color2):
    """Calculate CIEDE2000 color difference between two CIELAB colors.
    
    Args:
        color1 (tuple): First CIELAB color (L, a, b).
        color2 (tuple): Second CIELAB color (L, a, b).
    
    Returns:
        float: Delta E value.
    """
    return deltaE_ciede2000(np.array([color1]), np.array([color2]))[0]

def calculate_similarity(segmentation_data, target_colors):
    """Calculate similarity scores for segmented colors against target colors.
    
    Args:
        segmentation_data (dict): Contains 'avg_colors_lab' key with CIELAB colors.
        target_colors (list): List of target CIELAB colors.
    
    Returns:
        list: Similarity scores.
    """
    logging.info("Calculating similarity scores")
    similarity_scores = []
    avg_colors_lab = segmentation_data['avg_colors_lab']
    for color_lab in avg_colors_lab:
        if np.all(np.array(color_lab) <= np.array([5, 130, 130])):  # Check for nearly black segment
            similarity_scores.append(0)
            continue
        min_distance = float('inf')
        for target_color in target_colors:
            distance = ciede2000_distance(color_lab, target_color)
            if distance < min_distance:
                min_distance = distance
        similarity_score = max(0, 100 - min_distance)
        similarity_scores.append(similarity_score)
    logging.info("Similarity scores calculation completed")
    return similarity_scores

def find_best_matches(segmentation_data, reference_segmentation_data):
    """Find best matching segments between test and reference images.
    
    Args:
        segmentation_data (dict): Contains 'avg_colors_lab' for test image.
        reference_segmentation_data (dict): Contains 'avg_colors_lab' for reference.
    
    Returns:
        list: Tuple of (test_idx, ref_idx) pairs.
    """
    logging.info("Finding best matches for segments")
    avg_colors_lab = segmentation_data['avg_colors_lab']
    ref_avg_colors_lab = reference_segmentation_data['avg_colors_lab']

    best_matches = []
    for i, color_lab in enumerate(avg_colors_lab):
        if np.all(np.array(color_lab) <= np.array([5, 130, 130])):  # Ignore nearly black segments
            best_matches.append((i, -1))
            continue
        min_distance = float('inf')
        best_match_idx = -1
        for j, ref_color_lab in enumerate(ref_avg_colors_lab):
            distance = ciede2000_distance(color_lab, ref_color_lab)
            if distance < min_distance:
                min_distance = distance
                best_match_idx = j
        best_matches.append((i, best_match_idx))
    logging.info("Best matches found")
    return best_matches

def downsample_image(image, scale_factor=0.5):
    """Downsample an image by a given scale factor.
    
    Args:
        image (numpy.ndarray): Input image in BGR format.
        scale_factor (float): Factor to downsample (default: 0.5).
    
    Returns:
        numpy.ndarray: Downsampled image.
    """
    width = int(image.shape[1] * scale_factor)
    height = int(image.shape[0] * scale_factor)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

def gaussian_kernel(distance, bandwidth):
    """Compute Gaussian kernel for density estimation.
    
    Args:
        distance (float): Distance value.
        bandwidth (float): Bandwidth parameter.
    
    Returns:
        float: Kernel value.
    """
    return np.exp(-0.5 * (distance / bandwidth) ** 2)

def dpc_clustering(pixels, num_clusters, bandwidth=1.0):
    """Perform Density Peak Clustering on pixel data.
    
    Args:
        pixels (numpy.ndarray): Pixel data (n_samples, 3).
        num_clusters (int): Number of clusters.
        bandwidth (float): Bandwidth for Gaussian kernel.
    
    Returns:
        numpy.ndarray: Cluster labels.
    """
    logging.info("Starting Density Peak Clustering")
    distances = pairwise_distances_chunked(pixels)
    distances = np.vstack(list(distances))
    rho = np.zeros(len(pixels))
    delta = np.zeros(len(pixels))

    def compute_rho(distances_chunk):
        return np.sum(gaussian_kernel(distances_chunk, bandwidth), axis=1)

    for distances_chunk in pairwise_distances_chunked(pixels, working_memory=1024):
        rho_chunk = compute_rho(distances_chunk)
        rho[:len(rho_chunk)] += rho_chunk

    for i in range(len(rho)):
        mask = (rho > rho[i])
        if np.any(mask):
            delta[i] = np.min(distances[i, mask])
        else:
            delta[i] = np.max(distances[i])

    gamma = rho * delta
    centers = np.argsort(gamma)[-num_clusters:]
    labels = np.full(len(pixels), -1, dtype=int)

    for center in centers:
        labels[center] = center

    for i in np.argsort(-rho):
        if labels[i] == -1:
            labels[i] = labels[np.argmin(distances[i, centers])]

    logging.info("Density Peak Clustering completed")
    return labels

def optimal_clusters_dpc(pixels, min_k=2, max_k=10, subsample_threshold=1000, bandwidth=None):
    """Determine optimal number of clusters using DPC.
    
    Args:
        pixels (numpy.ndarray): Pixel data.
        min_k (int): Minimum number of clusters.
        max_k (int): Maximum number of clusters.
        subsample_threshold (int): Threshold for subsampling.
        bandwidth (float): Bandwidth parameter (optional).
    
    Returns:
        int: Optimal number of clusters.
    """
    unique_colors = np.unique(pixels, axis=0)
    dynamic_max_k = min(max_k, len(unique_colors))

    logging.info(f"Number of unique colors: {len(unique_colors)}. Adjusting max_k to: {dynamic_max_k}")

    if len(unique_colors) > subsample_threshold:
        logging.info(f"Subsampling unique colors from {len(unique_colors)} to {subsample_threshold}")
        subsample_indices = np.random.choice(len(unique_colors), subsample_threshold, replace=False)
        subsampled_colors = unique_colors[subsample_indices]
    else:
        subsampled_colors = unique_colors

    try:
        if bandwidth is None:
            pairwise_dists = pairwise_distances_chunked(subsampled_colors)
            bandwidth = np.median(pairwise_dists)
            logging.info(f"Calculated bandwidth: {bandwidth}")

        dpc_labels = dpc_clustering(subsampled_colors, dynamic_max_k, bandwidth=1.0 if bandwidth is None else bandwidth)
        n_clusters = len(np.unique(dpc_labels))
        logging.info(f"Optimal number of clusters determined by DPC: {n_clusters}")

        if n_clusters > max_k:
            n_clusters = max_k
            logging.info(f"Number of clusters exceeds max_k. Limiting to max_k: {max_k}")

        return n_clusters
    except Exception as e:
        logging.error(f"Error in calculating optimal clusters using DPC: {str(e)}. Falling back to default k: {min_k}")
        return min_k

def create_output_folders(base_path, techniques):
    """Create output folders for each segmentation technique.
    
    Args:
        base_path (str): Base directory for output.
        techniques (list): List of segmentation techniques.
    """
    for technique in techniques:
        technique_path = os.path.join(base_path, technique)
        os.makedirs(os.path.join(technique_path, 'optimal'), exist_ok=True)
        os.makedirs(os.path.join(technique_path, 'predefined'), exist_ok=True)
        os.makedirs(os.path.join(technique_path, 'summary'), exist_ok=True)

def save_delta_e_results(delta_e_results, output_path):
    """Save Delta E results to a CSV file.
    
    Args:
        delta_e_results (dict): Dictionary of Delta E values.
        output_path (str): Directory to save the CSV.
    """
    df = pd.DataFrame(delta_e_results)
    df.to_csv(os.path.join(output_path, 'overall_delta_e_results.csv'), index=False)

def save_results(segmentation_data, similarity_scores, method, output_dir):
    """Save segmentation results including images and data.
    
    Args:
        segmentation_data (dict): Segmentation data.
        similarity_scores (list): Similarity scores.
        method (str): Segmentation method.
        output_dir (str): Output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    cv2.imwrite(os.path.join(output_dir, f'segmented_image_{method}.png'), segmentation_data['segmented_image'])

    avg_colors_lab_df = pd.DataFrame(segmentation_data['avg_colors_lab'], columns=['L', 'a', 'b'])
    avg_colors_lab_dbn_df = pd.DataFrame(segmentation_data['avg_colors_lab_dbn'], columns=['L', 'a', 'b'])
    avg_colors_lab_df.to_csv(os.path.join(output_dir, f'average_colors_lab_{method}.csv'), index=False)
    avg_colors_lab_dbn_df.to_csv(os.path.join(output_dir, f'average_colors_lab_dbn_{method}.csv'), index=False)

    similarity_scores_df = pd.DataFrame(similarity_scores, columns=['Similarity_Score'])
    similarity_scores_df.to_csv(os.path.join(output_dir, f'similarity_scores_{method}.csv'), index=False)

    from src.utils.visualization import save_segmentation_summary_plot
    save_segmentation_summary_plot(segmentation_data, similarity_scores, method, output_dir=output_dir)

def compare_cielab_colors(test_avg_colors_lab, reference_avg_colors_lab):
    """Compare test and reference CIELAB colors to find best matches.
    
    Args:
        test_avg_colors_lab (list): Test image CIELAB colors.
        reference_avg_colors_lab (list): Reference image CIELAB colors.
    
    Returns:
        list: Tuple of (test_idx, ref_idx, distance) for each comparison.
    """
    comparisons = []
    for i, test_color in enumerate(test_avg_colors_lab):
        min_distance = float('inf')
        best_match_idx = -1
        for j, ref_color in enumerate(reference_avg_colors_lab):
            distance = deltaE_ciede2000(np.array([test_color]), np.array([ref_color]))[0]
            if distance < min_distance:
                min_distance = distance
                best_match_idx = j
        comparisons.append((i, best_match_idx, min_distance))
    return comparisons

def load_and_preprocess_image(image_path, preprocessor):
    """Load and preprocess the image.
    
    Args:
        image_path (str): Path to the image.
        preprocessor (ImagePreprocessor): Preprocessor instance.
    
    Returns:
        numpy.ndarray: Preprocessed image, or None if failed.
    """
    logging.info(f"Loading image from {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Failed to load image: {image_path}")
        return None
    logging.info("Starting preprocessing")
    preprocessed_image = preprocessor.preprocess(image)
    if preprocessed_image is None:
        logging.error("Preprocessing failed")
        return None
    logging.info("Preprocessing completed")
    return preprocessed_image

def determine_optimal_clusters(pixels_subsample, default_k, min_k=3, max_k=10):
    """Determine the optimal number of clusters.
    
    Args:
        pixels_subsample (numpy.ndarray): Subsampled pixel data.
        default_k (int): Default number of clusters.
        min_k (int): Minimum number of clusters.
        max_k (int): Maximum number of clusters.
    
    Returns:
        int: Optimal number of clusters.
    """
    logging.info("Determining optimal number of clusters")
    unique_colors = np.unique(pixels_subsample, axis=0)
    logging.info(f"Number of unique colors after quantization: {len(unique_colors)}")
    n_clusters = optimal_clusters(pixels_subsample, default_k, min_k=min_k, max_k=max_k)
    logging.info(f"Optimal number of clusters determined: {n_clusters}")
    return n_clusters

def perform_segmentation(image, n_clusters, dbn, scaler_x, scaler_y, scaler_y_ab):
    """Perform K-means and SOM segmentation on the image.
    
    Args:
        image (numpy.ndarray): Preprocessed image.
        n_clusters (int): Number of clusters.
        dbn (DBN): Trained DBN model.
        scaler_x (StandardScaler): Scaler for RGB input.
        scaler_y (MinMaxScaler): Scaler for CIELAB L channel.
        scaler_y_ab (MinMaxScaler): Scaler for CIELAB a, b channels.
    
    Returns:
        tuple: K-means and SOM segmentation data.
    """
    logging.info("Performing K-means segmentation with optimal k")
    segmented_image_kmeans_opt, avg_colors_kmeans_opt, labels_kmeans_opt = k_mean_segmentation(image, n_clusters)
    avg_colors_lab_kmeans_opt = convert_colors_to_cielab(avg_colors_kmeans_opt)
    avg_colors_lab_dbn_kmeans_opt = convert_colors_to_cielab_dbn(dbn, scaler_x, scaler_y, scaler_y_ab, avg_colors_kmeans_opt)
    reference_kmeans_opt = {
        'original_image': image,
        'segmented_image': segmented_image_kmeans_opt,
        'avg_colors': avg_colors_kmeans_opt,
        'avg_colors_lab': avg_colors_lab_kmeans_opt,
        'avg_colors_lab_dbn': avg_colors_lab_dbn_kmeans_opt,
        'labels': labels_kmeans_opt
    }

    logging.info("Performing SOM segmentation with optimal k")
    segmented_image_som_opt, avg_colors_som_opt, labels_som_opt = som_segmentation(image, n_clusters)
    avg_colors_lab_som_opt = convert_colors_to_cielab(avg_colors_som_opt)
    avg_colors_lab_dbn_som_opt = convert_colors_to_cielab_dbn(dbn, scaler_x, scaler_y, scaler_y_ab, avg_colors_som_opt)
    reference_som_opt = {
        'original_image': image,
        'segmented_image': segmented_image_som_opt,
        'avg_colors': avg_colors_som_opt,
        'avg_colors_lab': avg_colors_som_opt,  # [FIXED] Corrected to avg_colors_lab_som_opt
        'avg_colors_lab_dbn': avg_colors_lab_dbn_som_opt,
        'labels': labels_som_opt
    }

    return reference_kmeans_opt, reference_som_opt

def process_reference_image(reference_image_path, dbn, scaler_x, scaler_y, scaler_y_ab, default_k):
    """Process the reference image to generate segmentation data.
    
    Args:
        reference_image_path (str): Path to reference image.
        dbn (DBN): Trained DBN model.
        scaler_x (StandardScaler): Scaler for RGB input.
        scaler_y (MinMaxScaler): Scaler for CIELAB L channel.
        scaler_y_ab (MinMaxScaler): Scaler for CIELAB a, b channels.
        default_k (int): Default number of clusters.
    
    Returns:
        tuple: Reference K-means and SOM segmentation data, original image, DPC cluster count.
    """
    logging.info(f"Processing reference image: {reference_image_path}")
    
    # Load and preprocess the image
    preprocessor = Preprocessor()
    preprocessed_image = load_and_preprocess_image(reference_image_path, preprocessor)
    if preprocessed_image is None:
        return None, None, None, None
    
    original_image = preprocessed_image.copy()
    
    # Resize the preprocessed image
    logging.info("Resizing preprocessed image to 256x256")
    resized_image = preprocessor.resize_image(preprocessed_image, size=(256, 256))  # [ADDED]
    
    # Subsample pixels for cluster determination
    logging.info("Subsampling pixels for cluster determination")
    pixels_subsample = resized_image.reshape(-1, 3).astype(np.float32)[np.random.choice(resized_image.shape[0] * resized_image.shape[1], 2048, replace=False)]
    
    # Determine optimal number of clusters
    n_clusters = determine_optimal_clusters(pixels_subsample, default_k)
    
    # Determine the number of clusters using DPC
    logging.info("Determining the number of clusters using DPC")
    dpc_k = optimal_clusters_dpc(pixels_subsample, min_k=2, max_k=10, subsample_threshold=1000, bandwidth=1.0)
    
    # Perform segmentation
    reference_kmeans_opt, reference_som_opt = perform_segmentation(resized_image, n_clusters, dbn, scaler_x, scaler_y, scaler_y_ab)
    
    return reference_kmeans_opt, reference_som_opt, original_image, dpc_k