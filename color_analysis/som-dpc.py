import os
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.cluster import KMeans, DBSCAN
from joblib import Parallel, delayed
from skimage.color import rgb2lab, deltaE_ciede2000
from skimage.io import imread
from skimage.transform import resize
import pandas as pd
import seaborn as sns
from minisom import MiniSom
import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pyswarm import pso
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, pairwise_distances_chunked
import cProfile
import pstats
from PIL import Image

# Force TensorFlow to use the CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, 'config.yaml')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')

def exception_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {str(e)}")
            return None
    return wrapper

@exception_handler
def load_config(config_path=CONFIG_PATH):
    logging.info(f"Loading configuration from {config_path}...")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    logging.info("Configuration loaded successfully.")
    return config

def validate_config(config):
    required_keys = ['reference_image_path', 'test_images', 'distance_threshold', 'kmeans_clusters', 'predefined_k']
    for key in required_keys:
        if key not in config:
            logging.error(f"Missing required config key: {key}")
            return False
    return True

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """
    Apply unsharp masking to enhance image details.
    
    Parameters:
    - image: Input image
    - kernel_size: Size of the Gaussian kernel
    - sigma: Standard deviation of the Gaussian blur
    - amount: Weight of the added sharpness
    - threshold: Minimum difference in intensity to be considered for enhancement
    
    Returns:
    - sharpened: The sharpened image
    """
    # Step 1: Blur the image
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    
    # Step 2: Calculate the mask
    mask = cv2.subtract(image, blurred)
    
    # Step 3: Apply the mask with the specified amount
    sharpened = cv2.addWeighted(image, 1.0 + amount, mask, -amount, 0)
    
    # Optional: Apply a threshold to limit sharpening
    if threshold > 0:
        low_contrast_mask = np.absolute(mask) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    
    return sharpened

@exception_handler
def k_mean_segmentation(image, k):
    logging.info(f"Starting K-means segmentation with {k} clusters")
    pixels = image.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()].reshape(image.shape)
    avg_colors = [cv2.mean(image, mask=(labels.reshape(image.shape[:2]) == i).astype(np.uint8))[:3] for i in range(k)]
    logging.info(f"K-means segmentation completed with {k} clusters")
    return segmented_image, avg_colors, labels

def convert_colors_to_cielab(avg_colors):
    logging.info("Converting RGB colors to CIELAB")
    avg_colors_lab = []
    for color in avg_colors:
        color_rgb = np.uint8([[color]])
        color_lab = rgb2lab(color_rgb / 255.0)[0][0]
        avg_colors_lab.append(tuple(color_lab))
    logging.info("Conversion to CIELAB completed")
    return avg_colors_lab

def normalize_dbn_pso_output(dbn_pso_output, scaler_y, scaler_y_ab):
    L_predicted_scaled = dbn_pso_output[:, 0].reshape(-1, 1)
    ab_predicted_scaled = dbn_pso_output[:, 1:]
    L_predicted = scaler_y.inverse_transform(L_predicted_scaled)
    ab_predicted = scaler_y_ab.inverse_transform(ab_predicted_scaled)
    predicted_cielab = np.hstack((L_predicted, ab_predicted))
    return predicted_cielab

def convert_colors_to_cielab_dbn(dbn, scaler_x, scaler_y, scaler_y_ab, avg_colors):
    logging.info("Converting RGB colors to CIELAB using PSO-DBN")
    avg_colors_lab_dbn = []
    for color in avg_colors:
        color_rgb = np.array(color).reshape(1, -1)
        color_rgb_scaled = scaler_x.transform(color_rgb)
        color_lab_dbn_scaled = dbn.predict(color_rgb_scaled)
        color_lab_dbn = normalize_dbn_pso_output(color_lab_dbn_scaled, scaler_y, scaler_y_ab)[0]
        avg_colors_lab_dbn.append(tuple(color_lab_dbn))
    logging.info("Conversion using PSO-DBN completed")
    return avg_colors_lab_dbn

def bgr_to_rgb(color):
    return color[::-1]

def ciede2000_distance(color1, color2):
    return deltaE_ciede2000(np.array([color1]), np.array([color2]))[0]

def calculate_similarity(segmentation_data, target_colors):
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

def save_preprocessing_steps_plot(original_image, preprocessed_image, title_prefix='', output_dir=OUTPUT_DIR):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'{title_prefix} Original Image')
    axes[0].axis('off')
    axes[1].imshow(cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'{title_prefix} Preprocessed Image')
    axes[1].axis('off')
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f'{title_prefix}_preprocessing_steps.png')
    plt.savefig(plot_path, dpi=300)
    logging.info(f"Preprocessing steps plot saved to {plot_path}")
    plt.close(fig)

def save_segment_results_plot(segmentation_data, similarity_scores, test_image_path, reference_segmentation_data, best_matches, avg_colors_lab_dbn, method, output_dir=OUTPUT_DIR):
    for i, (test_segment_idx, ref_segment_idx) in enumerate(best_matches):
        if ref_segment_idx == -1:
            continue

        test_color_bgr = segmentation_data['avg_colors'][test_segment_idx]
        test_color_rgb = bgr_to_rgb(test_color_bgr)
        test_color_lab = segmentation_data['avg_colors_lab'][test_segment_idx]
        test_color_lab_dbn = avg_colors_lab_dbn[test_segment_idx]
        similarity = similarity_scores[test_segment_idx]

        mask = np.uint8(segmentation_data['labels'] == test_segment_idx).reshape(segmentation_data['original_image'].shape[:2])
        segment = cv2.bitwise_and(segmentation_data['original_image'], segmentation_data['original_image'], mask=mask)

        ref_color_bgr = reference_segmentation_data['avg_colors'][ref_segment_idx]
        ref_color_rgb = bgr_to_rgb(ref_color_bgr)
        ref_color_lab = reference_segmentation_data['avg_colors_lab'][ref_segment_idx]
        ref_color_lab_dbn = reference_segmentation_data['avg_colors_lab_dbn'][ref_segment_idx]

        ref_mask = np.uint8(reference_segmentation_data['labels'] == ref_segment_idx).reshape(reference_segmentation_data['original_image'].shape[:2])
        ref_segment = cv2.bitwise_and(reference_segmentation_data['original_image'], reference_segmentation_data['original_image'], mask=ref_mask)

        delta_e_cielab = ciede2000_distance(test_color_lab, ref_color_lab)
        delta_e_dbn = ciede2000_distance(test_color_lab_dbn, ref_color_lab_dbn)

        logging.info(f"Segment {test_segment_idx + 1} - Test Color (RGB): {test_color_rgb}, (CIELAB): {tuple(map(lambda x: round(x, 3), test_color_lab))}, (PSO-DBN CIELAB): {tuple(map(lambda x: round(x, 3), test_color_lab_dbn))}")
        logging.info(f"Reference Color (RGB): {ref_color_rgb}, (CIELAB): {tuple(map(lambda x: round(x, 3), ref_color_lab))}, (PSO-DBN CIELAB): {tuple(map(lambda x: round(x, 3), ref_color_lab_dbn))}")
        logging.info(f"Delta E CIELAB: {delta_e_cielab:.3f}, Delta E PSO-DBN: {delta_e_dbn:.3f}")

        fig, axes = plt.subplots(1, 5, figsize=(30, 6))

        axes[0].imshow(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f'Segment {test_segment_idx + 1}\nΔE CIELAB: {delta_e_cielab:.3f}, ΔE DBN: {delta_e_dbn:.3f}\n(Method: {method})')
        axes[0].axis('off')

        avg_color_rgb = np.uint8([[test_color_rgb]])
        axes[1].imshow(avg_color_rgb)
        axes[1].set_title(f'Segment {test_segment_idx + 1} Avg Color\n(RGB): {tuple(map(lambda x: round(x, 3), test_color_rgb))}\n(CIELAB): {tuple(map(lambda x: round(x, 3), test_color_lab))}\n(PSO-DBN CIELAB): {tuple(map(lambda x: round(x, 3), test_color_lab_dbn))}')
        axes[1].axis('off')

        axes[2].imshow(cv2.cvtColor(ref_segment, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f'Reference Segment {ref_segment_idx + 1}')
        axes[2].axis('off')

        ref_color_rgb_img = np.uint8([[ref_color_rgb]])
        axes[3].imshow(ref_color_rgb_img)
        axes[3].set_title(f'Reference Avg Color\n(RGB): {tuple(map(lambda x: round(x, 3), ref_color_rgb))}\n(CIELAB): {tuple(map(lambda x: round(x, 3), ref_color_lab))}\n(PSO-DBN CIELAB): {tuple(map(lambda x: round(x, 3), ref_color_lab_dbn))}')
        axes[3].axis('off')

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, f'{os.path.basename(test_image_path)}{method}_segment{test_segment_idx + 1}.png')
        plt.savefig(plot_path, dpi=300)
        logging.info(f"Segment results plot saved to {plot_path}")
        plt.close(fig)

@exception_handler
def som_segmentation(image, k):
    logging.info(f"Starting SOM segmentation with {k} clusters")
    pixels = image.reshape(-1, 3).astype(np.float32) / 255.0
    som = MiniSom(x=1, y=k, input_len=3, sigma=0.1, learning_rate=0.25)
    som.random_weights_init(pixels)
    som.train_random(pixels, 100)
    labels = np.array([som.winner(pixel)[1] for pixel in pixels])
    centers = np.array([som.get_weights()[0, i] for i in range(k)]) * 255
    centers = np.uint8(centers)
    segmented_image = centers[labels].reshape(image.shape)
    avg_colors = [cv2.mean(image, mask=(labels.reshape(image.shape[:2]) == i).astype(np.uint8))[:3] for i in range(k)]
    logging.info(f"SOM segmentation completed with {k} clusters")
    return segmented_image, avg_colors, labels

def dpc_clustering(pixels, num_clusters, subsample_size=1000, chunk_size=1024):
    logging.info("Starting Density Peak Clustering")

    # Subsampling
    if len(pixels) > subsample_size:
        pixels = pixels[np.random.choice(pixels.shape[0], subsample_size, replace=False)]

    distances_gen = pairwise_distances_chunked(pixels, working_memory=chunk_size)
    distances = np.vstack(list(distances_gen))  # Convert the generator to a numpy array
    dc = np.percentile(distances[distances > 0], 2)  # Distance cutoff, ignoring zeros

    rho = np.zeros(len(pixels))
    delta = np.zeros(len(pixels))

    def compute_rho(distances_chunk):
        valid_distances = distances_chunk / dc
        valid_distances[valid_distances == np.inf] = 0  # Handle division by zero
        return np.sum(np.exp(-(valid_distances) ** 2), axis=1)

    for distances_chunk in pairwise_distances_chunked(pixels, working_memory=chunk_size):
        rho_chunk = compute_rho(distances_chunk)
        rho[:len(rho_chunk)] += rho_chunk

    for i, distances_chunk in enumerate(pairwise_distances_chunked(pixels, working_memory=chunk_size)):
        if i == 0:
            distances = distances_chunk
        else:
            distances = np.vstack((distances, distances_chunk))

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

def optimal_clusters(pixels, default_k, min_k=3, max_k=10, n_runs=10):
    unique_colors = np.unique(pixels, axis=0)
    dynamic_max_k = min(max_k, len(unique_colors))

    logging.info(f"Number of unique colors: {len(unique_colors)}. Adjusting max_k to: {dynamic_max_k}")

    calinski_harabasz_scores = []
    davies_bouldin_scores = []
    silhouette_scores = []

    logging.info("Calculating optimal number of clusters using multiple metrics")

    try:
        for n_clusters in range(min_k, dynamic_max_k + 1):
            calinski_harabasz_avg = 0
            davies_bouldin_avg = 0
            silhouette_avg = 0

            for _ in range(n_runs):
                kmeans = KMeans(n_clusters=n_clusters, random_state=np.random.randint(0, 10000))
                labels = kmeans.fit_predict(pixels)
                calinski_harabasz_avg += calinski_harabasz_score(pixels, labels) / n_runs
                davies_bouldin_avg += davies_bouldin_score(pixels, labels) / n_runs
                silhouette_avg += silhouette_score(pixels, labels) / n_runs

            calinski_harabasz_scores.append(calinski_harabasz_avg)
            davies_bouldin_scores.append(davies_bouldin_avg)
            silhouette_scores.append(silhouette_avg)

            logging.info(f"Metrics for {n_clusters} clusters: Calinski-Harabasz={calinski_harabasz_avg}, Davies-Bouldin={davies_bouldin_avg}, Silhouette={silhouette_avg}")

        optimal_k_ch = calinski_harabasz_scores.index(max(calinski_harabasz_scores)) + min_k
        optimal_k_db = davies_bouldin_scores.index(min(davies_bouldin_scores)) + min_k

        # Prioritize Calinski-Harabasz
        if optimal_k_ch == optimal_k_db:
            optimal_k = optimal_k_ch
        else:
            optimal_k = optimal_k_ch  # Prioritize Calinski-Harabasz for now

        logging.info(f"Optimal number of clusters determined by Calinski-Harabasz: {optimal_k_ch}")
        logging.info(f"Optimal number of clusters determined by Davies-Bouldin: {optimal_k_db}")
        logging.info(f"Final optimal number of clusters: {optimal_k}")
        return optimal_k
    except Exception as e:
        logging.error(f"Error in calculating optimal clusters: {str(e)}. Falling back to default k: {default_k}")
        return default_k

@exception_handler
def preprocess(img):
    logging.info("Starting preprocessing")
    
    # Geometric transformation
    logging.info("Applying geometric transformation")
    img = cv2.resize(img, None, fx=1.1, fy=1.1, interpolation=cv2.INTER_LINEAR)
    logging.info("Geometric transformation completed")
    
    # Morphological operations
    logging.info("Applying morphological operations")
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    logging.info("Morphological operations completed")
    
    # Bilateral filtering
    logging.info("Applying bilateral filtering")
    img = cv2.bilateralFilter(img, 9, 75, 75)
    logging.info("Bilateral filtering completed")
    
    # Unsharp masking for sharpening
    logging.info("Applying unsharp masking for sharpening")
    img = unsharp_mask(img, kernel_size=(5, 5), sigma=1.0, amount=1.5, threshold=0)
    logging.info("Unsharp masking sharpening completed")
    
    logging.info("Preprocessing completed")
    return img

def resize_image(img, size=(256, 256)):
    return cv2.resize(img, size)

def downsample_image(image, scale_factor=0.5):
    width = int(image.shape[1] * scale_factor)
    height = int(image.shape[0] * scale_factor)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

def optimal_clusters_dpc(pixels, min_k=3, max_k=10, subsample_size=1000):
    if len(pixels) > subsample_size:
        pixels = pixels[np.random.choice(pixels.shape[0], subsample_size, replace=False)]

    unique_colors = np.unique(pixels, axis=0)
    dynamic_max_k = min(max_k, len(unique_colors))

    logging.info(f"Number of unique colors: {len(unique_colors)}. Adjusting max_k to: {dynamic_max_k}")

    try:
        dpc_labels = dpc_clustering(pixels, dynamic_max_k, subsample_size=subsample_size)
        n_clusters = len(np.unique(dpc_labels))
        logging.info(f"Optimal number of clusters determined by DPC: {n_clusters}")
        return n_clusters
    except Exception as e:
        logging.error(f"Error in calculating optimal clusters using DPC: {str(e)}. Falling back to default k: {min_k}")
        return min_k

def find_best_matches(segmentation_data, reference_segmentation_data):
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

def process_image(image_path, target_colors, distance_threshold, reference_kmeans, reference_som, dbn, scaler_x, scaler_y, scaler_y_ab, predefined_k, eps_values, min_samples_values, output_dir=OUTPUT_DIR):
    logging.info(f"Processing image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Failed to load image: {image_path}")
        return None

    original_image = image.copy()
    preprocessed_image = preprocess(image)
    if preprocessed_image is None or preprocessed_image.size == 0:
        logging.error("Preprocessed image is empty. Skipping image.")
        return None
    image = resize_image(preprocessed_image)

    # Save preprocessed image
    preprocessed_dir = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0], 'preprocessed')
    os.makedirs(preprocessed_dir, exist_ok=True)
    preprocessed_image_path = os.path.join(preprocessed_dir, f'preprocessed_{os.path.basename(image_path)}')
    cv2.imwrite(preprocessed_image_path, preprocessed_image)

    # Downsampling the image for faster processing
    downsampled_image = downsample_image(image, scale_factor=0.5)
    downsampled_pixels = downsampled_image.reshape(-1, 3).astype(np.float32)

    # K-means segmentation with optimal k
    segmented_image_kmeans_opt, avg_colors_kmeans_opt, labels_kmeans_opt = k_mean_segmentation(image, len(reference_kmeans['avg_colors']))
    avg_colors_lab_kmeans_opt = convert_colors_to_cielab(avg_colors_kmeans_opt)
    avg_colors_lab_dbn_kmeans_opt = convert_colors_to_cielab_dbn(dbn, scaler_x, scaler_y, scaler_y_ab, avg_colors_kmeans_opt)
    segmentation_data_kmeans_opt = {
        'original_image': image,
        'segmented_image': segmented_image_kmeans_opt,
        'avg_colors': avg_colors_kmeans_opt,
        'avg_colors_lab': avg_colors_lab_kmeans_opt,
        'avg_colors_lab_dbn': avg_colors_lab_dbn_kmeans_opt,
        'labels': labels_kmeans_opt
    }
    similarity_scores_kmeans_opt = calculate_similarity(segmentation_data_kmeans_opt, target_colors)
    best_matches_kmeans_opt = find_best_matches(segmentation_data_kmeans_opt, reference_kmeans)

    # Save results for K-means optimal
    save_segment_results_plot(segmentation_data_kmeans_opt, similarity_scores_kmeans_opt, image_path, reference_kmeans, best_matches_kmeans_opt, avg_colors_lab_dbn_kmeans_opt, method='K-means-Optimal', output_dir=os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0], 'kmeans', 'optimal'))

    # K-means segmentation with predefined k
    segmented_image_kmeans_predef, avg_colors_kmeans_predef, labels_kmeans_predef = k_mean_segmentation(image, predefined_k)
    avg_colors_lab_kmeans_predef = convert_colors_to_cielab(avg_colors_kmeans_predef)
    avg_colors_lab_dbn_kmeans_predef = convert_colors_to_cielab_dbn(dbn, scaler_x, scaler_y, scaler_y_ab, avg_colors_kmeans_predef)
    segmentation_data_kmeans_predef = {
        'original_image': image,
        'segmented_image': segmented_image_kmeans_predef,
        'avg_colors': avg_colors_kmeans_predef,
        'avg_colors_lab': avg_colors_lab_kmeans_predef,
        'avg_colors_lab_dbn': avg_colors_lab_dbn_kmeans_predef,
        'labels': labels_kmeans_predef
    }
    similarity_scores_kmeans_predef = calculate_similarity(segmentation_data_kmeans_predef, target_colors)
    best_matches_kmeans_predef = find_best_matches(segmentation_data_kmeans_predef, reference_kmeans)

    # Save results for K-means predefined
    save_segment_results_plot(segmentation_data_kmeans_predef, similarity_scores_kmeans_predef, image_path, reference_kmeans, best_matches_kmeans_predef, avg_colors_lab_dbn_kmeans_predef, method='K-means-Predefined', output_dir=os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0], 'kmeans', 'predefined'))

    # DBSCAN segmentation
    labels_dbscan, best_eps, best_min_samples = optimal_clusters_dbscan(downsampled_pixels, eps_values, min_samples_values)
    unique_labels = np.unique(labels_dbscan)
    segmented_image_dbscan = np.zeros_like(downsampled_image)
    for label in unique_labels:
        if label == -1:  # Skip noise
            continue
        mask = (labels_dbscan == label)
        color = np.mean(downsampled_pixels[mask], axis=0).astype(np.uint8)
        segmented_image_dbscan[mask.reshape(downsampled_image.shape[:2])] = color

    avg_colors_dbscan = [np.mean(downsampled_pixels[labels_dbscan == label], axis=0).astype(np.uint8) for label in unique_labels if label != -1]
    avg_colors_lab_dbscan = convert_colors_to_cielab(avg_colors_dbscan)
    avg_colors_lab_dbn_dbscan = convert_colors_to_cielab_dbn(dbn, scaler_x, scaler_y, scaler_y_ab, avg_colors_dbscan)
    segmentation_data_dbscan = {
        'original_image': downsampled_image,
        'segmented_image': segmented_image_dbscan,
        'avg_colors': avg_colors_dbscan,
        'avg_colors_lab': avg_colors_lab_dbscan,
        'avg_colors_lab_dbn': avg_colors_lab_dbn_dbscan,
        'labels': labels_dbscan
    }
    similarity_scores_dbscan = calculate_similarity(segmentation_data_dbscan, target_colors)
    best_matches_dbscan = find_best_matches(segmentation_data_dbscan, reference_kmeans)

    # Save results for DBSCAN
    save_segment_results_plot(segmentation_data_dbscan, similarity_scores_dbscan, image_path, reference_kmeans, best_matches_dbscan, avg_colors_lab_dbn_dbscan, method='DBSCAN', output_dir=os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0], 'dbscan'))

    # SOM segmentation with optimal k
    segmented_image_som_opt, avg_colors_som_opt, labels_som_opt = som_segmentation(image, len(reference_som['avg_colors']))
    avg_colors_lab_som_opt = convert_colors_to_cielab(avg_colors_som_opt)
    avg_colors_lab_dbn_som_opt = convert_colors_to_cielab_dbn(dbn, scaler_x, scaler_y, scaler_y_ab, avg_colors_som_opt)
    segmentation_data_som_opt = {
        'original_image': image,
        'segmented_image': segmented_image_som_opt,
        'avg_colors': avg_colors_som_opt,
        'avg_colors_lab': avg_colors_som_opt,
        'avg_colors_lab_dbn': avg_colors_lab_dbn_som_opt,
        'labels': labels_som_opt
    }
    similarity_scores_som_opt = calculate_similarity(segmentation_data_som_opt, target_colors)
    best_matches_som_opt = find_best_matches(segmentation_data_som_opt, reference_som)

    # Save results for SOM optimal
    save_segment_results_plot(segmentation_data_som_opt, similarity_scores_som_opt, image_path, reference_som, best_matches_som_opt, avg_colors_lab_dbn_som_opt, method='SOM-Optimal', output_dir=os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0], 'som', 'optimal'))

    # SOM segmentation with predefined k
    segmented_image_som_predef, avg_colors_som_predef, labels_som_predef = som_segmentation(image, predefined_k)
    avg_colors_lab_som_predef = convert_colors_to_cielab(avg_colors_som_predef)
    avg_colors_lab_dbn_som_predef = convert_colors_to_cielab_dbn(dbn, scaler_x, scaler_y, scaler_y_ab, avg_colors_som_predef)
    segmentation_data_som_predef = {
        'original_image': image,
        'segmented_image': segmented_image_som_predef,
        'avg_colors': avg_colors_som_predef,
        'avg_colors_lab': avg_colors_som_predef,
        'avg_colors_lab_dbn': avg_colors_lab_dbn_som_predef,
        'labels': labels_som_predef
    }
    similarity_scores_som_predef = calculate_similarity(segmentation_data_som_predef, target_colors)
    best_matches_som_predef = find_best_matches(segmentation_data_som_predef, reference_som)

    # Save results for SOM predefined
    save_segment_results_plot(segmentation_data_som_predef, similarity_scores_som_predef, image_path, reference_som, best_matches_som_predef, avg_colors_lab_dbn_som_predef, method='SOM-Predefined', output_dir=os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0], 'som', 'predefined'))

    return (
        preprocessed_image_path, 
        segmentation_data_kmeans_opt, similarity_scores_kmeans_opt, best_matches_kmeans_opt,
        segmentation_data_kmeans_predef, similarity_scores_kmeans_predef, best_matches_kmeans_predef,
        segmentation_data_dbscan, similarity_scores_dbscan, best_matches_dbscan,
        segmentation_data_som_opt, similarity_scores_som_opt, best_matches_som_opt,
        segmentation_data_som_predef, similarity_scores_som_predef, best_matches_som_predef
    )

def load_data(image_paths: list, target_size: tuple = (256, 256)) -> tuple:
    logging.info("Loading image data")
    rgb_data = []
    lab_data = []

    for image_path in image_paths:
        image = imread(image_path)
        image = resize(image, target_size, anti_aliasing=True)
        rgb_data.append(image.reshape(-1, 3))
        lab_data.append(rgb2lab(image).reshape(-1, 3))

    rgb_data = np.vstack(rgb_data)
    lab_data = np.vstack(lab_data)

    logging.info("Image data loading completed")
    return rgb_data, lab_data

class DBN:
    def __init__(self, input_size, hidden_layers, output_size):
        logging.info("Initializing DBN model")
        self.model = Sequential()
        self.model.add(Dense(hidden_layers[0], input_dim=input_size, activation='relu'))
        for layer_size in hidden_layers[1:]:
            self.model.add(Dense(layer_size, activation='relu'))
        self.model.add(Dense(output_size, activation='linear'))
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        logging.info("DBN model initialized")

    def train(self, x_train, y_train, epochs=50, batch_size=32):
        logging.info(f"Training DBN model for {epochs} epochs with batch size {batch_size}")
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        logging.info("DBN model training completed")

    def predict(self, x_test):
        return self.model.predict(x_test)

def pso_optimize(dbn, x_train, y_train, bounds):
    def objective(weights):
        reshaped_weights = []
        start = 0
        for w in dbn.model.get_weights():
            shape = w.shape
            size = np.prod(shape)
            reshaped_weights.append(weights[start:start + size].reshape(shape))
            start += size
        dbn.model.set_weights(reshaped_weights)
        predictions = dbn.model.predict(x_train)
        return np.mean((predictions - y_train) ** 2)
    
    initial_weights = dbn.model.get_weights()
    flat_weights = np.hstack([w.flatten() for w in initial_weights])
    
    flat_bounds = []
    epsilon = 1e-5
    for w in initial_weights:
        min_val = w.min()
        max_val = w.max()
        if min_val == max_val:
            min_val -= epsilon
            max_val += epsilon
        flat_bounds.extend([(min_val, max_val)] * w.size)
    
    lb = [b[0] for b in flat_bounds]
    ub = [b[1] for b in flat_bounds]

    swarmsize = 5
    maxiter = 5
    
    logging.info(f"Starting PSO optimization with swarmsize={swarmsize} and maxiter={maxiter}")
    optimized_weights, _ = pso(objective, lb=lb, ub=ub, swarmsize=swarmsize, maxiter=maxiter)
    logging.info("PSO optimization completed")
    
    start = 0
    new_weights = []
    for w in initial_weights:
        shape = w.shape
        size = np.prod(shape)
        new_weights.append(optimized_weights[start:start + size].reshape(shape))
        start += size
    
    return new_weights

def process_reference_image(reference_image_path, dbn, scaler_x, scaler_y, scaler_y_ab, default_k):
    logging.info(f"Processing reference image: {reference_image_path}")
    reference_image = cv2.imread(reference_image_path)
    if reference_image is None:
        logging.error("Failed to load reference image")
        return None

    original_image = reference_image.copy()
    preprocessed_image = preprocess(reference_image)
    if preprocessed_image is None or preprocessed_image.size == 0:
        logging.error("Preprocessed image is empty. Skipping image.")
        return None
    resized_image = resize_image(preprocessed_image)

    logging.info("Determining optimal number of clusters")
    pixels_subsample = resized_image.reshape(-1, 3).astype(np.float32)[np.random.choice(resized_image.shape[0] * resized_image.shape[1], 2048, replace=False)]
    unique_colors = np.unique(pixels_subsample, axis=0)
    logging.info(f"Number of unique colors after quantization: {len(unique_colors)}")
    n_clusters = optimal_clusters(pixels_subsample, default_k, min_k=3, max_k=10)
    logging.info(f"Optimal number of clusters determined: {n_clusters}")

    logging.info("Determining the number of clusters using DPC")
    dpc_k = optimal_clusters_dpc(pixels_subsample)
    
    # K-means segmentation for reference with optimal k
    segmented_image_kmeans_opt, avg_colors_kmeans_opt, labels_kmeans_opt = k_mean_segmentation(resized_image, n_clusters)
    avg_colors_lab_kmeans_opt = convert_colors_to_cielab(avg_colors_kmeans_opt)
    avg_colors_lab_dbn_kmeans_opt = convert_colors_to_cielab_dbn(dbn, scaler_x, scaler_y, scaler_y_ab, avg_colors_kmeans_opt)
    reference_kmeans_opt = {
        'original_image': resized_image,
        'segmented_image': segmented_image_kmeans_opt,
        'avg_colors': avg_colors_kmeans_opt,
        'avg_colors_lab': avg_colors_lab_kmeans_opt,
        'avg_colors_lab_dbn': avg_colors_lab_dbn_kmeans_opt,
        'labels': labels_kmeans_opt
    }

    # SOM segmentation for reference with optimal k
    segmented_image_som_opt, avg_colors_som_opt, labels_som_opt = som_segmentation(resized_image, n_clusters)
    avg_colors_lab_som_opt = convert_colors_to_cielab(avg_colors_som_opt)
    avg_colors_lab_dbn_som_opt = convert_colors_to_cielab_dbn(dbn, scaler_x, scaler_y, scaler_y_ab, avg_colors_som_opt)
    reference_som_opt = {
        'original_image': resized_image,
        'segmented_image': segmented_image_som_opt,
        'avg_colors': avg_colors_som_opt,
        'avg_colors_lab': avg_colors_som_opt,
        'avg_colors_lab_dbn': avg_colors_lab_dbn_som_opt,
        'labels': labels_som_opt
    }

    return reference_kmeans_opt, reference_som_opt, original_image, dpc_k

def save_results(segmentation_data, similarity_scores, method, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    cv2.imwrite(os.path.join(output_dir, f'segmented_image_{method}.png'), segmentation_data['segmented_image'])
    with open(os.path.join(output_dir, f'avg_colors_{method}.yaml'), 'w') as f:
        yaml.dump(segmentation_data['avg_colors'], f)
    with open(os.path.join(output_dir, f'similarity_scores_{method}.yaml'), 'w') as f:
        yaml.dump(similarity_scores, f)
    logging.info(f"Results saved for {method} method")

def save_delta_e_results(overall_delta_e, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'overall_delta_e.yaml'), 'w') as f:
        yaml.dump(overall_delta_e, f)
    logging.info("Overall Delta E results saved")

def main(config_path='config.yaml'):
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        logging.info(f"Current working directory: {os.getcwd()}")
        config = load_config(config_path)
        if config is None or not validate_config(config):
            logging.error("Invalid configuration. Exiting.")
            return

        reference_image_path = config['reference_image_path']
        test_images = config['test_images']
        distance_threshold = config['distance_threshold']
        k = config['kmeans_clusters']
        predefined_k = config['predefined_k']

        rgb_data, lab_data = load_data(test_images)
        x_train, x_test, y_train, y_test = train_test_split(rgb_data, lab_data, test_size=0.2, random_state=42)

        input_size = x_train.shape[1]
        output_size = y_train.shape[1]
        hidden_layers = [128, 64, 32]

        dbn = DBN(input_size, hidden_layers, output_size)

        scaler_x = StandardScaler().fit(x_train)
        scaler_y = MinMaxScaler(feature_range=(0, 100)).fit(y_train[:, 0].reshape(-1, 1))
        scaler_y_ab = MinMaxScaler(feature_range=(-128, 127)).fit(y_train[:, 1:])

        x_train_scaled = scaler_x.transform(x_train)
        y_train_scaled = np.hstack((scaler_y.transform(y_train[:, 0].reshape(-1, 1)), scaler_y_ab.transform(y_train[:, 1:])))

        sample_input = np.zeros((1, input_size))
        dbn.model(sample_input)
        initial_weights = dbn.model.get_weights()

        bounds = [(w.min(), w.max()) for w in initial_weights]

        optimized_weights = pso_optimize(dbn, x_train_scaled, y_train_scaled, bounds)
        dbn.model.set_weights(optimized_weights)

        reference_kmeans_opt, reference_som_opt, original_image, dpc_k = process_reference_image(reference_image_path, dbn, scaler_x, scaler_y, scaler_y_ab, k)
        if reference_kmeans_opt is None or reference_som_opt is None:
            logging.error("Failed to process reference image. Exiting.")
            return

        target_colors = reference_kmeans_opt['avg_colors_lab']

        save_reference_summary_plot(reference_kmeans_opt, reference_som_opt, original_image)

        preprocessed_image_paths = []
        results = []
        for image_path in test_images:
            result = process_image(image_path, target_colors, distance_threshold, reference_kmeans_opt, reference_som_opt, dbn, scaler_x, scaler_y, scaler_y_ab, predefined_k, eps_values=[10, 15, 20], min_samples_values=[5, 10, 20])
            if result:
                preprocessed_image_path, \
                segmentation_data_kmeans_opt, similarity_scores_kmeans_opt, best_matches_kmeans_opt, \
                segmentation_data_kmeans_predef, similarity_scores_kmeans_predef, best_matches_kmeans_predef, \
                segmentation_data_dbscan, similarity_scores_dbscan, best_matches_dbscan, \
                segmentation_data_som_opt, similarity_scores_som_opt, best_matches_som_opt, \
                segmentation_data_som_predef, similarity_scores_som_predef, best_matches_som_predef = result
                preprocessed_image_paths.append(preprocessed_image_path)
                results.append(result)

        overall_delta_e = {}

        for result in results:
            if result:
                preprocessed_image_path, \
                segmentation_data_kmeans_opt, similarity_scores_kmeans_opt, best_matches_kmeans_opt, \
                segmentation_data_kmeans_predef, similarity_scores_kmeans_predef, best_matches_kmeans_predef, \
                segmentation_data_dbscan, similarity_scores_dbscan, best_matches_dbscan, \
                segmentation_data_som_opt, similarity_scores_som_opt, best_matches_som_opt, \
                segmentation_data_som_predef, similarity_scores_som_predef, best_matches_som_predef = result
                
                image_name = os.path.splitext(os.path.basename(preprocessed_image_path))[0]

                save_results(segmentation_data_kmeans_opt, similarity_scores_kmeans_opt, 'kmeans_optimal', os.path.join(OUTPUT_DIR, image_name, 'kmeans', 'optimal'))
                save_results(segmentation_data_kmeans_predef, similarity_scores_kmeans_predef, 'kmeans_predefined', os.path.join(OUTPUT_DIR, image_name, 'kmeans', 'predefined'))
                save_results(segmentation_data_dbscan, similarity_scores_dbscan, 'dbscan', os.path.join(OUTPUT_DIR, image_name, 'dbscan'))
                save_results(segmentation_data_som_opt, similarity_scores_som_opt, 'som_optimal', os.path.join(OUTPUT_DIR, image_name, 'som', 'optimal'))
                save_results(segmentation_data_som_predef, similarity_scores_som_predef, 'som_predefined', os.path.join(OUTPUT_DIR, image_name, 'som', 'predefined'))

                overall_delta_e[image_name] = {
                    'kmeans_optimal': np.mean([ciede2000_distance(segmentation_data_kmeans_opt['avg_colors_lab'][i], reference_kmeans_opt['avg_colors_lab'][best_matches_kmeans_opt[i][1]]) for i in range(len(segmentation_data_kmeans_opt['avg_colors_lab'])) if best_matches_kmeans_opt[i][1] != -1]),
                    'kmeans_predefined': np.mean([ciede2000_distance(segmentation_data_kmeans_predef['avg_colors_lab'][i], reference_kmeans_opt['avg_colors_lab'][best_matches_kmeans_predef[i][1]]) for i in range(len(segmentation_data_kmeans_predef['avg_colors_lab'])) if best_matches_kmeans_predef[i][1] != -1]),
                    'dbscan': np.mean([ciede2000_distance(segmentation_data_dbscan['avg_colors_lab'][i], reference_kmeans_opt['avg_colors_lab'][best_matches_dbscan[i][1]]) for i in range(len(segmentation_data_dbscan['avg_colors_lab'])) if best_matches_dbscan[i][1] != -1]),
                    'som_optimal': np.mean([ciede2000_distance(segmentation_data_som_opt['avg_colors_lab'][i], reference_kmeans_opt['avg_colors_lab'][best_matches_som_opt[i][1]]) for i in range(len(segmentation_data_som_opt['avg_colors_lab'])) if best_matches_som_opt[i][1] != -1]),
                    'som_predefined': np.mean([ciede2000_distance(segmentation_data_som_predef['avg_colors_lab'][i], reference_kmeans_opt['avg_colors_lab'][best_matches_som_predef[i][1]]) for i in range(len(segmentation_data_som_predef['avg_colors_lab'])) if best_matches_som_predef[i][1] != -1]),
                }

        logging.info("Overall Delta E results:")
        for image_name, delta_e in overall_delta_e.items():
            logging.info(f"Image: {image_name}")
            for method, value in delta_e.items():
                logging.info(f"{method}: {value:.3f}")

        save_delta_e_results(overall_delta_e, OUTPUT_DIR)

    finally:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats(10)

if __name__ == "__main__":
    main(config_path='C:/Users/LENOVO/Desktop/prints/segmentation/block_config.yaml')
