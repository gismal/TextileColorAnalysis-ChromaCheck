import os
import cv2
import matplotlib

from src.data import preprocess
from src.data import load_data
from src.data.load_data import load_config, validate_config
from src.models.pso_dbn import DBN, convert_colors_to_cielab_dbn, pso_optimize
from src.models.segmentation import k_mean_segmentation, optimal_clusters, optimal_clusters_dbscan, som_segmentation
from src.utils.color_conversion import convert_colors_to_cielab
from src.utils.visualization import save_reference_summary_plot, save_segment_results_plot
from src.utils.processing import ciede2000_distance, calculate_similarity, find_best_matches, downsample_image, \
    gaussian_kernel, dpc_clustering, optimal_clusters_dpc, create_output_folders, save_delta_e_results, \
    save_results, compare_cielab_colors, process_reference_image

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
    image = preprocess.resize_image(preprocessed_image)

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
    main(config_path='C:/Users/LENOVO/Desktop/prints/configurations/block_config.yaml')