import os
import logging
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from preprocessing import preprocess_image, resize_image
from segmentation import k_mean_segmentation, som_segmentation, process_reference_image
from model import DBN, load_data, pso_optimize, compare_predictions_to_ground_truth
from utils import load_config, validate_config, save_preprocessing_steps_plot, save_segment_results_plot, save_reference_summary_plot, save_results, convert_colors_to_cielab, optimal_clusters, ciede2000_distance, convert_colors_to_cielab_dbn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def calculate_similarity(segmentation_data, target_colors):
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
    return similarity_scores

def find_best_matches(segmentation_data, reference_segmentation_data):
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

    return best_matches

def process_image(image_path, target_colors, distance_threshold, reference_kmeans, reference_som, dbn, scaler_x, scaler_y, preprocessing_config, output_dir):
    logging.info(f"Processing image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Failed to load image: {image_path}")
        return None

    preprocessed_image = preprocess_image(image, preprocessing_config)
    resized_image = resize_image(preprocessed_image)

    save_preprocessing_steps_plot(image, preprocessed_image, output_dir, title_prefix=os.path.basename(image_path))

    # K-means segmentation
    segmented_image_kmeans, avg_colors_kmeans, labels_kmeans = k_mean_segmentation(resized_image, len(reference_kmeans['avg_colors']))
    avg_colors_lab_kmeans = convert_colors_to_cielab(avg_colors_kmeans)
    avg_colors_lab_dbn_kmeans = convert_colors_to_cielab_dbn(dbn, scaler_x, scaler_y, avg_colors_kmeans)
    segmentation_data_kmeans = {
        'original_image': resized_image,
        'segmented_image': segmented_image_kmeans,
        'avg_colors': avg_colors_kmeans,
        'avg_colors_lab': avg_colors_lab_kmeans,
        'avg_colors_lab_dbn': avg_colors_lab_dbn_kmeans,
        'labels': labels_kmeans
    }
    similarity_scores_kmeans = calculate_similarity(segmentation_data_kmeans, target_colors)
    best_matches_kmeans = find_best_matches(segmentation_data_kmeans, reference_kmeans)

    save_segment_results_plot(segmentation_data_kmeans, similarity_scores_kmeans, image_path, reference_kmeans, best_matches_kmeans, avg_colors_lab_dbn_kmeans, 'kmeans', output_dir)

    # SOM segmentation
    segmented_image_som, avg_colors_som, labels_som = som_segmentation(resized_image, len(reference_som['avg_colors']))
    avg_colors_lab_som = convert_colors_to_cielab(avg_colors_som)
    avg_colors_lab_dbn_som = convert_colors_to_cielab_dbn(dbn, scaler_x, scaler_y, avg_colors_som)
    segmentation_data_som = {
        'original_image': resized_image,
        'segmented_image': segmented_image_som,
        'avg_colors': avg_colors_som,
        'avg_colors_lab': avg_colors_lab_som,
        'avg_colors_lab_dbn': avg_colors_lab_dbn_som,
        'labels': labels_som
    }
    similarity_scores_som = calculate_similarity(segmentation_data_som, target_colors)
    best_matches_som = find_best_matches(segmentation_data_som, reference_som)

    save_segment_results_plot(segmentation_data_som, similarity_scores_som, image_path, reference_som, best_matches_som, avg_colors_lab_dbn_som, 'som', output_dir)

    return resized_image, segmentation_data_kmeans, similarity_scores_kmeans, best_matches_kmeans, segmentation_data_som, similarity_scores_som, best_matches_som

def main(config_path='config.yaml'):
    config = load_config(config_path)
    if not validate_config(config):
        logging.error("Invalid configuration. Exiting.")
        return

    reference_image_path = config['reference_image_path']
    test_images = config['test_images']
    distance_threshold = config['distance_threshold']
    k = config['kmeans_clusters']
    preprocessing_config = config['preprocessing']
    output_dir = config['output_dir']

    # Load data for DBN training using raw test images
    rgb_data, lab_data = load_data(test_images)
    x_train, x_test, y_train, y_test = train_test_split(rgb_data, lab_data, test_size=0.2, random_state=42)

    input_size = x_train.shape[1]
    output_size = y_train.shape[1]
    hidden_layers = [128, 64, 32]  # Example architecture

    dbn = DBN(input_size, hidden_layers, output_size)

    # Normalize data
    scaler_x = StandardScaler().fit(x_train)
    scaler_y = StandardScaler().fit(y_train)

    x_train_scaled = scaler_x.transform(x_train)
    y_train_scaled = scaler_y.transform(y_train)

    # Call the model with a sample input to initialize the weights
    sample_input = np.zeros((1, input_size))
    dbn.model(sample_input)
    initial_weights = dbn.model.get_weights()

    bounds = [(w.min(), w.max()) for w in initial_weights]

    optimized_weights = pso_optimize(dbn, x_train_scaled, y_train_scaled, bounds)
    dbn.model.set_weights(optimized_weights)

    # Process the reference image
    reference_kmeans, reference_som, original_image = process_reference_image(reference_image_path, dbn, scaler_x, scaler_y)
    if reference_kmeans is None or reference_som is None:
        logging.error("Failed to process reference image. Exiting.")
        return

    target_colors = reference_kmeans['avg_colors_lab']

    save_reference_summary_plot(reference_kmeans, reference_som, original_image, output_dir)

    # Preprocess and segment test images
    results = []
    for image_path in test_images:
        result = process_image(image_path, target_colors, distance_threshold, reference_kmeans, reference_som, dbn, scaler_x, scaler_y, preprocessing_config, output_dir)
        if result:
            results.append(result)

    for result in results:
        if result:
            preprocessed_image_path, segmentation_data_kmeans, similarity_scores_kmeans, best_matches_kmeans, segmentation_data_som, similarity_scores_som, best_matches_som = result
            save_results(segmentation_data_kmeans, similarity_scores_kmeans, 'kmeans', os.path.join(output_dir, "kmeans"))
            save_results(segmentation_data_som, similarity_scores_som, 'som', os.path.join(output_dir, "som"))

if __name__ == "__main__":
    main()
