import logging
import os
import numpy as np
import sys
import cv2
import pandas as pd
from joblib import Parallel, delayed

# Add the parent directory to sys.path to recognize the 'segmentation' package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from segmentation.config_loader import load_config
from segmentation.image_processing import preprocess, resize_image, align_images, k_mean_segmentation
from segmentation.color_analysis import convert_colors_to_cielab, calculate_similarity, find_best_matches
from segmentation.plotting import plot_preprocessing_steps, plot_segment_results
from skimage.color import rgb2lab

def process_image(image_path, target_colors, distance_threshold, k, reference_image, reference_segmentation_data):
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Failed to load image: {image_path}")
        return None

    original_image = image.copy()
    preprocessed_image = preprocess(image)
    aligned_image = align_images(reference_image, preprocessed_image)
    image = resize_image(aligned_image)

    plot_preprocessing_steps(original_image, preprocessed_image, aligned_image, title_prefix=f'{os.path.basename(image_path)}')

    segmented_image, avg_colors, labels = k_mean_segmentation(image, k)
    avg_colors_lab = convert_colors_to_cielab(avg_colors)
    segmentation_data = {
        'original_image': image,
        'segmented_image': segmented_image,
        'avg_colors': avg_colors,
        'avg_colors_lab': avg_colors_lab,
        'labels': labels
    }
    similarity_scores = calculate_similarity(segmentation_data, target_colors)
    best_matches = find_best_matches(segmentation_data, reference_segmentation_data)

    for idx, color in enumerate(avg_colors):
        if np.all(np.array(color) <= np.array([5, 5, 5])):
            logging.warning(f"Warning: Segment {idx} in {image_path} is nearly black.")
            logging.warning(f"Segment {idx} average color (RGB): {color}, (CIELAB): {avg_colors_lab[idx]}")

    return segmentation_data, similarity_scores, best_matches, image_path

def process_reference_image(reference_image_path, k):
    reference_image = cv2.imread(reference_image_path)
    if reference_image is None:
        logging.error("Failed to load reference image")
        return None

    original_image = reference_image.copy()
    preprocessed_image = preprocess(reference_image)
    resized_image = resize_image(preprocessed_image)

    segmented_image, avg_colors, labels = k_mean_segmentation(resized_image, k)
    avg_colors_lab = convert_colors_to_cielab(avg_colors)
    reference_segmentation_data = {
        'original_image': resized_image,
        'segmented_image': segmented_image,
        'avg_colors': avg_colors,
        'avg_colors_lab': avg_colors_lab,
        'labels': labels
    }
    return resized_image, reference_segmentation_data

def save_results(segmentation_data, unsimilar_data, similar_data, similarity_scores, output_dir='output'):
    os.makedirs(output_dir, exist_ok=True)

    cv2.imwrite(os.path.join(output_dir, 'segmented_image.png'), segmentation_data['segmented_image'])

    avg_colors_df = pd.DataFrame(segmentation_data['avg_colors'], columns=['B', 'G', 'R'])
    avg_colors_df.to_csv(os.path.join(output_dir, 'average_colors.csv'), index=False)

    unsimilar_df = pd.DataFrame(unsimilar_data)
    similar_df = pd.DataFrame(similar_data)
    unsimilar_df.to_csv(os.path.join(output_dir, 'unsimilar_data.csv'), index=False)
    similar_df.to_csv(os.path.join(output_dir, 'similar_data.csv'), index=False)

    similarity_scores_df = pd.DataFrame(similarity_scores, columns=['Similarity_Score'])
    similarity_scores_df.to_csv(os.path.join(output_dir, 'similarity_scores.csv'), index=False)

def main(config_path='config.yaml'):
    logging.info(f"Current working directory: {os.getcwd()}")
    config = load_config(config_path)
    if config is None:
        logging.error("Failed to load configuration. Exiting.")
        return

    reference_image_path = config['reference_image_path']
    test_images = config['test_images']
    distance_threshold = config['distance_threshold']
    k = config['kmeans_clusters']

    reference_image, reference_segmentation_data = process_reference_image(reference_image_path, k)
    if reference_segmentation_data is None:
        logging.error("Failed to process reference image. Exiting.")
        return

    target_colors = reference_segmentation_data['avg_colors_lab']

    results = Parallel(n_jobs=-1)(delayed(process_image)(image_path, target_colors, distance_threshold, k, reference_image, reference_segmentation_data) for image_path in test_images)

    for result in results:
        if result:
            segmentation_data, similarity_scores, best_matches, image_path = result
            plot_segment_results(segmentation_data, similarity_scores, image_path, reference_segmentation_data, best_matches)
            save_results(segmentation_data, [], [], similarity_scores)

if __name__ == "__main__":
    main(config_path='config.yaml')
