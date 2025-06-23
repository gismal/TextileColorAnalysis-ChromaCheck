import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
from skimage.color import rgb2lab, deltaE_ciede2000
import pandas as pd
import logging
from minisom import MiniSom

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path='config.yaml'):
    try:
        logging.info(f"Loading configuration from {config_path}...")
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logging.info("Configuration loaded successfully.")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file {config_path} not found.")
        return None
    except yaml.YAMLError as exc:
        logging.error(f"Error parsing YAML file: {exc}")
        return None

def preprocess(img):
    try:
        img = cv2.GaussianBlur(img, (7, 7), 0)
        img = cv2.bilateralFilter(img, 9, 75, 75)
        return img
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        return None

def resize_image(img, size=(256, 256)):
    return cv2.resize(img, size)

def k_mean_segmentation(image, k):
    try:
        pixels = image.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        segmented_image = centers[labels.flatten()].reshape(image.shape)

        avg_colors = [cv2.mean(image, mask=(labels.reshape(image.shape[:2]) == i).astype(np.uint8))[:3] for i in range(k)]
        return segmented_image, avg_colors, labels
    except Exception as e:
        logging.error(f"Error in K-means segmentation: {str(e)}")
        return None, None, None

def som_segmentation(image, k):
    try:
        logging.info("Starting SOM segmentation")
        pixels = image.reshape(-1, 3).astype(np.float32) / 255.0  # Normalize pixels to [0, 1]
        som = MiniSom(x=1, y=k, input_len=3, sigma=0.5, learning_rate=0.5)  # Adjusted sigma to 0.5
        som.random_weights_init(pixels)
        som.train_random(pixels, 100)

        labels = np.array([som.winner(pixel)[1] for pixel in pixels])
        centers = np.array([som.get_weights()[0, i] for i in range(k)]) * 255
        centers = np.uint8(centers)

        segmented_image = centers[labels].reshape(image.shape)
        avg_colors = [cv2.mean(image, mask=(labels.reshape(image.shape[:2]) == i).astype(np.uint8))[:3] for i in range(k)]
        logging.info("SOM segmentation completed")
        return segmented_image, avg_colors, labels
    except Exception as e:
        logging.error(f"Error in SOM segmentation: {str(e)}")
        return None, None, None

def convert_colors_to_cielab(avg_colors):
    avg_colors_lab = []
    for color in avg_colors:
        color_rgb = np.uint8([[color]])
        color_lab = rgb2lab(color_rgb)[0][0]
        avg_colors_lab.append(tuple(color_lab))
    return avg_colors_lab

def plot_preprocessing_steps(original_image, preprocessed_image, title_prefix=''):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'{title_prefix} Original Image')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'{title_prefix} Preprocessed Image')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

def plot_segment_results(segmentation_data, similarity_scores, test_image_path, reference_segmentation_data, best_matches, method):
    for i, (test_segment_idx, ref_segment_idx) in enumerate(best_matches):
        if ref_segment_idx == -1:
            continue  # Skip segments without valid matches

        test_color = segmentation_data['avg_colors'][test_segment_idx]
        test_color_lab = segmentation_data['avg_colors_lab'][test_segment_idx]
        similarity = similarity_scores[test_segment_idx]

        mask = np.uint8(segmentation_data['labels'] == test_segment_idx).reshape(segmentation_data['original_image'].shape[:2])
        segment = cv2.bitwise_and(segmentation_data['original_image'], segmentation_data['original_image'], mask=mask)

        ref_color = reference_segmentation_data['avg_colors'][ref_segment_idx]
        ref_color_lab = reference_segmentation_data['avg_colors_lab'][ref_segment_idx]

        ref_mask = np.uint8(reference_segmentation_data['labels'] == ref_segment_idx).reshape(reference_segmentation_data['original_image'].shape[:2])
        ref_segment = cv2.bitwise_and(reference_segmentation_data['original_image'], reference_segmentation_data['original_image'], mask=ref_mask)

        # Calculate Delta E value
        delta_e = ciede2000_distance(test_color_lab, ref_color_lab)

        logging.info(f"Segment {test_segment_idx + 1} - Test Color (RGB): {test_color}, (CIELAB): {test_color_lab}")
        logging.info(f"Reference Color (RGB): {ref_color}, (CIELAB): {ref_color_lab}")
        logging.info(f"Delta E: {delta_e:.2f}")

        fig, axes = plt.subplots(1, 4, figsize=(24, 6))

        axes[0].imshow(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f'Test Image ({method}): {os.path.basename(test_image_path)}\nSegment {test_segment_idx + 1}, ΔE: {delta_e:.2f}')
        axes[0].axis('off')

        avg_color_rgb = np.uint8([[test_color]])
        axes[1].imshow(cv2.cvtColor(avg_color_rgb, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f'Segment {test_segment_idx + 1} Avg Color\n(RGB): {tuple(map(int, test_color))}\n(CIELAB): {tuple(map(int, test_color_lab))}')
        axes[1].axis('off')

        axes[2].imshow(cv2.cvtColor(ref_segment, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f'Reference Segment {ref_segment_idx + 1}')
        axes[2].axis('off')

        ref_color_rgb = np.uint8([[ref_color]])
        axes[3].imshow(cv2.cvtColor(ref_color_rgb, cv2.COLOR_BGR2RGB))
        axes[3].set_title(f'Reference Avg Color\n(RGB): {tuple(map(int, ref_color))}\n(CIELAB): {tuple(map(int, ref_color_lab))}')
        axes[3].axis('off')

        plt.tight_layout()
        plt.show()

def ciede2000_distance(color1, color2):
    return deltaE_ciede2000(np.array([color1]), np.array([color2]))[0]

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

def process_image(image_path, target_colors, distance_threshold, k, reference_image, reference_segmentation_data):
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Failed to load image: {image_path}")
        return None

    original_image = image.copy()
    preprocessed_image = preprocess(image)
    image = resize_image(preprocessed_image)

    plot_preprocessing_steps(original_image, preprocessed_image, title_prefix=f'{os.path.basename(image_path)}')

    # K-means segmentation
    kmeans_segmented_image, kmeans_avg_colors, kmeans_labels = k_mean_segmentation(image, k)
    kmeans_avg_colors_lab = convert_colors_to_cielab(kmeans_avg_colors)
    kmeans_segmentation_data = {
        'original_image': image,
        'segmented_image': kmeans_segmented_image,
        'avg_colors': kmeans_avg_colors,
        'avg_colors_lab': kmeans_avg_colors_lab,
        'labels': kmeans_labels
    }
    kmeans_similarity_scores = calculate_similarity(kmeans_segmentation_data, target_colors)
    kmeans_best_matches = find_best_matches(kmeans_segmentation_data, reference_segmentation_data)

    # SOM segmentation
    som_segmented_image, som_avg_colors, som_labels = som_segmentation(image, k)
    som_avg_colors_lab = convert_colors_to_cielab(som_avg_colors)
    som_segmentation_data = {
        'original_image': image,
        'segmented_image': som_segmented_image,
        'avg_colors': som_avg_colors,
        'avg_colors_lab': som_avg_colors_lab,
        'labels': som_labels
    }
    som_similarity_scores = calculate_similarity(som_segmentation_data, target_colors)
    som_best_matches = find_best_matches(som_segmentation_data, reference_segmentation_data)

    for idx, color in enumerate(kmeans_avg_colors):
        if np.all(np.array(color) <= np.array([5, 5, 5])):
            logging.warning(f"Warning: Segment {idx} in {image_path} is nearly black (K-means).")
            logging.warning(f"Segment {idx} average color (RGB): {color}, (CIELAB): {kmeans_avg_colors_lab[idx]}")

    for idx, color in enumerate(som_avg_colors):
        if np.all(np.array(color) <= np.array([5, 5, 5])):
            logging.warning(f"Warning: Segment {idx} in {image_path} is nearly black (SOM).")
            logging.warning(f"Segment {idx} average color (RGB): {color}, (CIELAB): {som_avg_colors_lab[idx]}")

    return (kmeans_segmentation_data, kmeans_similarity_scores, kmeans_best_matches, image_path), \
           (som_segmentation_data, som_similarity_scores, som_best_matches, image_path)

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

def save_results(segmentation_data, method, output_dir='output'):
    os.makedirs(output_dir, exist_ok=True)

    cv2.imwrite(os.path.join(output_dir, f'segmented_image_{method}.png'), segmentation_data['segmented_image'])

    avg_colors_df = pd.DataFrame(segmentation_data['avg_colors'], columns=['B', 'G', 'R'])
    avg_colors_df.to_csv(os.path.join(output_dir, f'average_colors_{method}.csv'), index=False)

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

    for kmeans_result, som_result in results:
        if kmeans_result:
            kmeans_segmentation_data, kmeans_similarity_scores, kmeans_best_matches, image_path = kmeans_result
            plot_segment_results(kmeans_segmentation_data, kmeans_similarity_scores, image_path, reference_segmentation_data, kmeans_best_matches, method='K-means')
            save_results(kmeans_segmentation_data, method='kmeans')
        if som_result:
            som_segmentation_data, som_similarity_scores, som_best_matches, image_path = som_result
            plot_segment_results(som_segmentation_data, som_similarity_scores, image_path, reference_segmentation_data, som_best_matches, method='SOM')
            save_results(som_segmentation_data, method='som')

if __name__ == "__main__":
    main(config_path='C:/Users/LENOVO/Desktop/prints/segmentation/stripes_config.yaml')
