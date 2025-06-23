import cv2
import numpy as np
import logging
from minisom import MiniSom
from skimage.color import rgb2lab
from preprocessing import preprocess_image, resize_image
from utils import convert_colors_to_cielab, convert_colors_to_cielab_dbn, optimal_clusters

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
        logging.error(f"Error in k_mean_segmentation: {e}")
        return None, None, None

def som_segmentation(image, k):
    try:
        pixels = image.reshape(-1, 3).astype(np.float32) / 255.0
        som = MiniSom(x=1, y=k, input_len=3, sigma=0.5, learning_rate=0.5)
        som.random_weights_init(pixels)
        som.train_random(pixels, 100)

        labels = np.array([som.winner(pixel)[1] for pixel in pixels])
        centers = np.array([som.get_weights()[0, i] for i in range(k)]) * 255
        centers = np.uint8(centers)

        segmented_image = centers[labels].reshape(image.shape)
        avg_colors = [cv2.mean(image, mask=(labels.reshape(image.shape[:2]) == i).astype(np.uint8))[:3] for i in range(k)]
        return segmented_image, avg_colors, labels
    except Exception as e:
        logging.error(f"Error in som_segmentation: {e}")
        return None, None, None

def process_reference_image(reference_image_path, dbn, scaler_x, scaler_y):
    logging.info(f"Processing reference image: {reference_image_path}")
    reference_image = cv2.imread(reference_image_path)
    if reference_image is None:
        logging.error("Failed to load reference image")
        return None

    preprocessed_image = preprocess_image(reference_image, {})
    resized_image = resize_image(preprocessed_image)

    n_clusters = optimal_clusters(resized_image.reshape(-1, 3).astype(np.float32))

    # K-means segmentation for reference
    segmented_image_kmeans, avg_colors_kmeans, labels_kmeans = k_mean_segmentation(resized_image, n_clusters)
    avg_colors_lab_kmeans = convert_colors_to_cielab(avg_colors_kmeans)
    avg_colors_lab_dbn_kmeans = convert_colors_to_cielab_dbn(dbn, scaler_x, scaler_y, avg_colors_kmeans)
    reference_kmeans = {
        'original_image': resized_image,
        'segmented_image': segmented_image_kmeans,
        'avg_colors': avg_colors_kmeans,
        'avg_colors_lab': avg_colors_lab_kmeans,
        'avg_colors_lab_dbn': avg_colors_lab_dbn_kmeans,
        'labels': labels_kmeans
    }

    # SOM segmentation for reference
    segmented_image_som, avg_colors_som, labels_som = som_segmentation(resized_image, n_clusters)
    avg_colors_lab_som = convert_colors_to_cielab(avg_colors_som)
    avg_colors_lab_dbn_som = convert_colors_to_cielab_dbn(dbn, scaler_x, scaler_y, avg_colors_som)
    reference_som = {
        'original_image': resized_image,
        'segmented_image': segmented_image_som,
        'avg_colors': avg_colors_som,
        'avg_colors_lab': avg_colors_lab_som,
        'avg_colors_lab_dbn': avg_colors_lab_dbn_som,
        'labels': labels_som
    }

    return reference_kmeans, reference_som, reference_image
