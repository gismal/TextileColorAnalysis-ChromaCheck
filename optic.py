import tensorflow as tf

from superpoint import SuperPoint
from lightglue import LightGlue

import cv2
import matplotlib
matplotlib.use('Agg')  # Use the non-interactive 'Agg' backend
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
from skimage.color import rgb2lab, deltaE_ciede2000
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Decorator for exception handling
def exception_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {str(e)}")
            return None
    return wrapper

# Load configuration from YAML file
@exception_handler
def load_config(config_path='config.yaml'):
    logging.info(f"Loading configuration from {config_path}...")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    logging.info("Configuration loaded successfully.")
    return config

# Validate the configuration file
def validate_config(config):
    required_keys = ['reference_image_path', 'test_images', 'distance_threshold', 'kmeans_clusters']
    for key in required_keys:
        if key not in config:
            logging.error(f"Missing required config key: {key}")
            return False
    return True

# Preprocess the image using GaussianBlur and bilateralFilter
@exception_handler
def preprocess(img):
    img = cv2.GaussianBlur(img, (7, 7), 0)
    img = cv2.bilateralFilter(img, 9, 75, 75)
    return img

# Resize the image to a given size
def resize_image(img, size=(256, 256)):
    return cv2.resize(img, size)

# Compute ORB descriptors for an image
def compute_descriptors(image):
    orb = cv2.ORB_create(nfeatures=2000)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

# Find good matches between descriptors using BFMatcher
def find_good_matches(des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches

# Align the test image with the reference image
@exception_handler
def align_images(reference_image, test_image):
    logging.info("Alignment starts")
    gray_ref = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    gray_test = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

    kp1, des1 = compute_descriptors(gray_ref)
    kp2, des2 = compute_descriptors(gray_test)

    if des1 is None or des2 is None:
        logging.error("Descriptor computation failed. Descriptors are None.")
        return None

    good_matches = find_good_matches(des1, des2)
    logging.info(f"Number of good matches found: {len(good_matches)}")

    if len(good_matches) < 10:
        logging.warning("Not enough good matches found between reference and test image.")
        return None

    pts_ref = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_test = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    try:
        H, mask = cv2.findHomography(pts_test, pts_ref, cv2.RANSAC, 5.0)
    except cv2.error as e:
        logging.error(f"Homography computation failed with error: {str(e)}")
        H = None

    if H is not None and H.shape == (3, 3) and not np.isclose(np.linalg.det(H), 0):
        logging.info(f"Homography matrix: \n{H}")
        height, width, _ = reference_image.shape
        aligned_test_image = cv2.warpPerspective(test_image, H, (width, height))
        # Verify if the resulting image is not distorted by checking the scale
        scale_x = np.sqrt(H[0, 0]**2 + H[0, 1]**2)
        scale_y = np.sqrt(H[1, 0]**2 + H[1, 1]**2)
        if scale_x < 0.1 or scale_x > 10 or scale_y < 0.1 or scale_y > 10:
            logging.error("Detected abnormal scale in homography transformation. Falling back to affine transformation.")
            H = None
    else:
        logging.error("Homography computation failed or resulted in an invalid transformation.")

    if H is None:
        if len(good_matches) < 3:
            logging.error("Not enough matches for affine transformation. Returning None.")
            return None
        pts_ref = np.float32([kp1[m.queryIdx].pt for m in good_matches[:3]]).reshape(-1, 2)
        pts_test = np.float32([kp2[m.trainIdx].pt for m in good_matches[:3]]).reshape(-1, 2)
        M = cv2.getAffineTransform(pts_test, pts_ref)
        height, width, _ = reference_image.shape
        aligned_test_image = cv2.warpAffine(test_image, M, (width, height))

    # Validate the transformation and crop black borders
    aligned_test_image_gray = cv2.cvtColor(aligned_test_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(aligned_test_image_gray, 1, 255, cv2.THRESH_BINARY)

    contours_info = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[0] if len(contours_info) >= 2 else []

    if not contours:
        logging.error("No contours found for cropping. Returning the aligned image without cropping.")
        return aligned_test_image

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    aligned_test_image = aligned_test_image[y:y+h, x:x+w]

    logging.info("Alignment done")
    return aligned_test_image

# Segment the image using K-means clustering
@exception_handler
def k_mean_segmentation(image, k):
    pixels = image.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()].reshape(image.shape)

    avg_colors = [cv2.mean(image, mask=(labels.reshape(image.shape[:2]) == i).astype(np.uint8))[:3] for i in range(k)]
    return segmented_image, avg_colors, labels

# Convert RGB colors to CIELAB color space
def convert_colors_to_cielab(avg_colors):
    avg_colors_lab = []
    for color in avg_colors:
        color_rgb = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2RGB)
        color_lab = rgb2lab(color_rgb / 255.0)[0][0]
        avg_colors_lab.append(tuple(color_lab))
    return avg_colors_lab

# Collect plot data for preprocessing steps
def collect_preprocessing_steps_plot_data(original_image, preprocessed_image, aligned_image, title_prefix=''):
    plot_data = {
        'original_image': original_image,
        'preprocessed_image': preprocessed_image,
        'aligned_image': aligned_image,
        'title_prefix': title_prefix
    }
    return plot_data

# Convert BGR color to RGB
def bgr_to_rgb(color):
    return color[::-1]

# Save the preprocessing steps plot
def save_preprocessing_steps_plot(plot_data, output_dir='output'):
    original_image = plot_data['original_image']
    preprocessed_image = plot_data['preprocessed_image']
    aligned_image = plot_data['aligned_image']
    title_prefix = plot_data['title_prefix']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'{title_prefix} Original Image')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'{title_prefix} Preprocessed Image')
    axes[1].axis('off')

    if aligned_image is not None:
        axes[2].imshow(cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f'{title_prefix} Aligned Image')
    else:
        axes[2].imshow(cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f'{title_prefix} Preprocessed Image (Alignment Failed)')
    axes[2].axis('off')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f'{title_prefix}_preprocessing_steps.png')
    plt.savefig(plot_path)
    logging.info(f"Preprocessing steps plot saved to {plot_path}")
    plt.close(fig)

# Collect plot data for segmentation results
def collect_segment_results_plot_data(segmentation_data, similarity_scores, test_image_path, reference_segmentation_data, best_matches):
    plot_data = {
        'segmentation_data': segmentation_data,
        'similarity_scores': similarity_scores,
        'test_image_path': test_image_path,
        'reference_segmentation_data': reference_segmentation_data,
        'best_matches': best_matches
    }
    return plot_data

# Save the segmentation results plot
def save_segment_results_plot(plot_data, output_dir='output'):
    segmentation_data = plot_data['segmentation_data']
    similarity_scores = plot_data['similarity_scores']
    test_image_path = plot_data['test_image_path']
    reference_segmentation_data = plot_data['reference_segmentation_data']
    best_matches = plot_data['best_matches']

    for i, (test_segment_idx, ref_segment_idx) in enumerate(best_matches):
        if ref_segment_idx == -1:
            continue  # Skip segments without valid matches

        test_color_bgr = segmentation_data['avg_colors'][test_segment_idx]
        test_color_rgb = bgr_to_rgb(test_color_bgr)
        test_color_lab = segmentation_data['avg_colors_lab'][test_segment_idx]
        similarity = similarity_scores[test_segment_idx]

        mask = np.uint8(segmentation_data['labels'] == test_segment_idx).reshape(segmentation_data['original_image'].shape[:2])
        segment = cv2.bitwise_and(segmentation_data['original_image'], segmentation_data['original_image'], mask=mask)

        ref_color_bgr = reference_segmentation_data['avg_colors'][ref_segment_idx]
        ref_color_rgb = bgr_to_rgb(ref_color_bgr)
        ref_color_lab = reference_segmentation_data['avg_colors_lab'][ref_segment_idx]

        ref_mask = np.uint8(reference_segmentation_data['labels'] == ref_segment_idx).reshape(reference_segmentation_data['original_image'].shape[:2])
        ref_segment = cv2.bitwise_and(reference_segmentation_data['original_image'], reference_segmentation_data['original_image'], mask=ref_mask)

        delta_e = ciede2000_distance(test_color_lab, ref_color_lab)

        logging.info(f"Segment {test_segment_idx + 1} - Test Color (RGB): {test_color_rgb}, (CIELAB): {test_color_lab}")
        logging.info(f"Reference Color (RGB): {ref_color_rgb}, (CIELAB): {ref_color_lab}")
        logging.info(f"Delta E: {delta_e:.2f}")

        fig, axes = plt.subplots(1, 4, figsize=(24, 6))

        axes[0].imshow(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f'Test Image: {os.path.basename(test_image_path)}\nSegment {test_segment_idx + 1}, ΔE: {delta_e:.2f}')
        axes[0].axis('off')

        avg_color_rgb = np.uint8([[test_color_rgb]])
        axes[1].imshow(avg_color_rgb)
        axes[1].set_title(f'Segment {test_segment_idx + 1} Avg Color\n(RGB): {tuple(map(int, test_color_rgb))}\n(CIELAB): {tuple(map(int, test_color_lab))}')
        axes[1].axis('off')

        axes[2].imshow(cv2.cvtColor(ref_segment, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f'Reference Segment {ref_segment_idx + 1}')
        axes[2].axis('off')

        ref_color_rgb_img = np.uint8([[ref_color_rgb]])
        axes[3].imshow(ref_color_rgb_img)
        axes[3].set_title(f'Reference Avg Color\n(RGB): {tuple(map(int, ref_color_rgb))}\n(CIELAB): {tuple(map(int, ref_color_lab))}')
        axes[3].axis('off')

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, f'{os.path.basename(test_image_path)}_segment_{test_segment_idx + 1}.png')
        plt.savefig(plot_path)
        logging.info(f"Segment results plot saved to {plot_path}")
        plt.close(fig)

# Plot matches between test and reference images for visualization
def plot_matches(test_image_path, reference_image_path, test_keypoints, ref_keypoints, matches, output_dir='output'):
    img_test = cv2.imread(test_image_path)
    img_ref = cv2.imread(reference_image_path)
    
    if img_test is None or img_ref is None:
        logging.error("Error loading images for plotting matches.")
        return

    img_matches = cv2.drawMatches(img_test, test_keypoints, img_ref, ref_keypoints, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.figure(figsize=(20, 10))
    plt.imshow(img_matches)
    plt.title("Keypoint Matches between Test and Reference Image")
    plt.axis('off')
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f'{os.path.basename(test_image_path)}_matches.png')
    plt.savefig(plot_path)
    logging.info(f"Matches plot saved to {plot_path}")
    plt.close()

# Calculate the CIEDE2000 color difference between two colors
def ciede2000_distance(color1, color2):
    return deltaE_ciede2000(np.array([color1]), np.array([color2]))[0]

# Calculate the similarity between the segmented image colors and target colors
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

# Find the best matching segments between the test and reference images
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

# Process a single image
def process_image(image_path, target_colors, distance_threshold, k, reference_image, reference_segmentation_data, output_dir='output'):
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Failed to load image: {image_path}")
        return None

    original_image = image.copy()
    preprocessed_image = preprocess(image)
    aligned_image = align_images(reference_image, preprocessed_image)

    if aligned_image is None:
        logging.warning(f"Alignment failed for image: {image_path}. Using preprocessed image instead.")
        aligned_image = preprocessed_image

    image = resize_image(aligned_image)

    plot_data = collect_preprocessing_steps_plot_data(original_image, preprocessed_image, aligned_image, title_prefix=f'{os.path.basename(image_path)}')

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
            logging.warning(f"Segment {idx} average color (RGB): {bgr_to_rgb(color)}, (CIELAB): {avg_colors_lab[idx]}")

    plot_results_data = collect_segment_results_plot_data(segmentation_data, similarity_scores, image_path, reference_segmentation_data, best_matches)

    return segmentation_data, similarity_scores, best_matches, plot_data, plot_results_data, image_path

# Process the reference image
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

# Save the results to files
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

# Main function to execute the script
def main(config_path='config.yaml'):
    logging.info(f"Current working directory: {os.getcwd()}")
    config = load_config(config_path)
    if config is None or not validate_config(config):
        logging.error("Invalid configuration. Exiting.")
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

    results = Parallel(n_jobs=-1, backend="threading")(delayed(process_image)(
        image_path, target_colors, distance_threshold, k, reference_image, reference_segmentation_data) for image_path in test_images)

    for result in results:
        if result:
            segmentation_data, similarity_scores, best_matches, plot_data, plot_results_data, image_path = result
            save_preprocessing_steps_plot(plot_data)
            save_segment_results_plot(plot_results_data)
            save_results(segmentation_data, [], [], similarity_scores)

if __name__ == "__main__":
    main(config_path='C:/Users/LENOVO/Desktop/prints/segmentation/unicorn_config.yaml')
