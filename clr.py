import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml
from skimage.color import rgb2lab, deltaE_ciede2000
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
import pandas as pd
import logging

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

def align_images(reference_image, test_image):
    try:
        gray_ref = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
        gray_test = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray_ref, None)
        kp2, des2 = sift.detectAndCompute(gray_test, None)

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) < 10:
            logging.warning("Not enough matches found between reference and test image.")
            return test_image

        pts_ref = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts_test = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(pts_test, pts_ref, cv2.RANSAC, 5.0)

        if H is None or H.shape != (3, 3) or np.linalg.det(H) == 0:
            logging.error("Homography computation failed.")
            return test_image

        height, width, _ = reference_image.shape
        aligned_test_image = cv2.warpPerspective(test_image, H, (width, height))

        # Remove black borders by cropping to the common area
        aligned_test_image_gray = cv2.cvtColor(aligned_test_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(aligned_test_image_gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        aligned_test_image = aligned_test_image[y:y+h, x:x+w]

        return aligned_test_image
    except Exception as e:
        logging.error(f"Error aligning images: {str(e)}")
        return test_image

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

def convert_colors_to_cielab(avg_colors):
    avg_colors_lab = []
    for color in avg_colors:
        color_rgb = np.uint8([[color]])
        color_lab = rgb2lab(color_rgb)[0][0]
        avg_colors_lab.append(tuple(color_lab))
    return avg_colors_lab

def convert_opencv_lab_to_standard(lab_color):
    # Convert to float to avoid overflow
    lab_color = lab_color.astype(np.float64)
    # Adjusting L* from [0, 255] to [0, 100]
    L = lab_color[0] * 100 / 255
    # Adjusting a* and b* by removing the offset 128
    a = lab_color[1] - 128
    b = lab_color[2] - 128
    return np.array([L, a, b], dtype=np.float64)

def plot_preprocessing_steps(original_image, preprocessed_image, aligned_image, title_prefix=''):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'{title_prefix} Original Image')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'{title_prefix} Preprocessed Image')
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f'{title_prefix} Aligned Image')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

def plot_segment_results(segmentation_data, similarity_scores, test_image_path, reference_segmentation_data, best_matches):
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
        axes[0].set_title(f'Test Image: {os.path.basename(test_image_path)}\nSegment {test_segment_idx + 1}, ΔE: {delta_e:.2f}')
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

def plot_matches(test_image_path, reference_image_path, test_keypoints, ref_keypoints, matches):
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
    aligned_image = align_images(reference_image, preprocessed_image)
    image = resize_image(aligned_image)

    plot_preprocessing_steps(original_image, preprocessed_image, aligned_image, title_prefix=f'{os.path.basename(image_path)}')

    segmented_image, avg_colors, labels = k_mean_segmentation(image, k)
    
    # Convert average colors to CIELAB using OpenCV and then normalize
    avg_colors_lab = [convert_opencv_lab_to_standard(cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2LAB)[0][0]) for color in avg_colors]

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
    
    # Convert average colors to CIELAB using OpenCV and then normalize
    avg_colors_lab = [convert_opencv_lab_to_standard(cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2LAB)[0][0]) for color in avg_colors]

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
    main(config_path='C:/Users/LENOVO/Desktop/prints/segmentation/unicorn_config.yaml')
