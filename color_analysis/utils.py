import os
import yaml
import logging
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from skimage.color import rgb2lab, deltaE_ciede2000
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

def load_config(config_path):
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logging.error(f"Error loading config: {e}")
        return None

def validate_config(config):
    required_keys = ['reference_image_path', 'test_images', 'distance_threshold', 'kmeans_clusters']
    for key in required_keys:
        if key not in config:
            logging.error(f"Missing required config key: {key}")
            return False
    return True

def convert_colors_to_cielab(avg_colors):
    avg_colors_lab = []
    for color in avg_colors:
        color_rgb = np.uint8([[color]])
        color_lab = rgb2lab(color_rgb / 255.0)[0][0]
        avg_colors_lab.append(tuple(color_lab))
    return avg_colors_lab

def convert_colors_to_cielab_dbn(dbn, scaler_x, scaler_y, avg_colors):
    avg_colors_lab_dbn = []
    for color in avg_colors:
        color_rgb = np.array(color).reshape(1, -1)
        color_rgb_scaled = scaler_x.transform(color_rgb)
        color_lab_dbn_scaled = dbn.predict(color_rgb_scaled)
        color_lab_dbn = scaler_y.inverse_transform(color_lab_dbn_scaled)[0]
        avg_colors_lab_dbn.append(tuple(color_lab_dbn))
    return avg_colors_lab_dbn

def optimal_clusters(pixels, max_k=10):
    silhouette_scores = []

    for n_clusters in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(pixels)
        silhouette_avg = silhouette_score(pixels, labels)
        silhouette_scores.append(silhouette_avg)

    optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
    return optimal_k

def save_preprocessing_steps_plot(original_image, preprocessed_image, output_dir, title_prefix=''):
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
    plt.savefig(plot_path)
    logging.info(f"Preprocessing steps plot saved to {plot_path}")
    plt.close(fig)

def save_segment_results_plot(segmentation_data, similarity_scores, test_image_path, reference_segmentation_data, best_matches, avg_colors_lab_dbn, method, output_dir):
    for i, (test_segment_idx, ref_segment_idx) in enumerate(best_matches):
        if ref_segment_idx == -1:
            continue  # Skip segments without valid matches

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
        axes[0].set_title(f'Segment {test_segment_idx + 1}\nΔE CIELAB: {delta_e_cielab:.3f}, ΔE DBN: {delta_e_dbn:.3f}')
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
        axes[3].set_title(f'Reference Avg Color\n(RGB): {tuple(map(lambda x: round(x, 3), ref_color_rgb))}\n(CIELAB): {tuple(map(lambda x: round(x, 3), ref_color_lab))}')
        axes[3].axis('off')

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, f'{os.path.basename(test_image_path)}_{method}_segment_{test_segment_idx + 1}.png')
        plt.savefig(plot_path)
        logging.info(f"Segment results plot saved to {plot_path}")
        plt.close(fig)

def save_reference_summary_plot(reference_kmeans, reference_som, original_image, output_dir):
    fig, axes = plt.subplots(2, len(reference_kmeans['avg_colors']) + 1, figsize=(20, 12))

    for i, (color, color_lab, color_lab_dbn) in enumerate(zip(reference_kmeans['avg_colors'], reference_kmeans['avg_colors_lab'], reference_kmeans['avg_colors_lab_dbn'])):
        color_rgb = bgr_to_rgb(color)
        avg_color_rgb_img = np.full((50, 100, 3), color_rgb, dtype=np.uint8)

        axes[0, i].imshow(avg_color_rgb_img)
        axes[0, i].set_title(f'Color {i + 1}\n(RGB): {tuple(map(lambda x: round(x, 3), color_rgb))}\n(CIELAB): {tuple(map(lambda x: round(x, 3), color_lab))}\n(PSO-DBN CIELAB): {tuple(map(lambda x: round(x, 3), color_lab_dbn))}')
        axes[0, i].axis('off')

    axes[0, -1].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0, -1].set_title('Reference Image')
    axes[0, -1].axis('off')

    for i, (color, color_lab, color_lab_dbn) in enumerate(zip(reference_som['avg_colors'], reference_som['avg_colors_lab'], reference_som['avg_colors_lab_dbn'])):
        color_rgb = bgr_to_rgb(color)
        avg_color_rgb_img = np.full((50, 100, 3), color_rgb, dtype=np.uint8)

        axes[1, i].imshow(avg_color_rgb_img)
        axes[1, i].set_title(f'Color {i + 1}\n(RGB): {tuple(map(lambda x: round(x, 3), color_rgb))}\n(CIELAB): {tuple(map(lambda x: round(x, 3), color_lab))}\n(PSO-DBN CIELAB): {tuple(map(lambda x: round(x, 3), color_lab_dbn))}')
        axes[1, i].axis('off')

    axes[1, -1].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[1, -1].set_title('Reference Image')
    axes[1, -1].axis('off')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'reference_summary.png')
    plt.savefig(plot_path)
    logging.info(f"Reference summary plot saved to {plot_path}")
    plt.close(fig)

def save_results(segmentation_data, similarity_scores, method, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    segmented_image_path = os.path.join(output_dir, f'segmented_image_{method}.png')
    cv2.imwrite(segmented_image_path, segmentation_data['segmented_image'])

    avg_colors_df = pd.DataFrame(segmentation_data['avg_colors'], columns=['B', 'G', 'R'])
    avg_colors_df.to_csv(os.path.join(output_dir, f'average_colors_{method}.csv'), index=False)

    similarity_scores_df = pd.DataFrame(similarity_scores, columns=['Similarity_Score'])
    similarity_scores_df.to_csv(os.path.join(output_dir, f'similarity_scores_{method}.csv'), index=False)

    if len(similarity_scores) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap([similarity_scores], annot=True, cmap="YlGnBu", cbar=True, ax=ax)
        ax.set_title(f'Similarity Scores Heatmap ({method})')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'similarity_heatmap_{method}.png'))
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, color in enumerate(segmentation_data['avg_colors']):
        ax.bar(i, 1, color=np.array(color) / 255.0, edgecolor='black')
    ax.set_xticks(range(len(segmentation_data['avg_colors'])))
    ax.set_xticklabels([f'Color {i + 1}' for i in range(len(segmentation_data['avg_colors']))], rotation=45)
    ax.set_title(f'Color Distribution Histogram ({method})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'color_distribution_{method}.png'))
    plt.close(fig)

def bgr_to_rgb(color):
    return color[::-1]

def ciede2000_distance(color1, color2):
    return deltaE_ciede2000(np.array([color1]), np.array([color2]))[0]
