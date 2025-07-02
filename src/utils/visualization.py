import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging

from src.config import BASE_OUTPUT_DIR
from src.utils.color_conversion import bgr_to_rgb
from src.utils.image_utils import ciede2000_distance

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_segment_results_plot(segmentation_data, similarity_scores, test_image_path, reference_segmentation_data, best_matches, avg_colors_lab_dbn, method, output_dir=BASE_OUTPUT_DIR):
    """Save segmentation results plot with comparison to reference."""
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
        plot_path = os.path.join(output_dir, f'{os.path.basename(test_image_path)}_{method}_segment{test_segment_idx + 1}.png')
        plt.savefig(plot_path, dpi=300)
        logging.info(f"Segment results plot saved to {plot_path}")
        plt.close(fig)

def save_reference_summary_plot(reference_kmeans_opt, reference_som_opt, original_image, output_dir=BASE_OUTPUT_DIR):
    """Save a summary plot for reference segmentation."""
    fig, axes = plt.subplots(2, len(reference_kmeans_opt['avg_colors']) + 1, figsize=(20, 12))

    for i, (color, color_lab, color_lab_dbn) in enumerate(zip(reference_kmeans_opt['avg_colors'], reference_kmeans_opt['avg_colors_lab'], reference_kmeans_opt['avg_colors_lab_dbn'])):
        color_rgb = bgr_to_rgb(color)
        avg_color_rgb_img = np.full((50, 100, 3), color_rgb, dtype=np.uint8)
        
        axes[0, i].imshow(avg_color_rgb_img)
        axes[0, i].set_title(f'Color {i + 1}\n(RGB): {tuple(map(lambda x: round(x, 3), color_rgb))}\n(CIELAB): {tuple(map(lambda x: round(x, 3), color_lab))}\n(PSO-DBN CIELAB): {tuple(map(lambda x: round(x, 3), color_lab_dbn))}', fontsize=8)
        axes[0, i].axis('off')

    axes[0, -1].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0, -1].set_title('Reference Image')
    axes[0, -1].axis('off')

    for i, (color, color_lab, color_lab_dbn) in enumerate(zip(reference_som_opt['avg_colors'], reference_som_opt['avg_colors_lab'], reference_som_opt['avg_colors_lab_dbn'])):
        color_rgb = bgr_to_rgb(color)
        avg_color_rgb_img = np.full((50, 100, 3), color_rgb, dtype=np.uint8)
        
        axes[1, i].imshow(avg_color_rgb_img)
        axes[1, i].set_title(f'Color {i + 1}\n(RGB): {tuple(map(lambda x: round(x, 3), color_rgb))}\n(CIELAB): {tuple(map(lambda x: round(x, 3), color_lab))}\n(PSO-DBN CIELAB): {tuple(map(lambda x: round(x, 3), color_lab_dbn))}', fontsize=8)
        axes[1, i].axis('off')

    axes[1, -1].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[1, -1].set_title('Reference Image')
    axes[1, -1].axis('off')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'reference_summary.png')
    plt.savefig(plot_path, dpi=300)
    logging.info(f"Reference summary plot saved to {plot_path}")
    plt.close(fig)

def save_preprocessing_steps_plot(original_image, preprocessed_image, title_prefix='', output_dir=BASE_OUTPUT_DIR):
    """Save plot comparing original and preprocessed images."""
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

def save_segmentation_summary_plot(segmentation_data, similarity_scores, method, output_dir=BASE_OUTPUT_DIR):
    """Save a summary plot for segmentation results."""
    fig, axes = plt.subplots(1, len(segmentation_data['avg_colors']) + 1, figsize=(20, 6))

    for i, (color, color_lab, color_lab_dbn) in enumerate(zip(segmentation_data['avg_colors'], segmentation_data['avg_colors_lab'], segmentation_data['avg_colors_lab_dbn'])):
        color_rgb = bgr_to_rgb(color)
        avg_color_rgb_img = np.full((50, 100, 3), color_rgb, dtype=np.uint8)
        
        axes[i].imshow(avg_color_rgb_img)
        axes[i].set_title(f'Color {i + 1}\n(RGB): {tuple(map(lambda x: round(x, 3), color_rgb))}\n(CIELAB): {tuple(map(lambda x: round(x, 3), color_lab))}\n(PSO-DBN CIELAB): {tuple(map(lambda x: round(x, 3), color_lab_dbn))}', fontsize=8)
        axes[i].axis('off')

    axes[-1].imshow(cv2.cvtColor(segmentation_data['segmented_image'], cv2.COLOR_BGR2RGB))
    axes[-1].set_title(f'Segmented Image\n(Method: {method})', fontsize=10)
    axes[-1].axis('off')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f'{method}_segmentation_summary.png')
    plt.savefig(plot_path, dpi=300)
    logging.info(f"{method} segmentation summary plot saved to {plot_path}")
    plt.close(fig)