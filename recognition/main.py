# main.py
from image_loader import ImageLoader
from image_preprocessor import ImagePreprocessor
from feature_matcher import FeatureMatcher
from image_aligner import ImageAligner
from contour_extractor import ContourExtractor
from segmenter import Segmenter
from color_analyzer import ColorAnalyzer
from pattern_detector import PatternDetector
from visualizer import Visualizer

import cv2
import numpy as np
from skimage import color
import logging

logging.basicConfig(level=logging.INFO)

def analyze_prints(reference_path, print_paths, resize_width=800, num_clusters=3):
    try:
        # Load and preprocess the reference image
        ref_loader = ImageLoader(reference_path)
        reference_img = ref_loader.load_image()
        if reference_img is None:
            logging.error("Error: Reference image could not be loaded.")
            return [], []
        preprocessor = ImagePreprocessor()
        reference_img_preprocessed = preprocessor.preprocess_for_alignment(reference_img)

        # Extract contours and apply mask to the reference image
        contour_extractor = ContourExtractor()
        ref_contours = contour_extractor.extract_and_smooth_contours(reference_img_preprocessed)
        ref_masked_image = contour_extractor.apply_mask(reference_img, ref_contours)

        correct_prints = []
        incorrect_prints = []

        for print_path in print_paths:
            # Load and preprocess the print image
            print_loader = ImageLoader(print_path)
            print_img = print_loader.load_image()
            if print_img is None:
                continue
            print_img_preprocessed = preprocessor.preprocess_for_alignment(print_img)

            # Feature matching
            matcher = FeatureMatcher()
            kp1, des1, kp2, des2, matches = matcher.orb_feature_matching(reference_img_preprocessed, print_img_preprocessed)

            visualizer = Visualizer()
            visualizer.visualize_keypoints(reference_img, kp1, 'Reference Image Keypoints')
            visualizer.visualize_keypoints(print_img, kp2, 'Print Image Keypoints')
            visualizer.visualize_matches(reference_img, kp1, print_img, kp2, matches, 'Keypoint Matches')

            # Align images
            aligner = ImageAligner()
            aligned_img = aligner.align_images(reference_img, print_img, kp1, kp2, matches)
            if aligned_img is None:
                logging.warning(f"Alignment failed for '{print_path}'. Skipping this print.")
                incorrect_prints.append(print_path)
                continue

            # Extract contours and apply mask to the aligned image
            aligned_contours = contour_extractor.extract_and_smooth_contours(preprocessor.preprocess_for_alignment(aligned_img))
            aligned_masked_image = contour_extractor.apply_mask(aligned_img, aligned_contours)

            visualizer.visualize_images([reference_img, ref_masked_image], ['Original Reference Image', 'Masked Reference Image'])
            visualizer.visualize_images([aligned_img, aligned_masked_image], ['Aligned Image', 'Masked Aligned Image'])
            
            segmenter = Segmenter()
            ref_segments = segmenter.adaptive_segment_image(reference_img)
            aligned_segments = segmenter.adaptive_segment_image(aligned_img)

            visualizer.visualize_segments(ref_segments, 'Reference Image Segments')
            visualizer.visualize_segments(aligned_segments, 'Aligned Image Segments')

            # Pattern detection and segmentation
            pattern_detector = PatternDetector()
            clustered_keypoints = pattern_detector.cluster_keypoints(kp2, des2, num_clusters=num_clusters)
            for cluster in clustered_keypoints:
                if len(cluster) > 0:
                    template = cv2.drawKeypoints(print_img, cluster, None, color=(0, 255, 0), flags=0)
                    loc = pattern_detector.template_matching(aligned_img, template)
                    template_size = template.shape[1::-1]
                    if loc[0].size > 0 and loc[1].size > 0:
                        visualizer.visualize_pattern_detection(aligned_img, loc, template_size, 'Pattern Detection')
                    else:
                        logging.info(f"No patterns detected for cluster with template size: {template_size}")

            # Color analysis
            color_analyzer = ColorAnalyzer()
            ref_colors = []
            aligned_colors = []
            for ref_seg, aligned_seg in zip(ref_segments, aligned_segments):
                ref_avg_color = color_analyzer.compute_average_color(ref_seg)
                aligned_avg_color = color_analyzer.compute_average_color(aligned_seg)

                ref_lab_color = color.rgb2lab(np.uint8([[ref_avg_color]]))[0][0]
                aligned_lab_color = color.rgb2lab(np.uint8([[aligned_avg_color]]))[0][0]

                ref_colors.append(ref_avg_color)
                aligned_colors.append(aligned_avg_color)

                logging.info(f"Reference segment average color in CIELAB: {ref_lab_color}")
                logging.info(f"Aligned segment average color in CIELAB: {aligned_lab_color}")

            color_analyzer.visualize_colors(ref_colors, aligned_colors, 'Average Colors of Segments')

            correct_prints.append(print_path)
    except Exception as e:
        logging.error(f"Error in the analyze_prints function: {e}")
        return [], []

    return correct_prints, incorrect_prints

# Example usage
if __name__ == "__main__":
    reference_path = 'C:/Users/LENOVO/Desktop/Thesis/zorlutekstil_1_2024-01-19_1042/1/1/orginal.jpg'
    print_paths = [
        'C:/Users/LENOVO/Desktop/Thesis/zorlutekstil_1_2024-01-19_1042/1/1/1.jpg',
        'C:/Users/LENOVO/Desktop/Thesis/zorlutekstil_1_2024-01-19_1042/1/1/2.jpg'
    ]
    correct_prints, incorrect_prints = analyze_prints(reference_path, print_paths)
    logging.info("Correct Prints: %s", correct_prints)
    logging.info("Incorrect Prints: %s", incorrect_prints)
