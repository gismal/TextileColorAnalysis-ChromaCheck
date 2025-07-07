import os
import cv2
import numpy as np
import logging
from src.models.segmentation.segmentation import k_mean_segmentation, som_segmentation, optimal_clusters_dbscan
from src.utils.image_utils import calculate_similarity, find_best_matches, downsample_image
from src.utils.visualization import save_segment_results_plot
from src.data import preprocess

class ImageProcessor:
    def __init__(self, image_path, target_colors, distance_threshold, reference_kmeans, reference_som, dbn, scalers, predefined_k, eps_values, min_samples_values, output_dir):
        self.image_path = image_path
        self.target_colors = target_colors
        self.distance_threshold = distance_threshold
        self.reference_kmeans = reference_kmeans
        self.reference_som = reference_som
        self.dbn = dbn
        self.scaler_x, self.scaler_y, self.scaler_y_ab = scalers
        self.predefined_k = predefined_k
        self.eps_values = eps_values
        self.min_samples_values = min_samples_values
        self.output_dir = output_dir
        self.image_name = os.path.splitext(os.path.basename(image_path))[0]

    def load_and_preprocess(self):
        """Load and preprocess the image, saving the preprocessed version."""
        logging.info(f"Processing image: {self.image_path}")
        image = cv2.imread(self.image_path)
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {self.image_path}")
        
        original_image = image.copy()
        preprocessed_image = preprocess(image)
        if preprocessed_image is None or preprocessed_image.size == 0:
            raise ValueError("Preprocessed image is empty")
        
        resized_image = preprocess.resize_image(preprocessed_image)
        
        # Save preprocessed image
        preprocessed_dir = os.path.join(self.output_dir, self.image_name, 'preprocessed')
        os.makedirs(preprocessed_dir, exist_ok=True)
        preprocessed_path = os.path.join(preprocessed_dir, f'preprocessed_{os.path.basename(self.image_path)}')
        cv2.imwrite(preprocessed_path, preprocessed_image)
        
        return original_image, resized_image, preprocessed_path

    ## belki segmentation türleri ayrılabilir
    def segment_and_analyze(self, image, downsampled_image, method, k=None):
        """Perform segmentation and analysis based on the specified method."""
        if method == 'kmeans_optimal':
            segmented_image, avg_colors, labels = k_mean_segmentation(image, len(self.reference_kmeans['avg_colors']))
        elif method == 'kmeans_predefined':
            segmented_image, avg_colors, labels = k_mean_segmentation(image, k or self.predefined_k)
        elif method == 'dbscan':
            downsampled_pixels = downsampled_image.reshape(-1, 3).astype(np.float32)
            labels, best_eps, best_min_samples = optimal_clusters_dbscan(downsampled_pixels, self.eps_values, self.min_samples_values)
            unique_labels = np.unique(labels)
            segmented_image = np.zeros_like(downsampled_image)
            for label in unique_labels:
                if label == -1:  # Skip noise
                    continue
                mask = (labels == label)
                color = np.mean(downsampled_pixels[mask], axis=0).astype(np.uint8)
                segmented_image[mask.reshape(downsampled_image.shape[:2])] = color
            avg_colors = [np.mean(downsampled_pixels[labels == label], axis=0).astype(np.uint8) for label in unique_labels if label != -1]
        elif method == 'som_optimal':
            segmented_image, avg_colors, labels = som_segmentation(image, len(self.reference_som['avg_colors']))
        elif method == 'som_predefined':
            segmented_image, avg_colors, labels = som_segmentation(image, k or self.predefined_k)
        else:
            raise ValueError(f"Unsupported segmentation method: {method}")

        from src.utils.color.color_conversion import convert_colors_to_cielab, convert_colors_to_cielab_dbn
        avg_colors_lab = convert_colors_to_cielab(avg_colors)
        avg_colors_lab_dbn = convert_colors_to_cielab_dbn(self.dbn, self.scaler_x, self.scaler_y, self.scaler_y_ab, avg_colors)
        segmentation_data = {
            'original_image': image if method != 'dbscan' else downsampled_image,
            'segmented_image': segmented_image,
            'avg_colors': avg_colors,
            'avg_colors_lab': avg_colors_lab,
            'avg_colors_lab_dbn': avg_colors_lab_dbn,
            'labels': labels
        }
        
        similarity_scores = calculate_similarity(segmentation_data, self.target_colors)
        reference = self.reference_kmeans if 'kmeans' in method or 'dbscan' in method else self.reference_som
        best_matches = find_best_matches(segmentation_data, reference)
        
        return segmentation_data, similarity_scores, best_matches

    def save_results(self, segmentation_data, similarity_scores, best_matches, method, subfolder):
        """Save segmentation results to the output directory."""
        output_path = os.path.join(self.output_dir, self.image_name, subfolder)
        save_segment_results_plot(
            segmentation_data, similarity_scores, self.image_path, 
            self.reference_kmeans if 'kmeans' in method or 'dbscan' in method else self.reference_som, 
            best_matches, segmentation_data['avg_colors_lab_dbn'], 
            method=method, output_dir=output_path
        )

    def process(self):
        """Process the image using multiple segmentation methods."""
        try:
            original_image, image, preprocessed_path = self.load_and_preprocess()
            downsampled_image = downsample_image(image, scale_factor=0.5)

            results = {}
            methods = [
                ('kmeans_optimal', 'kmeans/optimal'),
                ('kmeans_predefined', 'kmeans/predefined'),
                ('dbscan', 'dbscan'),
                ('som_optimal', 'som/optimal'),
                ('som_predefined', 'som/predefined')
            ]

            for method, subfolder in methods:
                seg_data, sim_scores, best_matches = self.segment_and_analyze(image, downsampled_image, method)
                self.save_results(seg_data, sim_scores, best_matches, method.capitalize().replace('_', '-'), subfolder)
                results[method] = (seg_data, sim_scores, best_matches)

            return (preprocessed_path, *(results[method] for method in ['kmeans_optimal', 'kmeans_predefined', 'dbscan', 'som_optimal', 'som_predefined']))

        except Exception as e:
            logging.error(f"Error processing image {self.image_path}: {e}")
            return None