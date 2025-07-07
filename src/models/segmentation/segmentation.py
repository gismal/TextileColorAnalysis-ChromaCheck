# src/models/segmentation.py
import logging
import os
import cv2
import numpy as np
from src.utils.segmentation_utils import k_mean_segmentation, som_segmentation, optimal_clusters, dbscan_clustering, optimal_clusters_dbscan, optimal_kmeans, optimal_dbscan, optimal_som, determine_optimal_clusters
from src.utils.image_utils import ciede2000_distance

class Segmenter:
    def __init__(self, preprocessed_image, target_colors, distance_threshold, reference_kmeans_opt, reference_som_opt, dbn, scalers, predefined_k, k_values, som_values, output_dir):
        """Initialize the segmenter for image segmentation and color conversion."""
        self.preprocessed_image = preprocessed_image
        self.target_colors = target_colors
        self.distance_threshold = distance_threshold
        self.reference_kmeans_opt = reference_kmeans_opt
        self.reference_som_opt = reference_som_opt
        self.dbn = dbn
        self.scalers = scalers  # Tuple of (scaler_x, scaler_y, scaler_y_ab)
        self.predefined_k = predefined_k
        self.k_values = k_values
        self.som_values = som_values
        self.output_dir = output_dir

    def compute_similarity(self, segmentation_result):
        """Compute similarity scores between segmented colors and target colors."""
        segmented_colors = segmentation_result[1]  # avg_colors from segmentation
        similarities = []
        for color in segmented_colors:
            min_distance = min(ciede2000_distance(color, target) for target in self.target_colors)
            similarities.append(min_distance)
        return similarities

    def find_best_matches(self, segmentation_result):
        """Find the best matches between segmented colors and target colors."""
        segmented_colors = segmentation_result[1]  # avg_colors
        best_matches = []
        for i, color in enumerate(segmented_colors):
            if not self.target_colors:  # Avoid division by zero or empty list
                best_matches.append((i, -1, float('inf')))
                continue
            min_distance = float('inf')
            best_target_idx = -1
            for j, target in enumerate(self.target_colors):
                distance = ciede2000_distance(color, target)
                if distance < min_distance:
                    min_distance = distance
                    best_target_idx = j
            best_matches.append((i, best_target_idx, min_distance))
        return best_matches

    def run_kmeans_optimal(self):
        """Run K-means with dynamically determined optimal clusters."""
        pixels = self.preprocessed_image.reshape(-1, 3).astype(np.float32)
        optimal_k = optimal_clusters(pixels, default_k=3, max_k=max(self.k_values))
        return k_mean_segmentation(self.preprocessed_image, optimal_k)

    def run_kmeans_predefined(self):
        """Run K-means with predefined number of clusters."""
        return k_mean_segmentation(self.preprocessed_image, self.predefined_k)

    def run_dbscan(self):
        """Run DBSCAN with optimal parameters."""
        pixels = self.preprocessed_image.reshape(-1, 3).astype(np.float32)
        labels = optimal_dbscan(self.preprocessed_image)
        # Convert DBSCAN labels to segmentation results
        unique_labels = np.unique(labels[labels >= 0])
        centers = np.array([np.mean(pixels[labels == label], axis=0) for label in unique_labels])
        centers = np.uint8(centers)
        segmented_image = centers[labels[labels >= 0]].reshape(self.preprocessed_image.shape)
        avg_colors = [cv2.mean(self.preprocessed_image, mask=(labels.reshape(self.preprocessed_image.shape[:2]) == i).astype(np.uint8))[:3] for i in unique_labels]
        return segmented_image, avg_colors, labels

    def run_som_optimal(self):
        """Run SOM with dynamically determined optimal clusters."""
        pixels = self.preprocessed_image.reshape(-1, 3).astype(np.float32) / 255.0
        optimal_k = optimal_clusters(pixels, default_k=3, max_k=max(self.som_values))
        return som_segmentation(self.preprocessed_image, optimal_k)

    def run_som_predefined(self):
        """Run SOM with predefined number of clusters."""
        return som_segmentation(self.preprocessed_image, self.predefined_k)

    def process(self):
        """Process the image with various segmentation methods.
        
        Returns:
            tuple: (preprocessed_path, kmeans_opt_results, kmeans_predef_results, dbscan_results,
                    som_opt_results, som_predef_results) where each _results is a tuple
                    (segmented_image, similarities, best_matches).
        """
        # Preprocessed path
        preprocessed_path = os.path.join(self.output_dir, "preprocessed_image.jpg")
        cv2.imwrite(preprocessed_path, self.preprocessed_image)

        # K-means with optimal clusters
        kmeans_opt_results = self.run_kmeans_optimal()
        sim_kmeans_opt = self.compute_similarity(kmeans_opt_results)
        best_kmeans_opt = self.find_best_matches(kmeans_opt_results)

        # K-means with predefined clusters
        kmeans_predef_results = self.run_kmeans_predefined()
        sim_kmeans_predef = self.compute_similarity(kmeans_predef_results)
        best_kmeans_predef = self.find_best_matches(kmeans_predef_results)

        # DBSCAN
        dbscan_results = self.run_dbscan()
        sim_dbscan = self.compute_similarity(dbscan_results)
        best_dbscan = self.find_best_matches(dbscan_results)

        # SOM with optimal clusters
        som_opt_results = self.run_som_optimal()
        sim_som_opt = self.compute_similarity(som_opt_results)
        best_som_opt = self.find_best_matches(som_opt_results)

        # SOM with predefined clusters
        som_predef_results = self.run_som_predefined()
        sim_som_predef = self.compute_similarity(som_predef_results)
        best_som_predef = self.find_best_matches(som_predef_results)

        return (
            preprocessed_path,
            (kmeans_opt_results[0], sim_kmeans_opt, best_kmeans_opt),  # segmented_image, similarities, best_matches
            (kmeans_predef_results[0], sim_kmeans_predef, best_kmeans_predef),
            (dbscan_results[0], sim_dbscan, best_dbscan),
            (som_opt_results[0], sim_som_opt, best_som_opt),
            (som_predef_results[0], sim_som_predef, best_som_predef)
        )