import logging
import os
import cv2
import numpy as np
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
from src.utils.segmentation_utils import k_mean_segmentation, som_segmentation, optimal_clusters, dbscan_clustering, optimal_clusters_dbscan, optimal_kmeans, optimal_dbscan, optimal_som, determine_optimal_clusters
from src.utils.image_utils import ciede2000_distance

# Abstract base class for cluster determination strategies
class ClusterStrategy(ABC):
    @abstractmethod
    def determine_k(self, pixels, default_k, min_k, max_k):
        pass

# Concrete strategy using the existing metric-based approach
class MetricBasedStrategy(ClusterStrategy):
    def determine_k(self, pixels, default_k, min_k=3, max_k=10):
        logging.info(f"Determining optimal k for pixels with shape {pixels.shape}")
        return optimal_clusters(pixels, default_k, min_k, max_k)

class Segmenter:
    def __init__(self, preprocessed_image, target_colors, distance_threshold, reference_kmeans_opt, reference_som_opt, dbn, scalers, predefined_k, k_values, som_values, output_dir, k_type='determined', cluster_strategy=None):
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
        self.k_type = k_type  # 'determined' or 'predefined'
        self.cluster_strategy = cluster_strategy or MetricBasedStrategy()

    def quantize_image(self, n_colors=50):
        """Quantize the image to reduce the number of unique colors."""
        logging.info(f"Quantizing image with {len(np.unique(self.preprocessed_image.reshape(-1, 3), axis=0))} unique colors to {n_colors}")
        pixels = self.preprocessed_image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=n_colors, n_init=3, random_state=42).fit(pixels)
        quantized_pixels = kmeans.cluster_centers_[kmeans.predict(pixels)]
        return quantized_pixels.reshape(self.preprocessed_image.shape).astype(np.uint8)

    def compute_similarity(self, segmentation_result):
        """Compute similarity scores between segmented colors and target colors."""
        segmented_colors = segmentation_result[1]  # avg_colors from segmentation
        similarities = []
        for color in segmented_colors:
            min_distance = min(ciede2000_distance(color, target) for target in self.target_colors)
            similarities.append(min_distance)
        return similarities

    def find_best_matches(self, segmentation_result):
        """Find the best matches between segmented colors and target colors.
        Args:
            segmentation_result: Tuple of (segmented_image, avg_colors, labels).
        
        Returns:
            list: List of (test_idx, ref_idx, distance) tuples.
        """
        segmented_colors = segmentation_result[1]  # avg_colors
        best_matches = []
        if self.target_colors.shape[0] == 0:
            logging.error("Target colors array has no entries. Cannot find best matches.")
            return []
        if len(segmented_colors) != len(range(len(segmented_colors))):
            logging.error("Mismatch between segmented colors and indices. Check segmentation result.")
            return []
        for i, color in enumerate(segmented_colors):
            if np.all(np.array(color) <= np.array([5, 130, 130])):  # Ignore nearly black segments
                best_matches.append((i, -1, float('inf')))
                continue
            min_distance = float('inf')
            best_target_idx = -1
            for j, target in enumerate(self.target_colors):
                distance = ciede2000_distance(color, target)
                if distance < min_distance:
                    min_distance = distance
                    best_target_idx = j
            if best_target_idx >= self.target_colors.shape[0]:
                logging.warning(f"Invalid best_target_idx {best_target_idx} for {self.target_colors.shape[0]} target colors. Setting to -1.")
                best_target_idx = -1
            best_matches.append((i, best_target_idx, min_distance))
        logging.debug(f"Best matches: {best_matches}")  # Debug output
        return best_matches

    def run_kmeans_optimal(self):
        """Run K-means with dynamically determined optimal clusters."""
        pixels = self.quantize_image().reshape(-1, 3).astype(np.float32)
        logging.info(f"Running K-means with optimal k determination")
        if self.k_type == 'determined':
            optimal_k = self.cluster_strategy.determine_k(pixels, default_k=3, min_k=3, max_k=max(self.k_values))
            logging.info(f"Optimal k determined: {optimal_k}")
        else:
            optimal_k = self.predefined_k
            logging.info(f"Using predefined k: {optimal_k}")
        return k_mean_segmentation(self.preprocessed_image, optimal_k)

    def run_kmeans_predefined(self):
        """Run K-means with predefined number of clusters."""
        pixels = self.quantize_image().reshape(-1, 3).astype(np.float32)
        logging.info(f"Running K-means with predefined k: {self.predefined_k}")
        return k_mean_segmentation(self.preprocessed_image, self.predefined_k)

    def run_dbscan(self):
        """Run DBSCAN with optimal parameters."""
        pixels = self.quantize_image().reshape(-1, 3).astype(np.float32)
        logging.info("Running DBSCAN with optimal parameters")
        labels = optimal_dbscan(self.preprocessed_image)
        unique_labels = np.unique(labels[labels >= 0])
        centers = np.array([np.mean(pixels[labels == label], axis=0) for label in unique_labels])
        centers = np.uint8(centers)
        segmented_image = centers[labels[labels >= 0]].reshape(self.preprocessed_image.shape)
        avg_colors = [cv2.mean(self.preprocessed_image, mask=(labels.reshape(self.preprocessed_image.shape[:2]) == i).astype(np.uint8))[:3] for i in unique_labels]
        return segmented_image, avg_colors, labels

    def run_som_optimal(self):
        """Run SOM with dynamically determined optimal clusters."""
        pixels = self.quantize_image().reshape(-1, 3).astype(np.float32) / 255.0
        logging.info(f"Running SOM with optimal k determination")
        if self.k_type == 'determined':
            optimal_k = self.cluster_strategy.determine_k(pixels, default_k=3, min_k=3, max_k=max(self.som_values))
            logging.info(f"Optimal k determined: {optimal_k}")
        else:
            optimal_k = self.predefined_k
            logging.info(f"Using predefined k: {optimal_k}")
        return som_segmentation(self.preprocessed_image, optimal_k)

    def run_som_predefined(self):
        """Run SOM with predefined number of clusters."""
        pixels = self.quantize_image().reshape(-1, 3).astype(np.float32) / 255.0
        logging.info(f"Running SOM with predefined k: {self.predefined_k}")
        return som_segmentation(self.preprocessed_image, self.predefined_k)

    def process(self):
        """Process the image with various segmentation methods.
        
        Returns:
            tuple: (preprocessed_path, kmeans_opt_results, kmeans_predef_results, dbscan_results,
                    som_opt_results, som_predef_results) where each _results is a tuple
                    (segmented_image, similarities, best_matches).
        """
        preprocessed_path = os.path.join(self.output_dir, "preprocessed_image.jpg")
        cv2.imwrite(preprocessed_path, quantized_image)

        kmeans_opt_results = self.run_kmeans_optimal()
        sim_kmeans_opt = self.compute_similarity(kmeans_opt_results)
        best_kmeans_opt = self.find_best_matches(kmeans_opt_results)

        kmeans_predef_results = self.run_kmeans_predefined()
        sim_kmeans_predef = self.compute_similarity(kmeans_predef_results)
        best_kmeans_predef = self.find_best_matches(kmeans_predef_results)

        dbscan_results = self.run_dbscan()
        sim_dbscan = self.compute_similarity(dbscan_results)
        best_dbscan = self.find_best_matches(dbscan_results)

        som_opt_results = self.run_som_optimal()
        sim_som_opt = self.compute_similarity(som_opt_results)
        best_som_opt = self.find_best_matches(som_opt_results)

        som_predef_results = self.run_som_predefined()
        sim_som_predef = self.compute_similarity(som_predef_results)
        best_som_predef = self.find_best_matches(som_predef_results)

        logging.info("Segmentation process completed")
        return (
            preprocessed_path,
            (kmeans_opt_results[0], sim_kmeans_opt, best_kmeans_opt),
            (kmeans_predef_results[0], sim_kmeans_predef, best_kmeans_predef),
            (dbscan_results[0], sim_dbscan, best_dbscan),
            (som_opt_results[0], sim_som_opt, best_som_opt),
            (som_predef_results[0], sim_som_predef, best_som_predef)
        )