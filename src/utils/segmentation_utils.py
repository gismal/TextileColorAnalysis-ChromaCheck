import logging
import cv2
import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from minisom import MiniSom

# Exception handler decorator
def exception_handler(func):
    """Decorator to handle exceptions in functions."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {str(e)}")
            return None
    return wrapper

@exception_handler
def k_mean_segmentation(image, k):
    """Segment image using K-means clustering."""
    logging.info(f"Starting K-means segmentation with {k} clusters")
    pixels = image.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()].reshape(image.shape)
    avg_colors = [cv2.mean(image, mask=(labels.reshape(image.shape[:2]) == i).astype(np.uint8))[:3] for i in range(k)]
    logging.info(f"K-means segmentation completed with {k} clusters")
    return segmented_image, avg_colors, labels

@exception_handler
def som_segmentation(image, k):
    """Segment image using Self-Organizing Maps (SOM)."""
    logging.info(f"Starting SOM segmentation with {k} clusters")
    pixels = image.reshape(-1, 3).astype(np.float32) / 255.0
    som = MiniSom(x=1, y=k, input_len=3, sigma=0.5, learning_rate=0.25)
    som.random_weights_init(pixels)
    som.train_random(pixels, 100)
    labels = np.array([som.winner(pixel)[1] for pixel in pixels])
    centers = np.array([som.get_weights()[0, i] for i in range(k)]) * 255
    centers = np.uint8(centers)
    segmented_image = centers[labels].reshape(image.shape)
    avg_colors = [cv2.mean(image, mask=(labels.reshape(image.shape[:2]) == i).astype(np.uint8))[:3] for i in range(k)]
    logging.info(f"SOM segmentation completed with {k} clusters")
    return segmented_image, avg_colors, labels

def optimal_clusters(pixels, default_k, min_k=2, max_k=10, n_runs=3):
    unique_colors = np.unique(pixels, axis=0)
    dynamic_max_k = min(max_k, max(min_k + 2, len(unique_colors) // 20))
    logging.info(f"Unique colors: {len(unique_colors)}. Adjusted max_k: {dynamic_max_k}")

    if len(pixels) > 10000:
        logging.info("Subsampling pixels for efficiency")
        pixels = pixels[np.random.choice(len(pixels), 10000, replace=False)]

    scores = {'silhouette': [], 'ch': []}
    k_range = range(min_k, dynamic_max_k + 1)

    for k in k_range:
        logging.info(f"Testing k={k}")
        kmeans = KMeans(n_clusters=k, n_init=n_runs, random_state=42).fit(pixels)
        labels = kmeans.labels_
        
        scores['silhouette'].append(silhouette_score(pixels, labels))
        scores['ch'].append(calinski_harabasz_score(pixels, labels))
        logging.info(f"Metrics for k={k}: Silhouette={scores['silhouette'][-1]}, CH={scores['ch'][-1]}")

    norm_sil = (np.array(scores['silhouette']) - min(scores['silhouette'])) / (max(scores['silhouette']) - min(scores['silhouette']) + 1e-10)
    norm_ch = (np.array(scores['ch']) - min(scores['ch'])) / (max(scores['ch']) - min(scores['ch']) + 1e-10)
    avg_scores = (norm_sil + norm_ch) / 2
    optimal_k = k_range[np.argmax(avg_scores)]
    logging.info(f"Optimal k determined: {optimal_k}")
    
    return optimal_k

def dbscan_clustering(pixels, eps, min_samples):
    """Cluster pixels using DBSCAN."""
    logging.info(f"Starting DBSCAN Clustering with eps={eps} and min_samples={min_samples}")
    try:
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(pixels)
        labels = db.labels_
    except MemoryError as e:
        logging.error(f"MemoryError during DBSCAN clustering: {str(e)}")
        labels = np.full(len(pixels), -1)
    except Exception as e:
        logging.error(f"Error during DBSCAN clustering: {str(e)}")
        labels = np.full(len(pixels), -1)

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels >= 0])
    logging.info(f"DBSCAN Clustering completed with {n_clusters} clusters")
    return labels

def optimal_clusters_dbscan(pixels, eps_values, min_samples_values):
    """Determine optimal DBSCAN parameters using silhouette score."""
    logging.info("Finding optimal DBSCAN parameters")
    best_eps = eps_values[0]
    best_min_samples = min_samples_values[0]
    best_silhouette = -1
    best_labels = None

    for eps in eps_values:
        for min_samples in min_samples_values:
            labels = dbscan_clustering(pixels, eps, min_samples)
            if len(set(labels)) > 1:
                silhouette_avg = silhouette_score(pixels, labels)
                logging.info(f"DBSCAN with eps={eps}, min_samples={min_samples}, silhouette={silhouette_avg}")
                if silhouette_avg > best_silhouette:
                    best_silhouette = silhouette_avg
                    best_eps = eps
                    best_min_samples = min_samples
                    best_labels = labels

    logging.info(f"Optimal DBSCAN parameters: eps={best_eps}, min_samples={best_min_samples}")
    return best_labels, best_eps, best_min_samples

def optimal_kmeans(image, max_clusters=10):
    """Determine optimal number of clusters for K-means."""
    pixels = image.reshape(-1, 3).astype(np.float32)
    return optimal_clusters(pixels, default_k=3, max_k=max_clusters)

def optimal_dbscan(image):
    """Determine optimal DBSCAN clustering."""
    pixels = image.reshape(-1, 3).astype(np.float32)
    eps_values = [10, 15, 20]
    min_samples_values = [5, 10, 20]
    labels, eps, min_samples = optimal_clusters_dbscan(pixels, eps_values, min_samples_values)
    return labels

def optimal_som(image, max_clusters=10):
    """Determine optimal SOM clustering."""
    pixels = image.reshape(-1, 3).astype(np.float32) / 255.0
    n_clusters = optimal_clusters(pixels, default_k=3, max_k=max_clusters)
    som = MiniSom(x=1, y=n_clusters, input_len=3, sigma=0.5, learning_rate=0.5)
    som.random_weights_init(pixels)
    som.train_random(pixels, 100)
    labels = np.array([som.winner(pixel)[1] for pixel in pixels])
    return labels

def determine_optimal_clusters(image, max_clusters=10):
    """Determine optimal clusters using multiple methods."""
    def safe_cluster(func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {e}")
            return None
    
    best_kmeans_k = safe_cluster(optimal_kmeans, image, max_clusters)
    best_dbscan_labels = safe_cluster(optimal_dbscan, image)
    best_som_labels = safe_cluster(optimal_som, image, max_clusters)
    return {
        'kmeans': best_kmeans_k,
        'dbscan': best_dbscan_labels,
        'som': best_som_labels
    }