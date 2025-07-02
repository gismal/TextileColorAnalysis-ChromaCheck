import logging
import cv2
import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from minisom import MiniSom

# Exception handler decorator
def exception_handler(func):
    """Decorator to handle exceptions in functions.
    
    Args:
        func: Function to wrap.
    
    Returns:
        wrapper: Function that catches and logs exceptions.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {str(e)}")
            return None
    return wrapper

@exception_handler
def k_mean_segmentation(image, k):
    """Segment image using K-means clustering.
    
    Args:
        image (numpy.ndarray): Input image in BGR format.
        k (int): Number of clusters.
    
    Returns:
        tuple: Segmented image, average colors, and labels.
    """
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
    """Segment image using Self-Organizing Maps (SOM).
    
    Args:
        image (numpy.ndarray): Input image in BGR format.
        k (int): Number of clusters.
    
    Returns:
        tuple: Segmented image, average colors, and labels.
    """
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

def optimal_clusters(pixels, default_k, min_k=3, max_k=10, n_runs=10):
    """Determine optimal number of clusters using multiple metrics.
    
    Args:
        pixels (numpy.ndarray): Pixel data in RGB format.
        default_k (int): Default number of clusters if optimization fails.
        min_k (int): Minimum number of clusters.
        max_k (int): Maximum number of clusters.
        n_runs (int): Number of runs for averaging metrics.
    
    Returns:
        int: Optimal number of clusters.
    """
    unique_colors = np.unique(pixels, axis=0)
    dynamic_max_k = min(max_k, len(unique_colors))

    logging.info(f"Number of unique colors: {len(unique_colors)}. Adjusting max_k to: {dynamic_max_k}")

    calinski_harabasz_scores = []
    davies_bouldin_scores = []
    silhouette_scores = []

    logging.info("Calculating optimal number of clusters using multiple metrics")

    try:
        def run_clustering(n_clusters):
            calinski_harabasz_avg = 0
            davies_bouldin_avg = 0
            silhouette_avg = 0

            for _ in range(n_runs):
                kmeans = KMeans(n_clusters=n_clusters, random_state=np.random.randint(0, 10000))
                labels = kmeans.fit_predict(pixels)
                calinski_harabasz_avg += calinski_harabasz_score(pixels, labels) / n_runs
                davies_bouldin_avg += davies_bouldin_score(pixels, labels) / n_runs
                silhouette_avg += silhouette_score(pixels, labels) / n_runs

            return calinski_harabasz_avg, davies_bouldin_avg, silhouette_avg

        results = Parallel(n_jobs=-1)(delayed(run_clustering)(n_clusters) for n_clusters in range(min_k, dynamic_max_k + 1))

        for ch_score, db_score, si_score in results:
            calinski_harabasz_scores.append(ch_score)
            davies_bouldin_scores.append(db_score)
            silhouette_scores.append(si_score)

        optimal_k_ch = calinski_harabasz_scores.index(max(calinski_harabasz_scores)) + min_k
        optimal_k_db = davies_bouldin_scores.index(min(davies_bouldin_scores)) + min_k
        optimal_k_si = silhouette_scores.index(max(silhouette_scores)) + min_k

        # Aggregate the three optimal k values
        aggregated_k = round((optimal_k_ch + optimal_k_db + optimal_k_si) / 3)
        aggregated_k = min(max(aggregated_k, min_k), dynamic_max_k)

        logging.info(f"Optimal number of clusters determined by Calinski-Harabasz: {optimal_k_ch}")
        logging.info(f"Optimal number of clusters determined by Davies-Bouldin: {optimal_k_db}")
        logging.info(f"Optimal number of clusters determined by Silhouette Score: {optimal_k_si}")
        logging.info(f"Final aggregated optimal number of clusters: {aggregated_k}")

        return aggregated_k

    except Exception as e:
        logging.error(f"Error in calculating optimal clusters: {str(e)}. Falling back to default k: {default_k}")
        return default_k

def dbscan_clustering(pixels, eps, min_samples):
    """Cluster pixels using DBSCAN.
    
    Args:
        pixels (numpy.ndarray): Pixel data in RGB format.
        eps (float): Maximum distance between two samples.
        min_samples (int): Minimum number of samples in a cluster.
    
    Returns:
        numpy.ndarray: Cluster labels.
    """
    logging.info(f"Starting DBSCAN Clustering with eps={eps} and min_samples={min_samples}")
    try:
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(pixels)
        labels = db.labels_
    except MemoryError as e:
        logging.error(f"MemoryError during DBSCAN clustering: {str(e)}")
        labels = np.full(len(pixels), -1)  # Assign all points as noise in case of memory error
    except Exception as e:
        logging.error(f"Error during DBSCAN clustering: {str(e)}")
        labels = np.full(len(pixels), -1)  # Assign all points as noise in case of other errors

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels >= 0])  # Exclude noise if present.

    logging.info(f"DBSCAN Clustering completed with {n_clusters} clusters")
    return labels

def optimal_clusters_dbscan(pixels, eps_values, min_samples_values):
    """Determine optimal DBSCAN parameters using silhouette score.
    
    Args:
        pixels (numpy.ndarray): Pixel data in RGB format.
        eps_values (list): List of epsilon values to test.
        min_samples_values (list): List of min_samples values to test.
    
    Returns:
        tuple: Best labels, best eps, best min_samples.
    """
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

# Define optimal clustering functions
def optimal_kmeans(image, max_clusters=10):
    """Determine optimal number of clusters for K-means.
    
    Args:
        image (numpy.ndarray): Input image in BGR format.
        max_clusters (int): Maximum number of clusters to consider.
    
    Returns:
        int: Optimal number of clusters.
    """
    pixels = image.reshape(-1, 3).astype(np.float32)
    return optimal_clusters(pixels, default_k=3, max_k=max_clusters)

def optimal_dbscan(image):
    """Determine optimal DBSCAN clustering.
    
    Args:
        image (numpy.ndarray): Input image in BGR format.
    
    Returns:
        numpy.ndarray: Optimal cluster labels.
    """
    pixels = image.reshape(-1, 3).astype(np.float32)
    eps_values = [10, 15, 20]
    min_samples_values = [5, 10, 20]
    labels, eps, min_samples = optimal_clusters_dbscan(pixels, eps_values, min_samples_values)
    return labels

def optimal_som(image, max_clusters=10):
    """Determine optimal SOM clustering.
    
    Args:
        image (numpy.ndarray): Input image in BGR format.
        max_clusters (int): Maximum number of clusters to consider.
    
    Returns:
        numpy.ndarray: Optimal cluster labels.
    """
    pixels = image.reshape(-1, 3).astype(np.float32) / 255.0
    n_clusters = optimal_clusters(pixels, default_k=3, max_k=max_clusters)
    som = MiniSom(x=1, y=n_clusters, input_len=3, sigma=0.5, learning_rate=0.5)
    som.random_weights_init(pixels)
    som.train_random(pixels, 100)
    labels = np.array([som.winner(pixel)[1] for pixel in pixels])
    return labels

def determine_optimal_clusters(image, max_clusters=10):
    """Determine optimal clusters using multiple methods.
    
    Args:
        image (numpy.ndarray): Input image in BGR format.
        max_clusters (int): Maximum number of clusters to consider.
    
    Returns:
        dict: Dictionary with optimal labels for K-means, DBSCAN, and SOM.
    """
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