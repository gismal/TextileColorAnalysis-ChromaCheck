import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def load_image(path):
    """Load an image from the given path."""
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"File '{path}' not found.")
    return image

def convert_to_cielab(image):
    """Convert BGR image to CIELAB color space."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

def reshape_image_for_clustering(image):
    """Reshape image to a 2D array of pixels for clustering."""
    return image.reshape((-1, 3))

def apply_kmeans_clustering(image, num_clusters=3):
    """Apply KMeans clustering to the image data."""
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(image)
    return kmeans

def create_clustered_image(kmeans, image_shape):
    """Create an image with clustered colors."""
    labels = kmeans.labels_
    clustered_image = kmeans.cluster_centers_[labels]
    clustered_image = clustered_image.reshape(image_shape)
    clustered_image = np.clip(clustered_image, 0, 255).astype(np.uint8)
    return clustered_image

def convert_lab_to_bgr(clustered_image, original_shape):
    """Convert the clustered image from CIELAB to BGR for visualization."""
    clustered_image = clustered_image.reshape(original_shape)
    clustered_image_bgr = cv2.cvtColor(clustered_image, cv2.COLOR_Lab2BGR)
    return clustered_image_bgr

def visualize_clusters(original_image, clustered_image_bgr):
    """Display the original and clustered images."""
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(clustered_image_bgr, cv2.COLOR_BGR2RGB))
    plt.title('Clustered Image in CIELAB')
    
    plt.show()
def main():
    image_path = 'C:/Users/LENOVO/Desktop/Thesis/zorlutekstil_1_2024-01-19_1042/1/1/orginal.jpg'  # Replace with your image path
    num_clusters = 5  # Number of color clusters

    # Load and process the image
    image = load_image(image_path)
    lab_image = convert_to_cielab(image)
    reshaped_image = reshape_image_for_clustering(lab_image)

    # Apply KMeans clustering
    kmeans = apply_kmeans_clustering(reshaped_image, num_clusters)

    # Create the clustered image
    clustered_image = create_clustered_image(kmeans, image.shape)

    # Convert clustered image back to BGR for proper visualization
    clustered_image_bgr = cv2.cvtColor(clustered_image, cv2.COLOR_Lab2BGR)

    # Visualize the original and clustered images
    visualize_clusters(image, clustered_image_bgr)

if __name__ == "__main__":
    main()
