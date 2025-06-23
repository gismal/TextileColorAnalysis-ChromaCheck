import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def load_image(path):
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Error: Unable to load image at '{path}'. Check file integrity.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def convert_to_lab(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Using adaptive thresholding to handle different lighting conditions
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return thresh

def detect_contours(thresh, min_area=500):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    return detected_contours

def compute_average_color_lab(image, mask):
    lab_image = convert_to_lab(image)
    masked_lab = cv2.bitwise_and(lab_image, lab_image, mask=mask)
    avg_color_per_row = np.mean(masked_lab[mask == 255], axis=0)
    return avg_color_per_row

def cluster_contours_by_color(avg_colors, n_clusters):
    if len(avg_colors) < n_clusters:
        n_clusters = len(avg_colors)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(avg_colors)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    return labels, cluster_centers

def visualize_clusters(image, contours, labels, cluster_centers):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    unique_labels = np.unique(labels)
    for cluster_idx in unique_labels:
        clustered_contours = [contour for i, contour in enumerate(contours) if labels[i] == cluster_idx]
        for contour in clustered_contours:
            cv2.drawContours(image, [contour], -1, (255, 0, 0), 2)
        avg_color = cluster_centers[cluster_idx]
        x, y, w, h = cv2.boundingRect(np.vstack(clustered_contours))
        ax.text(x + w / 2, y + h / 2, f'Cluster {cluster_idx} Avg Color: {avg_color.astype(int)}', color='white', ha='center', va='center', fontsize=8, 
                bbox=dict(facecolor='black', alpha=0.5))
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_avg_colors(cluster_centers):
    fig, ax = plt.subplots(figsize=(10, 2))
    for idx, color in enumerate(cluster_centers):
        rect = plt.Rectangle((idx, 0), 1, 1, color=np.array(color) / 255.0)
        ax.add_patch(rect)
        ax.text(idx + 0.5, 0.5, f'{color.astype(int)}', color='white', ha='center', va='center', fontsize=8, 
                bbox=dict(facecolor='black', alpha=0.5))
    ax.set_xlim(0, len(cluster_centers))
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

def main(image_path, n_clusters=5):
    image = load_image(image_path)
    thresh = preprocess_image(image)
    contours = detect_contours(thresh)
    avg_colors = []
    for contour in contours:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        avg_color = compute_average_color_lab(image, mask)
        avg_colors.append(avg_color)
    avg_colors = np.array(avg_colors)
    labels, cluster_centers = cluster_contours_by_color(avg_colors, n_clusters)
    visualize_clusters(image, contours, labels, cluster_centers)
    visualize_avg_colors(cluster_centers)

# Path to the input image
image_path = 'C:/Users/LENOVO/Desktop/Thesis/zorlutekstil_1_2024-01-19_1042/1/1/orginal.jpg'

# Run the main function
main(image_path, n_clusters=5)
