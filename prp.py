import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

def load_image(path):
    image = cv2.imread(path)
    if image is None:
        print(f"Error: Unable to load image at '{path}'. Check file integrity.")
        return None
    print(f"Loaded image '{path}' successfully.")
    return image

def resize_image(image, width):
    height = int(image.shape[0] * width / image.shape[1])
    return cv2.resize(image, (width, height))

def preprocess_image_for_alignment(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized_image = cv2.equalizeHist(gray_image)
    bilateral_filtered_image = cv2.bilateralFilter(equalized_image, 9, 75, 75)
    blurred_image = cv2.GaussianBlur(bilateral_filtered_image, (5, 5), 0)
    return blurred_image

def preprocess(img):
    blurred = cv2.GaussianBlur(img, (7,7), 0)
    return blurred

def k_mean_segmentation(image, k=10):
    pixels = image.reshape(-1, 3)
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()].reshape(image.shape)
    return segmented_image, centers, labels.reshape(image.shape[:2])

def calculate_color_difference(color1, color2):
    # Using Euclidean distance for simplicity
    return np.linalg.norm(np.array(color1) - np.array(color2))

def compare_colors(reference_colors, test_colors):
    color_differences = []
    for ref_color, test_color in zip(reference_colors, test_colors):
        diff = calculate_color_difference(ref_color, test_color)
        color_differences.append(diff)
    return color_differences

def analyze_images(reference_path, test_paths):
    reference_img = load_image(reference_path)
    if reference_img is None:
        return

    reference_img = resize_image(reference_img, 800)  # Example width
    reference_img_blurred = preprocess(reference_img)
    ref_segmented_image, ref_centers, ref_labels = k_mean_segmentation(reference_img_blurred)

    plt.imshow(cv2.cvtColor(ref_segmented_image, cv2.COLOR_BGR2RGB))
    plt.title("Reference Image - Segmented")
    plt.axis('off')
    plt.show()

    reference_colors = [np.mean(reference_img[ref_labels == i], axis=0) for i in range(ref_centers.shape[0])]

    for test_path in test_paths:
        test_img = load_image(test_path)
        if test_img is None:
            continue

        test_img = resize_image(test_img, 800)  # Example width
        test_img_blurred = preprocess(test_img)
        test_segmented_image, test_centers, test_labels = k_mean_segmentation(test_img_blurred)

        plt.imshow(cv2.cvtColor(test_segmented_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Test Image - Segmented ({os.path.basename(test_path)})")
        plt.axis('off')
        plt.show()

        test_colors = [np.mean(test_img[test_labels == i], axis=0) for i in range(test_centers.shape[0])]

        color_differences = compare_colors(reference_colors, test_colors)
        print(f"Color differences for {os.path.basename(test_path)}:", color_differences)

if __name__ == "__main__":
# Example usage
    reference_path = 'C:/Users/LENOVO/Desktop/Thesis/zorlutekstil_1_2024-01-19_1042/1/1/orginal.jpg'
    test_paths = [
        'C:/Users/LENOVO/Desktop/Thesis/zorlutekstil_1_2024-01-19_1042/1/1/1.jpg',
        'C:/Users/LENOVO/Desktop/Thesis/zorlutekstil_1_2024-01-19_1042/1/1/2.jpg'
    ]
    analyze_images(reference_path, test_paths)
