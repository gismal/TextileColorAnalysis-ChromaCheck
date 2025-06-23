import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_image(image_path):
    return cv2.imread(image_path)

def preprocess_image(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return lab_image, gray_image

def equalize_and_filter_image(gray_image):
    equalized_image = cv2.equalizeHist(gray_image)
    bilateral_filtered_image = cv2.bilateralFilter(equalized_image, 9, 75, 75)
    blurred_image = cv2.GaussianBlur(bilateral_filtered_image, (5, 5), 0)
    return blurred_image

def detect_edges(blurred_image):
    edges = cv2.Canny(blurred_image, 50, 150)
    return edges

def clean_edges(edges):
    kernel = np.ones((3, 3), np.uint8)
    morphed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    morphed_edges = cv2.morphologyEx(morphed_edges, cv2.MORPH_OPEN, kernel)
    morphed_edges = cv2.dilate(morphed_edges, kernel, iterations=3)
    morphed_edges = cv2.erode(morphed_edges, kernel, iterations=1)
    return morphed_edges

def find_and_filter_contours(morphed_edges, gray_image):
    contours, hierarchy = cv2.findContours(morphed_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [
        cnt for i, cnt in enumerate(contours)
        if cv2.contourArea(cnt) > 3000  # Slightly increased area threshold
    ]
    smoothed_contours = [cv2.approxPolyDP(cnt, 3, True) for cnt in filtered_contours]

    mask = np.zeros_like(gray_image)
    cv2.drawContours(mask, smoothed_contours, -1, 255, thickness=cv2.FILLED)
    
    return smoothed_contours, mask

def analyze_segments(lab_image, smoothed_contours, mask):
    contour_features = []
    for i, cnt in enumerate(smoothed_contours):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        moments = cv2.moments(cnt)
        hu_moments = cv2.HuMoments(moments).flatten()
        
        individual_mask = np.zeros_like(mask)
        cv2.drawContours(individual_mask, [cnt], -1, 255, thickness=cv2.FILLED)
        
        mean_color = cv2.mean(lab_image, mask=individual_mask)[:3]
        std_color = cv2.meanStdDev(lab_image, mask=individual_mask)
        std_color = std_color[1].flatten()[:3]
        
        contour_features.append({
            'contour_id': i,
            'area': area,
            'perimeter': perimeter,
            'mean_color_L': mean_color[0],
            'mean_color_A': mean_color[1],
            'mean_color_B': mean_color[2],
            'std_color_L': std_color[0],
            'std_color_A': std_color[1],
            'std_color_B': std_color[2],
            'hu_moments': hu_moments.tolist()
        })
    
    return pd.DataFrame(contour_features)

def display_images(image, mask, masked_lab_image):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    plt.subplot(1, 3, 2)
    plt.title('Refined Mask')
    plt.imshow(mask, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title('Masked LAB Image')
    plt.imshow(cv2.cvtColor(masked_lab_image, cv2.COLOR_LAB2RGB))

    plt.show()

def main(image_path):
    image = load_image(image_path)
    lab_image, gray_image = preprocess_image(image)
    blurred_image = equalize_and_filter_image(gray_image)
    edges = detect_edges(blurred_image)
    morphed_edges = clean_edges(edges)
    smoothed_contours, mask = find_and_filter_contours(morphed_edges, gray_image)
    masked_lab_image = cv2.bitwise_and(lab_image, lab_image, mask=mask)
    df_contour_features = analyze_segments(lab_image, smoothed_contours, mask)
    display_images(image, mask, masked_lab_image)
    return df_contour_features

image_path = 'C:/Users/LENOVO/Desktop/Thesis/zorlutekstil_1_2024-01-19_1042/1/1/orginal.jpg'

