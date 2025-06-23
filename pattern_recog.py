import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def load_image(path):
    if not os.path.exists(path):
        print(f"Error: File '{path}' does not exist.")
        return None
    image = cv2.imread(path)
    if image is None:
        print(f"Error: Unable to load image at '{path}'. Check file integrity.")
    else:
        print(f"Loaded image '{path}' successfully.")
    return image

def resize_image(image, width):
    height = int(image.shape[0] * width / image.shape[1])
    return cv2.resize(image, (width, height))

def preprocess_image_for_alignment(image):
    bilateral_filtered_image = cv2.bilateralFilter(image, 9, 75, 75)
    blurred_image = cv2.GaussianBlur(bilateral_filtered_image, (5, 5), 0)
    sobelx = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_edges = cv2.magnitude(sobelx, sobely)
    sobel_edges = cv2.convertScaleAbs(sobel_edges)
    _, sobel_binary = cv2.threshold(sobel_edges, 100, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    morphed_edges = cv2.morphologyEx(sobel_binary, cv2.MORPH_CLOSE, kernel)
    morphed_edges = cv2.morphologyEx(morphed_edges, cv2.MORPH_OPEN, kernel)

    # Visualization of preprocessing steps
    visualize_images([bilateral_filtered_image, blurred_image, sobel_edges, sobel_binary, morphed_edges],
                     ['Bilateral Filtered', 'Blurred Image', 'Sobel Edges', 'Binary Sobel Edges', 'Morphed Edges'])
    
    return morphed_edges

def extract_and_smooth_contours(image, min_area=2000):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for i, cnt in enumerate(contours)
                         if cv2.contourArea(cnt) > min_area and hierarchy[0, i, 3] == -1]
    smoothed_contours = [cv2.approxPolyDP(cnt, 0.03 * cv2.arcLength(cnt, True), True) for cnt in filtered_contours]
    return smoothed_contours

def apply_mask(image, contours):
    mask = np.zeros_like(image)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def orb_feature_matching(img1, img2, max_features=5000):
    orb = cv2.ORB_create(nfeatures=max_features)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        raise ValueError("No descriptors found for one or both images.")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return kp1, kp2, matches

def align_images(img1, img2, kp1, kp2, matches):
    if len(matches) < 4:
        raise ValueError("Not enough matches to compute homography.")
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    height, width = img1.shape[:2]
    aligned_img = cv2.warpPerspective(img2, H, (width, height))
    return aligned_img

def segment_image_by_contours(image, contours):
    segments = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        segment = image[y:y+h, x:x+w]
        segments.append(segment)
    return segments

def compute_average_color(image):
    avg_color_per_row = np.mean(image, axis=0)
    avg_color = np.mean(avg_color_per_row, axis=0)
    return avg_color

def visualize_images(images, titles, figsize=(20, 5)):
    plt.figure(figsize=figsize)
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title, fontsize=10)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_contours(image, contours, title):
    img_contours = image.copy()
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img_contours, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

def visualize_segments(image, segments, title):
    fig, axes = plt.subplots(1, len(segments), figsize=(20, 5))
    for ax, segment in zip(axes, segments):
        ax.imshow(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
        ax.axis('off')
    plt.suptitle(title)
    plt.show()

def visualize_colors(ref_colors, aligned_colors, title):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, colors, label in zip(axes, [ref_colors, aligned_colors], ['Reference', 'Aligned']):
        color_patches = np.zeros((100, 100 * len(colors), 3), dtype=np.uint8)
        for i, color in enumerate(colors):
            color_patches[:, i * 100:(i + 1) * 100] = color
        ax.imshow(color_patches)
        ax.set_title(f'{label} Image Colors')
        ax.axis('off')
    plt.suptitle(title)
    plt.show()

def analyze_prints(reference_path, print_paths, resize_width=800):
    reference_img = load_image(reference_path)
    if reference_img is None:
        print("Error: Reference image could not be loaded.")
        return [], []

    reference_img = resize_image(reference_img, resize_width)

    # Preprocess reference image for contour detection
    ref_preprocessed_for_alignment = preprocess_image_for_alignment(reference_img)
    ref_contours = extract_and_smooth_contours(ref_preprocessed_for_alignment)
    ref_masked_image = apply_mask(reference_img, ref_contours)

    # Visualize contours on the reference image
    visualize_contours(reference_img, ref_contours, 'Reference Image Contours')

    # Segment and visualize the reference image based on contours
    ref_segments = segment_image_by_contours(ref_masked_image, ref_contours)
    visualize_segments(ref_masked_image, ref_segments, 'Reference Image Segments')

    correct_prints = []
    incorrect_prints = []

    for print_path in print_paths:
        print_img = load_image(print_path)
        if print_img is None:
            continue

        print_img = resize_image(print_img, resize_width)

        try:
            kp1, kp2, matches = orb_feature_matching(ref_masked_image, print_img)
            aligned_img = align_images(ref_masked_image, print_img, kp1, kp2, matches)
            visualize_images([reference_img, print_img, aligned_img], ['Reference Image', f'Print Image - {os.path.basename(print_path)}', f'Aligned Image - {os.path.basename(print_path)}'])

            # Extract contours from the aligned image
            aligned_preprocessed_for_alignment = preprocess_image_for_alignment(aligned_img)
            aligned_contours = extract_and_smooth_contours(aligned_preprocessed_for_alignment)
            aligned_masked_image = apply_mask(aligned_img, aligned_contours)

            # Visualize contours on the aligned image
            visualize_contours(aligned_img, aligned_contours, f'Aligned Image Contours - {os.path.basename(print_path)}')

            # Segment and visualize the aligned image based on contours
            aligned_segments = segment_image_by_contours(aligned_masked_image, aligned_contours)
            visualize_segments(aligned_masked_image, aligned_segments, f'Aligned Image Segments - {os.path.basename(print_path)}')

            # Compute average colors for each segment and visualize
            ref_colors = []
            aligned_colors = []
            for ref_seg, aligned_seg in zip(ref_segments, aligned_segments):
                ref_avg_color = compute_average_color(cv2.cvtColor(ref_seg, cv2.COLOR_BGR2Lab))
                aligned_avg_color = compute_average_color(cv2.cvtColor(aligned_seg, cv2.COLOR_BGR2Lab))

                ref_colors.append(ref_avg_color)
                aligned_colors.append(aligned_avg_color)
                
                print(f"Reference segment average color in CIELAB: {ref_avg_color}")
                print(f"Aligned segment average color in CIELAB: {aligned_avg_color}")

            visualize_colors(ref_colors, aligned_colors, f'Average Colors of Segments - {os.path.basename(print_path)}')

            correct_prints.append(print_path)
        except ValueError as e:
            print(f"Error aligning print '{print_path}': {e}")
            incorrect_prints.append(print_path)

    return correct_prints, incorrect_prints

# Example usage
reference_path = 'C:/Users/LENOVO/Desktop/Thesis/zorlutekstil_1_2024-01-19_1042/1/1/orginal.jpg'
print_paths = [
    'C:/Users/LENOVO/Desktop/Thesis/zorlutekstil_1_2024-01-19_1042/1/1/1.jpg',
    'C:/Users/LENOVO/Desktop/Thesis/zorlutekstil_1_2024-01-19_1042/1/1/2.jpg'
]

correct_prints, incorrect_prints = analyze_prints(reference_path, print_paths)
print("Correct Prints:", correct_prints)
print("Incorrect Prints:", incorrect_prints)
