import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import metrics

def load_and_resize_images(ref_image_path, sample_image_paths, width):
    ref_image = cv2.imread(ref_image_path)
    ref_image = cv2.resize(ref_image, (width, int(ref_image.shape[0] * width / ref_image.shape[1])))

    sample_images = []
    for path in sample_image_paths:
        img = cv2.imread(path)
        img = cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])))
        sample_images.append(img)
    return ref_image, sample_images

def convert_to_cielab(images):
    return [cv2.cvtColor(image, cv2.COLOR_BGR2LAB) for image in images]

def align_images(ref_image, sample_images, method='SIFT'):
    aligned_images = []
    if method == 'SIFT':
        feature_detector = cv2.SIFT_create()
    elif method == 'ORB':
        feature_detector = cv2.ORB_create()
    else:
        raise ValueError("Unsupported method. Use 'SIFT' or 'ORB'.")

    ref_kp, ref_desc = feature_detector.detectAndCompute(ref_image, None)

    for img in sample_images:
        kp, desc = feature_detector.detectAndCompute(img, None)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(ref_desc, desc, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) > 4:
            src_pts = np.float32([ref_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            if M is not None and M.shape == (3, 3):
                height, width, channels = ref_image.shape
                aligned_image = cv2.warpPerspective(img, M, (width, height))
                aligned_images.append(aligned_image)
            else:
                print(f"Homography matrix for image is None or not 3x3.")
                aligned_images.append(None)
        else:
            print(f"Not enough good matches for image. Number of good matches: {len(good_matches)}")
            aligned_images.append(None)
    return aligned_images

def calculate_delta_e(ref_image, aligned_images):
    delta_e_values = []
    ref_lab = cv2.cvtColor(ref_image, cv2.COLOR_BGR2LAB)
    for img in aligned_images:
        if img is None:
            delta_e_values.append(None)
            continue
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        delta_e = np.mean([np.linalg.norm(ref_lab[i, j] - lab[i, j]) for i in range(ref_lab.shape[0]) for j in range(ref_lab.shape[1])])
        delta_e_values.append(delta_e)
    return delta_e_values

def calculate_ssim(ref_image, aligned_images):
    ssim_values = []
    ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    for img in aligned_images:
        if img is None:
            ssim_values.append(None)
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ssim = metrics.structural_similarity(ref_gray, gray)
        ssim_values.append(ssim)
    return ssim_values

def segment_and_analyze(images, segments):
    segment_means = []
    segmented_images = []
    height, width = images[0].shape[:2]
    seg_h = height // segments
    seg_w = width // segments
    for img in images:
        if img is None:
            segment_means.append(None)
            segmented_images.append(None)
            continue
        segments_means = []
        segmented_image = img.copy()
        for i in range(segments):
            for j in range(segments):
                segment = img[i*seg_h:(i+1)*seg_h, j*seg_w:(j+1)*seg_w]
                mean_color = cv2.mean(segment)[:3]
                segments_means.append(mean_color)
                cv2.rectangle(segmented_image, (j*seg_w, i*seg_h), ((j+1)*seg_w, (i+1)*seg_h), mean_color[::-1], -1)
        segment_means.append(segments_means)
        segmented_images.append(segmented_image)
    return segment_means, segmented_images

def visualize_segmented_images(ref_image, segmented_images):
    num_images = len(segmented_images)
    fig, axes = plt.subplots(1, num_images + 1, figsize=(20, 5))

    # Display reference image
    axes[0].imshow(cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Reference Image')
    axes[0].axis('off')

    for i, img in enumerate(segmented_images):
        if img is None:
            continue
        axes[i+1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i+1].set_title(f'Segmented Sample {i+1}')
        axes[i+1].axis('off')

    plt.tight_layout()
    plt.show()

def visualize_results(ref_image, aligned_images, delta_e_values, ssim_values, segments_means):
    num_images = len(aligned_images)
    fig, axes = plt.subplots(3, num_images + 1, figsize=(20, 15))

    # Display reference image
    axes[0, 0].imshow(cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Reference Image')
    axes[0, 0].axis('off')

    for i, (img, delta_e, ssim, seg_means) in enumerate(zip(aligned_images, delta_e_values, ssim_values, segments_means)):
        if img is None:
            continue

        # Display aligned sample image
        axes[0, i+1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, i+1].set_title(f'Sample {i+1}\nDelta E: {delta_e:.2f}, SSIM: {ssim:.2f}')
        axes[0, i+1].axis('off')

        # Display segment means in the second and third rows
        for j, mean_color in enumerate(seg_means):
            seg_color = np.ones((50, 50, 3), dtype=np.uint8)
            seg_color[:, :] = mean_color[::-1]
            row = 1 + j // segments
            col = (i + 1) % (num_images + 1)
            if row < 3 and col < (num_images + 1):
                axes[row, col].imshow(seg_color)
                axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()

# Example usage
ref_image_path = 'C:/Users/LENOVO/Desktop/Thesis/zorlutekstil_1_2024-01-19_1042/1/1/orginal.jpg'
sample_image_paths = [
    'C:/Users/LENOVO/Desktop/Thesis/zorlutekstil_1_2024-01-19_1042/1/1/1.jpg',
    'C:/Users/LENOVO/Desktop/Thesis/zorlutekstil_1_2024-01-19_1042/1/1/2.jpg'
]
width = 800
segments = 4

ref_image, sample_images = load_and_resize_images(ref_image_path, sample_image_paths, width)
ref_image_lab = convert_to_cielab([ref_image])[0]
sample_images_lab = convert_to_cielab(sample_images)

aligned_images = align_images(ref_image, sample_images, method='SIFT')

delta_e_values = calculate_delta_e(ref_image, aligned_images)
ssim_values = calculate_ssim(ref_image, aligned_images)

segment_means, segmented_images = segment_and_analyze(aligned_images, segments)

# Visualize the segmented images
visualize_segmented_images(ref_image, segmented_images)

# Visualize the results
visualize_results(ref_image, aligned_images, delta_e_values, ssim_values, segment_means)
