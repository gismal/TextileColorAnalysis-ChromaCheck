import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

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

def orb_feature_matching(img1, img2, max_features=5000):
    orb = cv2.ORB_create(nfeatures=max_features)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    if des1 is None or des2 is None:
        raise ValueError("No descriptors found for one or both images.")
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    good_matches = matches[:50]  # Taking top 50 matches
    print(f"Found {len(good_matches)} good matches between the images using ORB.")
    return kp1, kp2, good_matches

def validate_homography(H):
    if H is None or H.shape != (3, 3):
        print("Homography matrix is not valid.")
        return False
    # Check the determinant to avoid singular matrices
    if np.linalg.det(H) < 1e-10:
        print("Homography matrix determinant is too small.")
        return False
    # Check for large translations and scaling factors
    if np.any(np.abs(H[:2, :2]) > 1e3) or np.any(np.abs(H[:2, 2]) > 1e4):
        print("Homography matrix has large translation or scaling factors.")
        return False
    return True

def align_images(img1, img2, kp1, kp2, matches):
    if len(matches) < 4:
        print("Not enough matches to compute homography.")
        return None, None
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if not validate_homography(H):
        return None, None
    print(f"Homography matrix:\n{H}")
    height, width = img1.shape[:2]
    aligned_img = cv2.warpPerspective(img2, H, (width, height))
    print(f"Aligned image using homography matrix.")
    return aligned_img, H

def visualize_keypoints(img, keypoints, title):
    img_with_kp = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(cv2.cvtColor(img_with_kp, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()
    plt.close()

def visualize_matches(img1, kp1, img2, kp2, matches, title):
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()
    plt.close()

def visualize_homography(reference_path, print_paths):
    reference_img = load_image(reference_path)
    if reference_img is None:
        print("Error: Reference image could not be loaded.")
        return

    for print_path in print_paths:
        print_img = load_image(print_path)
        if print_img is None:
            continue

        try:
            kp1, kp2, matches = orb_feature_matching(reference_img, print_img)
        except ValueError as e:
            print(f"Error during feature matching: {e}")
            continue

        aligned_img, H = align_images(reference_img, print_img, kp1, kp2, matches)

        # Create a figure to visualize the images and matches
        plt.figure(figsize=(20, 10))

        # Display the reference image
        plt.subplot(1, 4, 1)
        plt.imshow(cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB))
        plt.title('Reference Image')
        plt.axis('off')

        # Display the example image
        plt.subplot(1, 4, 2)
        plt.imshow(cv2.cvtColor(print_img, cv2.COLOR_BGR2RGB))
        plt.title('Example Image')
        plt.axis('off')

        # Display the matches
        img_matches = cv2.drawMatches(reference_img, kp1, print_img, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.subplot(1, 4, 3)
        plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        plt.title('Matches')
        plt.axis('off')

        # Display the aligned image
        if aligned_img is not None:
            plt.subplot(1, 4, 4)
            plt.imshow(cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB))
            plt.title('Aligned Image')
            plt.axis('off')

        plt.tight_layout()
        plt.show()
        plt.close()

# Paths to the provided images
reference_path = 'C:/Users/LENOVO/Desktop/Thesis/zorlutekstil_1_2024-01-19_1042/2/orginal.95.jpg'
print_paths = [
    'C:/Users/LENOVO/Desktop/Thesis/zorlutekstil_1_2024-01-19_1042/2/7-2.jpg' ,
    'C:/Users/LENOVO/Desktop/Thesis/zorlutekstil_1_2024-01-19_1042/2/60-1.jpg' 

]




# Visualize homography
visualize_homography(reference_path, print_paths)


# Example usage
src_image_path = r'C:/Users/LENOVO/Desktop/Thesis/zorlutekstil_1_2024-01-19_1042/1/1/1.jpg'  # Image to be aligned
ref_image_path = 'C:/Users/LENOVO/Desktop/Thesis/zorlutekstil_1_2024-01-19_1042/1/1/orginal.jpg'  # Reference image

reference_path = 'C:/Users/LENOVO/Desktop/Thesis/zorlutekstil_1_2024-01-19_1042/2/orginal.95.jpg'
print_paths = [
    'C:/Users/LENOVO/Desktop/Thesis/zorlutekstil_1_2024-01-19_1042/2/7-2.jpg' ,
    'C:/Users/LENOVO/Desktop/Thesis/zorlutekstil_1_2024-01-19_1042/2/60-1.jpg' 

]



# Paths to the provided images
image_path = 'C:/Users/LENOVO/Desktop/Thesis/wetransfer_jpg_2024-04-02_1007/OK.jpg'
print_paths = [
    'C:/Users/LENOVO/Desktop/Thesis/wetransfer_jpg_2024-04-02_1007/1.jpg',
    'C:/Users/LENOVO/Desktop/Thesis/wetransfer_jpg_2024-04-02_1007/2.jpg',
    'C:/Users/LENOVO/Desktop/Thesis/wetransfer_jpg_2024-04-02_1007/3.jpg',
    'C:/Users/LENOVO/Desktop/Thesis/wetransfer_jpg_2024-04-02_1007/4.jpg',
    'C:/Users/LENOVO/Desktop/Thesis/wetransfer_jpg_2024-04-02_1007/5.jpg'
]


'''
Sure, let's break down the visualization to understand what each part represents:

    First Row:
        The first row shows the whole reference image (on the left) and the whole aligned image (on the right).
        The titles of these images provide information about the reference image, the aligned image, and the correlation between them.

    Second Row:
        The second row contains the segments of the reference image. Each segment is a smaller part of the reference image, divided to allow detailed comparison with corresponding segments from the aligned image.
        Each segment is titled with its segment number and the average color of that segment in RGB format.

    Third Row:
        The third row contains the segments of the aligned image that correspond to the reference image segments shown in the second row.
        If the segment was successfully aligned, it shows the aligned segment with its average color in RGB format.
        If the segment could not be aligned due to insufficient matches or errors in homography computation, it shows a black image with the title "Not Aligned."

Explanation of "Not Aligned":

    "Not Aligned" means that the algorithm could not find enough matching features between the reference segment and the corresponding segment in the print image to compute a valid homography matrix. As a result, it was unable to align that segment of the print image with the reference segment.

Summary of the Visualization:

    Whole Images:
        Top-left: Reference Image.
        Top-right: Aligned Image with the correlation value.
    Reference Segments:
        Middle row: Segments of the reference image with their average colors.
    Aligned Segments:
        Bottom row: Corresponding segments of the aligned image or a black image if the segment was not aligned, along with their average colors.

        GPT

Certainly! Let's break down what is shown in the visualization and the significance of each part.
Visualization Breakdown:

    First Row: Whole Images
        Reference Image: The first plot on the left in the first row is the entire reference image.
        Aligned Image: The second plot on the right in the first row is the entire image from the print set that has been aligned to the reference image. The title includes the file path of the aligned image and the correlation score between the reference and aligned images.

    Second Row: Segments of the Reference Image
        Segments 1 to 9: These plots show the segments of the reference image. Each segment is extracted from the reference image to allow for detailed, segment-by-segment comparison.
        Average Color: Each segment title shows the segment number and the average color of that segment in RGB format.

    Third Row: Segments of the Aligned Image
        Segments 1 to 9: These plots show the segments of the aligned image. Each segment corresponds to the same region as the reference image segments.
        Not Aligned: If the homography (alignment) process did not produce a valid result for a segment, it will show a black square and "Not Aligned" in the title.
        Average Color: For successfully aligned segments, the title shows the segment number and the average color of that segment in RGB format. For segments that were not aligned, "Not Aligned" is displayed.

Explanation of "Not Aligned":

    Homography Process: This process involves finding the transformation matrix that maps points from the reference image to the corresponding points in the print image. If there are not enough good matches between features in a segment, the homography process cannot compute a valid transformation.
    Black Segments: When a segment could not be aligned, it is represented by a black square in the visualization to indicate failure in alignment.
    Reasons for Failure: Reasons could include lack of distinctive features in that segment, poor quality of the print image, or significant differences between the reference and print images.

Summary:

    First Row: Shows the overall reference image and the aligned print image.
    Second Row: Shows the segments of the reference image with their average colors.
    Third Row: Shows the corresponding segments of the aligned image. Successfully aligned segments display the actual content and their average colors, while segments that could not be aligned display a black square with "Not Aligned."
'''