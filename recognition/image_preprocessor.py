import cv2
import matplotlib.pyplot as plt

class ImagePreprocessor:
    @staticmethod
    def preprocess_for_alignment(image):
        if image is None:
            raise ValueError("Invalid image provided for preprocessing.")
        try:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            equalized_image = cv2.equalizeHist(gray_image)
            bilateral_filtered_image = cv2.bilateralFilter(equalized_image, 9, 75, 75)
            blurred_image = cv2.GaussianBlur(bilateral_filtered_image, (5, 5), 0)
            laplacian_edges = cv2.Laplacian(blurred_image, cv2.CV_64F, ksize=3)
            laplacian_edges = cv2.convertScaleAbs(laplacian_edges)
            sobelx = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
            sobel_edges = cv2.magnitude(sobelx, sobely)
            sobel_edges = cv2.convertScaleAbs(sobel_edges)
            combined_edges = cv2.bitwise_or(laplacian_edges, sobel_edges)
            adaptive_thresh = cv2.adaptiveThreshold(combined_edges, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
             # Visualization
            titles = ['Gray Image', 'Equalized Image', 'Bilateral Filtered Image', 'Blurred Image', 
                    'Laplacian Edges', 'Sobel Edges', 'Combined Edges', 'Adaptive Threshold']
            images = [gray_image, equalized_image, bilateral_filtered_image, blurred_image, 
                    laplacian_edges, sobel_edges, combined_edges, adaptive_thresh]
            
            plt.figure(figsize=(15, 10))
            for i in range(len(images)):
                plt.subplot(2, 4, i+1)
                plt.imshow(images[i], cmap='gray')
                plt.title(titles[i])
                plt.axis('off')
            plt.show()
            
            return adaptive_thresh
        except Exception as e:
            print(f"Error during image preprocessing: {e}")
            return None

