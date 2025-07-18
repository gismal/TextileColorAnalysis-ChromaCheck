import logging
import cv2
import numpy as np
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def exception_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {str(e)}")
            return None
    return wrapper

class Preprocessor:
    def __init__(self, initial_resize=512, target_size=(128, 128), denoise_h=10, max_colors=8, edge_enhance=False, unsharp_amount=0.0, unsharp_threshold=0):
        """Initialize the preprocessor with configurable parameters."""
        self.initial_resize = initial_resize
        self.target_size = target_size
        self.denoise_h = denoise_h  # Parameter for non-local means denoising
        self.max_colors = max_colors  # Maximum number of colors to allow
        self.edge_enhance = edge_enhance
        self.unsharp_amount = unsharp_amount  # Disabled by default
        self.unsharp_threshold = unsharp_threshold

    def estimate_n_colors(self, image):
        """Estimate the number of colors based on unique pixel values."""
        unique_colors = len(np.unique(image.reshape(-1, 3), axis=0))
        # Set n_colors to 1.5x unique colors, capped at max_colors
        n_colors = min(int(unique_colors * 1.5), self.max_colors)
        logging.info(f"Estimated {unique_colors} unique colors, setting n_colors to {n_colors}")
        return max(n_colors, 2)  # Ensure at least 2 colors

    @exception_handler
    def quantize_image(self, image):
        """Quantize the image to a dynamically estimated number of colors using K-means."""
        n_colors = self.estimate_n_colors(image)
        pixels = image.reshape(-1, 3)  # Flatten to (n_pixels, 3) for clustering
        if pixels.shape[0] > 10000:  # Subsample for efficiency
            indices = np.random.choice(pixels.shape[0], 10000, replace=False)
            pixels = pixels[indices]
        kmeans = KMeans(n_clusters=n_colors, n_init=3, random_state=42).fit(pixels)
        labels = kmeans.predict(image.reshape(-1, 3))
        quantized = kmeans.cluster_centers_[labels].reshape(image.shape).astype(np.uint8)
        logging.info(f"Quantized to {n_colors} colors, unique colors: {len(np.unique(quantized.reshape(-1, 3), axis=0))}")
        return quantized

    @exception_handler
    def unsharp_mask(self, image):
        """Apply unsharp masking to enhance image details (disabled by default)."""
        if self.unsharp_amount > 0:
            blurred = cv2.GaussianBlur(image, (5, 5), 1.0)
            mask = cv2.subtract(image, blurred)
            sharpened = cv2.addWeighted(image, 1.0 + self.unsharp_amount, mask, -self.unsharp_amount, 0)
            if self.unsharp_threshold > 0:
                low_contrast_mask = np.absolute(mask) < self.unsharp_threshold
                np.copyto(sharpened, image, where=low_contrast_mask)
            logging.info("Applied unsharp masking")
            return sharpened
        return image

    @exception_handler
    def preprocess(self, img):
        """Preprocess the image with denoising, quantization, and resizing."""
        logging.info("Starting preprocessing")
        
        # Initial resize to reduce computation
        scale_factor = self.initial_resize / min(img.shape[:2])
        resized = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        logging.info(f"Initial resize to approx {self.initial_resize}x{self.initial_resize}")

        # Denoise with non-local means (replaces bilateral filter)
        denoised = cv2.fastNlMeansDenoisingColored(resized, None, h=self.denoise_h, templateWindowSize=7, searchWindowSize=21)
        logging.info(f"Applied non-local means denoising with h={self.denoise_h}")

        # Apply unsharp masking (optional)
        sharpened_image = self.unsharp_mask(denoised)
        
        # Quantize colors dynamically
        quantized_image = self.quantize_image(sharpened_image)
        
        # Final resize to target size
        final_image = cv2.resize(quantized_image, self.target_size, interpolation=cv2.INTER_AREA)
        logging.info(f"Resized to {self.target_size[0]}x{self.target_size[1]}")

        logging.info("Preprocessing completed")
        return final_image
