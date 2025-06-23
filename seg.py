"""image'i gridlere bölüp öyle avg rengi hesaplıyor. çalışıyor iyi"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def load_image(path):
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Error: Unable to load image at '{path}'. Check file integrity.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def convert_to_lab(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2Lab)

def segment_image(image, num_segments=3):
    height, width, _ = image.shape
    segment_height = height // num_segments
    segment_width = width // num_segments
    segments = []
    for i in range(num_segments):
        for j in range(num_segments):
            start_row = i * segment_height
            end_row = (i + 1) * segment_height if (i + 1) * segment_height < height else height
            start_col = j * segment_width
            end_col = (j + 1) * segment_width if (j + 1) * segment_width < width else width
            segment = image[start_row:end_row, start_col:end_col]
            segments.append(segment)
    return segments

def compute_average_color(image):
    avg_color_per_row = np.mean(image, axis=0)
    avg_color = np.mean(avg_color_per_row, axis=0)
    return avg_color

def visualize_segments(image, segments, avg_colors, num_segments=3):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    height, width, _ = image.shape
    segment_height = height // num_segments
    segment_width = width // num_segments

    for i in range(num_segments):
        for j in range(num_segments):
            start_row = i * segment_height
            start_col = j * segment_width
            rect = Rectangle((start_col, start_row), segment_width, segment_height, 
                             linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            avg_color = avg_colors[i * num_segments + j]
            ax.text(start_col + segment_width / 2, start_row + segment_height / 2, 
                    f'Avg Color: {avg_color.astype(int)}', color='white', ha='center', va='center', fontsize=8, 
                    bbox=dict(facecolor='black', alpha=0.5))

    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main(image_path):
    image = load_image(image_path)
    lab_image = convert_to_lab(image)
    num_segments = 4  # Increase the number of segments for finer analysis
    segments = segment_image(image, num_segments=num_segments)
    avg_colors = [compute_average_color(segment) for segment in segments]
    visualize_segments(image, segments, avg_colors, num_segments=num_segments)

# Path to the input image
image_path = 'C:/Users/LENOVO/Desktop/Thesis/zorlutekstil_1_2024-01-19_1042/1/1/orginal.jpg'

# Run the main function
main(image_path)
