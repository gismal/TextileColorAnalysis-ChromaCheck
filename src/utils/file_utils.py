import os
from datetime import datetime
import cv2

def save_output(dataset_name, method, file_name, content, base_dir="output"):
    """
    Save content to a file in a structured directory: output/dataset_name/method/.
    
    Args:
        dataset_name (str): Name of the dataset (e.g., 'block').
        method (str): Segmentation method (e.g., 'kmeans_optimal').
        file_name (str): Name of the file (e.g., 'segmented_image.png').
        content: The content to save (e.g., image data, DataFrame).
        base_dir (str): Base directory for outputs (default: 'output').
    
    Returns:
        str: Path where the file was saved.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_file_name = f"{timestamp}_{file_name}"
    
    method_dir = os.path.join(base_dir, dataset_name, method)
    os.makedirs(method_dir, exist_ok=True)
    
    file_path = os.path.join(method_dir, unique_file_name)
    
    if file_name.endswith('.png'):
        cv2.imwrite(file_path, content)
    elif file_name.endswith('.csv'):
        content.to_csv(file_path, index=False)
    else:
        with open(file_path, 'w') as f:
            f.write(str(content))
    
    print(f"Saved {file_name} to {file_path}")
    return file_path