import os
import cv2
import pandas as pd
import logging
from pathlib import Path

class OutputManager:
    def __init__(self, base_dir, dataset_name):
        """Initialize OutputManager with base directory and dataset name."""
        self.base_dir = Path(base_dir)
        self.dataset_dir = self.base_dir / 'datasets' / dataset_name
        self.inputs_dir = self.dataset_dir / 'inputs'
        self.test_images_dir = self.inputs_dir / 'test_images'
        self.processed_dir = self.dataset_dir / 'processed'
        self.summaries_dir = self.dataset_dir / 'summaries'
        self.analysis_dir = self.base_dir / 'analysis'
        self.delta_e_dir = self.analysis_dir / 'delta_e'

        # Create directories
        for directory in [self.test_images_dir, self.processed_dir, self.summaries_dir, self.delta_e_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def save_reference_image(self, image_path, image):
        """Save the reference image to inputs directory."""
        save_path = self.inputs_dir / 'reference.png'
        cv2.imwrite(str(save_path), image)
        logging.info(f"Saved reference image to {save_path}")
        return save_path

    def save_test_image(self, image_path, image):
        """Save a test image to inputs/test_images directory."""
        image_name = Path(image_path).name
        save_path = self.test_images_dir / image_name
        cv2.imwrite(str(save_path), image)
        logging.info(f"Saved test image to {save_path}")
        return save_path

    def save_preprocessed_image(self, image_name, preprocessed_image):
        """Save preprocessed image to processed/<image_name>/."""
        image_dir = self.processed_dir / image_name
        image_dir.mkdir(exist_ok=True)
        save_path = image_dir / 'preprocessed.png'
        cv2.imwrite(str(save_path), preprocessed_image)
        logging.info(f"Saved preprocessed image to {save_path}")
        return save_path

    def save_segmentation_image(self, image_name, method, segmented_image):
        """Save segmentation result to processed/<image_name>/segmented/."""
        segmented_dir = self.processed_dir / image_name / 'segmented'
        segmented_dir.mkdir(exist_ok=True)
        save_path = segmented_dir / f"{method}.png"
        cv2.imwrite(str(save_path), segmented_image)
        logging.info(f"Saved {method} segmentation to {save_path}")
        return save_path

    def save_reference_summary(self, image):
        """Save reference summary to summaries directory."""
        save_path = self.summaries_dir / 'reference_summary.png'
        cv2.imwrite(str(save_path), image)
        logging.info(f"Saved reference summary to {save_path}")
        return save_path

    def save_delta_e_results(self, dataset_name, delta_e_data):
        """Save all Delta E results to a single CSV file."""
        df_delta_e = pd.DataFrame(delta_e_data)
        delta_e_path = self.delta_e_dir / f"{dataset_name}_delta_e.csv"
        df_delta_e.to_csv(str(delta_e_path), index=False)
        logging.info(f"Saved all Delta E results to {delta_e_path}")
        return delta_e_path