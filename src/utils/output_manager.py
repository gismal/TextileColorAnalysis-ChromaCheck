# src/utils/output_manager.py (NEW FILE)

import logging
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class OutputManager:
    """Handles saving all output files in an organized structure."""

    def __init__(self, base_output_dir: Path, dataset_name: str):
        """
        Initializes the OutputManager.

        Args:
            base_output_dir (Path): The root directory for all outputs (e.g., ./output).
            dataset_name (str): The name of the current dataset (e.g., 'block').
        """
        self.base_output_dir = Path(base_output_dir)
        self.dataset_name = dataset_name
        # Define the main directory for this dataset's outputs
        self.dataset_dir = self.base_output_dir / "datasets" / self.dataset_name
        self._create_directories()
        logger.info(f"OutputManager initialized for dataset '{dataset_name}' at {self.dataset_dir}")

    def _create_directories(self):
        """Creates the necessary subdirectories for the dataset."""
        try:
            # Create base dataset directory
            self.dataset_dir.mkdir(parents=True, exist_ok=True)
            
            # Subdirectories based on potential outputs
            # (Adjust these as needed based on your actual saving methods)
            (self.dataset_dir / "inputs" / "reference_image").mkdir(parents=True, exist_ok=True)
            (self.dataset_dir / "inputs" / "test_images").mkdir(parents=True, exist_ok=True)
            (self.dataset_dir / "processed" / "preprocessed").mkdir(parents=True, exist_ok=True)
            (self.dataset_dir / "processed" / "segmented").mkdir(parents=True, exist_ok=True)
            (self.dataset_dir / "analysis").mkdir(parents=True, exist_ok=True)
            (self.dataset_dir / "summaries").mkdir(parents=True, exist_ok=True)

        except Exception as e:
            logger.error(f"Failed to create output directories for {self.dataset_name}: {e}", exc_info=True)
            # Depending on severity, you might want to raise the error
            # raise

    def _get_safe_filename(self, original_filename: str, suffix: str = ".png") -> str:
         """Sanitizes a filename and ensures it has the correct suffix."""
         # Basic sanitization: remove path components, replace problematic chars
         safe_name = Path(original_filename).stem
         safe_name = safe_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
         # Add suffix if it doesn't have one or has the wrong one
         if not safe_name.lower().endswith(suffix.lower()):
              safe_name += suffix
         return safe_name


    def save_reference_image(self, original_filename: str, image: np.ndarray):
        """Saves a copy of the reference image."""
        try:
            filename = self._get_safe_filename(original_filename, ".png") # Save as PNG for consistency
            save_path = self.dataset_dir / "inputs" / "reference_image" / filename
            cv2.imwrite(str(save_path), image)
            logger.info(f"Saved reference image copy to {save_path.relative_to(self.base_output_dir)}")
        except Exception as e:
            logger.error(f"Failed to save reference image {original_filename}: {e}", exc_info=True)

    def save_test_image(self, original_filename: str, image: np.ndarray):
        """Saves a copy of a test image."""
        try:
            filename = self._get_safe_filename(original_filename, Path(original_filename).suffix or ".jpg") # Keep original suffix if possible
            save_path = self.dataset_dir / "inputs" / "test_images" / filename
            cv2.imwrite(str(save_path), image)
            logger.info(f"Saved test image copy to {save_path.relative_to(self.base_output_dir)}")
        except Exception as e:
            logger.error(f"Failed to save test image {original_filename}: {e}", exc_info=True)

    def save_preprocessed_image(self, image_name_stem: str, image: np.ndarray):
        """Saves the preprocessed image."""
        try:
            filename = f"{image_name_stem}_preprocessed.png" # Use stem + suffix
            save_path = self.dataset_dir / "processed" / "preprocessed" / filename
            cv2.imwrite(str(save_path), image)
            logger.info(f"Saved preprocessed image to {save_path.relative_to(self.base_output_dir)}")
        except Exception as e:
            logger.error(f"Failed to save preprocessed image for {image_name_stem}: {e}", exc_info=True)

    def get_preprocessed_image_path(self, image_name_stem: Optional[str] = None) -> str:
         """Returns the expected path for a preprocessed image (used by Segmenter)."""
         # This might need refinement - depends if Segmenter needs a specific file
         if image_name_stem:
              filename = f"{image_name_stem}_preprocessed.png"
              return str(self.dataset_dir / "processed" / "preprocessed" / filename)
         else:
              # Return a generic path or the directory path if no name is given
              return str(self.dataset_dir / "processed" / "preprocessed")


    def save_segmentation_result(self, image: np.ndarray, method_name: str, k_type: str):
        """Saves the segmented image output."""
        try:
            # Include k_type in the filename for clarity
            filename = f"{method_name}_{k_type}_segmented.png"
            save_path = self.dataset_dir / "processed" / "segmented" / filename
            cv2.imwrite(str(save_path), image)
            logger.info(f"Saved segmented image to {save_path.relative_to(self.base_output_dir)}")
        except Exception as e:
            logger.error(f"Failed to save segmented image for {method_name} ({k_type}): {e}", exc_info=True)

    def save_delta_e_results(self, dataset_name_override: Optional[str], delta_e_list: List[Dict[str, Any]]):
        """Saves the collected Delta E results to a CSV file."""
        if not delta_e_list:
            logger.warning("No Delta E results provided to save.")
            return
        try:
            df = pd.DataFrame(delta_e_list)
            # Use override if provided, else use instance dataset_name
            filename = f"{dataset_name_override or self.dataset_name}_delta_e_results.csv"
            save_path = self.dataset_dir / "analysis" / filename
            df.to_csv(save_path, index=False)
            logger.info(f"Saved Delta E results ({len(delta_e_list)} entries) to {save_path.relative_to(self.base_output_dir)}")
        except Exception as e:
            logger.error(f"Failed to save Delta E results CSV: {e}", exc_info=True)
            
    def save_reference_summary(self, original_filename: str, image: np.ndarray):
         """Saves summary information related to reference image processing."""
         # Placeholder - This might save plots or specific data.
         # For now, let's save the 'original_image' from reference processing
         try:
             filename = self._get_safe_filename(original_filename, "_summary.png")
             save_path = self.dataset_dir / "summaries" / filename
             cv2.imwrite(str(save_path), image)
             logger.info(f"Saved reference summary image to {save_path.relative_to(self.base_output_dir)}")
         except Exception as e:
             logger.error(f"Failed to save reference summary for {original_filename}: {e}", exc_info=True)

    # Add other saving methods as needed (e.g., for plots, specific analysis dataframes)