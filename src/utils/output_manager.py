import logging
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class OutputManager:
    """
    Handles saving all output files (images, CSVs) to an organized directory structure.

    This class centralizes file path generation and saving operations, ensuring
    consistency across different datasets and processing steps. It creates a
    dedicated subdirectory for each dataset under a base output directory.
    """

    def __init__(self, base_output_dir: Path, dataset_name: str):
        """
        Initializes the OutputManager for a specific dataset.

        Creates the necessary directory structure for the dataset if it doesn't exist.

        Args:
            base_output_dir: The root directory where all outputs will be stored
                             (e.g., Path("./output")).
            dataset_name: The name of the current dataset being processed
                          (e.g., 'block', 'flowers'). This will be used as a
                          subdirectory name.
        """
        if not isinstance(base_output_dir, Path):
             # Ensure the input is a Path object for consistency
             base_output_dir = Path(base_output_dir)
             
        self.base_output_dir = base_output_dir
        self.dataset_name = dataset_name
        # Define the main directory for this specific dataset's outputs
        self.dataset_dir = self.base_output_dir / "datasets" / self.dataset_name

        # Keep track of the image currently being processed (set via pipeline)
        self.current_image_stem: Optional[str] = None # Stores the filename without extension

        try:
            self._create_directories()
            logger.info(f"OutputManager initialized for dataset '{dataset_name}'. Output directory: {self.dataset_dir}")
        except IOError as e:
            # If directories cannot be created, it's a critical failure.
            logger.critical(f"OutputManager failed to initialize directories: {e}", exc_info=True)
            raise # Re-raise the exception to stop the process

    def _create_directories(self):
        """
        Creates the standard subdirectory structure for the current dataset.

        Ensures that folders for inputs, preprocessed images, segmented images (per method),
        analysis results (CSVs), and summaries exist.

        Raises:
            IOError: If any directory cannot be created due to permissions or other issues.
        """
        try:
            # Base directories for different output types
            dirs_to_create = [
                self.dataset_dir / "inputs" / "reference_image",
                self.dataset_dir / "inputs" / "test_images",
                self.dataset_dir / "processed" / "preprocessed",
                self.dataset_dir / "analysis",
                self.dataset_dir / "summaries"
            ]
            for dir_path in dirs_to_create:
                dir_path.mkdir(parents=True, exist_ok=True)

            # Subdirectories within 'segmented' for each possible method
            segmented_base_dir = self.dataset_dir / "processed" / "segmented"
            # Define known methods here. Add more if new segmenters are introduced.
            possible_methods = [
                'kmeans_opt', 'kmeans_predef',
                'som_opt', 'som_predef',
                'dbscan'
            ]
            for method in possible_methods:
                (segmented_base_dir / method).mkdir(parents=True, exist_ok=True)

            logger.debug(f"Ensured output directory structure exists for dataset '{self.dataset_name}'.")

        except OSError as e: # Catch specific OS errors related to file operations
            logger.error(f"Failed to create output directories under {self.dataset_dir}: {e}", exc_info=True)
            # Wrap the OSError in an IOError for consistent error handling upstream
            raise IOError(f"Could not create output directories: {e}")
        except Exception as e: # Catch any other unexpected errors
            logger.error(f"An unexpected error occurred creating directories: {e}", exc_info=True)
            raise IOError(f"Unexpected error creating directories: {e}")


    def _get_safe_filename(self, original_filename: str, suffix: str = ".png") -> str:
        """
        Sanitizes a filename string and ensures it has the specified suffix.

        Removes the original extension, replaces spaces and slashes with underscores,
        and appends the desired suffix (case-insensitive).

        Args:
            original_filename: The original filename (e.g., "block 1.jpg", "reference.png").
            suffix: The desired file extension (e.g., ".png", ".csv"). Defaults to ".png".

        Returns:
            A sanitized filename string (e.g., "block_1.png", "reference.png").
        """
        safe_stem = Path(original_filename).stem # Get name without extension
        # Replace potentially problematic characters
        safe_stem = safe_stem.replace(" ", "_").replace("/", "_").replace("\\", "_")
        # Ensure the correct suffix exists, case-insensitively
        if not safe_stem.lower().endswith(suffix.lower()):
              safe_stem += suffix
        return safe_stem

    def set_current_image_stem(self, image_stem: Optional[str]):
        """
        Sets the stem (filename without extension) of the image currently being processed.

        This is typically called by the pipeline before processing each test image,
        so that `save_segmentation_result` knows the correct base filename to use.

        Args:
            image_stem: The filename stem (e.g., "block1"), or None when processing ends.
        """
        self.current_image_stem = image_stem
        if image_stem:
            logging.debug(f"Current image stem set to: '{image_stem}'")
        else:
            logging.debug("Current image stem cleared.")

    def get_current_image_name(self) -> str:
        """
        Gets the stem of the image currently being processed.

        Used by methods like `save_segmentation_result` to construct filenames.
        Returns a placeholder if no stem is currently set.

        Returns:
            The current image stem, or "unknown_image" if not set.
        """
        return self.current_image_stem or "unknown_image"

    def get_preprocessed_image_path(self, image_name_stem: str) -> str:
        """
        Constructs the full, absolute path for saving a preprocessed image.

        Args:
            image_name_stem: The stem of the original image filename.

        Returns:
            The absolute path (as a string) where the preprocessed image should be saved.
        """
        filename = f"{image_name_stem}_preprocessed.png"
        save_path = self.dataset_dir / "processed" / "preprocessed" / filename
        return str(save_path.resolve()) # Return absolute path as string

    def save_reference_image(self, original_filename: str, image: np.ndarray):
        """
        Saves a copy of the original reference image to the 'inputs' directory.

        Args:
            original_filename: The original filename of the reference image.
            image: The image data (NumPy array) to save.
        """
        if image is None:
            logger.warning(f"Attempted to save None reference image ('{original_filename}'). Skipping.")
            return
        try:
            filename = self._get_safe_filename(original_filename, ".png") # Ensure consistent format
            save_path = self.dataset_dir / "inputs" / "reference_image" / filename
            success = cv2.imwrite(str(save_path), image)
            if success:
                # Log relative path for conciseness
                logging.info(f"Saved reference image copy to: {save_path.relative_to(self.base_output_dir)}")
            else:
                 logging.error(f"cv2.imwrite failed to save reference image to: {save_path}")
        except Exception as e:
            # Catch potential errors during path creation or writing
            logging.error(f"Failed to save reference image '{original_filename}': {e}", exc_info=True)

    def save_test_image(self, original_filename: str, image: np.ndarray):
        """
        Saves a copy of an original test image to the 'inputs' directory.

        Preserves the original file extension if possible.

        Args:
            original_filename: The original filename of the test image.
            image: The image data (NumPy array) to save.
        """
        if image is None:
            logger.warning(f"Attempted to save None test image ('{original_filename}'). Skipping.")
            return
        try:
            # Try to keep the original suffix (like .jpg)
            original_suffix = Path(original_filename).suffix or ".jpg"
            filename = self._get_safe_filename(original_filename, original_suffix)
            save_path = self.dataset_dir / "inputs" / "test_images" / filename
            success = cv2.imwrite(str(save_path), image)
            if success:
                logging.info(f"Saved test image copy to: {save_path.relative_to(self.base_output_dir)}")
            else:
                 logging.error(f"cv2.imwrite failed to save test image to: {save_path}")
        except Exception as e:
            logging.error(f"Failed to save test image '{original_filename}': {e}", exc_info=True)

    def save_preprocessed_image(self, image_name_stem: str, image: np.ndarray):
        """
        Saves the preprocessed version of a test image.

        Args:
            image_name_stem: The stem of the original image filename.
            image: The preprocessed image data (NumPy array) to save.
        """
        if image is None:
            logger.warning(f"Attempted to save None preprocessed image for '{image_name_stem}'. Skipping.")
            return
        try:
            # Use the helper to get the consistent save path
            save_path_str = self.get_preprocessed_image_path(image_name_stem)
            save_path = Path(save_path_str)
            success = cv2.imwrite(save_path_str, image)
            if success:
                logging.info(f"Saved preprocessed image to: {save_path.relative_to(self.base_output_dir)}")
            else:
                 logging.error(f"cv2.imwrite failed to save preprocessed image to: {save_path}")
        except Exception as e:
            logger.error(f"Failed to save preprocessed image for '{image_name_stem}': {e}", exc_info=True)

    def save_segmentation_result(self,
                                 image: np.ndarray,
                                 method_name: str, # e.g., "kmeans_opt"
                                 k_type: str):     # e.g., "determined"
        """
        Saves the segmented image output into the appropriate method subdirectory.

        The filename will typically be '{image_stem}_{k_type}.png'.

        Args:
            image: The segmented image array to save.
            method_name: The key identifying the segmentation method (used for subdirectory).
            k_type: The k determination type ('determined' or 'predefined').
        """
        if image is None:
            logger.warning(f"Attempted to save None segmented image for method '{method_name}'. Skipping.")
            return

        try:
            image_name_stem = self.get_current_image_name()
            if not image_name_stem or image_name_stem == "unknown_image":
                logger.error(f"Cannot save segmentation result: current image stem not set.")
                return

            # Determine the target directory based on the method name
            target_dir = self.dataset_dir / "processed" / "segmented" / method_name
            # Ensure the directory exists (it should have been created by __init__)
            target_dir.mkdir(parents=True, exist_ok=True)

            # Construct the filename (method name is now part of the path)
            filename = f"{image_name_stem}_{k_type}.png"
            save_path = target_dir / filename

            success = cv2.imwrite(str(save_path), image)
            if success:
                relative_path = save_path.relative_to(self.base_output_dir)
                logger.info(f"Saved segmented image ({method_name}) to: {relative_path}")
            else:
                 logging.error(f"cv2.imwrite failed to save segmented image to: {save_path}")

        except Exception as e:
            # Catch potential errors during path creation or writing
            current_stem = self.current_image_stem or "unknown"
            logger.error(f"Failed to save segmented image for method '{method_name}' (k_type: {k_type}, image: '{current_stem}'): {e}", exc_info=True)

    def save_delta_e_results(self, delta_e_list: List[Dict[str, Any]]):
        """
        Saves the collected Delta E results (list of dictionaries) to a CSV file.

        The CSV file is saved in the 'analysis' subdirectory for the dataset.

        Args:
            delta_e_list: A list where each item is a dictionary containing
                          Delta E results for a specific image and method
                          (keys should match expected columns like 'image', 'method',
                          'traditional_avg_delta_e', etc.).
        """
        if not delta_e_list:
            logger.warning("No Delta E results provided to save.")
            return
            
        # Check if the input is actually a list of dictionaries
        if not isinstance(delta_e_list, list) or not all(isinstance(item, dict) for item in delta_e_list):
             logger.error("Invalid format for delta_e_list. Expected a list of dictionaries.")
             return
             
        try:
            df = pd.DataFrame(delta_e_list)
            # Use a filename that clearly identifies the dataset
            filename = f"{self.dataset_name}_delta_e_results.csv" # Removed leading underscore
            save_path = self.dataset_dir / "analysis" / filename
            df.to_csv(save_path, index=False)
            logging.info(f"Saved Delta E results ({len(delta_e_list)} entries) to: {save_path.relative_to(self.base_output_dir)}")
        except ImportError:
             logger.error("Pandas library not found. Cannot save Delta E results to CSV. Please install pandas.")
        except Exception as e:
            # Catch potential errors during DataFrame creation or CSV writing
            logger.error(f"Failed to save Delta E results CSV: {e}", exc_info=True)

    # TODO: Update this method to accept SegmentationResult objects
    #       (kmeans_result, som_result from segment_reference_image)
    #       and potentially call a plotting function from visualization.py
    #       to create a more informative summary image instead of just saving
    #       the original.
    def save_reference_summary(self, original_filename: str, image: np.ndarray, *args):
        """
        Saves summary information related to reference image processing.

        Currently saves only a copy of the original reference image as a placeholder.
        Should be updated to generate a more informative summary plot.

        Args:
            original_filename: The original filename of the reference image.
            image: The original reference image data (NumPy array).
            *args: Placeholder for potential future arguments like segmentation results.
        """
        if image is None:
             logger.warning(f"Attempted to save None reference summary image for '{original_filename}'. Skipping.")
             return
        try:
             filename = self._get_safe_filename(original_filename, "_summary.png")
             save_path = self.dataset_dir / "summaries" / filename
             # Currently, just save the original image
             success = cv2.imwrite(str(save_path), image)
             if success:
                  logging.info(f"Reference summary (placeholder image) saved to: {save_path.relative_to(self.base_output_dir)}")
             else:
                  logging.error(f"cv2.imwrite failed to save reference summary image to: {save_path}")
        except Exception as e:
             logger.error(f"Failed to save reference summary for '{original_filename}': {e}", exc_info=True)