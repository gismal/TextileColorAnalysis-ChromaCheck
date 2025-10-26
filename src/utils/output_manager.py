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
        self.dataset_dir = self.base_output_dir / "datasets" / self.dataset_name
        self._create_directories()
        self.current_image_stem: Optional[str] = None # İşlenen mevcut resim adını sakla
        logger.info(f"OutputManager initialized for dataset '{dataset_name}' at {self.dataset_dir}")

    def _create_directories(self):
        """Creates the necessary subdirectories for the dataset."""
        try:
            (self.dataset_dir / "inputs" / "reference_image").mkdir(parents=True, exist_ok=True)
            (self.dataset_dir / "inputs" / "test_images").mkdir(parents=True, exist_ok=True)
            (self.dataset_dir / "processed" / "preprocessed").mkdir(parents=True, exist_ok=True)
            (self.dataset_dir / "processed" / "segmented").mkdir(parents=True, exist_ok=True)
            (self.dataset_dir / "analysis").mkdir(parents=True, exist_ok=True)
            (self.dataset_dir / "summaries").mkdir(parents=True, exist_ok=True)
            
            segmented_base_dir = self.dataset_dir / "processed" / "segmented"
            possible_methods = [
                'kmeans_opt', 'kmeans_predef', 
                'som_opt', 'som_predef', 
                'dbscan'
            ]
            
            for method in possible_methods:
                (segmented_base_dir / method).mkdir(parents=True, exist_ok=True)
            
        except Exception as e:
            logger.error(f"Failed to create output directories for {self.dataset_name}: {e}", exc_info=True)
            raise IOError(f"Could not create output directories: {e}")

    def _get_safe_filename(self, original_filename: str, suffix: str = ".png") -> str:
         """Sanitizes a filename and ensures it has the correct suffix."""
         safe_name = Path(original_filename).stem
         safe_name = safe_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
         if not safe_name.lower().endswith(suffix.lower()):
              safe_name += suffix
         return safe_name

    def set_current_image_stem(self, image_stem: str):
        """İşlenmekte olan mevcut resmin adını (uzantısız) ayarlar."""
        self.current_image_stem = image_stem
        logging.debug(f"Current image stem set to: {image_stem}")

    def get_current_image_name(self) -> str:
        """İşlenmekte olan mevcut resmin adını döndürür (Segmenter tarafından kullanılır)."""
        return self.current_image_stem or "unknown_image"

    def get_preprocessed_image_path(self, image_name_stem: str) -> str:
         """Ön işlenmiş görüntü için tam kayıt yolunu döndürür."""
         filename = f"{image_name_stem}_preprocessed.png"
         save_path = self.dataset_dir / "processed" / "preprocessed" / filename
         return str(save_path)

    def save_reference_image(self, original_filename: str, image: np.ndarray):
        """Saves a copy of the reference image."""
        try:
            filename = self._get_safe_filename(original_filename, ".png")
            save_path = self.dataset_dir / "inputs" / "reference_image" / filename
            cv2.imwrite(str(save_path), image)
            logging.info(f"Saved reference image copy to {save_path.relative_to(self.base_output_dir)}")
        except Exception as e:
            logging.error(f"Failed to save reference image {original_filename}: {e}", exc_info=True)

    def save_test_image(self, original_filename: str, image: np.ndarray):
        """Saves a copy of a test image."""
        try:
            filename = self._get_safe_filename(original_filename, Path(original_filename).suffix or ".jpg")
            save_path = self.dataset_dir / "inputs" / "test_images" / filename
            cv2.imwrite(str(save_path), image)
            logging.info(f"Saved test image copy to {save_path.relative_to(self.base_output_dir)}")
        except Exception as e:
            logging.error(f"Failed to save test image {original_filename}: {e}", exc_info=True)

    def save_preprocessed_image(self, image_name_stem: str, image: np.ndarray):
        """Saves the preprocessed image."""
        try:
            save_path = Path(self.get_preprocessed_image_path(image_name_stem))
            cv2.imwrite(str(save_path), image)
            logging.info(f"Saved preprocessed image to {save_path.relative_to(self.base_output_dir)}")
        except Exception as e:
            logger.error(f"Failed to save preprocessed image for {image_name_stem}: {e}", exc_info=True)

    def save_segmentation_result(self,
                                 image: np.ndarray,
                                 method_name: str,
                                 k_type: str):
        """
        Saves the segmented image output into the subdirectory.
        the filename will now typically be '{image_stem}_{k_type}.png'.
        
        Args:
            image (np.ndarray): The segmented image array to save.
            method_name (str): The key identifying the segmentation method 
                               (used to determine the subdirectory).
            k_type (str): The type of k determination used ('determined' or 'predefined').
                          This becomes part of the filename.
        """
        if image is None:
            logger.warning(f"Attempted to save None image for method {method_name}. Skipping")
            return
        
        try:
            image_name_stem = self.get_current_image_name() 
            if not image_name_stem or image_name_stem == "unknown_image":
                logger.error(f"Cannot save segmentation result: current image stem is not set in OutputManager.")
                # Geçici bir isimle kaydetmeyi deneyebilir veya atlayabiliriz. Şimdilik atlayalım.
                return 

            # 1. Doğru Alt Klasörü Belirle
            # Neden? method_name'e göre sonuçları gruplamak için.
            target_dir = self.dataset_dir / "processed" / "segmented" / method_name
            # (Klasörün _create_directories'de oluşturulduğunu varsayıyoruz)
            if not target_dir.exists():
                 logger.warning(f"Directory {target_dir} does not exist. Attempting to create.")
                 target_dir.mkdir(parents=True, exist_ok= True)
                 target_dir = self.dataset_dir / "processed" / "segmented" / method_name
            # (Klasörün _create_directories'de oluşturulduğunu varsayıyoruz)
            if not target_dir.exists():
                 logger.warning(f"Directory {target_dir} does not exist. Attempting to create.")
                 target_dir.mkdir(parents=True, exist_ok=True) # Oluşturmayı dene

            # 2. Yeni Dosya Adını Oluştur
            # Neden? method_name artık klasör adında olduğu için dosya adında tekrarlamaya gerek yok.
            # k_type'ı eklemek, aynı resmin farklı k belirleme sonuçlarını ayırt eder.
            filename = f"{image_name_stem}_{k_type}.png"
            save_path = target_dir / filename

            # 3. Görüntüyü Kaydet
            cv2.imwrite(str(save_path), image)
            
            relative_path = save_path.relative_to(self.base_output_dir)
            logger.info(f"Saved segmented image ({method_name}) to: {relative_path}")
            
        except Exception as e:
            logger.error(f"Failed to save segmented image for {method_name} ({k_type}): {e}", exc_info=True)

    def save_delta_e_results(self, delta_e_list: List[Dict[str, Any]]):
        """Saves the collected Delta E results to a CSV file."""
        if not delta_e_list:
            logger.warning("No Delta E results provided to save.")
            return
        try:
            df = pd.DataFrame(delta_e_list)
            filename = f"_{self.dataset_name}_delta_e_results.csv" # Başına _ koydum
            save_path = self.dataset_dir / "analysis" / filename
            df.to_csv(save_path, index=False)
            logging.info(f"Saved Delta E results ({len(delta_e_list)} entries) to {save_path.relative_to(self.base_output_dir)}")
        except Exception as e:
            logger.error(f"Failed to save Delta E results CSV: {e}", exc_info=True)
            
    def save_reference_summary(self, original_filename: str, image: np.ndarray, *args):
         """
         Saves summary information related to reference image processing.
         TODO: (Madde 2 & 4) Bu fonksiyonu *args (kmeans_result, som_result vb.) 
         ile gelen verileri kullanarak güzel bir plot oluşturacak şekilde güncelle.
         """
         try:
             filename = self._get_safe_filename(original_filename, "_summary.png")
             save_path = self.dataset_dir / "summaries" / filename
             # Şimdilik sadece orijinal görüntüyü kaydediyoruz
             cv2.imwrite(str(save_path), image)
             logging.info(f"Reference summary (placeholder) saved to {save_path.relative_to(self.base_output_dir)}")
         except Exception as e:
             logger.error(f"Failed to save reference summary for {original_filename}: {e}", exc_info=True)