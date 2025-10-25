# src/utils/output_manager.py (KÜÇÜK GÜNCELLEME)

import logging
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class OutputManager:
    def __init__(self, base_output_dir: Path, dataset_name: str):
        self.base_output_dir = Path(base_output_dir)
        self.dataset_name = dataset_name
        self.dataset_dir = self.base_output_dir / "datasets" / self.dataset_name
        self._create_directories()
        self.current_image_name: Optional[str] = None # İşlenen mevcut resim adını sakla
        logger.info(f"OutputManager initialized for dataset '{dataset_name}' at {self.dataset_dir}")

    def _create_directories(self):
        (self.dataset_dir / "inputs" / "reference_image").mkdir(parents=True, exist_ok=True)
        (self.dataset_dir / "inputs" / "test_images").mkdir(parents=True, exist_ok=True)
        (self.dataset_dir / "processed" / "preprocessed").mkdir(parents=True, exist_ok=True)
        (self.dataset_dir / "processed" / "segmented").mkdir(parents=True, exist_ok=True)
        (self.dataset_dir / "analysis").mkdir(parents=True, exist_ok=True)
        (self.dataset_dir / "summaries").mkdir(parents=True, exist_ok=True)

    def _get_safe_filename(self, original_filename: str, suffix: str = ".png") -> str:
         safe_name = Path(original_filename).stem
         safe_name = safe_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
         if not safe_name.lower().endswith(suffix.lower()):
              safe_name += suffix
         return safe_name
         
    # --- YENİ METOTLAR (veya güncellenmiş) ---
    def set_current_image(self, image_name_stem: str):
        """İşlenmekte olan mevcut resmin adını ayarlar."""
        self.current_image_name = image_name_stem

    def get_current_image_name(self) -> str:
        """İşlenmekte olan mevcut resmin adını döndürür."""
        return self.current_image_name or "unknown_image"

    def get_preprocessed_image_path(self, image_name_stem: str) -> str:
         """Ön işlenmiş görüntü için tam kayıt yolunu döndürür."""
         filename = f"{image_name_stem}_preprocessed.png"
         save_path = self.dataset_dir / "processed" / "preprocessed" / filename
         return str(save_path)
    # --- BİTİŞ ---

    def save_reference_image(self, original_filename: str, image: np.ndarray):
        try:
            filename = self._get_safe_filename(original_filename, ".png")
            save_path = self.dataset_dir / "inputs" / "reference_image" / filename
            cv2.imwrite(str(save_path), image)
            logger.info(f"Saved reference image copy to {save_path.relative_to(self.base_output_dir)}")
        except Exception as e:
            logger.error(f"Failed to save reference image {original_filename}: {e}", exc_info=True)

    def save_test_image(self, original_filename: str, image: np.ndarray):
        try:
            filename = self._get_safe_filename(original_filename, Path(original_filename).suffix or ".jpg")
            save_path = self.dataset_dir / "inputs" / "test_images" / filename
            cv2.imwrite(str(save_path), image)
            logger.info(f"Saved test image copy to {save_path.relative_to(self.base_output_dir)}")
        except Exception as e:
            logger.error(f"Failed to save test image {original_filename}: {e}", exc_info=True)

    def save_preprocessed_image(self, image_name_stem: str, image: np.ndarray):
        try:
            # get_preprocessed_image_path ile aynı yolu kullan
            save_path = Path(self.get_preprocessed_image_path(image_name_stem))
            cv2.imwrite(str(save_path), image)
            logger.info(f"Saved preprocessed image to {save_path.relative_to(self.base_output_dir)}")
        except Exception as e:
            logger.error(f"Failed to save preprocessed image for {image_name_stem}: {e}", exc_info=True)

    def save_segmentation_result(self, image: np.ndarray, method_name: str, k_type: str):
        try:
            # Mevcut işlenen resmin adını kullan
            image_name_stem = self.get_current_image_name()
            filename = f"{image_name_stem}_{method_name}_{k_type}.png"
            save_path = self.dataset_dir / "processed" / "segmented" / filename
            cv2.imwrite(str(save_path), image)
            logger.info(f"Saved segmented image to {save_path.relative_to(self.base_output_dir)}")
        except Exception as e:
            logger.error(f"Failed to save segmented image for {method_name} ({k_type}): {e}", exc_info=True)

    def save_delta_e_results(self, dataset_name_override: Optional[str], delta_e_list: List[Dict[str, Any]]):
        if not delta_e_list:
            logger.warning("No Delta E results provided to save.")
            return
        try:
            df = pd.DataFrame(delta_e_list)
            filename = f"{dataset_name_override or self.dataset_name}_delta_e_results.csv"
            save_path = self.dataset_dir / "analysis" / filename
            df.to_csv(save_path, index=False)
            logger.info(f"Saved Delta E results ({len(delta_e_list)} entries) to {save_path.relative_to(self.base_output_dir)}")
        except Exception as e:
            logger.error(f"Failed to save Delta E results CSV: {e}", exc_info=True)
            
    def save_reference_summary(self, original_filename: str, image: np.ndarray):
         try:
             filename = self._get_safe_filename(original_filename, "_summary.png")
             save_path = self.dataset_dir / "summaries" / filename
             cv2.imwrite(str(save_path), image)
             logger.info(f"Saved reference summary image to {save_path.relative_to(self.base_output_dir)}")
         except Exception as e:
             logger.error(f"Failed to save reference summary for {original_filename}: {e}", exc_info=True)