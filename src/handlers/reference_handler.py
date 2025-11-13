# src/handlers/reference_handler.py
import logging
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path

# Gerekli proje içi importlar
from src.data.load_data import load_image
from src.data.preprocess import Preprocessor, PreprocessingConfig
from src.models.pso_dbn import DBN
from src.models.segmentation import SegmentationResult
from src.models.segmentation.reference import segment_reference_image
from src.utils.output_manager import OutputManager
from src.utils.color.color_conversion import convert_colors_to_cielab
from src.utils.visualization import plot_reference_summary

# Scaler tipi için type hinting
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

class ReferenceHandler:
    """
    Handles all processing related to the reference image.

    This class encapsulates the logic for:
    1. Loading the reference image.
    2. Preprocessing the reference image.
    3. Running the specialized `segment_reference_image` function (KMeans & SOM).
    4. Extracting the target CIELAB colors from the K-Means result.
    5. Plotting and saving the reference summary visual.
    """

    def __init__(self,
                 preprocess_config: PreprocessingConfig,
                 output_manager: OutputManager):
        """
        Initializes the ReferenceHandler.

        Args:
            preprocess_config: The configuration object for the Preprocessor.
            output_manager: The instance of OutputManager to save outputs.
        """
        self.preprocess_config = preprocess_config
        self.output_manager = output_manager
        self.ref_preprocessor = Preprocessor(config=self.preprocess_config)
        self.original_ref_image: Optional[np.ndarray] = None # Orijinal görüntüyü saklamak için
        
        logger.debug("ReferenceHandler initialized.")

    def execute(self,
                ref_image_path: str,
                dbn: DBN,
                scalers: Dict[str, MinMaxScaler],
                seg_params: Dict[str, Any]
                ) -> Tuple[np.ndarray, Optional[SegmentationResult], Optional[SegmentationResult]]:
        """
        Executes the entire reference image processing workflow.

        This method mirrors the logic previously in `pipeline._run_reference_processing`.

        Args:
            ref_image_path: Absolute path to the reference image.
            dbn: The trained DBN model.
            scalers: The dictionary of fitted scalers.
            seg_params: The 'segmentation_params' dictionary from the main config.

        Returns:
            A tuple containing:
            - Target LAB colors (np.ndarray) derived from K-Means.
            - The raw SegmentationResult from K-Means.
            - The raw SegmentationResult from SOM.

        Raises:
            ValueError: If loading, preprocessing, segmentation, or color extraction fails.
        """
        logger.info(f"Processing reference image specified in config: {ref_image_path}")

        # 1. Load the reference image
        ref_image_bgr = load_image(ref_image_path)
        if ref_image_bgr is None:
            raise ValueError(f"Failed to load reference image: {ref_image_path}")
        
        self.original_ref_image = ref_image_bgr
        
        self.output_manager.save_reference_image(Path(ref_image_path).name, ref_image_bgr)

        # 2. Preprocess the reference image
        logger.info("Preprocessing reference image...")
        try:
            preprocessed_ref_image = self.ref_preprocessor.preprocess(ref_image_bgr)
            if preprocessed_ref_image is None:
                raise ValueError("Preprocessing returned None for the reference image.")
        except Exception as e:
            logger.error(f"Preprocessing failed for reference image: {e}", exc_info=True)
            raise ValueError(f"Preprocessing failed for reference: {e}")

        # 3. Segment the preprocessed reference
        # Gerekli parametreleri config'den al
        default_k = seg_params.get('predefined_k', 2)
        k_range_ref = seg_params.get('k_values', list(range(2, 9)))

        # Gerekli scaler'ları listeye al
        try:
            scaler_list = [scalers['scaler_x'], scalers['scaler_y'], scalers['scaler_y_ab']]
        except KeyError as e:
            logger.error(f"Missing required scaler key: {e}. Cannot proceed with reference segmentation.")
            raise ValueError(f"Scalers dictionary is missing required keys: {e}")
            
        kmeans_result, som_result, determined_k = segment_reference_image(
            preprocessed_image=preprocessed_ref_image,
            dbn=dbn,
            scalers=scaler_list,
            default_k=default_k,
            k_range=k_range_ref
        )

        # 4. Extract Target Colors
        if not kmeans_result or not kmeans_result.is_valid():
            raise ValueError("Reference K-Means segmentation failed or was invalid, cannot extract target colors.")

        target_colors_lab = np.array([])
        try:
            # Segmenter avg_colors'ı (R, G, B) tuple listesi olarak döndürmeli
            avg_colors_rgb = [tuple(c) for c in kmeans_result.avg_colors]
            target_colors_lab = convert_colors_to_cielab(avg_colors_rgb)
            if not isinstance(target_colors_lab, np.ndarray) or target_colors_lab.size == 0:
                raise ValueError("convert_colors_to_cielab failed to produce valid LAB colors from K-Means result.")
        except Exception as e:
            logger.error(f"Error extracting target LAB colors from reference K-Means result: {e}", exc_info=True)
            raise ValueError(f"Target color extraction failed: {e}")

        # 5. Create and Save Summary Plot
        try:
            summary_filename = "reference_summary.png"
            summary_output_path = self.output_manager.dataset_dir / "summaries" / summary_filename

            plot_reference_summary(
                kmeans_result=kmeans_result,
                som_result=som_result,
                original_image=ref_image_bgr,
                target_colors_lab=target_colors_lab,
                output_path=summary_output_path
            )
        except Exception as plot_err:
            # A plotting error shouldn't stop the whole pipeline, just log it.
            logger.error(f"Failed to generate reference summary plot: {plot_err}", exc_info=True)

        logger.info(f"Reference image processed successfully. Determined k={determined_k}. "
                    f"Extracted {target_colors_lab.shape[0]} target LAB colors.")

        return target_colors_lab, kmeans_result, som_result
    
    def get_original_image(self) -> Optional[np.ndarray]:
        """
        Getter method for AnalysisHandler
        """
        return self.original_ref_image