import logging 
import time
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path
import numpy as np
from contextlib import contextmanager

# from data
from src.data.load_data import load_config, load_data, load_image
from src.data.preprocess import Preprocessor, PreprocessingConfig

# the models
from src.models.dbn_trainer import DBNTrainer, DBNConfig, PSOConfig, TrainConfig
from src.models.segmentation import (
    Segmenter,
    SegmentationConfig, ModelConfig, SegmentationResult
)
from src.models.segmentation.reference import segment_reference_image
from src.models.pso_dbn import DBN
# the utils
from src.utils.output_manager import OutputManager
from src.utils.color.color_analysis import ColorMetricCalculator
from src.utils.color.color_conversion import convert_colors_to_cielab, convert_colors_to_cielab_dbn
from src.utils.setup import validate_processing_config

try:
    from src.utils.setup import validate_processing_config # Yeni setup modülümüz
except ImportError:
    logging.warning("src.utils.setup.py is not here. please configure")
    # temporary fallback
    def validate_processing_config(config: Dict,
                                   project_root: Path) -> bool:
        logging.error("please setup the setup.py and carry the validation method into")
        return False

logger = logging.getLogger(__name__)

# timer
@contextmanager
def timer(operation_name: str):
    """ keeps log of the code blocks"""
    start_time = time.perf_counter()
    logger.info(f"Starting: {operation_name}...")
    try:
        yield
    finally:
        duration = time.perf_counter() - start_time
        logger.info(f"Completed: {operation_name} in {duration: .2f} seconds")

# main pioeline class
class ProcessingPipeline:
    """ manages all the process and workflow"""
    
    def __init__(self,
                 config_path: str,
                 output_dir: Path,
                 project_root: Path):
        """ intiates the pipeline, loads the configs and validates
        
        Args:
            config_path (str): YAML file provided by the system
            output_dir (Path): the output file all the results saved into 
            project_root (Path): roots of the project
        """
        self.config_path = config_path
        self.output_dir = output_dir
        self.project_root = project_root
        
        self.output_manager : Optional[OutputManager] = None
        self.config : Dict[str, Any] = self._load_and_validate_config()
        
        self.dbn_config = DBNConfig(**self.config.get('dbn_params', {}))
        self.pso_config = PSOConfig(**self.config.get('pso_params', {}))
        self.train_config = TrainConfig(**self.config.get('training_params', {}))
        
        preproc_cfg_dict = self.config.get('preprocess_params', {})
        preproc_cfg_dict['target_size'] = tuple(preproc_cfg_dict.get('target_size', [128, 128]))
        self.preprocess_config = PreprocessingConfig(**preproc_cfg_dict)
        
        # placeholders for the models
        self.dbn: Optional[DBN] = None
        self.scalers : Optional[Dict[str, Any]] = None
    
    def _load_and_validate_config(self) -> Dict[str, Any]:
        """ loads and validates the config and intiates the OutputManager"""
        logger.info(f"loading the config from: {self.config_path}")
        config = load_config(self.config_path)
        if config is None:
            raise ValueError("Failed to load config file")
        
        if not validate_processing_config(config, self.project_root):
            raise ValueError("Processing config validation failed")
        
        # start the outputmanager
        dataset_name = Path(self.config_path).stem.replace('_config', '')
        self.output_manager = OutputManager(self.output_dir, dataset_name)
        logger.info(f"OutputManager intialized for dataset: {dataset_name}")
        
        return config
    
    def run(self):
        logger.info("="*50)
        logger.info(f"Processing Pipeline RUN starting for dataset: {self.output_manager.dataset_name}")
        logger.info("="*50)
        try:
            with timer("Total Pipeline"):
                # --- Adım 1: DBN Eğitim Aşaması ---
                with timer("DBN Training"):
                    self.dbn, self.scalers = self._train_dbn_model()
                
                # --- Adım 2: Referans Görüntü İşleme ---
                with timer("Reference Image Processing"):
                    target_colors_lab, ref_kmeans_result, ref_som_result = self._run_reference_processing()

                # --- Adım 3: Test Görüntü İşleme ---
                with timer("Test Image Analysis Loop"):
                    all_delta_e_results = self._run_test_image_analysis(
                        target_colors_lab, 
                        ref_kmeans_result,  # Öncelik 4: Nesneyi iletiyoruz
                        ref_som_result      # Öncelik 4: Nesneyi iletiyoruz
                    )
                
                # --- Adım 4: Sonuçları Kaydetme ---
                with timer("Saving Final Results"):
                    self._save_and_summarize_results(all_delta_e_results)

            logger.info("="*50)
            logger.info(f"Processing Pipeline RUN completed for: {self.output_manager.dataset_name}")
            logger.info("="*50)
            
        except Exception as e:
            logger.critical(f"Pipeline failed critically: {e}", exc_info=True)
            # Hata durumunda bile logların kaydedildiğinden emin olmak için
            logging.shutdown()
            raise
        
    def _train_dbn_model(self) -> Tuple[DBN, Dict[str, Any]]:
        """ loads the data for DBN training and initiates the training """
        
        logger.info("loading images to generate training data...")
        valid_test_images = []
        for image_path in self.config['test_images']:
            image = load_image(image_path)
            if image is not None:
                valid_test_images.append(image_path)
                self.output_manager.save_test_image(Path(image_path).name, image)
            else:
                logging.warning(f"Skipping invalid test image: {image_path}")
        if not valid_test_images:
            raise ValueError("No valid test images could be loaded for training")
        
        logger.info(f"loading and processing {len(valid_test_images)} images into training data...")
        load_data_target_size = tuple(self.config.get('load_data_resize', [100, 100]))
        rgb_data, lab_data = load_data(valid_test_images, target_size= load_data_target_size)
        
        # this part can be moved to dbn_trainer
        if rgb_data.size == 0 or lab_data.size == 0:
            raise ValueError("loaded training data arrays are empty")
        
        logger.info(f"initializing DBNTrainer (Samples  = {self.train_config.n_samples}...)")
        trainer = DBNTrainer(self.dbn_config, self.pso_config, self.train_config)
        dbn, scalers = trainer.train(rgb_data, lab_data)
        
        logger.info("DBN training and scaling complete")
        return dbn, scalers
    
    def _run_reference_processing(self) -> Tuple[np.ndarray, Optional[SegmentationResult], Optional[SegmentationResult]]:
        """
        Loads, preprocesses, and segments the reference image using the dedicated
        segmentation function, then extracts target LAB colors.

        Returns:
            Tuple[np.ndarray, Optional[SegmentationResult], Optional[SegmentationResult]]:
                - Target LAB colors derived from K-Means result.
                - The raw SegmentationResult from K-Means.
                - The raw SegmentationResult from SOM.

        Raises:
            ValueError: If loading, preprocessing, segmentation, or color extraction fails.
        """
        ref_image_path = self.config['reference_image_path']
        logger.info(f"Processing reference image: {ref_image_path}")
        
        # load
        ref_image_bgr = load_image(ref_image_path)
        if ref_image_bgr is None:
            raise ValueError(f"Failed to load reference image: {ref_image_path}")
        # save the original copy
        self.output_manager.save_reference_image(Path(ref_image_path).name, ref_image_bgr)
        
        # preprocess
        ref_preprocessor = Preprocessor(config= self.preprocess_config)
        try:
            preprocessed_ref_image = ref_preprocessor.preprocess(ref_image_bgr)
            if preprocessed_ref_image is None:
                raise ValueError("Preprocessing returned None for reference image.")
            # Ön işlenmiş halini de (debug için) kaydedebiliriz
            # self.output_manager.save_preprocessed_image("reference", preprocessed_ref_image)
        except Exception as e:
            logger.error(f"Preprocessing failed for reference image: {e}", exc_info=True)
            raise ValueError(f"Preprocessing failed for reference: {e}")
        
        # segmentation starts
        ref_seg_params = self.config.get('segmentation_params', {})
        default_k = ref_seg_params.get('predefined_k', 2)
        k_range_ref = ref_seg_params.get('k_values', list(range(2, 9)))
        
        kmeans_result, som_result, determined_k = segment_reference_image(
            preprocessed_image=preprocessed_ref_image,
            dbn=self.dbn,
            scalers=[self.scalers['scaler_x'], self.scalers['scaler_y'], self.scalers['scaler_y_ab']],
            default_k=default_k,
            k_range=k_range_ref
        )
        
        # Bu fonksiyonu visualization.py'de güncelledikten sonra buraya ekleyebiliriz.
        # Şimdilik sadece orijinali kaydeden OutputManager metodunu çağırıyoruz.
        if ref_image_bgr is not None:
            self.output_manager.save_reference_summary(
                Path(ref_image_path).name, ref_image_bgr # Şimdilik sadece orijinali kaydet
                # TODO: Update save_reference_summary in OutputManager/Visualization 
                #       to accept kmeans_result and som_result for plotting.
            )
            
         # 4. Hedef Renkleri Çıkar (K-Means sonucundan)
        if not kmeans_result or not kmeans_result.is_valid():
             raise ValueError("Reference K-Means segmentation failed, cannot extract target colors.")

        try:
            avg_colors_rgb = [tuple(c) for c in kmeans_result.avg_colors]
            target_colors_lab = convert_colors_to_cielab(avg_colors_rgb)
            if not isinstance(target_colors_lab, np.ndarray) or target_colors_lab.size == 0:
                raise ValueError("Could not extract valid LAB target colors from K-Means result.")
        except Exception as e:
            logger.error(f"Error extracting target colors from reference: {e}", exc_info=True)
            raise ValueError(f"Error extracting target colors: {e}")

        logger.info(f"Reference processed. Target LAB colors shape: {target_colors_lab.shape}, Determined k: {determined_k}")

        # Target renkleri, K-Means sonucunu ve SOM sonucunu döndür
        return target_colors_lab, kmeans_result, som_result
    
       
    def _run_test_image_analysis(
        self,
        target_colors_lab: np.ndarray,
        ref_kmeans_result: Optional[SegmentationResult],
        ref_som_result: Optional[SegmentationResult]
    ) -> List[Dict[str, Any]]:
        """ loops all the test images and analyses"""
        
        logger.info("Starting test image analysis loop...")
        all_delta_e_results = []
        
        # load the images and process
        for image_path_str in self.config['test_images']:
            image_name_stem = Path(image_path_str).stem
            logger.info(f"--- Processing test image: {image_name_stem} ---")
            
            image_data = load_image(image_path_str)
            if image_data is None:
                logger.warning(f"Skipping {image_name_stem}, could not load image.")
                continue
            
            # inform the OutputManager which image is processed
            self.output_manager.set_current_image_stem(image_name_stem)
            
            results_for_image = self._process_single_test_image(
                image_path_str,
                image_data,
                target_colors_lab,
                ref_kmeans_result,
                ref_som_result
            )
            all_delta_e_results.extend(results_for_image)
            
        self.output_manager.set_current_image_stem(None) # İşlem bitince temizle
        logger.info("Test image analysis loop finished.")
        return all_delta_e_results
    
    def _process_single_test_image(
        self,
        image_path: str,
        image_data: np.ndarray,
        target_colors_lab: np.ndarray,
        reference_kmeans_result: Optional[SegmentationResult],
        reference_som_result: Optional[SegmentationResult]
    ) -> List[Dict[str, Any]]:
        """
        Tek bir test görüntüsünü işler (ön işleme, segmentasyon, Delta E).
        Bu, eski main.py'deki 'process_single_test_image' fonksiyonunun TAMAMLANMIŞ halidir.
        """
        image_name = Path(image_path).stem
        single_image_delta_e_results = []
        
        with timer(f"Single image processing for {image_name}"):
            preprocessor = Preprocessor(config = self.preprocess_config)
            try:
                preprocessed_image = preprocessor.preprocess(image_data)
                if preprocessed_image is None:
                    raise ValueError("Preprocessing returned None")
                self.output_manager.save_preprocessed_image(image_name, preprocessed_image) 
            except Exception as e:
                logger.error(f"Preprocessing failed for {image_name}: {e}", exc_info=True)
                return [] # Bu görüntü için boş liste döndür
            
            # 2. Segmentasyon (k_type döngüsü)
            seg_params = self.config.get('segmentation_params', {})
            
            for k_type in ['determined', 'predefined']:
                with timer(f"Segmentation ({image_name}) k_type: {k_type}"):
                    try:
                        # --- Öncelik 3: Config nesnelerini burada oluştur ---
                        seg_config = SegmentationConfig(
                            target_colors=target_colors_lab, 
                            distance_threshold=seg_params.get('distance_threshold', 0.7),
                            predefined_k=seg_params.get('predefined_k', 2),
                            k_values=seg_params.get('k_values', [2, 3, 4, 5]),
                            som_values=seg_params.get('som_values', [2, 3, 4, 5]),
                            k_type=k_type,
                            methods=seg_params.get('methods', ['kmeans_opt', 'kmeans_predef', 'som_opt', 'som_predef', 'dbscan']),
                            dbscan_eps=seg_params.get('dbscan_eps', 10.0),
                            dbscan_min_samples=seg_params.get('dbscan_min_samples', 5)
                        )
                        
                        # --- Öncelik 4: Yeni ModelConfig ---
                        model_config = ModelConfig(
                            dbn=self.dbn,
                            scalers=[self.scalers['scaler_x'], self.scalers['scaler_y'], self.scalers['scaler_y_ab']],
                            reference_kmeans_result=reference_kmeans_result, 
                            reference_som_result=reference_som_result
                        )

                        # Segmenter, ModelConfig aracılığıyla artık ham nesnelere sahip.
                        segmenter = Segmenter(preprocessed_image, seg_config, model_config, self.output_manager)
                        
                        # Segmenter'ın process metodu, 'save_segmentation_result' 
                        # fonksiyonunu (self.output_manager aracılığıyla) zaten çağırıyor.
                        processing_result = segmenter.process() 

                        if not processing_result.results:
                            logger.warning(f"No segmentation results for {image_name} k_type={k_type}")
                            continue

                        # 3. Delta E Hesaplanması
                        # Neden? Segmentasyon bitti, şimdi bu sonuçların "ne kadar iyi"
                        # olduğunu referans renklerimizle (target_colors_lab) karşılaştırıyoruz.
                        color_metric_calculator = ColorMetricCalculator(target_colors_lab) 

                        for method_name, result in processing_result.results.items():
                            if not result.is_valid():
                                logger.warning(f"Invalid result for {method_name} on {image_name}")
                                continue
                            
                            segmented_rgb_colors = result.avg_colors
                            if not segmented_rgb_colors: 
                                logger.warning(f"No avg RGB colors for {method_name} on {image_name}")
                                continue
                            
                            try:
                                # Eski main.py'deki Delta E hesaplama mantığı (olduğu gibi taşındı)
                                segmented_lab_traditional = convert_colors_to_cielab(segmented_rgb_colors)
                                segmented_lab_dbn = convert_colors_to_cielab_dbn(
                                    self.dbn, self.scalers['scaler_x'], self.scalers['scaler_y'], self.scalers['scaler_y_ab'], 
                                    segmented_rgb_colors
                                )

                                if segmented_lab_traditional.size == 0 or segmented_lab_dbn.size == 0:
                                     logger.warning(f"Color conversion failed for {method_name} on {image_name}. Skipping Delta E.")
                                     continue

                                delta_e_traditional_list = color_metric_calculator.compute_all_delta_e(segmented_lab_traditional)
                                delta_e_dbn_list = color_metric_calculator.compute_all_delta_e(segmented_lab_dbn)

                                avg_delta_e_traditional = np.mean([d for d in delta_e_traditional_list if d != float('inf')])
                                avg_delta_e_dbn = np.mean([d for d in delta_e_dbn_list if d != float('inf')])

                                if np.isnan(avg_delta_e_traditional): avg_delta_e_traditional = float('inf')
                                if np.isnan(avg_delta_e_dbn): avg_delta_e_dbn = float('inf')

                                # Sonuçları (CSV için) listeye ekle
                                single_image_delta_e_results.append({
                                    'dataset': self.output_manager.dataset_name, 
                                    'image': image_name,
                                    'method': method_name.replace('_opt', '').replace('_predef', ''),
                                    'k_type': k_type,
                                    'n_clusters': result.n_clusters,
                                    'traditional_avg_delta_e': avg_delta_e_traditional,
                                    'pso_dbn_avg_delta_e': avg_delta_e_dbn,
                                    'processing_time': result.processing_time
                                })
                                logger.info(f"-> {method_name} ({k_type}) on {image_name}: "
                                           f"Avg Delta E Traditional={avg_delta_e_traditional:.2f}, "
                                           f"Avg Delta E DBN={avg_delta_e_dbn:.2f}, "
                                           f"k={result.n_clusters}")

                            except Exception as e:
                                logger.error(f"Delta E calculation failed for {method_name} on {image_name}: {e}", exc_info=True)
                                continue # Sonraki metoda geç

                    except Exception as e:
                        logger.error(f"Segmentation loop failed for {image_name} with {k_type}: {e}", exc_info=True)
                        continue # Sonraki k_type'a geç
        
        return single_image_delta_e_results

    def _save_and_summarize_results(self, all_delta_e: List[Dict[str, Any]]):
        """
        Toplanan tüm Delta E sonuçlarını kaydeder ve konsola özetler.
        Bu, eski main.py'deki 'save_and_summarize_results' fonksiyonunun TAMAMLANMIŞ halidir.
        """
        if not all_delta_e:
            logger.warning("No Delta E results to save or summarize.")
            return
        
        logger.info(f"Saving {len(all_delta_e)} total Delta E results to CSV...")
        self.output_manager.save_delta_e_results(all_delta_e)
        
        # Konsola özet yazdırma mantığı (eski main.py'den)
        try:
            import pandas as pd
            df = pd.DataFrame(all_delta_e)
            
            logger.info("--- Overall Results Summary ---")
            
            # Yöntemlere göre grupla (k_type'ı ayırmadan)
            summary = df.groupby('method').agg(
                avg_traditional_delta_e=('traditional_avg_delta_e', 'mean'),
                avg_pso_dbn_delta_e=('pso_dbn_avg_delta_e', 'mean'),
                avg_processing_time=('processing_time', 'mean')
            ).reset_index()
            
            logger.info("\n" + summary.to_string(float_format="%.3f"))
            
            # k_type'a göre detaylı grupla
            logger.info("--- Detailed Results by k_type ---")
            detailed_summary = df.groupby(['method', 'k_type']).agg(
                avg_traditional_delta_e=('traditional_avg_delta_e', 'mean'),
                avg_pso_dbn_delta_e=('pso_dbn_avg_delta_e', 'mean'),
                avg_processing_time=('processing_time', 'mean'),
                avg_n_clusters=('n_clusters', 'mean')
            ).reset_index()

            logger.info("\n" + detailed_summary.to_string(float_format="%.3f"))
            logger.info("--- End of Summary ---")

        except ImportError:
            logger.warning("Pandas not installed. Skipping console summary.")
        except Exception as e:
            logger.error(f"Failed to generate console summary: {e}", exc_info=True)