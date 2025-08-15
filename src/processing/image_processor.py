import os
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import gc
from functools import lru_cache

from config import PreprocessingConfig, SegmentationConfig, ProcessingConfig
from preprocessing import Preprocessor, PreprocessingResult, PreprocessingError
from segmentation import SegmentationOrchestrator, SegmentationResult, SegmentationConfig as SegConfig

logger = logging.getLogger(__name__)

## Type Definitions and Protocols

class ProgressCallback(Protocol):
    """Protocol for progress callback functions"""
    def __call__(sefl, stage: str, progress: float, message: str = "") -> None:
            """Called to report processing progress"""
            ...
            
class ResultCallback(Protocol):
    """Protocol for result callback functions"""
    def __call__(self, result: Any) -> None:
        """Called when a processing stage completes"""
        ...

#Type aliases       
ImageArray = np.ndarray
ColorArray = np.nd.array

## ENUMS
class ProcessingStage(Enum):
    """Enumeration of processing pipeline stages"""
    LOADING = "loading"
    PREPROCESSING = "preprocessing"
    SEGMENTATION = "segmentation"
    COLOR_CONVERSION = "color_conversion"
    SIMILARITY_ANALYSIS = "similarity_analysis"
    RESULT_COMPILATION = "result_compilation"
    SAVING = "saving"
    
class ProcessingStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    
class SegmentationMethod(Enum):
    """Enumeration of available segmentation methods"""
    KMEANS_OPTIMAL = "kmeans_optimal"
    KMEANS_PREDEFINED = "kmeans_predefined"
    DBSCAN = "dbscan"
    SOM_OPTIMAL = "som_optimal"
    SOM_PREDEFINED = "som_predefined"
    
## Exceptions
class ProcessingError(Exception):
    def __init__(self, message: str, stage: Optional[ProcessingStage] = None, **kwargs):
        super().__init__(message)
        self.stage = stage
        self.details = kwargs
        
class ResourceError(ProcessingError):
    """Exception for resource-related errors"""
    pass

class ValidationError(ProcessingError):
    """Exception for validation errors"""
    pass

## Result classes
@dataclass
class StageResult:
    """Result of a single processing stage"""
    stage: ProcessingStage
    status: ProcessingStatus
    data: Any = None
    processing_time: float = 0.0
    memory_used: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_successful(self) -> bool:
        """Check if the stage completed successfully"""
        return self.status = ProcessingStatus.COMPLETED and self.data is not None
    
@dataclass
class ProcessingContext:
    """Context info passed between processing stages"""
    image_path: str 
    original_image: Optional[ImageArray] = None
    preprocessed_image: Optional[ImageArray] = None
    target_colors: Optional[ColorArray] = None
    reference_data: Dict[str, Any] = field(default_factory = dict)
    processing_config: Optional[ProcessingConfig] = None
    stage_results: Dict[ProcessingStage, StageResult] = field(default_factory = dict)
    
    def add_stage_result(self, result: StageResult) -> None:
        """Add a stage result to the context"""
        self.stage_result[result.stage] = result
        
@dataclass
class SegmentationResult:
    method: str
    segmented_image: ImageArray
    avg_colors: List[ColorArray]
    avg_colors_lab: List[ColorArray]
    avg_colors_lab_dbn: List[ColorArray]
    labels: np.ndarray
    n_cluster: int
    processing_time: float
    quality_score: float = 0.0
    similarity_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory = dict)
    
    def is_valid(self) -> bool:
        return (
            self.segmented_image is not None and
            self.segmented_image.size > 0 and
            len(self.avg_colors) > 0 and
            self.labels is not None
        )
        
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the segmentation result"""
        return {
            'method': self.method,
            'n_clusters': self.n_clusters,
            'processing_time': self.processing_time,
            'quality_score': self.quality_score,
            'similarity_score': self.similarity_score
            'valid': self.is_valid()
        }
        
@dataclass
class ComprehensiveProcessingResult:
    """Comprehensive result of the entire processing pipeline"""
    image_path: str
    processing_status: ProcessingStatus
    stage_results: Dict[ProcessingStage, StageResult] = field(default_factory = dict)
    segmentation_results: Dict[str, SegmentationResult] = field(default_factory = dict)
    similarity_scores: Dict[str, float] = field(default_factory = dict)
    best_matches: Dict[str, List] = field(default_factory = dict)
    best_method: Optional[str] = None
    total_processing_time: float = 0.0
    memory_peak: float = 0.0
    error_log: List[str] = field(default_factory = list)
    
    def get_successful_methods(self) -> List[str]:
        """Get list of successfully processed methods"""
        return [method for method, result in self.segmentation_results.items()
                if result.is_valid()]
        
    def get_best_result(self) -> Optional[SegmentationResult]:
        """Get the best segmentation result"""
        successful_results = {k: v for k, v in self.segmentation_results.items()
                              if v.is_valid()}
        
        if not successful_results:
            return None
        
        best_method = max(successful_results.keys(),
                          key = lambda k: (successful_results[k].quality_score + 
                                      successful_results[k].similarity_score) / 2)
        
        self.best_method = best_method
        return successful_results[best_method]
    
    def export_summary(self) -> Dict[str, Any]:
        """export a summary of the processing results"""
        return {
            'image_path': self.image_path,
            'status': self.processing_status.value,
            'successful_methods': self.get_successful_methods(),
            'best_method': self.best_method,
            'total_time': self.total_processing_time,
            'memory_peak': self.memory_peak,
            'stage_summary': {
                stage.value: result.is_successful()
                for stage, result in self.stage_results.items()
            },
            'method_summaries': {
                method: result.get_summary()
                for method, result in self.segmentation_results.items()
            }
        }
        
## Processing Stages - Strategy Pattern
class ProcessingStageBase(ABC):
    """Abstract base class for processing steps"""
    
    def __init__(self, stage: ProcessingStage):
        self.stage = stage
        self.logger = logging.getLogger(f"{__name__.{self.__class__.__name__}")
        
    @abstractmethod
    def execute(self, context: ProcessingContext, progress_callback: Optional[ProgressCallback] = None) -> StageResults:
        """execute the processing stage"""
        pass
    
    def _create_result(self, status: ProcessingStatus, data: Any = None, processing_time: float = 0.0,
                       error_msg: str = None, **metadata) -> StageResult:
        """Helper to create stage result"""
        return StageResult(
            stage = self.stage,
            status = self.status,
            data = data,
            processing_time = processing_time,
            error_message = error_message,
            metadata = metadata
        )
        
class ImageLoadingStage(ProcessingStageBase):
    """Stage for loading and validating images"""
    
    def __init__(self):
        super().__init__(ProcessingStage.LOADING)
        
    def execute(self, context: ProcessingContext, progress_callback: Optional[ProgressCallback] = None)
        """load and validate the image"""
        start_time = time.time()
        
        try:
            if progress_callback:
                progress_callback(self.stage.value, 0.0, "Starting image loading")
                
            image = cv2.imread(context.image_path)
            if image is None:
                raise ValidationError(f"Failed to load image: {context.image_path}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if not self._validate_image(image):
                raise ValidationError("Invalid image format or size")
            
            context.original_image = image
            
            if progress_callback:
                progress_callback(self.stage.value, 1.0, f"Image loaded: {image.shape}")
            
            processing_time = time.time() - start_time
            self.logger.info(f"Image loaded successfully: {context.image_path}, shape: {image.shape}")
            
            
            return self._create_result(
                ProcessingStatus.COMPLETED,
                data = image,
                processing_time = processing_time,
                width = image_shape[1],
                height = image_shape[0],
                channels = image_shape[2]
            )
            
        except Exception as e:
            processing_time= time.time() - start_time
            error_msg = f"Image loading failed: {str(e)}"
            self.logger.error(error_msg)
            
            return self._create_result(
                ProcessingStatus.FAILED
                processing_time = processing_time,
                error_message = error_msg
            )
        
        def _validate_image(self, image: ImageArray) -> bool:
            """Validate image format and properties"""
            return (image is not None and 
                    len(image.shape) == 3 and
                    image.shape[2] == 3 and
                    image.size > 0 and
                    image.shape[0] > 0 and image.shape[1] > 0)
            
class PreprocessingStage(ProcessingStageBase):
    """Stage for image preprocessing"""
    
    def __init__(self, preprocessing_config: PreprocessingConfig):
        super().__init__(ProcessingStage.PREPROCESSING)
        self.config = preprocessing_config
        self.preprocessor = Preprocessor(preprocessing_config)
        
    def execute(self, context: ProcessingContext, progress_callback: Optional[ProgressCallback] = None) -> StageResult:
        """Execute preprocessing on the image"""
        start_time = time.time()
        
        try:
            if context.original_image is None:
                raise ProcessingError("No image available for preprocessing")
            
            if progress_callback:
                progress_callback(self.stage.value, 0.0, "Starting preprocessing")
                
            preprocessing_result = self.preprocessor.preprocess(context.original_image)
            
            context.preprocessed_image = preprocessing_result.processed_image
            
            if progress_callback:
                progress_callback(self.stage.value, 1.0, "Preprocessing completed")
                
            processing_time = time.time() - start_time
            self.logger.info(f"Preprocessing completed in {processing_time: .2f}s")
            
            return self._crete_result(
                ProcessingStatus.COMPLETED,
                data = preprocessing_result,
                processing_time = processing_time,
                original_shape = context.original_image.shape,
                processed_shape = preprocessing_result.processed_image.shape,
                preprocessing_info = preprocessing_result.processing_info
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Preprocessing failed: {str(e)}"
            self.logger.error(error_msg)
            
            return self._create_result(
                ProcessingStatus.FAILED,
                processing_time = processing.time,
                error_message = error_msg
            )
            
class SegmentationStage(ProcessingStageBase):
    """Stage for image segmentation with multiple methods"""
    
    def __init__(self, segmentation_config: SegmentationConfig, methods: List[SegmentationMethod]):
        super().__init__(ProcessingStage.SEGMENTATION)
        self.config = segmentation_config
        self.methods = methods
        
    def execute(self, context: ProcessingContext, progress_callback: Optional[ProgressCallback] = None) -> StageResult:
        """execute segmentation using configuration methods"""
        start_time = time.time()
        
        try:
            if context.preprocessed_image is None:
                raise ProcessingError("No preprocessed image available for segmentation")
            
            if progress_callback:
                progress_callback(self.stage.value, 0.0, "Starting segmentation")
                
            seg_config = self._convert_to_seg_config(context)
            
            from segmentation import ModelConfig
            models = ModelConfig()
            
            class DummyOutputManager:
                def get_current_image_name(self):
                    return Path(context.image_path).stem
                def save_segmentation_image(self, name, method, image):
                    pass
                processed_dir = "/tmp"
                
                output_manager = DummyOutputManager()
                orchestrator = SegmentationOrchestrator(
                    context.preprocessed_image, seg_config, models, output_manager
                )
                
                method_names = [method.value for method in self.models]
                seg_config.methods = method_names
                
                results = {}
                total_methods = len(method_names)
                
                for i, method_name in enumerate(method_names):
                    try:
                        if progress_callback:
                            progress = i / total_methods
                            progress_callback(self.stage.value, progress, f"Processing {method_name}")
                            
                        method_result, _ = orchestrator._process_single_method(method_name)     
                        
                        if method_result.is_valid():
                            enhanced_result = self._convert_to_enhanced_result(method_result)
                            results[method_name] = enhanced_result
                            self.logger.info(f"Segmentation method {method_name} completed successfully")
                        else:
                            self.logger.warning(f"Error in segmentation method {method_name}: {e}")
                            continue
                        
                context.intermediate_results['segmentation_results'] = results
                
                if progress_callback:
                    progress_class(self.stage.value, 1.0, f"Segmentation completed: {len(results)} methods")
                    
                progressing_time = time.time() - start_time
                self.logger.info(f"Segmentation stage completed in {processing_time:.2f}s with {len(results)} successful methods")
            
                return self._create_result(
                    ProcessingStatus.COMPLETED,
                    data = results
                    processing_time = processing_time
                    successful_methods= len(results)
                    total_methods = total_methods
                    method_names = list(results.keys())
                )
                
            except Exception as e:
                processing_time = time.time() - start_time
                error_msg = f"Segmentation stage failed: {str(e)}"
                self.logger.error(error_msg)
                
                return self._create_result(
                ProcessingStatus.FAILED,
                processing_time=processing_time,
                error_message=error_msg
            )
    
    def _convert_to_seg_config(self, context: ProcessingContext) -> SegConfig:
        """Convert processing config to segmentation config."""
        target_colors = context.target_colors if context.target_colors is not None else np.array([])
        
        return SegConfig(
            target_colors = target_colors
            distance_threshold = getattr(context.processing_config, 'distance_threshold', 50.0),
            predefined_k = getattr(self.config, 'predefined_k', 5),
            k_values = getattr(self.config, 'k_value', list(range(3,11))),
            som_values = getattr(self.config, 'som_values', list(range(3,11))),
            k_type = 'determined'
            methods = [method.value for method in self.methods],
            quantization_colors = getattr(self.config, 'quantization_color', 50 ),
            parallel_processing = getattr(self.config, 'parallel_processing', True)            
        )
    
    def _convert_to_enhanced_results(self, seg_result) -> SegmentationResultEnhanced:
        lab_colors = []
        for rgb_color in seg_result.avg_colors:
            try:
                rgb_array = np.array([[[rgb_color]]], dtype= np.uint8)
                lab_array = cv2.cvtColor(rgb_array, cv2.COLORRGB2LAB)
                lab_colors.append(lab_array[0,0].astype(np.float32))
            except:
                lab_colors.append(np.array([50.0, 0.0, 0.0]))
                
        return SegmentationResultEnhanced(
            method = seg_result.method,
            segmented_image = seg_result.segmented_image,
            avg_colors = seg_result.avg_colors,
            avg_colors_lab = lab_colors,
            avg_colors_lab_dbn = None,
            labels =seg_result.labels,
            n_clusters = seg_result.n_clusters,
            processing_time = seg_result.processing_time,
            similarity_score=np.mean(seg_result.similarity) if seg_result.similarity else 0.0,
            metadata=getattr(seg_result, 'metadata', {})
        )
        
class ColorConversionStage(ProcessingStageBase):
    """Stage for advanced color space conversion using DBN"""
    
    def __init__(self, dbn= None, scalers= None):
        super().__init__(ProcessingStage.COLOR_CONVERSION)
        self.dbn = dbn
        self.scalers = scalers
        self.has_dbn = dbn is not None and scalers is not None
        
    def execute(self, context: ProcessingContext, progress_callback: Optimal[ProgressCallback] = None) -> StageResults:
        start_time = time.time()
        
        try: 
            segmentation_result = context.intermediate_results.get('segmentation_results', {})
            
            if not segmentation_results:
                raise ProcessingError("No segmentation results available for color conversion")
            
            if progress_callback:
                progress_callback(self.stage.value, 0.0, "Starting color conversion")
            
            conversion_results = {}
            
            if self.has_dbn:
                # dbn color conversion
                total_results = len(segmentation_results)
                
                for i, (method_name, result) in enumerate(segmentation_results.items())
                    if progress_callback:
                        progress = i / total_results
                        progress_callback(self.stage.value, progress, f"Converting colors")
                        
                    conversion_results = {}
                    
                    if self.has_dbn:
                        total_results = len(segmentation_results)
                        
                        for i, (method_name, result) in enumerate(segmentation_results.items()):
                            if progress_callback:
                                progress = i / total_results
                                progress_callback(self.stage.value, progress, f"Converting colors for {method_name}")
                            
                            try:
                                dbn_colors = self._convert_colors_dbn(result.avg_colors)
                                result.avg_colors_lab_dbn = dbn_colors
                                conversion_results[method_name] = len(dbn_colors)
                                
                            except Exception as e:
                                self.logger.warning(f"DBN conversion failed for {method_name}: {e}")
                                conversion_results[method_name] = 0
                    else:
                        # No DBN available, skip conversion
                        self.logger.info("No DBN available, skipping advanced color conversion")
                        for method_name in segmentation_results.keys():
                            conversion_results[method_name] = 0
                            
                    if progress_callback:
                        progress_callback(self.stage.value, 1.0, "Color conversion completed" )

                    processing_time = time.time() - start_time
                    self.logger.info(f"Color conversion completed in {processing_time: .2f}s")
                    
                    return self._create_result(
                        ProcessingStatus.COMPLETED,
                        data = conversion_result, 
                        processing_time = processing_time,
                        dbn_available = self.has_dbn,
                        methods_processed = len(conversion_results)
                    )
                
                except Exception as e:
                    processing_time = time.time() - start_time
                    error_msg = f"Color conversion failed: {str(e)}"
                    self.logger.error(error_msg)
                    
                    return self._create_result(
                        ProcessingStatus.FAILED, 
                        processing_time= processing_time,
                        error_message = error_msg
                    )
    
    def _convert_colors_dbn(self, avg_colors: List[ColorArray]) -> List[ColorArray]:
        """Convert colors using PSO-optimized DBN."""
        try:
            from src.models.pso_dbn import convert_colors_to_cielab_dbn
            scaler_x, scaler_y, scaler_y_ab = self.scalers
            
            return convert_colors_to_cielab_dbn(
                self.dbn, scaler_x, scaler_y, scaler_y_ab, avg_colors
            )
        except Exception as e:
            self.logger.error(f"DBN color conversion error: {e}")
            return []

class SimilarityAnalysisStage(ProcessingStageBase):
    """Stage for analysing color similarity and finding best matches"""
    def __init__(self, target_colors: ColorArray, reference_data: Dict):
        super().__init__(ProcessingStage.SIMILARITY_ANALYSIS)
        self.target_colors = target_colors
        self.reference_data = reference_data
        
    def execute(self, context: ProcessingContext, progress_callback: Optional[ProgressCallback] = None) -> StageResult:
        """Execute similarity analysis"""
        start_time = time.time()
        
        try:
            segmentation_results = context.intermediate_results.get('segmentation_results', {})
            
            if not segmentation_results:
                raise ProcessingError("No segmentation results available for similar analysis")
            
            if progress_callback:
                progress_callback(self.stage.value, 0.0, "Starting similarity analysis")
                
            similarity_scores = {}
            best_matches  = {}
            total_methods = len(segmentation_results)
            
            for i, (method_name, result) in enumerate(segmentation_results.items()):
                if progress_callback:
                    progress = i / total_methods
                    progress_callback(self.stage.value, progress, f"Analyzing {method_name}")
                
                try:
                    # Calculate similarity score
                    similarity_score = self._calculate_similarity(result)
                    similarity_scores[method_name] = similarity_score
                    
                    # Find best matches
                    best_match = self._find_best_matches(result, method_name)
                    best_matches[method_name] = best_match
                    
                    # Update result with similarity score
                    result.similarity_score = similarity_score
                    
                except Exception as e:
                    self.logger.warning(f"Similarity analysis failed for {method_name}: {e}")
                    similarity_scores[method_name] = 0.0
                    best_matches[method_name] = []
                    
            context.intermediate_results['similarity_scores'] = similarity_scores
            context.intermediate_results['best_matches'] = best_matches
            
            if progress_callback:
                progress_callback(self.stage.value, 1.0, "Similarity analysis completed")
            
            processing_time = time.time() - start_time
            self.logger.info(f"Similarity analysis completed in {processing_time:.2f}s")
            
            return self._create_result(
                ProcessingStatus.COMPLETED,
                data={'similarity_scores': similarity_scores, 'best_matches': best_matches},
                processing_time=processing_time,
                methods_analyzed=len(similarity_scores),
                average_similarity=np.mean(list(similarity_scores.values())) if similarity_scores else 0.0
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Similarity analysis failed: {str(e)}"
            self.logger.error(error_msg)
            
            return self._create_result(
                ProcessingStatus.FAILED,
                processing_time=processing_time,
                error_message=error_msg
            )
            
    def _calculate_similarity(self, result: SegmentationResultEnhanced) -> float:
        """Calculate similarity score for segmentation"""
        try:
            from src.utils.image_utils import calculate_similarity
            
            segmentation_data = {
                'avg_colors_lab': result.avg_colors_lab,
                'avg_colors': result.avg_colors
            }
            
            return calculate_similarity(segmentation_data, self.target_colors)
            
        except Exception as e:
            self.logger.warning(f"Similarity calculation error: {e}")
            return 0.0
    
    def _find_best_matches(self, result: SegmentationResultEnhanced, method_name: str) -> List:
        """Find best matches between segmented colors and reference colors."""
        try:
            from src.utils.image_utils import find_best_matches
            
            segmentation_data = {
                'avg_colors_lab': result.avg_colors_lab,
                'avg_colors': result.avg_colors,
                'segmented_image': result.segmented_image,
                'labels': result.labels
            }
            
            # Use appropriate reference data
            reference = (self.reference_data.get('reference_kmeans', {}) 
                        if 'kmeans' in method_name or 'dbscan' in method_name 
                        else self.reference_data.get('reference_som', {}))
            
            return find_best_matches(segmentation_data, reference)
            
        except Exception as e:
            self.logger.warning(f"Best match calculation error: {e}")
            return []

# Processing Pipeline
class ProcessingPipeline:
    """Processing pipeline that orchestrates all stages"""
    
    def __init__(self, preprocessing_config: Optional[PreprocessingConfig] = None,
                 segmentation_config: Optional[SegmentationConfig] = None,
                 processing_config: Optional[ProcessingConfig] = None
                 ):
        """initialize processing pipeline with configurations"""
        self.preprocessing_config = processing_config or PreprocessingConfig()
        self.segmentation_config = segmentation_config or SegmentationConfig()
        self.processing_config = processing_config or ProcessingConfig()
        
        self.logger = logging.getLogger(f"{__name__}.ProcessingPipeline")
        
        # pipeline stages
        self.stages = []
        self._setup_stages()
        
        self.progress_callback: Optional[ProgressCallback] = None
        self.result_callback: Optional[ResultCallback] = None
        
        self._is_cancelled = False
        
    def _setup_stages(self):
        """setup the processing pipeline stages"""
        self.stages = [
            ImageLoadingStage(),
            PreprocessingStage(self.preprocessing_config)
        ]

    def set_progress_callback(self, callback: ProgressCallback):
        self.progress_callback = callback
        
    def set_result_callback(self, callback: ResultCallback):
        self.result_callback = callback
        
    def cancel(self):
        self._is_cancelled = True
        self.logger.info("Processing pipeline cancellation requested")
        
    @contextmanager
    def _memory_monitor(self):
        """monitor memory usage"""
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024 #MB
        
        try:
            yield
        finally:
            current_memory = process.memory_info().rss / 1024 / 1024 #MB
            memory_increase = current - initial_memory
            if memory_increase > 100: # More than 100MB increase
                self.logger.warning(f"High memory usage detected: +{memory_increase:.1f}MB")
                gc.collect()
                
    def process_image(self,
                      image_path: str,
                      target_colors: ColorArray,
                      reference_data: Dict[str, Any],
                      segmentation_methods: List[SegmentationMethod],
                      dbn= None,
                      scalers = None) -> ComprehensiveProcessingResult:
        """Process a single image tru the complete pipeline"""
        start_time = time.time()
        result = ComprehensiveProcessingResult(
            image_path = image_path,
            processing_status = ProcessingStatus.RUNNING
        )
        
        # Create processing context
        context = ProcessingContext(
            image_path=image_path,
            target_colors=target_colors,
            reference_data=reference_data,
            processing_config=self.processing_config
        )
        
        try:
            self.logger.info(f"Starting processing pipeline for: {image_path}")
            
            with self._memory_monitor():
                total_stages = len(self.stages) + 3
                current_stage = 0
                
                if self._is_cancelled:
                    result.processing_status = ProcessingStatus.CANCELLED
                    return result
                
                loading_stage = ImageLoadingStage()
                loading_result = loading_stage.execute(context, self._create_stage_progress_callback(current_stage, total_stages))
                result.stage_results[ProcessingStage.LOADING] = loading_result
                context.add_stage_result(loading_result)
                current_stage += 1

                if not loading_result.is_successful():
                    raise ProcessingError("Image loading failed", ProcessingStage.LOADING)
                
                # Stage 2: Preprocessing
                if self._is_cancelled:
                    result.processing_status = ProcessingStatus.CANCELLED
                    return result
                
                preprocessing_stage = PreprocessingStage(self.preprocessing_config)
                preprocessing_result = preprocessing_stage.execute(context, self._create_stage_progress_callback(current_stage, total_stages))
                result.stage_results[ProcessingStage.PREPROCESSING] = preprocessing_result
                context.add_stage_result(preprocessing_result)
                current_stage += 1
                
                if not preprocessing_result.is_successful():
                    raise ProcessingError("Preprocessing failed", ProcessingStage.PREPROCESSING)
                
                # Stage 3: Segmentation
                if self._is_cancelled:
                    result.processing_status = ProcessingStatus.CANCELLED
                    return result
                
                segmentation_stage = SegmentationStage(self.segmentation_config, segmentation_methods)
                segmentation_result = segmentation_stage.execute(context, self._create_stage_progress_callback(current_stage, total_stages))
                result.stage_results[ProcessingStage.SEGMENTATION] = segmentation_result
                context.add_stage_result(segmentation_result)
                current_stage += 1
                
                if not segmentation_result.is_successful():
                    raise ProcessingError("Segmentation failed", ProcessingStage.SEGMENTATION)
                
                # Store segmentation results
                segmentation_results = context.intermediate_results.get('segmentation_results', {})
                result.segmentation_results = segmentation_results
                
                # Stage 4: Color Conversion
                if self._is_cancelled:
                    result.processing_status = ProcessingStatus.CANCELLED
                    return result
                
                color_conversion_stage = ColorConversionStage(dbn, scalers)
                conversion_result = color_conversion_stage.execute(context, self._create_stage_progress_callback(current_stage, total_stages))
                result.stage_results[ProcessingStage.COLOR_CONVERSION] = conversion_result
                context.add_stage_result(conversion_result)
                current_stage += 1
                
                # Stage 5: Similarity Analysis
                if self._is_cancelled:
                    result.processing_status = ProcessingStatus.CANCELLED
                    return result
                
                similarity_stage = SimilarityAnalysisStage(target_colors, reference_data)
                similarity_result = similarity_stage.execute(context, self._create_stage_progress_callback(current_stage, total_stages))
                result.stage_results[ProcessingStage.SIMILARITY_ANALYSIS] = similarity_result
                context.add_stage_result(similarity_result)
                current_stage += 1
                
                if similarity_result.is_successful():
                    analysis_data = similarity_result.data
                    result.similarity_scores = analysis_data['similarity_scores']
                    result.best_matches = analysis_data['best_matches']
                    
                result.total_processing_time = time.time() - start_time
            result.processing_status = ProcessingStatus.COMPLETED
            
            # Find best method
            best_result = result.get_best_result()
            if best_result:
                result.best_method = best_result.method
            
            self.logger.info(f"Processing pipeline completed successfully for {image_path} in {result.total_processing_time:.2f}s")
            
            # Call result callback if available
            if self.result_callback:
                self.result_callback(result)
            
            return result
            
        except Exception as e:
            result.total_processing_time = time.time() - start_time
            result.processing_status = ProcessingStatus.FAILED
            error_msg = f"Processing pipeline failed: {str(e)}"
            result.error_log.append(error_msg)
            
            self.logger.error(error_msg)
            return result
        
    
    def _create_stage_progress_callback(self, 
                                        stage_index: int, 
                                        total_stages: int):
        """Create a progress callback for a specific stage"""
        def stage_progress_callback(stage: str, progress: float, message: str = ""):
            if self.progress_callback:
                overall_progress = (stage_index + progress) / total_stages
                self.progress_callback(stage, overall_progress, message)
            return stage_progress_callback
        
        def process_batch(self, 
                          image_path: List[str], 
                          target_colors: ColorArray,
                          reference_data: Dict[str, Any],
                          segmentation_methods: List[SegmentationMethod],
                          dbn = None,
                          scalers = None,
                          max_workers: int = 4) -> List[ComprehensiveProcessingResult]:
            """Process multiple images in parallel"""
            results = []
            
            self.logger.info(f"Starting batch processing of {len(image_paths)} images with {max_workers} workers")
        
            def process_single_image(image_path: str) -> ComprehensiveProcessingResult:
                """Process a single image in the batch."""
                return self.process_image(image_path, target_colors, reference_data, 
                                        segmentation_methods, dbn, scalers)
            with ThreadPoolExecutor(max_workers  =max_workers) as executor:
                # submit all tasks
                future_to_path = {
                    executor.submit(process_single_image, path): path
                    for path in image_paths
                }
                
                for future in as_completed(future_to_path):
                    image_path = future_to_path[future]
                    try: 
                        result = future.result(timeout= 600)
                        results.append(result)
                        self.logger.info(f"Batch processing completed for {image_path}")
                    
                except Exception as e:
                    error_msg = f"Batch processing failed for {image_path}: {str(e)}"
                    self.logger.error(error_msg)
                    
                    # Create failed result
                    failed_result = ComprehensiveProcessingResult(
                        image_path=image_path,
                        processing_status=ProcessingStatus.FAILED
                    )
                    failed_result.error_log.append(error_msg)
                    results.append(failed_result)
        
        successful_count = sum(1 for r in results if r.processing_status == ProcessingStatus.COMPLETED)
        self.logger.info(f"Batch processing completed: {successful_count}/{len(image_paths)} successful")
        
        return results
        
## Enhanced Image Processor - Main Class
class EnhancedImageProcessor:
    def __init__(self,
                 preprocessing_config: Optional[PreprocessingConfig] = None,
                 segmentation_config: Optional[SegmentationConfig] = None,
                 processing_config: Optional[ProcessingConfig] = None):
        """initialize the image processor"""
        self.pipeline = ProcessingPipeline(
            preprocessing_config, segmentation_config, processing_config
        )
        # Reference data
        self.target_colors: Optional[ColorArray] = None
        self.reference_data: Dict[str, Any] = {}
        
        # DBN components
        self.dbn = None
        self.scalers = None
        
        # Default segmentation methods
        self.default_methods = [
            SegmentationMethod.KMEANS_OPTIMAL,
            SegmentationMethod.DBSCAN,
            SegmentationMethod.SOM_OPTIMAL
        ]
        
        self.logger = logging.getLogger(f"{__name__}.EnhancedImageProcessor")
        self.logger.info("Enhanced Image Processor initialized")
    
    def set_reference_data(self, target_colors: ColorArray, 
                          reference_kmeans: Dict, reference_som: Dict):
        """Set reference data for similarity calculations."""
        self.target_colors = np.array(target_colors)
        self.reference_data = {
            'reference_kmeans': reference_kmeans,
            'reference_som': reference_som
        }
        self.logger.info(f"Reference data set with {len(target_colors)} target colors")
    
    def set_dbn_converter(self, dbn, scalers):
        """Set DBN model and scalers for advanced color conversion."""
        self.dbn = dbn
        self.scalers = scalers
        self.logger.info("DBN color converter initialized")
    
    def set_progress_callback(self, callback: ProgressCallback):
        """Set progress callback for processing updates."""
        self.pipeline.set_progress_callback(callback)
    
    def set_result_callback(self, callback: ResultCallback):
        """Set result callback for processing completion."""
        self.pipeline.set_result_callback(callback)
    
    def process_image(self, 
                     image_path: str,
                     methods: Optional[List[SegmentationMethod]] = None) -> ComprehensiveProcessingResult:
        """
        Process a single image through the complete pipeline.
        
        Args:
            image_path: Path to the image file
            methods: List of segmentation methods to use
            
        Returns:
            Comprehensive processing results
        """
        if self.target_colors is None:
            raise ValueError("Target colors must be set before processing")
        
        if methods is None:
            methods = self.default_methods
        
        return self.pipeline.process_image(
            image_path=image_path,
            target_colors=self.target_colors,
            reference_data=self.reference_data,
            segmentation_methods=methods,
            dbn=self.dbn,
            scalers=self.scalers
        )
    
    def process_batch(self, 
                     image_paths: List[str],
                     methods: Optional[List[SegmentationMethod]] = None,
                     max_workers: int = 4) -> List[ComprehensiveProcessingResult]:
        """
        Process multiple images in batch.
        
        Args:
            image_paths: List of image file paths
            methods: List of segmentation methods to use
            max_workers: Number of parallel workers
            
        Returns:
            List of comprehensive processing results
        """
        if self.target_colors is None:
            raise ValueError("Target colors must be set before processing")
        
        if methods is None:
            methods = self.default_methods
        
        return self.pipeline.process_batch(
            image_paths=image_paths,
            target_colors=self.target_colors,
            reference_data=self.reference_data,
            segmentation_methods=methods,
            dbn=self.dbn,
            scalers=self.scalers,
            max_workers=max_workers
        )
    
    def cancel_processing(self):
        """Cancel any ongoing processing."""
        self.pipeline.cancel()
    
    def get_available_methods(self) -> List[SegmentationMethod]:
        """Get list of available segmentation methods."""
        return list(SegmentationMethod)
    
    def validate_image(self, image_path: str) -> bool:
        """Validate if an image can be processed."""
        try:
            image = cv2.imread(image_path)
            return (image is not None and 
                    len(image.shape) == 3 and 
                    image.shape[2] == 3 and
                    image.size > 0)
        except Exception:
            return False
    
    def export_results(self, results: List[ComprehensiveProcessingResult], 
                      output_path: str, format: str = 'json'):
        """Export processing results to file."""
        try:
            export_data = {
                'metadata': {
                    'total_images': len(results),
                    'successful_images': sum(1 for r in results if r.processing_status == ProcessingStatus.COMPLETED),
                    'export_timestamp': time.time(),
                    'processor_version': '2.0.0'
                },
                'results': [result.export_summary() for result in results]
            }
            
            if format.lower() == 'json':
                import json
                with open(output_path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            elif format.lower() == 'csv':
                import pandas as pd
                # Flatten results for CSV export
                flattened_results = []
                for result in results:
                    summary = result.export_summary()
                    flattened_results.append({
                        'image_path': summary['image_path'],
                        'status': summary['status'],
                        'best_method': summary.get('best_method', ''),
                        'total_time': summary['total_time'],
                        'memory_peak': summary['memory_peak'],
                        'successful_methods': len(summary['successful_methods'])
                    })
                
                df = pd.DataFrame(flattened_results)
                df.to_csv(output_path, index=False)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            self.logger.info(f"Results exported to {output_path} in {format} format")
            
        except Exception as e:
            self.logger.error(f"Failed to export results: {e}")
            raise


# ==============================================================================
# UTILITY FUNCTIONS AND BACKWARD COMPATIBILITY
# ==============================================================================

def create_simple_processor(target_colors: ColorArray,
                           reference_kmeans: Dict,
                           reference_som: Dict,
                           dbn=None,
                           scalers=None) -> EnhancedImageProcessor:
    """Create a simple processor instance with minimal configuration."""
    
    processor = EnhancedImageProcessor()
    processor.set_reference_data(target_colors, reference_kmeans, reference_som)
    
    if dbn is not None and scalers is not None:
        processor.set_dbn_converter(dbn, scalers)
    
    return processor


def create_processor_from_legacy_params(
    target_colors: ColorArray,
    distance_threshold: float,
    reference_kmeans: Dict,
    reference_som: Dict,
    dbn,
    scalers: Tuple,
    predefined_k: int,
    eps_values: List[float],
    min_samples_values: List[int],
    output_dir: str
) -> EnhancedImageProcessor:
    """Create processor from legacy parameters for backward compatibility."""
    
    # Create configurations from legacy parameters
    preprocessing_config = PreprocessingConfig()
    
    segmentation_config = SegmentationConfig()
    segmentation_config.predefined_k = predefined_k
    if hasattr(segmentation_config, 'eps_values'):
        segmentation_config.eps_values = eps_values
    if hasattr(segmentation_config, 'min_samples_values'):
        segmentation_config.min_samples_values = min_samples_values
    
    processing_config = ProcessingConfig()
    if hasattr(processing_config, 'target_colors'):
        processing_config.target_colors = target_colors.tolist()
    if hasattr(processing_config, 'distance_threshold'):
        processing_config.distance_threshold = distance_threshold
    if hasattr(processing_config, 'output_dir'):
        processing_config.output_dir = output_dir
    
    # Create processor
    processor = EnhancedImageProcessor(
        preprocessing_config=preprocessing_config,
        segmentation_config=segmentation_config,
        processing_config=processing_config
    )
    
    # Set reference data and DBN
    processor.set_reference_data(target_colors, reference_kmeans, reference_som)
    if dbn is not None and scalers is not None:
        processor.set_dbn_converter(dbn, scalers)
    
    return processor


class ProgressTracker:
    """Simple progress tracker for monitoring processing."""
    
    def __init__(self, total_images: int = 1):
        self.total_images = total_images
        self.current_image = 0
        self.current_stage_progress = 0.0
        self.current_stage = ""
        self.start_time = time.time()
        self.logger = logging.getLogger(f"{__name__}.ProgressTracker")
    
    def __call__(self, stage: str, progress: float, message: str = ""):
        """Progress callback implementation."""
        self.current_stage = stage
        self.current_stage_progress = progress
        
        overall_progress = (self.current_image + progress) / self.total_images
        elapsed_time = time.time() - self.start_time
        
        if progress > 0:
            estimated_total_time = elapsed_time / overall_progress
            remaining_time = estimated_total_time - elapsed_time
        else:
            remaining_time = 0
        
        self.logger.info(f"Progress: {overall_progress*100:.1f}% | Stage: {stage} | "
                        f"Elapsed: {elapsed_time:.1f}s | ETA: {remaining_time:.1f}s | {message}")
    
    def next_image(self):
        """Move to next image in batch processing."""
        self.current_image += 1
        self.current_stage_progress = 0.0
    
    def reset(self, total_images: int = 1):
        """Reset progress tracker."""
        self.total_images = total_images
        self.current_image = 0
        self.current_stage_progress = 0.0
        self.start_time = time.time()


# ==============================================================================
# EXAMPLE USAGE AND TESTING UTILITIES
# ==============================================================================

def create_test_processor() -> EnhancedImageProcessor:
    """Create a processor instance for testing purposes."""
    # Create dummy configurations
    preprocessing_config = PreprocessingConfig(
        target_size=(256, 256),
        max_colors=16
    )
    
    segmentation_config = SegmentationConfig()
    processing_config = ProcessingConfig()
    
    processor = EnhancedImageProcessor(
        preprocessing_config=preprocessing_config,
        segmentation_config=segmentation_config,
        processing_config=processing_config
    )
    
    # Set dummy reference data
    dummy_colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    dummy_reference = {'avg_colors': dummy_colors}
    
    processor.set_reference_data(dummy_colors, dummy_reference, dummy_reference)
    
    return processor


def benchmark_processing(processor: EnhancedImageProcessor,
                        image_paths: List[str],
                        methods: List[SegmentationMethod],
                        output_file: Optional[str] = None) -> Dict[str, Any]:
    """Benchmark processing performance."""
    
    benchmark_results = {
        'total_images': len(image_paths),
        'methods': [method.value for method in methods],
        'results': [],
        'summary': {}
    }
    
    # Setup progress tracker
    progress_tracker = ProgressTracker(len(image_paths))
    processor.set_progress_callback(progress_tracker)
    
    start_time = time.time()
    
    # Process each image
    for i, image_path in enumerate(image_paths):
        try:
            result = processor.process_image(image_path, methods)
            
            benchmark_results['results'].append({
                'image_path': image_path,
                'processing_time': result.total_processing_time,
                'status': result.processing_status.value,
                'successful_methods': result.get_successful_methods(),
                'best_method': result.best_method,
                'memory_peak': result.memory_peak
            })
            
            progress_tracker.next_image()
            
        except Exception as e:
            logger.error(f"Benchmark error for {image_path}: {e}")
            benchmark_results['results'].append({
                'image_path': image_path,
                'processing_time': 0.0,
                'status': 'error',
                'error': str(e)
            })
    
    total_time = time.time() - start_time
    
    # Calculate summary statistics
    successful_results = [r for r in benchmark_results['results'] if r['status'] == 'completed']
    
    if successful_results:
        processing_times = [r['processing_time'] for r in successful_results]
        benchmark_results['summary'] = {
            'total_time': total_time,
            'successful_images': len(successful_results),
            'failed_images': len(image_paths) - len(successful_results),
            'average_processing_time': np.mean(processing_times),
            'median_processing_time': np.median(processing_times),
            'min_processing_time': np.min(processing_times),
            'max_processing_time': np.max(processing_times),
            'images_per_second': len(successful_results) / total_time
        }
    
    # Save benchmark results if output file is specified
    if output_file:
        import json
        with open(output_file, 'w') as f:
            json.dump(benchmark_results, f, indent=2, default=str)
        
        logger.info(f"Benchmark results saved to {output_file}")
    
    return benchmark_results


# ==============================================================================
# MAIN EXECUTION AND EXAMPLES
# ==============================================================================

if __name__ == "__main__":
    """Example usage and testing."""
    
    # Create a test processor
    processor = create_test_processor()
    
    # Setup progress tracking
    def progress_callback(stage: str, progress: float, message: str = ""):
        print(f"[{progress*100:.1f}%] {stage}: {message}")
    
    processor.set_progress_callback(progress_callback)
    
    # Example: Process a single image
    try:
        # You would replace this with actual image path
        # result = processor.process_image("path/to/your/image.jpg")
        # print(f"Processing completed. Best method: {result.best_method}")
        print("Enhanced Image Processor initialized successfully!")
        print("Available methods:", [method.value for method in processor.get_available_methods()])
        
    except Exception as e:
        print(f"Error in example: {e}")

            