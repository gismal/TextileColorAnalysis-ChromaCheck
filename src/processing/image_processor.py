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