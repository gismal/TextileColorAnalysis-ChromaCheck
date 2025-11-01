import logging
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path

from src.data.load_data import load_data, load_image
from src.models.dbn_trainer import DBNTrainer, DBNConfig, PSOConfig
from src.models.pso_dbn import DBN
from src.config_types import TrainConfig
from src.utils.output_manager import OutputManager

logger = logging.getLogger(__name__)

class TrainingHandler:
    """
    Handles the DBN model training workflow
    
    This class encapsulates all logic related to DBN training, including:
    1. Loading and validating the images specified for training
    2. Calling 'load_data' to process images into flattened arrays
    3. Initializing the 'DBNTrainer'
    4. Executing the 'dbn_trainer.train()' method
    
    This abstracts the "how"  of training away from the main pipeline
    """
    def __init__(self, 
                 dbn_config: DBNConfig,
                 pso_config: PSOConfig,
                 train_config: TrainConfig,
                 output_manager: OutputManager):
        """
        Initalizes the TrainingHandler
        
        Args:
            dbn_config: Config onject for the DBN architecture
            pso_config: Config object for the PSO optimizer
            train_config: Config object for training params
            output_manager: The instance of OutputManager to save input image copies
        """
        self.dbn_config = dbn_config
        self.pso_config = pso_config
        self.train_config = train_config
        self.output_manager = output_manager
        logger.debug("TrainingHandler initialized")
        
    def execute(self,
                image_paths: List[str],
                load_data_target_size: Tuple[int, int]) -> Tuple[DBN, Dict[str, Any]]:
        """
        Executes the entire DBN training and scaling process
        
        Args:
           image_paths: List of absolute paths to the test images used for training data.
            load_data_target_size: The (W, H) tuple for resizing images during data loading.

        Returns:
            A tuple containing:
            - The trained and optimized DBN model.
            - A dictionary of the fitted scalers ('scaler_x', 'scaler_y', 'scaler_y_ab').

        Raises:
            ValueError: If no valid training images are found or data loading fails.
            RuntimeError: If DBNTrainer fails to produce a valid model or scalers. 
        """
        logger.info("Loading images specified in config to generate training data...")
        valid_test_image_paths = []
        
        for image_path_str in image_paths:
            image = load_image(image_path_str)
            if image is not None:
                valid_test_image_paths.append(image_path_str)
                self.output_manager.save_test_image(Path(image_path_str).name, image)
            else:
                logging.warning(f"Skipping image (cannot be loaded): {image_path_str}")

        if not valid_test_image_paths:
            raise ValueError("No valid images found in 'test_images' list. Cannot generate training data.")

        logger.info(f"Loading and processing {len(valid_test_image_paths)} valid images into flattened training arrays...")
        
        # load_data returns flattened (n_images, H*W*3) arrays
        rgb_data, lab_data = load_data(valid_test_image_paths, target_size=load_data_target_size)

        if rgb_data.size == 0 or lab_data.size == 0:
            raise ValueError("Loading training data resulted in empty arrays.")
        logger.debug(f"Loaded training data shapes - RGB: {rgb_data.shape}, LAB: {lab_data.shape}")

        logger.info(f"Initializing DBNTrainer (Target samples: {self.train_config.n_samples})...")
        trainer = DBNTrainer(self.dbn_config, self.pso_config, self.train_config)
        
        # The train method handles sampling, splitting, scaling, init, and PSO
        dbn, scalers = trainer.train(rgb_data, lab_data)

        if dbn is None or scalers is None:
             raise RuntimeError("DBN training failed to return a valid model or scalers.")

        logger.info("DBN training and scaling completed.")
        return dbn, scalers