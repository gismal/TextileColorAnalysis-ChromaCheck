# src/models/dbn_trainer.py
# UPDATED WITH DOCSTRINGS

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# Configuration types
from src.config_types import TrainConfig # TrainConfig now comes from here
from src.models.pso_dbn import DBN, DBNConfig, PSOConfig, PSOOptimizer 
# Data sampling function
from src.data.sampling import efficient_data_sampling 

logger = logging.getLogger(__name__)

class DBNTrainer:
    """
    Manages the DBN model training workflow, including data preparation and PSO optimization. 
    
    This class takes the raw pixel data, samples it, splits it, scales it, 
    initializes a DBN model according to the provided configuration, and then 
    uses Particle Swarm Optimization (PSO) to fine-tune the DBN's weights for 
    the RGB to CIELAB conversion task.
    """
    
    def __init__(self, dbn_config: DBNConfig, pso_config: PSOConfig, train_config: TrainConfig):
        """
        Initializes the DBNTrainer with necessary configuration objects.

        Args:
            dbn_config (DBNConfig): Settings for the DBN architecture 
                                    (e.g., hidden layers, activation).
            pso_config (PSOConfig): Parameters for the PSO algorithm 
                                    (e.g., swarm size, max iterations).
            train_config (TrainConfig): General training parameters like 
                                        sampling size, test split ratio, and PSO retries.
        """
        self.dbn_config = dbn_config
        self.pso_config = pso_config
        self.train_config = train_config
        
        # Placeholders for fitted scalers and the trained model
        self.scalers: Dict[str, MinMaxScaler] = {}
        self.dbn: Optional[DBN] = None
        
        # Scaled data is stored for use during PSO
        self.x_train_scaled: Optional[np.ndarray] = None
        self.y_train_scaled: Optional[np.ndarray] = None
        logger.info("DBNTrainer initialized.")

    def train(self, rgb_data: np.ndarray, lab_data: np.ndarray) -> Tuple[DBN, Dict[str, MinMaxScaler]]:
        """
        Executes the full DBN training and PSO optimization pipeline.

        This is the main public method of the class. It orchestrates the 
        sampling, splitting, scaling, DBN initialization, and PSO weight tuning.

        Args:
            rgb_data (np.ndarray): Flattened RGB pixel data (shape: n_images, n_pixels*3) 
                                   as loaded by `load_data`, with values in [0, 255] range.
            lab_data (np.ndarray): Flattened LAB pixel data (shape: n_images, n_pixels*3) 
                                   as loaded by `load_data`, potentially in OpenCV's range. 
                                   `efficient_data_sampling` will convert it to standard CIELAB.

        Returns:
            Tuple[DBN, Dict[str, MinMaxScaler]]: 
                A tuple containing:
                - The trained and PSO-optimized DBN model instance.
                - A dictionary holding the fitted scalers ('scaler_x', 'scaler_y', 'scaler_y_ab').

        Raises:
            RuntimeError: If the training process fails to produce a valid model or scalers.
            ValueError: If data sampling or splitting fails (e.g., due to empty input).
            Exception: Propagates exceptions from underlying steps like PSO optimization.
        """
        try:
            logger.info("Starting DBN training and PSO optimization process...")
            
            # 1. Sample Data: Select a representative subset for efficiency.
            logger.debug("Sampling raw data...")
            rgb_samples, lab_samples = efficient_data_sampling(
                rgb_data, lab_data, train_config=self.train_config
            )
            logger.debug(f"Sampled data shapes - RGB: {rgb_samples.shape}, LAB: {lab_samples.shape}")
            
            # 2. Split Data: Create training and testing sets (test set currently unused here).
            logger.debug("Splitting data into training/test sets...")
            x_train, x_test, y_train, y_test = train_test_split(
                rgb_samples, lab_samples, 
                test_size=self.train_config.test_size, 
                random_state=self.train_config.random_state
            )
            logger.info(f"Training set shape: {x_train.shape}, Test set shape: {x_test.shape}")

            # 3. Scale Data: Normalize features to the [0, 1] range for the neural network.
            logger.debug("Preparing data scalers...")
            self._prepare_scalers(x_train, y_train)
            
            # 4. Initialize DBN Model: Build the neural network structure.
            logger.debug("Initializing DBN model...")
            self.dbn = DBN(input_size=3, output_size=3, config=self.dbn_config)
            # Perform a dummy forward pass to build the model and initialize weights
            sample_input = np.zeros((1, 3))
            self.dbn.model(sample_input) 
            initial_weights = self.dbn.model.get_weights()
            logger.info(f"DBN model initialized with {len(initial_weights)} weight layers.")
            # Log initial weight details for debugging (optional)
            # for i, w in enumerate(initial_weights):
            #     logger.debug(f"Initial Layer {i}: shape {w.shape}, range [{w.min():.3f}, {w.max():.3f}]")

            # 5. Optimize Weights with PSO: Fine-tune the initialized weights using PSO.
            logger.debug("Starting PSO weight optimization...")
            optimized_weights = self._run_pso_with_retries(initial_weights)
            self.dbn.model.set_weights(optimized_weights)
            
            logger.info("DBN training and PSO optimization completed successfully.")
            
            # Final check
            if not self.dbn or not self.scalers:
                 raise RuntimeError("Training failed: DBN model or scalers were not properly generated.")
                 
            return self.dbn, self.scalers

        except Exception as e: # Catch any exception during the process
            logger.error(f"Error during DBN training pipeline: {e}", exc_info=True)
            raise # Re-raise the exception to be handled by the caller (pipeline.py)

    def _prepare_scalers(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Initializes and fits MinMaxScaler instances for input (RGB) and output (LAB) data.

        Scales RGB (0-255) to [0, 1].
        Scales LAB (L: 0-100, a/b: -128-127 approx.) also to [0, 1] using separate
        scalers for L and a/b channels, as they have different original ranges.
        Stores the fitted scalers in `self.scalers` and the scaled training data
        in `self.x_train_scaled` and `self.y_train_scaled` for PSO.

        Args:
            x_train (np.ndarray): The training RGB data (samples, 3).
            y_train (np.ndarray): The training LAB data (samples, 3), standard CIELAB range.
        """
        logger.info("Preparing and fitting data scalers...")
        
        # Initialize scalers to map features to [0, 1]
        scaler_x = MinMaxScaler(feature_range=(0, 1)) # For RGB input
        scaler_y_l = MinMaxScaler(feature_range=(0, 1))      # For L channel output
        scaler_y_ab = MinMaxScaler(feature_range=(0, 1))   # For a, b channel outputs
        
        # Fit and transform RGB training data
        # Store scaled data because PSO needs it for the objective function
        self.x_train_scaled = scaler_x.fit_transform(x_train)
        
        # Fit and transform LAB channels separately
        y_l_scaled = scaler_y_l.fit_transform(y_train[:, [0]]) # Select L channel (column 0)
        y_ab_scaled = scaler_y_ab.fit_transform(y_train[:, 1:]) # Select a, b channels (columns 1, 2)
        # Combine scaled L and a/b back together
        # Store scaled data because PSO needs it for the objective function
        self.y_train_scaled = np.hstack((y_l_scaled, y_ab_scaled))
        
        # Store the FITTED scalers for later use (e.g., prediction, inverse transform)
        self.scalers = {
            'scaler_x': scaler_x,
            'scaler_y': scaler_y_l, # Store L scaler as 'scaler_y' for consistency
            'scaler_y_ab': scaler_y_ab
        }
        logger.info(f"Data scaling complete. Scaled X shape: {self.x_train_scaled.shape}, Scaled Y shape: {self.y_train_scaled.shape}")
        logger.debug(f"Scaled X range: [{self.x_train_scaled.min():.3f}, {self.x_train_scaled.max():.3f}]") # Use debug for ranges
        logger.debug(f"Scaled Y range: [{self.y_train_scaled.min():.3f}, {self.y_train_scaled.max():.3f}]") # Use debug for ranges

    def _run_pso_with_retries(self, initial_weights: List[np.ndarray]) -> List[np.ndarray]:
        """
        Attempts PSO optimization and retries with slight weight perturbation if it fails.

        This helps to avoid getting stuck in poor local optima during PSO by 
        adding a small amount of noise to the weights before retrying.

        Args:
            initial_weights (List[np.ndarray]): The initial weights obtained from the DBN model.

        Returns:
            List[np.ndarray]: The best weights found by PSO after all attempts, 
                              or the initial weights if all attempts failed critically.
                              
        Raises:
            RuntimeError: If DBN model or scaled training data is missing when needed.
            ValueError: If PSO produces invalid (NaN or Inf) weights and retries fail.
        """
        logger.info(f"Starting PSO optimization with up to {self.train_config.pso_retries} retries.")
        current_weights = initial_weights
        best_weights_so_far = initial_weights # Keep track of the best valid weights found
        
        for attempt in range(self.train_config.pso_retries):
            try:
                logger.info(f"PSO optimization attempt {attempt + 1}/{self.train_config.pso_retries}")
                
                # Create a PSOOptimizer instance (handles the PSO logic)
                optimizer = PSOOptimizer(self.pso_config)
                
                # Ensure necessary data and model are available
                if self.dbn is None or self.x_train_scaled is None or self.y_train_scaled is None:
                    raise RuntimeError("DBN model or scaled data not available for PSO objective function.")
                
                # Run the optimization
                optimized_weights = optimizer.optimize(
                    self.dbn, self.x_train_scaled, self.y_train_scaled
                )
                
                # Validate the results from PSO
                if not isinstance(optimized_weights, list) or not optimized_weights:
                     raise ValueError("PSO returned empty or invalid weights structure.")
                if any(np.isnan(w).any() or np.isinf(w).any() for w in optimized_weights):
                    raise ValueError("PSO produced invalid weights containing NaN or Inf.")
                
                logger.info(f"PSO attempt {attempt + 1} completed successfully.")
                return optimized_weights # Success, return the optimized weights
                
            except (ValueError, RuntimeError, Exception) as e: # Catch potential errors
                logger.warning(f"PSO attempt {attempt + 1} failed: {e}")
                if attempt < self.train_config.pso_retries - 1:
                    logger.info("Retrying PSO with slightly perturbed weights...")
                    # Add small random noise to weights for the next attempt
                    current_weights = [w + np.random.normal(0, 0.001, w.shape) for w in current_weights]
                    if self.dbn:
                        self.dbn.model.set_weights(current_weights) # Update model for next try
                else:
                    # All retries failed
                    logger.error("All PSO optimization attempts failed. Could not find valid optimized weights.")
                    # Decide what to return: initial weights or raise an error?
                    # Returning initial weights might be safer but could lead to poor performance.
                    # Raising an error stops the pipeline. Let's return the last known good weights.
                    logger.warning("Falling back to the initial weights provided to PSO.")
                    return best_weights_so_far # Return the initial weights as a fallback
        
        # This part should ideally not be reached if logic is correct, but as a safeguard:
        logger.error("PSO loop finished unexpectedly without success or failure. Returning initial weights.")
        return initial_weights