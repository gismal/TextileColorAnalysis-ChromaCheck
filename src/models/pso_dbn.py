import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pyswarm import pso
import logging
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass, field # Import field
import time
import warnings

# --- Setup ---
# Suppress excessive TensorFlow logging and warnings
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Note: The implementation is based on the concepts described in:
# "A precise method of color space conversion in the digital printing process based on PSO-DBN"
# by Su et al., 2022, adapted here for RGB-to-CIELAB.
# DOI: 10.1007/s00170-022-08729-7


# --- Configuration Dataclasses ---
# TODO: Consider moving these dataclasses to src/config_types.py for centralization.

@dataclass
class PSOConfig:
    """
    Configuration parameters for the Particle Swarm Optimization (PSO) algorithm.

    These parameters control the behavior of the particle swarm as it searches
    for optimal DBN weights.

    Attributes:
        swarmsize: Number of particles (potential solutions) in the swarm.
        maxiter: Maximum number of iterations for the swarm.
        minstep: Minimum step size change for stopping criterion.
        minfunc: Minimum objective function change for stopping criterion.
        debug: If True, enables detailed output from the `pyswarm` library.
        w: Inertia weight, controlling exploration vs. exploitation.
        c1: Cognitive coefficient (particle's attraction to its own best).
        c2: Social coefficient (particle's attraction to the swarm's best).
    """
    swarmsize: int = 20
    maxiter: int = 50
    minstep: float = 1e-8
    minfunc: float = 1e-8
    debug: bool = False
    w: float = 0.7
    c1: float = 2.0
    c2: float = 2.0

    def to_dict(self) -> Dict[str, Any]:
        """Converts the PSOConfig object to a dictionary for logging."""
        return {
            'swarmsize': self.swarmsize,
            'maxiter': self.maxiter,
            'w': self.w,
            'c1': self.c1,
            'c2': self.c2,
        }

@dataclass
class DBNConfig:
    """
    Configuration parameters for the DBN (Feedforward Network) architecture.

    Note: Training-related parameters (epochs, batch_size, optimizer, etc.)
    are included but primarily relevant if using the standard `DBN.train()` method,
    not when weights are set via PSO.

    Attributes:
        hidden_layers: List defining neuron counts for each hidden layer.
        dropout_rate: Dropout rate applied after hidden layers (0 disables).
        batch_normalization: Whether to use Batch Normalization.
        activation: Activation function for hidden layers (e.g., 'relu').
        output_activation: Activation for the output layer (e.g., 'sigmoid' for [0,1]).
        optimizer: Keras optimizer name (used by `DBN.train()`).
        learning_rate: Learning rate for the optimizer (used by `DBN.train()`).
        epochs: Max epochs for standard training (used by `DBN.train()`).
        batch_size: Batch size for standard training (used by `DBN.train()`).
        validation_split: Fraction of data for validation (used by `DBN.train()`).
        early_stopping_patience: Patience for early stopping (used by `DBN.train()`).
    """
    hidden_layers: List[int] = field(default_factory=lambda: [100, 50, 25])
    dropout_rate: float = 0.1
    batch_normalization: bool = True
    activation: str = 'relu'
    output_activation: str = 'sigmoid' # For [0, 1] scaled output
    optimizer: str = 'adam'
    learning_rate: float = 0.001
    epochs: int = 50
    batch_size: int = 32
    validation_split: float = 0.2
    early_stopping_patience: int = 10


# --- DBN Model Class ---

class DBN:
    """
    Represents the Feedforward Neural Network used for RGB to CIELAB conversion.

    Builds a Keras Sequential model based on DBNConfig. While named DBN following
    the source paper's terminology, it functions as a standard MLP whose weights
    are optimized using the PSO algorithm.
    """
    def __init__(self,
                 input_size: int = 3,
                 output_size: int = 3,
                 config: Optional[DBNConfig] = None):
        """
        Initializes and builds the DBN model structure.

        Args:
            input_size: Number of input features (should be 3 for RGB).
            output_size: Number of output features (should be 3 for LAB).
            config: Configuration for the network architecture. Uses defaults if None.
        """
        self.input_size = input_size
        self.output_size = output_size
        self.config = config or DBNConfig()

        self.model: Optional[Sequential] = None
        self.training_history: Optional[Dict[str, Any]] = None
        self._weights_are_set = False # Renamed from _is_trained

        logger.info(f"Initializing DBN: Input({input_size}) -> Hidden({self.config.hidden_layers}) -> Output({output_size})")
        self._build_model()

    def _build_model(self) -> None:
        """Constructs the Keras Sequential model based on the DBNConfig."""
        self.model = Sequential(name='PSO-DBN')

        # Input Layer (defined implicitly by the first Dense layer's input_dim)
        self.model.add(Dense(
            self.config.hidden_layers[0],
            input_shape=(self.input_size,), # Use input_shape for clarity
            activation=self.config.activation,
            name='input_dense'
        ))
        if self.config.batch_normalization:
            self.model.add(BatchNormalization(name='input_bn'))
        if self.config.dropout_rate > 0:
            self.model.add(Dropout(self.config.dropout_rate, name='input_dropout'))

        # Hidden Layers
        for i, layer_size in enumerate(self.config.hidden_layers[1:], 1):
            self.model.add(Dense(
                layer_size,
                activation=self.config.activation,
                name=f"hidden_{i}"
            ))
            if self.config.batch_normalization:
                self.model.add(BatchNormalization(name=f"hidden_bn_{i}"))
            if self.config.dropout_rate > 0:
                self.model.add(Dropout(self.config.dropout_rate, name=f'hidden_dropout_{i}'))

        # Output Layer
        self.model.add(Dense(
            self.output_size,
            activation=self.config.output_activation,
            name='output_dense'
        ))

        # Compile Model (Necessary even if weights are set externally by PSO)
        optimizer_instance = Adam(learning_rate=self.config.learning_rate)
        self.model.compile(
            optimizer=optimizer_instance,
            loss='mean_squared_error',
            metrics=['mae', 'mse']
        )
        logger.info(f"DBN Keras model built successfully. Total parameters: {self.model.count_params()}")

    def train(self,
              x_train: np.ndarray,
              y_train: np.ndarray,
              x_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              verbose: int = 0
              ) -> Dict[str, Any]:
        """
        Trains the DBN model using standard backpropagation (e.g., Adam optimizer).

        NOTE: This method is NOT used by the current pipeline which utilizes PSO.
        It is provided for potential experimentation or comparison.

        Args:
            x_train: Scaled training input data (samples, features).
            y_train: Scaled training target data (samples, outputs).
            x_val: Optional scaled validation input data.
            y_val: Optional scaled validation target data.
            verbose: Keras verbosity level (0=silent, 1=progress bar, 2=one line/epoch).

        Returns:
            A dictionary containing training metrics (duration, epochs, loss, etc.).

        Raises:
            RuntimeError: If the model hasn't been built.
        """
        if self.model is None:
             raise RuntimeError("Model has not been built. Call _build_model() first.")

        logger.info(f"Starting standard DBN training for max {self.config.epochs} epochs...")

        callbacks = []
        if self.config.early_stopping_patience > 0:
            early_stopping = EarlyStopping(
                monitor='val_loss' if x_val is not None else 'loss',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True, # Restore best weights found
                verbose=verbose
            )
            callbacks.append(early_stopping)

            lr_scheduler = ReduceLROnPlateau(
                monitor='val_loss' if x_val is not None else 'loss',
                factor=0.5, # Reduce LR by half
                patience=5, # If no improvement for 5 epochs
                min_lr=1e-6,
                verbose=verbose
            )
            callbacks.append(lr_scheduler)

        validation_data = (x_val, y_val) if x_val is not None and y_val is not None else None

        start_time = time.time()
        history = self.model.fit(
            x_train, y_train,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_data=validation_data,
            # Use validation_split only if validation_data is not provided
            validation_split=self.config.validation_split if validation_data is None else 0.0,
            callbacks=callbacks,
            verbose=verbose
        )
        training_time = time.time() - start_time
        self.training_history = history.history
        self._weights_are_set = True # Mark weights as set after training

        final_loss = history.history['loss'][-1]
        final_val_loss = history.history.get('val_loss', [np.nan])[-1] # Use NaN if no validation

        metrics = {
            'training_time': training_time,
            'epochs_trained': len(history.history['loss']),
            'final_loss': final_loss,
            'final_val_loss': final_val_loss,
            'best_epoch': np.argmin(history.history.get('val_loss', history.history['loss'])) + 1
        }

        logger.info(f"Standard DBN training completed in {training_time:.2f}s. "
                    f"Final loss: {final_loss:.6f}, Val loss: {final_val_loss:.6f}")
        return metrics

    def predict(self,
                x_test: np.ndarray,
                batch_size: Optional[int] = None) -> np.ndarray:
        """
        Generates predictions (scaled LAB values) for given scaled RGB inputs.

        Args:
            x_test (np.ndarray): Scaled input RGB data (samples, 3).
            batch_size (Optional[int]): Batch size for prediction. Uses config default if None.

        Returns:
            np.ndarray: Predicted scaled LAB values (samples, 3).

        Raises:
            RuntimeError: If the model hasn't been built.
            ValueError: If input data shape is incorrect.
        """
        if self.model is None:
             raise RuntimeError("Model has not been built.")
        
        if x_test.ndim != 2 or x_test.shape[1] != self.input_size:
             raise ValueError(f"Input data shape mismatch. Expected (samples, {self.input_size}), got {x_test.shape}")

        effective_batch_size = batch_size or self.config.batch_size
        return self.model.predict(x_test, batch_size=effective_batch_size, verbose=0)

    def get_model_summary(self) -> str:
        """Returns the Keras model summary as a string."""
        if self.model is None:
             return "Model not built yet."
        import io
        import contextlib
        string_buffer = io.StringIO()
        with contextlib.redirect_stdout(string_buffer):
            self.model.summary()
        return string_buffer.getvalue()

    def set_weights(self, weights: List[np.ndarray]):
        """Sets the model weights and marks the model as having weights set."""
        if self.model is None:
            raise RuntimeError("Model must be built before setting weights.")
        try:
            self.model.set_weights(weights)
            self._weights_are_set = True
            logger.info("Successfully set weights in the DBN model.")
        except ValueError as e:
             logger.error(f"Failed to set weights: {e}. Check weight shapes.")
             self._weights_are_set = False # Mark as unset if failed
             raise
        except Exception as e:
             logger.error(f"An unexpected error occurred while setting weights: {e}")
             self._weights_are_set = False
             raise

# --- PSO Optimizer Class ---

class PSOOptimizer:
    """
    Optimizes the weights of a DBN model using the Particle Swarm Optimization algorithm.

    This class encapsulates the PSO logic, including objective function definition,
    bound calculation, and running the optimization using the `pyswarm` library.
    """
    def __init__(self, config: Optional[PSOConfig] = None):
        """
        Initializes the PSOOptimizer.

        Args:
            config (Optional[PSOConfig]): Configuration for the PSO algorithm.
                                          Uses defaults if None.
        """
        self.config = config or PSOConfig()
        # self.optimization_history = [] # Could be used to store fitness per iteration
        self.best_fitness = float('inf')
        self.best_weights = None # Store the best valid weights found during optimization

    def optimize(self,
                 dbn: DBN,
                 x_train: np.ndarray, y_train: np.ndarray,
                 x_val: Optional[np.ndarray] = None,
                 y_val: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """
        Performs PSO to find optimal weights for the provided DBN model.

        Args:
            dbn (DBN): The DBN model instance whose weights need optimization.
            x_train (np.ndarray): Scaled training input data (samples, 3).
            y_train (np.ndarray): Scaled training target data (samples, 3).
            x_val (Optional[np.ndarray]): Scaled validation input data.
            y_val (Optional[np.ndarray]): Scaled validation target data.

        Returns:
            List[np.ndarray]: The list of optimized weight matrices/vectors
                              found by PSO. Returns initial weights if optimization fails.
                              
        Raises:
             RuntimeError: If the DBN model is not provided correctly.
        """
        if dbn.model is None:
             raise RuntimeError("DBN model is not built, cannot optimize.")
             
        logger.info(f"Starting PSO optimization with parameters: {self.config.to_dict()}")

        initial_weights = dbn.model.get_weights()
        if not initial_weights:
            raise RuntimeError("Could not retrieve initial weights from the DBN model.")
            
        self.best_weights = [w.copy() for w in initial_weights] # Initialize best weights
        self.best_fitness = float('inf')

        # Define bounds adaptively based on initial weights
        bounds = self._calculate_adaptive_bounds(initial_weights)
        if not bounds:
             raise RuntimeError("Failed to calculate PSO bounds.")
        lb = [b[0] for b in bounds] # Lower bounds
        ub = [b[1] for b in bounds] # Upper bounds

        # --- Objective Function (Fitness Function) ---
        # This function evaluates how 'good' a set of weights (particle position) is.
        # It needs access to dbn, x_train, y_train, etc., from the outer scope.
        def objective_function(flat_weights: np.ndarray) -> float:
            """Calculates the fitness (loss + penalties) for a given set of weights."""
            nonlocal dbn, x_train, y_train, x_val, y_val, initial_weights # Allow access
            try:
                # 1. Reshape flat weights back into Keras layer structure
                reshaped_weights = self._reshape_weights(flat_weights, initial_weights)
                # Check shapes before setting
                if len(reshaped_weights) != len(initial_weights) or any(r.shape != i.shape for r, i in zip(reshaped_weights, initial_weights)):
                     logger.warning("PSO generated weights with incompatible shapes. Returning high penalty.")
                     return 1e7 # High penalty
                     
                # 2. Temporarily set these weights in the model
                dbn.set_weights(reshaped_weights) # Use the DBN class's method

                # 3. Calculate Loss (MSE on training and optionally validation set)
                train_pred = dbn.predict(x_train) # Use DBN class's predict
                train_loss = np.mean((train_pred - y_train) ** 2)

                total_loss = train_loss
                if x_val is not None and y_val is not None:
                    val_pred = dbn.predict(x_val)
                    val_loss = np.mean((val_pred - y_val) ** 2)
                    # Combine train and validation loss (e.g., weighted average)
                    total_loss = 0.7 * train_loss + 0.3 * val_loss
                    
                # 4. Calculate Penalties (optional, e.g., for weight decay or output range)
                penalty = self._calculate_penalties(train_pred, reshaped_weights)

                # 5. Fitness = Loss + Penalty (PSO minimizes this value)
                fitness = total_loss + penalty

                # Keep track of the best valid weights found so far
                if fitness < self.best_fitness:
                    if not any(np.isnan(w).any() for w in reshaped_weights): # Ensure weights are valid
                         self.best_fitness = fitness
                         # Deep copy is important here
                         self.best_weights = [w.copy() for w in reshaped_weights]
                         logger.debug(f"PSO New best fitness: {fitness:.6f}")

                return fitness

            except Exception as e:
                # Log errors occurring within the objective function
                logger.warning(f"Error during PSO objective function evaluation: {e}")
                return 1e6 # Return a high penalty value for invalid solutions
        # --- End of Objective Function ---

        # Flatten initial weights for PSO input
        flat_initial = np.concatenate([w.flatten() for w in initial_weights])

        start_time = time.time()
        optimized_flat_weights = None
        final_fitness = float('inf')
        
        try:
            # --- Run PSO ---
            optimized_flat_weights, final_fitness = pso(
                objective_function,
                lb,
                ub,
                
                swarmsize=self.config.swarmsize,
                maxiter=self.config.maxiter,
                minstep=self.config.minstep,
                minfunc=self.config.minfunc,
                debug=self.config.debug,
                omega=self.config.w, # Inertia weight
                phip=self.config.c1, # Cognitive coefficient
                phig=self.config.c2  # Social coefficient
            )
            optimization_time = time.time() - start_time
            logger.info(f"PSO optimization process completed in {optimization_time:.2f}s. "
                        f"Reported final fitness: {final_fitness:.6f}, Best fitness found: {self.best_fitness:.6f}")

            # --- Process Results ---
            # Reshape the best flat weights found by PSO (or the best stored ones)
            if optimized_flat_weights is not None and final_fitness <= self.best_fitness:
                 # If pyswarm's result is valid and better or equal
                 result_weights = self._reshape_weights(optimized_flat_weights, initial_weights)
                 if self._validate_weights(result_weights, initial_weights):
                      self.best_weights = result_weights # Update if valid
                 else:
                      logger.warning("pyswarm returned invalid weights despite good fitness. Using previously stored best weights.")
            elif self.best_weights:
                 logger.info("Using the best valid weights found during the PSO run.")
                 # best_weights already holds the correct structure
            else:
                 logger.error("PSO failed to find any valid weights. Returning initial weights.")
                 self.best_weights = initial_weights # Fallback


        except Exception as e:
            logger.error(f"PSO optimization using 'pyswarm' failed critically: {e}", exc_info=True)
            logger.warning("Returning initial weights due to PSO failure.")
            self.best_weights = initial_weights # Fallback to initial weights

        # Return the best valid weights found
        dbn.set_weights(self.best_weights) # Ensure the DBN object has the best weights set
        return self.best_weights


    def _calculate_adaptive_bounds(self, weights: List[np.ndarray]) -> List[Tuple[float, float]]:
        """
        Calculates adaptive search bounds for PSO based on initial weight statistics.

        For each weight/bias, it defines a search range (e.g., mean +/- 3 * std_dev)
        around its initial value. This can help guide the PSO search more effectively
        than using fixed, arbitrary bounds.

        Args:
            weights (List[np.ndarray]): The initial weights from the DBN model.

        Returns:
            List[Tuple[float, float]]: A flat list where each tuple represents the
                                       (lower_bound, upper_bound) for a single weight parameter.
        """
        bounds = []
        bound_multiplier = 3.0 # How many standard deviations to allow
        min_bound_range = 0.2 # Minimum range if std_dev is zero

        for w_layer in weights:
            if w_layer.size == 0: continue # Skip empty layers if any
            w_flat = w_layer.flatten()
            mean_w = np.mean(w_flat)
            std_w = np.std(w_flat)

            if std_w > 1e-9: # If there's variation
                margin = bound_multiplier * std_w
                min_val = mean_w - margin
                max_val = mean_w + margin
            else: # If all weights are the same initially (e.g., biases are zero)
                margin = max(min_bound_range / 2.0, abs(mean_w) * 0.1) # Add a small fixed or relative range
                min_val = mean_w - margin
                max_val = mean_w + margin

            # Extend the bounds list for every parameter in this layer
            bounds.extend([(min_val, max_val)] * w_flat.size)

        if not bounds:
             logger.error("Could not generate any bounds for PSO.")
        else:
             logger.debug(f"Calculated {len(bounds)} adaptive bounds for PSO.")
        return bounds

    def _reshape_weights(self, flat_weights: np.ndarray,
                         reference_weights: List[np.ndarray]) -> List[np.ndarray]:
        """
        Reshapes a flat array of weights back into the original Keras layer structure.

        Args:
            flat_weights (np.ndarray): A 1D array containing all weight parameters concatenated.
            reference_weights (List[np.ndarray]): The original list of weight matrices/vectors
                                                  used to determine the shapes for reshaping.

        Returns:
            List[np.ndarray]: A list of weight matrices/vectors with the correct shapes.

        Raises:
            ValueError: If the length of flat_weights doesn't match the total number
                        of parameters in reference_weights.
        """
        reshaped_weights = []
        start_idx = 0
        total_params_expected = sum(w.size for w in reference_weights)

        if len(flat_weights) != total_params_expected:
             raise ValueError(f"Weight reshaping error: Mismatch in parameter count. "
                              f"Expected {total_params_expected}, got {len(flat_weights)}.")

        for w_ref in reference_weights:
            size = w_ref.size
            # Extract the segment and reshape it to match the reference layer's shape
            reshaped_w = flat_weights[start_idx : start_idx + size].reshape(w_ref.shape)
            reshaped_weights.append(reshaped_w)
            start_idx += size

        return reshaped_weights

    def _calculate_penalties(self,
                             predictions: np.ndarray,
                             weights: List[np.ndarray]
                             ) -> float:
        """
        Calculates penalty terms to add to the PSO objective function.

        Includes penalties for:
        - Predictions falling outside the expected [0, 1] range (with some tolerance).
        - L2 regularization on weights (weight decay) to prevent overly large weights.

        Args:
            predictions (np.ndarray): The model's scaled output predictions ([0, 1] range).
            weights (List[np.ndarray]): The current weights being evaluated.

        Returns:
            float: The calculated penalty value.
        """
        penalty = 0.0
        range_tolerance = 0.2 # Allow predictions slightly outside [0, 1]
        weight_decay_factor = 1e-6 # L2 regularization strength

        # --- Output Range Penalty ---
        # Penalize if L channel prediction is significantly outside [0, 1]
        l_penalty = np.mean(
            np.maximum(0, predictions[:, 0] - (1.0 + range_tolerance)) + # Penalty for being > 1 + tolerance
            np.maximum(0, (0.0 - range_tolerance) - predictions[:, 0])   # Penalty for being < 0 - tolerance
        )
        # Penalize if a/b channel predictions are significantly outside [0, 1]
        ab_penalty = np.mean(
            np.maximum(0, predictions[:, 1:] - (1.0 + range_tolerance)) +
            np.maximum(0, (0.0 - range_tolerance) - predictions[:, 1:])
        )
        # Combine range penalties (adjust multiplier if needed)
        range_penalty_multiplier = 0.1
        penalty += range_penalty_multiplier * (l_penalty + ab_penalty)

        # --- L2 Weight Regularization Penalty ---
        # Helps prevent overfitting and keeps weights from growing too large.
        weight_penalty = sum(np.sum(w**2) for w in weights) * weight_decay_factor
        penalty += weight_penalty

        return penalty

    def _validate_weights(self,
                          optimized_weights: List[np.ndarray],
                          initial_weights: List[np.ndarray]) -> bool:
        """
        Performs basic checks to ensure the optimized weights are reasonable.

        Checks for NaN/Inf values and excessively large weight magnitudes compared
        to the initial weights.

        Args:
            optimized_weights (List[np.ndarray]): The weights returned by PSO.
            initial_weights (List[np.ndarray]): The weights before optimization.

        Returns:
            bool: True if weights seem valid, False otherwise.
        """
        try:
            if not optimized_weights or len(optimized_weights) != len(initial_weights):
                logger.warning("Weight validation failed: Structure mismatch.")
                return False

            extreme_weight_threshold = 100.0 # How much larger can weights get?

            for opt_w, init_w in zip(optimized_weights, initial_weights):
                if opt_w.shape != init_w.shape:
                    logger.warning(f"Weight validation failed: Shape mismatch {opt_w.shape} vs {init_w.shape}.")
                    return False
                if np.isnan(opt_w).any() or np.isinf(opt_w).any():
                    logger.warning("Weight validation failed: Found NaN or Inf.")
                    return False

                # Check magnitude (optional, disable if weights can legitimately grow large)
                init_max = np.max(np.abs(init_w)) if init_w.size > 0 else 0
                opt_max = np.max(np.abs(opt_w)) if opt_w.size > 0 else 0
                # Avoid division by zero and check only if initial max was non-negligible
                if init_max > 1e-9 and opt_max > extreme_weight_threshold * init_max:
                    logger.warning(f"Weight validation warning: Optimized weights magnitude ({opt_max:.2f}) "
                                   f"is significantly larger than initial ({init_max:.2f}).")
                    # return False # Decide if this should be a failure or just a warning

            # logger.debug("Weight validation passed.")
            return True
        except Exception as e:
            logger.error(f"Error during weight validation: {e}", exc_info=True)
            return False

# --- Convenience Function (kept as it was used in DBNTrainer) ---

def optimize_dbn_with_pso(dbn: DBN,
                          x_train: np.ndarray,
                          y_train: np.ndarray,
                          x_val: Optional[np.ndarray] = None,
                          y_val: Optional[np.ndarray] = None,
                          pso_config: Optional[PSOConfig] = None
                          ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    A helper function to create a PSOOptimizer and run the optimization.
     This simplifies the call from external modules like DBNTrainer.

    Args:
        dbn: The DBN model instance to optimize.
        x_train: Scaled training input data.
        y_train: Scaled training target data.
        x_val: Optional scaled validation input data.
        y_val: Optional scaled validation target data.
        pso_config: Configuration for the PSO algorithm.

    Returns:
        Tuple containing the optimized weights (List[ndarray]) and a dictionary
        of optimization metrics (like best fitness).
    """
    optimizer = PSOOptimizer(pso_config)
    optimized_weights = optimizer.optimize(dbn, x_train, y_train, x_val, y_val)

    metrics = {
        'best_fitness': optimizer.best_fitness,
        'pso_config': optimizer.config.to_dict()
    }

    return optimized_weights, metrics


# --- Color Conversion Function ---

def convert_colors_to_cielab_dbn(dbn: DBN,
                                 scaler_x: MinMaxScaler,
                                 scaler_y_l: MinMaxScaler, # Renamed for clarity
                                 scaler_y_ab: MinMaxScaler,
                                 rgb_colors: List[Tuple[float, float, float]] # Expect tuples or list of lists
                                 ) -> List[Tuple[float, float, float]]:
    """
    Converts a list of RGB colors to CIELAB using the trained PSO-DBN model and scalers.
    Handles input normalization, prediction, and inverse scaling of the output.

    Args:
        dbn (DBN): The trained and optimized DBN model.
        scaler_x (MinMaxScaler): The scaler fitted on the RGB training data ([0, 255] -> [0, 1]).
        scaler_y_l (MinMaxScaler): The scaler fitted on the L channel training data ([0, 100] -> [0, 1]).
        scaler_y_ab (MinMaxScaler): The scaler fitted on the a/b channel training data (approx [-128, 127] -> [0, 1]).
        rgb_colors (List[Tuple[float, float, float]]): A list of RGB color tuples, expected range [0, 255].

    Returns:
        List[Tuple[float, float, float]]: A list of predicted CIELAB color tuples,
                                           clipped to standard ranges (L: 0-100, a/b: -128-127).
                                           Returns a list of fallback colors [(50.0, 0.0, 0.0)] on failure.
    """

    if not rgb_colors:
        logger.warning("No RGB colors provided for DBN conversion.")
        return []

    try:
        logger.debug(f"Converting {len(rgb_colors)} RGB colors to CIELAB using PSO-DBN model...")

        # Convert list of tuples/lists to a NumPy array
        rgb_array = np.array(rgb_colors, dtype=np.float32)

        # Basic shape check
        if rgb_array.ndim != 2 or rgb_array.shape[1] != 3:
             if rgb_array.ndim == 1 and rgb_array.size == 3: # Handle single color input
                  rgb_array = rgb_array.reshape(1, 3)
             else:
                  raise ValueError(f"Input rgb_colors should be convertible to shape (n, 3), but got shape {rgb_array.shape}")

        # Ensure input is in [0, 255] range (scaler_x expects this)
        # Check if already normalized (max <= 1.0 is a heuristic)
        if np.max(rgb_array) <= 1.0 and np.min(rgb_array) >= 0.0:
            logger.warning("Input RGB colors to convert_colors_to_cielab_dbn seem to be in [0, 1] range. Scaling to [0, 255].")
            rgb_array = rgb_array * 255.0
        # Clip to ensure valid range after potential scaling
        rgb_array = np.clip(rgb_array, 0, 255)

        # 1. Scale RGB inputs using the fitted scaler_x
        rgb_scaled = scaler_x.transform(rgb_array)

        # 2. Predict scaled LAB values using the DBN model
        lab_dbn_scaled = dbn.predict(rgb_scaled)

        # Check prediction shape
        if lab_dbn_scaled.ndim != 2 or lab_dbn_scaled.shape[1] != 3:
             raise ValueError(f"DBN prediction has unexpected shape: {lab_dbn_scaled.shape}. Expected (n, 3).")

        # 3. Inverse transform the predictions to get CIELAB values
        L_predicted = scaler_y_l.inverse_transform(lab_dbn_scaled[:, [0]])
        ab_predicted = scaler_y_ab.inverse_transform(lab_dbn_scaled[:, 1:])
        lab_predicted = np.hstack((L_predicted, ab_predicted))

        # 4. Clip results to valid CIELAB ranges as a safety measure
        lab_predicted[:, 0] = np.clip(lab_predicted[:, 0], 0, 100)    # L channel: [0, 100]
        lab_predicted[:, 1:] = np.clip(lab_predicted[:, 1:], -128, 127) # a, b channels: approx [-128, 127]

        # Convert back to a list of tuples
        result_list = [tuple(color) for color in lab_predicted]

        logger.debug("PSO-DBN color conversion successful.")
        return result_list

    except Exception as e:
        logger.error(f"Error during DBN color conversion: {e}", exc_info=True)
        # Return a list of fallback colors (mid-gray) with the same length as input
        fallback_color = (50.0, 0.0, 0.0)
        return [fallback_color] * len(rgb_colors)
