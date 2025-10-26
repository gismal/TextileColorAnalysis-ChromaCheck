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
from dataclasses import dataclass
import time
import warnings

tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# PSO-optimized DBN for RGB to CIELAB conversion
# Based on "A precise method of color space conversion in the digital printing process based on PSO-DBN"
# by Su et al., 2022, adapted for RGB-to-CIELAB
# DOI: 10.1007/s00170-022-08729-7


@dataclass
class PSOConfig:
    """Configuration for PSO optimization params"""
    swarmsize: int = 20
    maxiter: int = 50
    minstep: float = 1e-8
    minfunc: float = 1e-8
    debug: bool = False
    # PSO specific params
    w: float = 0.7
    c1: float = 2.0
    c2: float = 2.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            'swarmsize': self.swarmsize,
            'maxiter': self.maxiter,
            'w': self.w,
            'c1': self.c1,
            'c2': self.c2
        }

@dataclass
class DBNConfig:
    """Config for DBN architecture and training"""
    hidden_layers: List[int] = None
    dropout_rate: float = 0.1
    batch_normalization: bool = True
    activation: str = 'relu'
    output_activation: str = 'sigmoid'  # DÜZELTME: [0, 1] aralığı için 'sigmoid'
    optimizer: str = 'adam'
    learning_rate: float = 0.001
    epochs: int = 50
    batch_size: int = 32
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    
    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [100, 50, 25]

class DBN:
    """Deep Belief Network for RGB to CIELAB conversion."""
    def __init__(self, 
                 input_size: int = 3, 
                 output_size: int = 3,
                 config: Optional[DBNConfig] = None):
        """Initialize the DBN model.

        Args:
            input_size: Number of input features (3 for RGB)
            output_size: Number of output features (3 for CIELAB)
            config: DBN configuration parameters
        """
        self.input_size = input_size
        self.output_size = output_size
        self.config = config or DBNConfig() # DÜZELTME: 'confif' -> 'config'
        
        self.model = None
        self.training_history = None
        self._is_trained = False
        
        logger.info(f"Initializing DBN with architecture: {input_size} -> {self.config.hidden_layers} -> {output_size}")
        self._build_model()
        
    # DÜZELTME: Bu fonksiyonun girintisi azaltıldı (__init__ ile aynı seviyede)
    def _build_model(self) -> None:
        """Build the nn architecture"""
        self.model = Sequential(name = 'PSO-DBN')
        
        # input layer with optional batch normalization
        self.model.add(Dense(
            self.config.hidden_layers[0],
            input_dim = self.input_size,
            activation = self.config.activation,
            name = 'input_dense'
        ))
        
        if self.config.batch_normalization:
            self.model.add(BatchNormalization(name = 'input_bn'))
            
        if self.config.dropout_rate > 0:
            self.model.add(Dropout(self.config.dropout_rate, name= 'input_dropout'))
            
        # hidden layers
        # DÜZELTME: 'hidden_layer' -> 'hidden_layers'
        for i, layer_size in enumerate(self.config.hidden_layers[1:], 1):
            self.model.add(Dense(
                layer_size,
                activation = self.config.activation,
                name = f"hidden_{i}"
            ))
            
            if self.config.batch_normalization:
                self.model.add(BatchNormalization(name = f"hidden_bn{i}"))
                
            if self.config.dropout_rate > 0:
                self.model.add(Dropout(self.config.dropout_rate, name=f'hidden_dropout_{i}'))
        
        # Output layer
        self.model.add(Dense(
            self.output_size, 
            activation=self.config.output_activation,
            name='output_dense'
        ))
        
        # Compile model
        optimizer = Adam(learning_rate = self.config.learning_rate)
        self.model.compile(
            optimizer = optimizer, # DÜZELTME: 'optimizear' -> 'optimizer'
            loss = 'mean_squared_error',
            metrics = ['mae', 'mse']
        )
        
        logger.info(f"DBN model built with {self.model.count_params()} parameters")   
        
    # DÜZELTME: Bu fonksiyonun girintisi azaltıldı
    def train(self,
              x_train: np.ndarray,
              y_train: np.ndarray,
              x_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              verbose: int = 0
              ) -> Dict[str, Any]:
        """Train the DBN model."""
        # (Bu fonksiyon DBNTrainer sınıfı ile şu anda kullanılmıyor,
        # ancak gelecekteki kullanım için burada durması sorun değil)
        logger.info(f"Training DBN for {self.config.epochs} epochs")
        
        # Setup callbacks
        callbacks = []
        
        if self.config.early_stopping_patience > 0:
            early_stopping = EarlyStopping(
                monitor='val_loss' if x_val is not None else 'loss',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=verbose
            )
            callbacks.append(early_stopping)
            
            lr_scheduler = ReduceLROnPlateau(
                monitor='val_loss' if x_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=verbose
            )
            callbacks.append(lr_scheduler)
        
        validation_data = None
        if x_val is not None and y_val is not None:
            validation_data = (x_val, y_val)
        
        start_time = time.time()
        
        history = self.model.fit(
            x_train, y_train,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_data=validation_data,
            validation_split=self.config.validation_split if validation_data is None else 0.0,
            callbacks=callbacks,
            verbose=verbose
        )    
        
        training_time = time.time() - start_time
        self.training_history = history.history
        self._is_trained = True
        
        final_loss = history.history['loss'][-1]
        final_val_loss = history.history.get('val_loss', [final_loss])[-1]
        
        metrics = {
            'training_time': training_time,
            'epochs_trained': len(history.history['loss']),
            'final_loss': final_loss,
            'final_val_loss': final_val_loss,
            'best_epoch': np.argmin(history.history.get('val_loss', history.history['loss'])) + 1
        }
        
        logger.info(f"DBN training completed in {training_time:.2f}s - "
                    f"Final loss: {final_loss:.6f}, Val loss: {final_val_loss:.6f}")
        return metrics
    
    # DÜZELTME: Bu fonksiyonun girintisi azaltıldı
    def predict(self,
                x_test: np.ndarray,
                batch_size: Optional[int] = None) -> np.ndarray:
        """Predict CIELAB values for RGB input."""
        if not self._is_trained:
            # PSO optimizasyonu ağırlıkları belirlediği için bu bir "eğitim" sayılır.
            # logger.warning("Model has not been trained via .train(), but weights may be set by PSO.")
            pass
        
        batch_size = batch_size or self.config.batch_size # DÜZELTME: 'eslf' -> 'self'
        return self.model.predict(x_test, batch_size = batch_size, verbose=0)
    
    # DÜZELTME: Bu fonksiyonun girintisi azaltıldı
    def get_model_summary(self) -> str:
        """Get model architectrue summary"""
        import io
        import contextlib
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            self.model.summary()
        return f.getvalue()


class PSOOptimizer:
    """Enhanced PSO optimizer with better bounds calculation and monitoring"""
    def __init__(self, config: Optional[PSOConfig] = None):
        self.config = config or PSOConfig()
        self.optimization_history = []
        self.best_fitness = float('inf')
        self.best_weights = None
        
    def optimize(self,
                 dbn: DBN,
                 x_train: np.ndarray, y_train: np.ndarray,
                 x_val: Optional[np.ndarray] = None,
                 y_val: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """
        Optimize DBN weights using PSO with enhanced objective function
        """
        logger.info(f"Starting PSO optimization with config: {self.config.to_dict()}")
        
        initial_weights = dbn.model.get_weights()
        self.best_weights = initial_weights # Başlangıçta en iyi ağırlık olarak ata
        bounds = self._calculate_adaptive_bounds(initial_weights) 
        
        def objective_function(flat_weights: np.ndarray) -> float:
            try:
                # Reshape weights and set in model
                reshaped_weights = self._reshape_weights(flat_weights, initial_weights)
                dbn.model.set_weights(reshaped_weights)
                
                # Calculate training loss
                train_pred = dbn.model.predict(x_train, verbose=0)
                train_loss = np.mean((train_pred - y_train) ** 2)
                
                val_loss = 0.0
                if x_val is not None and y_val is not None:
                    val_pred = dbn.model.predict(x_val, verbose = 0)
                    val_loss = np.mean((val_pred - y_val) ** 2)
                    total_loss = 0.7 * train_loss + 0.3 * val_loss
                else:
                    total_loss = train_loss
                
                # DÜZELTME: 'penaly' -> 'penalty'
                penalty = self._calculate_penalties(train_pred, reshaped_weights)
                
                fitness = total_loss + penalty
                
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_weights = reshaped_weights.copy()
                    
                return fitness
            
            except Exception as e:
                logger.warning(f"PSO objective function error: {e}")
                return 1e6 #return high penalty for invalid solutions
            
        flat_initial = np.hstack([w.flatten() for w in initial_weights])
        lb = [b[0] for b in bounds]
        ub = [b[1] for b in bounds] 
        
        start_time = time.time()
        
        try:
            # Run PSO opti
            optimized_flat, final_fitness = pso(
                objective_function,
                lb = lb, ub = ub,
                swarmsize = self.config.swarmsize,
                maxiter = self.config.maxiter,
                minstep = self.config.minstep,
                minfunc = self.config.minfunc,
                debug = self.config.debug            
            )
            
            optimization_time = time.time() - start_time
            
            optimized_weights = self._reshape_weights(optimized_flat, initial_weights)
            
            # DÜZELTME: 'optimizaed' -> 'optimized'
            if not self._validate_weights(optimized_weights, initial_weights):
                logger.warning("PSO produced invalid weights, using best found weights")
                if self.best_weights:
                    optimized_weights = self.best_weights
                else:
                    optimized_weights = initial_weights # Fallback
            
            # Ekstra güvenlik kontrolü: En iyi ağırlıklar NaN değilse onu kullan
            elif self.best_weights and not any(np.isnan(w).any() for w in self.best_weights):
                 optimized_weights = self.best_weights
            
            logger.info(f"PSO optimization completed in {optimization_time:.2f}s - "
                          f"Final fitness: {final_fitness:.6f}")
            
            return optimized_weights
            
        except Exception as e:
            logger.error(f"PSO optimization failed: {e}")
            return initial_weights


    def _calculate_adaptive_bounds(self, weights: List[np.ndarray]) -> List[Tuple[float, float]]:
        """Calculate adaptive bounds based on weight statistics."""
        bounds = []
        
        for w in weights:
            w_flat = w.flatten()
            
            mean_w = np.mean(w_flat)
            std_w = np.std(w_flat)
            
            if std_w > 0:
                margin = 3 * std_w
                min_val = mean_w - margin
                max_val = mean_w + margin
            else:
                margin = max(0.1, abs(mean_w) * 0.1)
                min_val = mean_w - margin
                max_val = mean_w + margin
                
            bounds.extend([(min_val, max_val)] * w.size) 
            
        return bounds

    def _reshape_weights(self, flat_weights: np.ndarray, 
                         reference_weights: List[np.ndarray]) -> List[np.ndarray]:
        """Reshape flat weights back to original weight structure."""
        reshaped_weights = []
        start_idx = 0
        
        for w in reference_weights:
            size = w.size
            # DÜZELTME: 'strt_idx' -> 'start_idx' ve 'w.reshape' -> 'w.shape'
            reshaped_w = flat_weights[start_idx: start_idx + size].reshape(w.shape)
            reshaped_weights.append(reshaped_w)
            start_idx += size
            
        return reshaped_weights
    
    # DÜZELTME: '_calcualte' -> '_calculate'
    def _calculate_penalties(self,
                             predictions: np.ndarray,
                             weights: List[np.ndarray]
                             ) -> float:
        """Calculate penalty terms for PSO objective"""
        penalty = 0.0
        
        # CIELAB range penalties (Model [0, 1] aralığında tahmin yapıyor)
        l_penalty = np.mean(np.maximum(0, predictions[:, 0] - 1.2) + # 1.0'ın biraz üstü
                             np.maximum(0, -0.2 - predictions[:,0])) # 0.0'ın biraz altı
        
        ab_penalty = np.mean(np.maximum(0, predictions[:, 1:] - 1.2) +
                              np.maximum(0, -0.2 - predictions[:, 1:]))
        
        weight_penalty = sum(np.sum(w**2) for w in weights) * 1e-6
        penalty = 0.1 * (l_penalty + ab_penalty) + weight_penalty
        
        return penalty
    
    # DÜZELTME: '_validate_weigths' -> '_validate_weights'
    def _validate_weights(self, 
                          # DÜZELTME: 'optimizaed' -> 'optimized'
                          optimized_weights: List[np.ndarray],
                          initial_weights: List[np.ndarray]) -> bool:
        """Validate that optimized weights are reasonable"""
        try:
            if not initial_weights: # Başlangıç ağırlıkları boşsa kontrol etme
                 return True
                 
            for opt_w, init_w in zip(optimized_weights, initial_weights):
                if np.isnan(opt_w).any() or np.isinf(opt_w).any():
                    logger.warning("Invalid weights found (NaN or Inf)")
                    return False
                
                # Başlangıç ağırlıklarının max'ı 0 ise bu kontrolü atla
                init_max = np.abs(init_w).max()
                if init_max > 1e-9 and np.abs(opt_w).max() > 100 * init_max:
                    logger.warning(f"Weights seem extreme: {np.abs(opt_w).max()} vs {init_max}")
                    return False
                
            return True
        except Exception as e:
            logger.error(f"Error during weight validation: {e}")
            return False
        
def optimize_dbn_with_pso(dbn: DBN,
                          x_train: np.ndarray,
                          y_train: np.ndarray,
                          x_val: Optional[np.ndarray] = None,
                          y_val: Optional[np.ndarray] = None,
                          pso_config: Optional[PSOConfig] = None
                          ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Convenience function to optimize DBN with PSO."""
    optimizer = PSOOptimizer(pso_config)
    optimized_weights = optimizer.optimize(dbn, x_train, y_train, x_val, y_val)
    
    metrics = {
        'best_fitness': optimizer.best_fitness,
        'pso_config': optimizer.config.to_dict() # DÜZELTME: 'optimier' -> 'optimizer'
    }   
    
    return optimized_weights, metrics

    
def convert_colors_to_cielab_dbn(dbn: DBN, 
                                 scaler_x: MinMaxScaler, # Type hint'i düzeltildi
                                 scaler_y: MinMaxScaler, 
                                 scaler_y_ab: MinMaxScaler, 
                                 avg_colors: List) -> List[Tuple[float, float, float]]:
    """Convert RGB colors to CIELAB using PSO-optimized DBN with enhanced error handling."""
    
    if not avg_colors:
        logger.warning("No colors provided for conversion")
        return []
    
    try:
        logger.debug(f"Converting {len(avg_colors)} RGB colors to CIELAB using PSO-DBN")
        
        avg_colors_array = np.array(avg_colors, dtype=np.float32)
        
        if avg_colors_array.max() <= 1.0:
            avg_colors_array = avg_colors_array * 255.0
        
        avg_colors_array = np.clip(avg_colors_array, 0, 255)
        
        # Scale RGB inputs
        color_rgb_scaled = scaler_x.transform(avg_colors_array)
        
        # Predict using DBN
        color_lab_dbn_scaled = dbn.predict(color_rgb_scaled)
        
        # Inverse transform to get original CIELAB scale
        L_predicted = scaler_y.inverse_transform(color_lab_dbn_scaled[:, [0]])
        ab_predicted = scaler_y_ab.inverse_transform(color_lab_dbn_scaled[:, 1:])
        color_lab_dbn = np.hstack((L_predicted, ab_predicted))
        
        # Ensure CIELAB values are in valid ranges
        color_lab_dbn[:, 0] = np.clip(color_lab_dbn[:, 0], 0, 100)    # L: 0-100
        color_lab_dbn[:, 1:] = np.clip(color_lab_dbn[:, 1:], -128, 127)  # a,b: -128 to 127
        
        result = [tuple(color.astype(float)) for color in color_lab_dbn]
        
        logger.debug("Color conversion using PSO-DBN completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"DBN color conversion failed: {e}")
        fallback_color = (50.0, 0.0, 0.0)
        return [fallback_color] * len(avg_colors)

# Compatibility function for legacy code
def pso_optimize(dbn, x_train, y_train, bounds=None):
    """Legacy compatibility function for PSO optimization."""
    logger.warning("Using legacy pso_optimize function. Consider upgrading to PSOOptimizer class.")
    
    optimizer = PSOOptimizer()
    
    # bounds argümanı artık optimize() metoduna geçirilmiyor.
    # PSOOptimizer kendi _calculate_adaptive_bounds metodunu kullanıyor.
    return optimizer.optimize(dbn, x_train, y_train)