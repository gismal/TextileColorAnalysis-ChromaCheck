import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pyswarm import pso
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# PSO-optimized DBN for RGB to CIELAB conversion
# Based on "A precise method of color space conversion in the digital printing process based on PSO-DBN"
# by Su et al., 2022, adapted for RGB-to-CIELAB
# DOI: 10.1007/s00170-022-08729-7

class DBN:
    """Deep Belief Network for RGB to CIELAB conversion."""
    def __init__(self, input_size=3, hidden_layers=[100, 50, 25], output_size=3):
        """Initialize the DBN model.

        Args:
            input_size (int): Number of input features (3 for RGB). [UPDATED]
            hidden_layers (list): List of hidden layer sizes (default: [100, 50, 25]). [UPDATED]
            output_size (int): Number of output features (3 for CIELAB).
        """
        logging.info("Initializing DBN model")
        self.model = Sequential()
        self.model.add(Dense(hidden_layers[0], input_dim=input_size, activation='relu'))
        for layer_size in hidden_layers[1:]:
            self.model.add(Dense(layer_size, activation='relu'))
        self.model.add(Dense(output_size, activation='linear'))
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        logging.info("DBN model initialized")

    def train(self, x_train, y_train, epochs=50, batch_size=32):
        """Train the DBN model.

        Args:
            x_train (numpy.ndarray): Input RGB data (shape: (n_samples, 3)).
            y_train (numpy.ndarray): Target CIELAB data (shape: (n_samples, 3)).
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
        """
        logging.info(f"Training DBN model for {epochs} epochs with batch size {batch_size}")
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        logging.info("DBN model training completed")

    def predict(self, x_test):
        """Predict CIELAB values for RGB input.

        Args:
            x_test (numpy.ndarray): Input RGB data (shape: (n_samples, 3)).

        Returns:
            numpy.ndarray: Predicted CIELAB values (shape: (n_samples, 3)).
        """
        return self.model.predict(x_test, verbose=0)

def pso_optimize(dbn, x_train, y_train, bounds):
    """Optimize DBN weights using Particle Swarm Optimization.

    Args:
        dbn (DBN): DBN model instance.
        x_train (numpy.ndarray): Scaled RGB input data (shape: (n_samples, 3)).
        y_train (numpy.ndarray): Scaled CIELAB target data (shape: (n_samples, 3)).
        bounds (list): Bounds for PSO optimization (e.g., [(min, max), ...]).

    Returns:
        list: Optimized weights for the DBN.
    """
    def objective(weights):
        reshaped_weights = []
        start = 0
        for w in dbn.model.get_weights():
            shape = w.shape
            size = np.prod(shape)
            reshaped_weights.append(weights[start:start + size].reshape(shape))
            start += size
        dbn.model.set_weights(reshaped_weights)
        predictions = dbn.model.predict(x_train, verbose=0)
        return np.mean((predictions - y_train) ** 2)

    initial_weights = dbn.model.get_weights()
    flat_weights = np.hstack([w.flatten() for w in initial_weights])

    flat_bounds = []
    epsilon = 1e-5
    for w in initial_weights:
        min_val = w.min()
        max_val = w.max()
        if min_val == max_val:
            min_val -= epsilon
            max_val += epsilon
        flat_bounds.extend([(min_val, max_val)] * w.size)

    lb = [b[0] for b in flat_bounds]
    ub = [b[1] for b in flat_bounds]

    swarmsize = 5
    maxiter = 5

    logging.info(f"Starting PSO optimization with swarmsize={swarmsize} and maxiter={maxiter}")
    optimized_weights, _ = pso(objective, lb=lb, ub=ub, swarmsize=swarmsize, maxiter=maxiter)
    logging.info("PSO optimization completed")

    start = 0
    new_weights = []
    for w in initial_weights:
        shape = w.shape
        size = np.prod(shape)
        new_weights.append(optimized_weights[start:start + size].reshape(shape))
        start += size

    return new_weights

def convert_colors_to_cielab_dbn(dbn, scaler_x, scaler_y, scaler_y_ab, avg_colors):
    """Convert RGB colors to CIELAB using PSO-optimized DBN.

    Args:
        dbn (DBN): Trained DBN model.
        scaler_x (StandardScaler): Scaler for RGB input (fitted on 3D data).
        scaler_y (MinMaxScaler): Scaler for CIELAB L channel.
        scaler_y_ab (MinMaxScaler): Scaler for CIELAB a, b channels.
        avg_colors (numpy.ndarray): RGB color values (shape: (n_colors, 3)).

    Returns:
        list: CIELAB colors predicted by DBN as list of tuples.
    """
    logging.info(f"Converting {len(avg_colors)} RGB colors to CIELAB using PSO-DBN")
    avg_colors_array = np.array(avg_colors)
    if len(avg_colors_array.shape) == 1:
        avg_colors_array = avg_colors_array.reshape(1, -1)

    # Apply scaling to RGB inputs
    color_rgb_scaled = scaler_x.transform(avg_colors_array)  # [UPDATED] Expects 3 features

    # Predict using DBN
    color_lab_dbn_scaled = dbn.predict(color_rgb_scaled)

    # Inverse transform to get original CIELAB scale
    L_predicted_scaled = color_lab_dbn_scaled[:, [0]]
    ab_predicted_scaled = color_lab_dbn_scaled[:, 1:]
    L_predicted = scaler_y.inverse_transform(L_predicted_scaled)
    ab_predicted = scaler_y_ab.inverse_transform(ab_predicted_scaled)
    color_lab_dbn = np.hstack((L_predicted, ab_predicted))

    # Convert to list of tuples
    avg_colors_lab_dbn = [tuple(color) for color in color_lab_dbn]
    logging.info("Conversion using PSO-DBN completed")
    return avg_colors_lab_dbn