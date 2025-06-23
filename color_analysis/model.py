import numpy as np
import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from pyswarm import pso
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2lab
from sklearn.model_selection import train_test_split
from utils import ciede2000_distance
class DBN:
    def __init__(self, input_size, hidden_layers, output_size):
        logging.info("Initializing DBN model")
        self.model = Sequential()
        self.model.add(Dense(hidden_layers[0], input_dim=input_size, activation='relu'))
        for layer_size in hidden_layers[1:]:
            self.model.add(Dense(layer_size, activation='relu'))
        self.model.add(Dense(output_size, activation='linear'))
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        logging.info("DBN model initialized")

    def train(self, x_train, y_train, epochs=50, batch_size=32):
        logging.info(f"Training DBN model for {epochs} epochs with batch size {batch_size}")
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        logging.info("DBN model training completed")

    def predict(self, x_test):
        return self.model.predict(x_test)

def load_data(image_paths, target_size=(256, 256)):
    logging.info("Loading image data")
    rgb_data = []
    lab_data = []

    for image_path in image_paths:
        image = imread(image_path)
        image = resize(image, target_size, anti_aliasing=True)
        rgb_data.append(image.reshape(-1, 3))
        lab_data.append(rgb2lab(image).reshape(-1, 3))

    rgb_data = np.vstack(rgb_data)
    lab_data = np.vstack(lab_data)

    logging.info("Image data loading completed")
    return rgb_data, lab_data

def pso_optimize(dbn, x_train, y_train, bounds):
    def objective(weights):
        reshaped_weights = []
        start = 0
        for w in dbn.model.get_weights():
            shape = w.shape
            size = np.prod(shape)
            reshaped_weights.append(weights[start:start + size].reshape(shape))
            start += size
        dbn.model.set_weights(reshaped_weights)
        predictions = dbn.model.predict(x_train)
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

    logging.info(f"Starting PSO optimization")
    optimized_weights, _ = pso(objective, lb=lb, ub=ub, swarmsize=15, maxiter=15)
    logging.info("PSO optimization completed")

    start = 0
    new_weights = []
    for w in initial_weights:
        shape = w.shape
        size = np.prod(shape)
        new_weights.append(optimized_weights[start:start + size].reshape(shape))
        start += size

    return new_weights

def compare_predictions_to_ground_truth(avg_colors, avg_colors_lab_dbn, scaler_y):
    logging.info("Comparing PSO-DBN CIELAB predictions to ground truth CIELAB values")
    for i, (rgb_color, predicted_lab_dbn) in enumerate(zip(avg_colors, avg_colors_lab_dbn)):
        color_rgb = np.array(rgb_color).reshape(1, -1)
        ground_truth_lab = rgb2lab(color_rgb / 255.0)[0][0]
        predicted_lab_dbn = scaler_y.inverse_transform([predicted_lab_dbn])[0]
        delta_e = ciede2000_distance(ground_truth_lab, predicted_lab_dbn)
        logging.info(f"Segment {i+1} - RGB: {rgb_color}, Ground Truth CIELAB: {tuple(ground_truth_lab)}, Predicted CIELAB: {tuple(predicted_lab_dbn)}, ΔE: {delta_e:.3f}")
