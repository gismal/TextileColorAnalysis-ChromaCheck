import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib
matplotlib.use('Agg')
import numpy as np
import logging
import cProfile
import pstats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.data.load_data import load_config, validate_config, load_data
from src.models.pso_dbn import DBN, pso_optimize
from src.processing.image_processor import ImageProcessor
from src.utils.image_utils import ciede2000_distance, save_delta_e_results, save_results, process_reference_image
from src.utils.visualization import save_reference_summary_plot

# Force TensorFlow to use the CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set up logging
logging.basicConfig(filename='output/log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, 'config.yaml')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')

def main(config_path='config.yaml'):
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        logging.info(f"Current working directory: {os.getcwd()}")
        config = load_config(config_path)
        if config is None or not validate_config(config):
            logging.error("Invalid configuration. Exiting.")
            return

        reference_image_path = config['reference_image_path']
        test_images = config['test_images']
        distance_threshold = config['distance_threshold']
        k = config['kmeans_clusters']
        predefined_k = config['predefined_k']

        # Load and prepare data
        rgb_data, lab_data = load_data(test_images)
        print(f"rgb_data shape: {rgb_data.shape}")  # Debug print
        print(f"lab_data shape: {lab_data.shape}")  # Debug print
        x_train, x_test, y_train, y_test = train_test_split(rgb_data, lab_data, test_size=0.2, random_state=42)
        print(f"x_train shape: {x_train.shape}")  # Debug print
        print(f"y_train shape: {y_train.shape}")  # Debug print

        input_size = x_train.shape[1]  # Should be 30000
        output_size = y_train.shape[1]  # Should be 30000
        hidden_layers = [128, 64, 32]

        # Initialize and train DBN
        dbn = DBN(input_size, hidden_layers, output_size)
        scaler_x = StandardScaler().fit(x_train)
        scaler_y = MinMaxScaler(feature_range=(0, 100)).fit(y_train[:, 0].reshape(-1, 1))  # Adjust for L channel
        scaler_y_ab = MinMaxScaler(feature_range=(-128, 127)).fit(y_train[:, 1:])  # Adjust for a,b channels
        x_train_scaled = scaler_x.transform(x_train)
        y_train_scaled = np.hstack((scaler_y.transform(y_train[:, 0].reshape(-1, 1)), scaler_y_ab.transform(y_train[:, 1:])))

        sample_input = np.zeros((1, input_size))
        dbn.model(sample_input)
        initial_weights = dbn.model.get_weights()
        bounds = [(w.min(), w.max()) for w in initial_weights]

        try:
            optimized_weights = pso_optimize(dbn, x_train_scaled, y_train_scaled, bounds)
            dbn.model.set_weights(optimized_weights)
        except Exception as e:
            logging.error(f"Failed to optimize DBN: {e}")
            return

        # Process reference image
        reference_kmeans_opt, reference_som_opt, original_image, dpc_k = process_reference_image(reference_image_path, dbn, scaler_x, scaler_y, scaler_y_ab, k)
        if reference_kmeans_opt is None or reference_som_opt is None:
            logging.error("Failed to process reference image. Exiting.")
            return

        target_colors = reference_kmeans_opt['avg_colors_lab']
        save_reference_summary_plot(reference_kmeans_opt, reference_som_opt, original_image)

        # Process test images
        preprocessed_image_paths = []
        results = []
        for image_path in test_images:
            processor = ImageProcessor(
                image_path, target_colors, distance_threshold, reference_kmeans_opt, reference_som_opt,
                dbn, (scaler_x, scaler_y, scaler_y_ab), predefined_k, [10, 15, 20], [5, 10, 20], OUTPUT_DIR
            )
            result = processor.process()
            if result:
                preprocessed_image_paths.append(result[0])
                results.append(result)

        # Aggregate and save results
        overall_delta_e = {}
        for result in results:
            if result:
                preprocessed_path, \
                (seg_kmeans_opt, sim_kmeans_opt, best_kmeans_opt), \
                (seg_kmeans_predef, sim_kmeans_predef, best_kmeans_predef), \
                (seg_dbscan, sim_dbscan, best_dbscan), \
                (seg_som_opt, sim_som_opt, best_som_opt), \
                (seg_som_predef, sim_som_predef, best_som_predef) = result
                
                image_name = os.path.splitext(os.path.basename(preprocessed_path))[0]
                
                save_results(seg_kmeans_opt, sim_kmeans_opt, 'kmeans_optimal', os.path.join(OUTPUT_DIR, image_name, 'kmeans', 'optimal'))
                save_results(seg_kmeans_predef, sim_kmeans_predef, 'kmeans_predefined', os.path.join(OUTPUT_DIR, image_name, 'kmeans', 'predefined'))
                save_results(seg_dbscan, sim_dbscan, 'dbscan', os.path.join(OUTPUT_DIR, image_name, 'dbscan'))
                save_results(seg_som_opt, sim_som_opt, 'som_optimal', os.path.join(OUTPUT_DIR, image_name, 'som', 'optimal'))
                save_results(seg_som_predef, sim_som_predef, 'som_predefined', os.path.join(OUTPUT_DIR, image_name, 'som', 'predefined'))

                overall_delta_e[image_name] = {
                    'kmeans_optimal': np.mean([ciede2000_distance(seg_kmeans_opt['avg_colors_lab'][i], reference_kmeans_opt['avg_colors_lab'][best_kmeans_opt[i][1]]) for i in range(len(seg_kmeans_opt['avg_colors_lab'])) if best_kmeans_opt[i][1] != -1]),
                    'kmeans_predefined': np.mean([ciede2000_distance(seg_kmeans_predef['avg_colors_lab'][i], reference_kmeans_opt['avg_colors_lab'][best_kmeans_predef[i][1]]) for i in range(len(seg_kmeans_predef['avg_colors_lab'])) if best_kmeans_predef[i][1] != -1]),
                    'dbscan': np.mean([ciede2000_distance(seg_dbscan['avg_colors_lab'][i], reference_kmeans_opt['avg_colors_lab'][best_dbscan[i][1]]) for i in range(len(seg_dbscan['avg_colors_lab'])) if best_dbscan[i][1] != -1]),
                    'som_optimal': np.mean([ciede2000_distance(seg_som_opt['avg_colors_lab'][i], reference_kmeans_opt['avg_colors_lab'][best_som_opt[i][1]]) for i in range(len(seg_som_opt['avg_colors_lab'])) if best_som_opt[i][1] != -1]),
                    'som_predefined': np.mean([ciede2000_distance(seg_som_predef['avg_colors_lab'][i], reference_kmeans_opt['avg_colors_lab'][best_som_predef[i][1]]) for i in range(len(seg_som_predef['avg_colors_lab'])) if best_som_predef[i][1] != -1]),
                }

        logging.info("Overall Delta E results:")
        for image_name, delta_e in overall_delta_e.items():
            logging.info(f"Image: {image_name}")
            for method, value in delta_e.items():
                logging.info(f"{method}: {value:.3f}")

        save_delta_e_results(overall_delta_e, OUTPUT_DIR)

    finally:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats(10)

if __name__ == "__main__":
    main(config_path='C:/Users/LENOVO/Desktop/prints/configurations/block_config.yaml')