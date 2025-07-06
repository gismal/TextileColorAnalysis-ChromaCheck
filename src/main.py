import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib
matplotlib.use('Agg')
import numpy as np
import cv2
import logging
import cProfile
import pstats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from skimage.color import deltaE_ciede2000
from src.data.load_data import load_config, validate_config, load_data
from src.models.pso_dbn import DBN, pso_optimize, convert_colors_to_cielab_dbn
from src.data.preprocess import Preprocessor
from src.models.segmentation.segmentation import Segmenter 
from src.utils.image_utils import ciede2000_distance, save_delta_e_results, save_results, process_reference_image
from src.utils.visualization import save_reference_summary_plot

# Force TensorFlow to use CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set up logging
logging.basicConfig(filename='output/log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'reports', 'figures')

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

        # Load and prepare data for DBN training
        rgb_data, lab_data = load_data(test_images)
        print(f"rgb_data shape: {rgb_data.shape}")
        print(f"lab_data shape: {lab_data.shape}")

        # Subsample 800 RGB-to-CIELAB pairs for training
        n_samples = 800
        rgb_samples = []
        lab_samples = []
        for img_rgb, img_lab in zip(rgb_data, lab_data):
            n_per_image = n_samples // len(test_images)
            indices = np.random.choice(img_rgb.shape[0], n_per_image, replace=False)
            rgb_samples.append(img_rgb[indices])
            lab_samples.append(img_lab[indices])
        rgb_samples = np.vstack(rgb_samples)[:, :3]  # Shape: (800, 3)
        lab_samples = np.vstack(lab_samples)[:, :3]  # Shape: (800, 3)
        print(f"rgb_samples shape: {rgb_samples.shape}")
        print(f"lab_samples shape: {lab_samples.shape}")

        # Train-test split
        x_train, x_test, y_train, y_test = train_test_split(rgb_samples, lab_samples, test_size=0.2, random_state=42)

        # Initialize DBN for RGB-to-CIELAB
        input_size = 3
        output_size = 3
        hidden_layers = [100, 50, 25]
        dbn = DBN(input_size, hidden_layers, output_size)

        # Scale data
        scaler_x = StandardScaler().fit(x_train)
        scaler_y = MinMaxScaler(feature_range=(0, 100)).fit(y_train[:, [0]])  # L channel
        scaler_y_ab = MinMaxScaler(feature_range=(-128, 127)).fit(y_train[:, 1:])  # a,b channels
        x_train_scaled = scaler_x.transform(x_train)
        y_train_scaled = np.hstack((scaler_y.transform(y_train[:, [0]]), scaler_y_ab.transform(y_train[:, 1:])))

        # Build and optimize DBN with PSO
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
        overall_delta_e = {}
        for image_path in test_images:
            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                logging.error(f"Failed to load image: {image_path}")
                continue

            # Preprocess the image
            preprocessor = Preprocessor(
                resize_factor=1.1,  # Adjust these based on your Preprocessor definition
                kernel_size=(3, 3),
                bilateral_d=9,
                bilateral_sigma_color=75,
                bilateral_sigma_space=75
            )
            preprocessed_image = preprocessor.preprocess(image)

            # Segment the preprocessed image
            segmenter = Segmenter(
                preprocessed_image,
                target_colors,
                distance_threshold,
                reference_kmeans_opt,
                reference_som_opt,
                dbn,
                (scaler_x, scaler_y, scaler_y_ab),
                predefined_k,
                [10, 15, 20],  # kmeans_k_values
                [5, 10, 20],   # som_k_values
                OUTPUT_DIR
            )
            result = segmenter.process()

            if result:
                preprocessed_path, \
                (seg_kmeans_opt, sim_kmeans_opt, best_kmeans_opt), \
                (seg_kmeans_predef, sim_kmeans_predef, best_kmeans_predef), \
                (seg_dbscan, sim_dbscan, best_dbscan), \
                (seg_som_opt, sim_som_opt, best_som_opt), \
                (seg_som_predef, sim_som_predef, best_som_predef) = result
                
                image_name = os.path.splitext(os.path.basename(image_path))[0]
                
                # Save segmentation results
                save_results(seg_kmeans_opt, sim_kmeans_opt, 'kmeans_optimal', os.path.join(OUTPUT_DIR, image_name, 'kmeans', 'optimal'))
                save_results(seg_kmeans_predef, sim_kmeans_predef, 'kmeans_predefined', os.path.join(OUTPUT_DIR, image_name, 'kmeans', 'predefined'))
                save_results(seg_dbscan, sim_dbscan, 'dbscan', os.path.join(OUTPUT_DIR, image_name, 'dbscan'))
                save_results(seg_som_opt, sim_som_opt, 'som_optimal', os.path.join(OUTPUT_DIR, image_name, 'som', 'optimal'))
                save_results(seg_som_predef, sim_som_predef, 'som_predefined', os.path.join(OUTPUT_DIR, image_name, 'som', 'predefined'))

                # Calculate Delta E
                rgb_colors = seg_kmeans_opt['avg_colors_rgb']
                lab_traditional = [cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2LAB)[0][0] for color in rgb_colors]
                lab_dbn = convert_colors_to_cielab_dbn(dbn, scaler_x, scaler_y, scaler_y_ab, rgb_colors)
                
                delta_e_traditional = np.mean([ciede2000_distance(lab_traditional[i], target_colors[best_kmeans_opt[i][1]]) 
                                             for i in range(len(lab_traditional)) if best_kmeans_opt[i][1] != -1])
                delta_e_dbn = np.mean([ciede2000_distance(lab_dbn[i], target_colors[best_kmeans_opt[i][1]]) 
                                     for i in range(len(lab_dbn)) if best_kmeans_opt[i][1] != -1])
                
                overall_delta_e[image_name] = {
                    'traditional': delta_e_traditional,
                    'pso_dbn': delta_e_dbn
                }

        # Log and save results
        logging.info("Overall Delta E results:")
        for image_name, delta_e in overall_delta_e.items():
            logging.info(f"Image: {image_name}")
            logging.info(f"Traditional: {delta_e['traditional']:.3f}")
            logging.info(f"PSO-DBN: {delta_e['pso_dbn']:.3f}")
        save_delta_e_results(overall_delta_e, OUTPUT_DIR)

    finally:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats(10)

if __name__ == "__main__":
    main(config_path='C:/Users/LENOVO/Desktop/prints/configurations/block_config.yaml')