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
from src.utils.color.color_analysis import ColorMetricCalculator
from src.utils.image_utils import process_reference_image
from src.utils.file_utils import save_output
from src.utils.visualization import save_reference_summary_plot

# Force TensorFlow to use CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Define absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Ensure output directory exists

# Set up logging
log_file = os.path.join(OUTPUT_DIR, 'log.txt')
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(config_path='config.yaml'):
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        logging.info(f"Current working directory: {os.getcwd()}")
        config = load_config(config_path)
        if config is None or not validate_config(config):
            logging.error("Invalid configuration. Exiting.")
            return

        # Derive dataset name from config path
        dataset_name = os.path.basename(os.path.dirname(config_path))

        reference_image_path = config['reference_image_path']
        test_images = config['test_images']
        distance_threshold = config['distance_threshold']
        k = config['kmeans_clusters']
        predefined_k = config['predefined_k']

        # Load and prepare data for DBN training
        rgb_data, lab_data = load_data(test_images)
        logging.info(f"rgb_data shape: {rgb_data.shape if hasattr(rgb_data, 'shape') else 'None'}")
        logging.info(f"lab_data shape: {lab_data.shape if hasattr(lab_data, 'shape') else 'None'}")

        if rgb_data.size == 0 or lab_data.size == 0:
            logging.error("No valid data loaded from test_images. Check file paths.")
            return
        
        # Subsample 800 RGB-to-CIELAB pairs for training
        n_samples = 800
        rgb_samples, lab_samples = [], []
        for img_rgb, img_lab in zip(rgb_data, lab_data):
            logging.info(f"Processing img_rgb shape: {img_rgb.shape}, img_lab shape: {img_lab.shape}")
            n_per_image = n_samples // len(test_images)
            if n_per_image > 0 and img_rgb.size >= n_per_image * 3:
                indices = np.random.choice(img_rgb.size // 3, n_per_image, replace=False)
                rgb_samples.append(img_rgb.reshape(-1, 3)[indices])
                lab_samples.append(img_lab.reshape(-1, 3)[indices])
            else:
                logging.warning(f"Insufficient samples in image. n_per_image: {n_per_image}, img_rgb.size: {img_rgb.size}")

        if not rgb_samples:
            logging.error("No samples collected for rgb_samples. Exiting.")
            return

        rgb_samples = np.vstack(rgb_samples)[:, :3]  # Shape: (800, 3)
        lab_samples = np.vstack(lab_samples)[:, :3]  # Shape: (800, 3)
        logging.info(f"rgb_samples shape: {rgb_samples.shape}")
        logging.info(f"lab_samples shape: {lab_samples.shape}")
        
        # Train-test split
        x_train, x_test, y_train, y_test = train_test_split(rgb_samples, lab_samples, test_size=0.2, random_state=42)

        # Initialize DBN for RGB-to-CIELAB
        input_size = 3
        output_size = 3
        hidden_layers = [100, 50, 25]
        dbn = DBN(input_size, hidden_layers, output_size)
        logging.info("DBN model initialized")

        # Scale data
        scaler_x = StandardScaler().fit(x_train)
        scaler_y = MinMaxScaler(feature_range=(0, 100)).fit(y_train[:, [0]])  # L channel
        scaler_y_ab = MinMaxScaler(feature_range=(-128, 127)).fit(y_train[:, 1:])  # a,b channels
        x_train_scaled = scaler_x.transform(x_train)
        y_train_scaled = np.hstack((scaler_y.transform(y_train[:, [0]]), scaler_y_ab.transform(y_train[:, 1:])))
        logging.info(f"x_train_scaled shape: {x_train_scaled.shape}, y_train_scaled shape: {y_train_scaled.shape}")

        # Build and optimize DBN with PSO
        sample_input = np.zeros((1, input_size))
        dbn.model(sample_input)
        initial_weights = dbn.model.get_weights()
        bounds = [(w.min(), w.max()) for w in initial_weights]
        try:
            optimized_weights = pso_optimize(dbn, x_train_scaled, y_train_scaled, bounds)
            dbn.model.set_weights(optimized_weights)
            logging.info("PSO optimization completed")
        except Exception as e:
            logging.error(f"Failed to optimize DBN: {e}")
            return

        # Process reference image
        reference_kmeans_opt, reference_som_opt, original_image, dpc_k = process_reference_image(reference_image_path, dbn, scaler_x, scaler_y, scaler_y_ab, k)
        if reference_kmeans_opt is None or reference_som_opt is None:
            logging.error("Failed to process reference image. Exiting.")
            return

        target_colors = reference_kmeans_opt['avg_colors_lab']
        save_output(dataset_name, "reference_summary", "reference_summary.png", original_image, output_dir=OUTPUT_DIR)
        logging.info(f"Reference image processed, target_colors shape: {len(target_colors)}")

        # Process test images
        overall_delta_e = {}
        for image_path in test_images:
            logging.info(f"Processing test image: {image_path}")
            image = cv2.imread(image_path)
            if image is None:
                logging.error(f"Failed to load image: {image_path}")
                continue

            # Preprocess the image
            preprocessor = Preprocessor(
                initial_resize=512,
                target_size=(128, 128),
                denoise_h=10,
                max_colors=8,
                edge_enhance=False,
                unsharp_amount=0.0,
                unsharp_threshold=0
            )
            preprocessed_image = preprocessor.preprocess(image)
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            save_output(dataset_name, "preprocessed", f"{image_name}_preprocessed.png", preprocessed_image, output_dir=OUTPUT_DIR)
            logging.info(f"Preprocessing completed, unique colors: {len(np.unique(preprocessed_image.reshape(-1, 3), axis=0))}")

            # Test both determined and predefined k types
            for k_type in ['determined', 'predefined']:
                logging.info(f"Starting segmentation with k_type: {k_type}")
                segmenter = Segmenter(
                    preprocessed_image,
                    target_colors,
                    distance_threshold,
                    reference_kmeans_opt,
                    reference_som_opt,
                    dbn,
                    (scaler_x, scaler_y, scaler_y_ab),
                    predefined_k,
                    k_values=[10, 15, 20],
                    som_values=[5, 10, 20],
                    output_dir=OUTPUT_DIR,
                    k_type=k_type
                )
                result = segmenter.process()

                if result:
                    preprocessed_path, \
                    (seg_kmeans_opt, sim_kmeans_opt, best_kmeans_opt), \
                    (seg_kmeans_predef, sim_kmeans_predef, best_kmeans_predef), \
                    (seg_dbscan, sim_dbscan, best_dbscan), \
                    (seg_som_opt, sim_som_opt, best_som_opt), \
                    (seg_som_predef, sim_som_predef, best_som_predef) = result

                    # Calculate LAB colors for Delta E
                    rgb_colors = seg_kmeans_opt[1] if isinstance(seg_kmeans_opt[1], (list, np.ndarray)) else seg_kmeans_opt.get('avg_colors_rgb', [])
                    if not rgb_colors:
                        raise ValueError("No RGB colors available for Delta E calculation.")
                    lab_traditional_converter = lambda x: cv2.cvtColor(np.uint8([[x]]), cv2.COLOR_RGB2LAB)[0][0]
                    lab_dbn_converter = lambda x: convert_colors_to_cielab_dbn(dbn, scaler_x, scaler_y, scaler_y_ab, [x])[0]

                    # Use ColorMetricCalculator for Delta E
                    color_metric_calculator = ColorMetricCalculator(target_colors)
                    delta_e_traditional = color_metric_calculator.compute_delta_e(rgb_colors, lab_traditional_converter, best_kmeans_opt)
                    delta_e_dbn = color_metric_calculator.compute_delta_e(rgb_colors, lab_dbn_converter, best_kmeans_opt)

                    overall_delta_e[image_name] = {
                        'traditional': delta_e_traditional,
                        'pso_dbn': delta_e_dbn
                    }
                    save_output(dataset_name, "delta_e", f"{image_name}_delta_e.csv", overall_delta_e[image_name], output_dir=OUTPUT_DIR)

        # Log and save overall Delta E results
        logging.info("Overall Delta E results:")
        for image_name, delta_e in overall_delta_e.items():
            logging.info(f"Image: {image_name}, Traditional: {delta_e['traditional']:.3f}, PSO-DBN: {delta_e['pso_dbn']:.3f}")
        save_output(dataset_name, "delta_e", "overall_delta_e.csv", overall_delta_e, output_dir=OUTPUT_DIR)

    finally:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats(10)

if __name__ == "__main__":
    main(config_path='C:/Users/LENOVO/Desktop/prints/configurations/block_config.yaml')