# Textile Pattern Color Analysis using Image Segmentation and PSO-DBN

## 👋 Overview

Welcome! This project explores an automated approach for **analyzing color quality in printed textiles**, a crucial aspect of the textile industry. The core idea is to segment colors in digital images of test patterns using various clustering algorithms (K-Means, SOM, DBSCAN) and quantitatively compare them against a reference image using the **CIEL*a*b\* color space** and the **Delta E 2000** color difference formula.

A key contribution of this work, developed as part of a Master's thesis in collaboration with Zorluteks Textile (Türkiye), is the investigation of a novel **Particle Swarm Optimized Deep Belief Network (PSO-DBN)** as a potentially more precise method for RGB-to-CIEL*a*b\* color space conversion compared to standard library functions, specifically within the context of textile color assessment.

---

## 🎯 The Challenge: Color Consistency in Textiles

Maintaining consistent color quality is a major challenge in textile printing. Variations can arise from dyes, printing methods, and environmental factors. While **spectrophotometers** are the industry standard for precise color measurement, they struggle with **small or complex patterns** due to their limited scanning area. Human visual inspection, though common, can be subjective.

This project addresses these limitations by:

1.  Using **high-resolution digital images** captured under controlled lighting to analyze patterns regardless of complexity.
2.  Employing **image segmentation** techniques to isolate distinct color regions.
3.  Calculating color differences (Delta E) in the perceptually uniform **CIEL*a*b\* color space**.
4.  Exploring an **alternative, data-driven CIEL*a*b\* conversion method (PSO-DBN)** inspired by recent research to potentially improve accuracy over standard conversions.

---

## ✨ Key Features & Methodology

- **Industry Collaboration:** Developed with insights and real-world data from Zorluteks Textile.
- **Controlled Image Acquisition:** Utilized specific camera, lens, and lighting setups for high-quality, consistent image capture.
- **Preprocessing Pipeline:** Implements configurable steps:
  - Initial Resizing
  - Denoising (FastNLMeans) (Similar to Bilateral Filtering mentioned in thesis text)
  - Optional Unsharp Masking
  - Color Quantization (using K-Means) (Reduces color complexity for faster segmentation)
  - Final Resizing
- **Segmentation Algorithms:** Implements and compares:
  - **K-Means:** Partition-based clustering. Explored with both optimal 'k' (using validity metrics) and predefined 'k' (based on known number of colors).
  - **DBSCAN:** Density-based clustering, capable of finding arbitrarily shaped clusters and handling noise. Parameters (`eps`, `min_samples`) can be determined automatically (using Silhouette score heuristic) or predefined.
  - **SOM (Self-Organizing Maps):** Neural network-based unsupervised learning, good for visualizing high-dimensional data. Explored with optimal 'k' and predefined 'k'.
  - _(Note: DPC was explored in the thesis but the current codebase primarily uses K-Means, DBSCAN, and SOM)._
- **CIEL\*a\*b\* Color Space:** All color comparisons are performed in this perceptually uniform and industry-standard space.
- **Dual Color Conversion:**
  - **Standard:** Uses `skimage.color.rgb2lab`.
  - **PSO-DBN:** Implements a feedforward neural network (DBN architecture inspired by Su et al.) whose weights are optimized using Particle Swarm Optimization (PSO) to learn the RGB -> CIEL*a*b\* mapping from the image data.
- **Delta E 2000 Calculation:** Uses the standard formula to quantify the perceptual difference between segmented colors and reference colors, calculated using _both_ CIEL*a*b\* conversion methods for comparison.
- **Structured Output:** Organizes all generated files (input copies, preprocessed images, segmented images per method/k-type, analysis CSV) into dataset-specific directories.
- **YAML Configuration:** Allows easy definition of datasets, file paths, and parameters for all pipeline stages.

---

## 📁 Dataset

- **Source:** Provided by Zorluteks Textile Trade and Industry Inc..
- **Content:** High-resolution (300 ppi JPG) images of various printed textile patterns (e.g., simple two-color blocks, stripes, complex multi-color florals, unicorns).
- **Structure:** Organized by pattern type in the `dataset/` directory. Each pattern includes a `reference.jpg` and corresponding test sample images (e.g., `dataset/block/reference.jpg`, `dataset/block/block1.jpg`).

---

## ⚙️ Workflow Pipeline

The execution flow managed by `src/main.py` and `src/pipeline.py`:

1.  **Load & Validate Config:** Reads `defaults.yaml`, merges the specified pattern config (e.g., `block_config.yaml`), validates all parameters, resolves relative file paths to absolute paths, and initializes the `OutputManager`.
2.  **Load Training Data:** Loads images specified in `test_images`, resizes them, and converts to flattened RGB & LAB arrays for DBN training data generation.
3.  **Train PSO-DBN:** Samples pixels, splits data, fits `MinMaxScaler`s, initializes DBN (MLP) architecture, and runs PSO to optimize DBN weights.
4.  **Process Reference Image:** Loads the `reference_image`, applies the full preprocessing pipeline, segments using K-Means and SOM (determining optimal 'k'), extracts average RGB colors (from K-Means), converts to standard CIEL*a*b\* to establish **target colors**.
5.  **Analyze Test Images (Loop):** For each image in `test_images`:
    - Preprocesses the image fully and saves it.
    - Runs configured segmentation methods (KMeans, SOM, DBSCAN) for both `k_type='determined'` and `k_type='predefined'`. Saves segmented images.
    - Calculates average Delta E 2000 between segmented colors and target colors using _both_ standard CIEL*a*b* and PSO-DBN CIEL*a*b* values.
6.  **Save & Summarize Results:** Saves all Delta E results to a CSV file and prints summary tables to the console.

---

## 🌳 Directory Structure

````plaintext
prints/
|
+-- .venv/                  # Virtual environment
+-- configurations/
|   +-- defaults.yaml       # Default parameters for all pipelines
|   +-- pattern_configs/    # Specific configurations per dataset
|       +-- block_config.yaml
|       +-- flowers_config.yaml
|       +-- ... (other pattern configs)
+-- dataset/                # Raw image data
|   +-- block/
|   +-- flowers/
|   +-- ... (other pattern datasets)
+-- notebooks/              # Jupyter notebooks for experimentation/analysis (optional)
+-- output/                 # Generated output files (created on run)
|   +-- datasets/
|       +-- [dataset_name]/ # e.g., block
|           |
|           +-- analysis/       # CSV results (e.g., block_delta_e_results.csv)
|           +-- inputs/
|           |   +-- reference_image/ # Copy of the reference image used
|           |   +-- test_images/     # Copies of the test images used
|           +-- processed/
|           |   +-- preprocessed/    # Preprocessed test images (e.g., block1_preprocessed.png)
|           |   +-- segmented/       # Segmented test images, organized by method
|           |       +-- kmeans_opt/      # e.g., block1_determined.png
|           |       +-- kmeans_predef/   # e.g., block1_predefined.png
|           |       +-- som_opt/
|           |       +-- som_predef/
|           |       +-- dbscan/          # e.g., block1_determined.png, block1_predefined.png
|           +-- summaries/           # Summary plots/images (e.g., reference_summary.png)
+-- src/                    # Source code
|   +-- __init__.py         # Makes 'src' a package
|   +-- data/               # Data loading, preprocessing, sampling modules
|   +-- models/             # ML models (DBN, PSO) and Segmentation logic (package)
|   +-- utils/              # Helper modules (color, image, output, setup)
|   +-- config_types.py     # Dataclasses for configuration objects
|   +-- main.py             # Main script entry point (starts the application)
|   +-- pipeline.py         # Core processing workflow class (orchestrator)
+-- reports/                # Project reports, summaries (optional - e.g., thesis PDF)
+-- requirements.txt        # Project dependencies
+-- README.md               # This file
+-- LICENSE                 # License file (e.g., MIT)

## Setup and Installation

1.  **Clone the Repository:**
    ```
    git clone [your_repository_url]
    cd prints
    ```
2.  **Create and Activate Virtual Environment:** (Recommended)
    ```
    python -m venv .venv
    # Windows
    .\.venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```
3.  **Install Dependencies:**
    ```
    pip install -r requirements.txt
    ```

---

## Usage

Run the analysis pipeline from the project root directory (`prints/`) using `src/main.py`. You **must** specify a configuration file via the `--config` argument.

**Basic Example:**
python src/main.py --config configurations/pattern_configs/block_config.yaml

**Run with Different Pattern and Log Level:**
python src/main.py --config configurations/pattern_configs/flowers_config.yaml --log-level DEBUG
**Command-line Arguments:**

* `--config` (Required): Path to the specific `.yaml` configuration file.
* `--log-level` (Optional): Console logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). Default: `INFO`.
* `--output-dir` (Optional): Override the default output directory (`./output/`).
* `--profile` (Optional): Enable performance profiling. Results saved to `profile_stats.txt`.

---

## Configuration

* **`configurations/defaults.yaml`:** Contains default parameters for all steps.
* **`configurations/pattern_configs/*.yaml`:** Specific files for each dataset. Must define `reference_image_path` and `test_images` (relative paths). Can optionally override any parameter from `defaults.yaml`.

---

## Output

After a run for a dataset (e.g., 'block'), the `output/datasets/block/` directory will contain:

* **`analysis/*.csv`:** CSV file with detailed Delta E scores.
* **`inputs/`:** Copies of the original images used.
* **`processed/preprocessed/`:** Preprocessed test images.
* **`processed/segmented/[method_name]/`:** Segmented images, organized by method and containing k-type in the filename.
* **`summaries/`:** Summary images/plots (currently placeholder).

---

## Results & Analysis

Analyze the `_delta_e_results.csv` file to compare:

* Different segmentation methods (KMeans, SOM, DBSCAN).
* Determined 'k' vs. predefined 'k'.
* Standard CIELAB vs. PSO-DBN CIELAB for Delta E calculation.

Lower Delta E scores indicate higher color similarity to the reference. Summary statistics are also printed to the console.

*(Optional: Add your preliminary findings here.)*

---

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
````
