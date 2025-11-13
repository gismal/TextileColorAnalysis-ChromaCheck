import logging
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# --- Type Hinting Imports ---
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.models.pso_dbn import DBN
    from sklearn.preprocessing import MinMaxScaler

# --- Project-Specific Imports ---
from src.models.segmentation import SegmentationResult
from src.utils.color.color_conversion import bgr_to_rgb, convert_colors_to_cielab, convert_colors_to_cielab_dbn
from src.utils.color.color_analysis import ColorMetricCalculator

# --- Matplotlib Imports ---
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec # For advanced layout control
    from matplotlib.patches import Rectangle
    from mpl_toolkits.mplot3d import Axes3D
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
    SEABORN_AVAILABLE = True
except ImportError as e:
    # Log an error if matplotlib isn't installed, as it's critical for this module
    logging.getLogger(__name__).error(f"Matplotlib, Seaborn, or Mpl_toolkits could not be imported. Visualization will fail. Error: {e}"
        "Please run 'pip install matplotlib seaborn'"
    )
    MATPLOTLIB_AVAILABLE = False
    SEABORN_AVAILABLE = False
    # Create dummy classes/objects so functions can be defined without crashing
    plt = type('obj', (object,), {'figure': lambda: None, 'subplots': lambda: (None, None), 'Rectangle': object, 'close': lambda x: None})
    sns = type('obj', (object,), {'heatmap': lambda: None})
    Axes3D = type('obj', (object,), {})

logger = logging.getLogger(__name__)

# --- MATPLOTLIB STYLE CONFIGURATION ---
# TODO: yaml dosyasına taşı
if MATPLOTLIB_AVAILABLE:
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
        'font.size': 10,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelsize': 11,
        'axes.labelweight': 'bold',
        'figure.facecolor': 'white',
        'axes.facecolor': '#f8f9fa',
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'lines.linewidth': 2,
    })

# --- COLOR PALETTE for Consistent Visualizations ---
SEGMENT_COLORS = plt.cm.Set3(np.linspace(0, 1, 12))  # Up to 12 distinct colors

# Renk paletleri
COLOR_TRADITIONAL = '#3498db'
COLOR_PSO_DBN = '#e67e22'
COLOR_IMPROVEMENT_POSITIVE = '#2ecc71'
COLOR_IMPROVEMENT_NEGATIVE = '#e74c3c'

# --- Helper Function ---
def _bgr_to_rgb_tuple(bgr_tuple: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    Converts a single OpenCV (BGR) tuple to a Mathplotlib (RGB) tuple
    
    Args:
        bgr_tuple (Tuple[float, float, float]): BGR values as tuple
        
    Returns:
        Tuple[float, float, float]: RGB values represented as tuples.
    """
    return (bgr_tuple[2], bgr_tuple[1], bgr_tuple[0])   

def _plot_color_patch(ax,
                      color_rgb: Tuple[float, float, float],
                      title: str,
                      percentage: Optional[float] = None,
                      fontsize: int = 8):
    """
    Internal helper to draw a single color swatch on a matplotlib axis.
    
    This function is the workhorse for creating the color palettes seen in
    the summary plots. It takes a color, draws a rectangle of that color,
    and prints a title underneath it.

    Args:
        ax: The matplotlib Axes (subplot) to draw on.
        color_rgb: A tuple of (R, G, B) values, expected in [0, 255] range.
        title: The text to display below the color patch (e.g., RGB/LAB values).
        percentage (Optional[float]): area percentage of the segment
        fontsize: The font size for the title text.
    """
    # Create a small, solid-color image 
    color_patch_img = np.full((80, 150, 3), np.clip(color_rgb, 0, 255), dtype=np.uint8)
    ax.imshow(color_patch_img)
    
    # Add edges
    rect = plt.Rectangle((0, 0), 150, 80, fill=False, 
                         edgecolor='black', linewidth=2.5)
    ax.add_patch(rect)
    
    # Add percentage info if available
    if percentage is not None:
        title = f"{title}\n(Area: {percentage:.1f}%)"
    
    ax.set_title(title, fontsize=fontsize, y=-0.4, weight = 'bold', ha= 'center', multialignment= 'center') # Position title below the patch
    ax.axis('off') # Hide the black box/axis lines
    
def _create_segment_mask(
    image : np.ndarray,
    labels: np.ndarray,
    segment_index: int
) -> np.ndarray:
    """
    Creates a masked image showing only the pixels belonging to a specific segment
    
    Args:
        image: The original (perprocessed) BGR image (H, W, 3)
        labels: The flattened array of labels (size H*W) from SegmentationResult
        segment_index: The specific cluster index (e.g. 0, 1, 2 ...) to isolate
        
    Returns:
        A BGR image where only the pixels matching the segmetn_index are visible (all others are black) 
    """
    try:
        if labels is None:
            raise ValueError("Labels are None, can't create mask")
        
        mask_2d = labels.reshape(image.shape[:2])
        segment_mask = (mask_2d == segment_index)
        
        mask_3d = np.zeros_like(image, dtype=np.uint8)
        mask_3d[segment_mask] = 255
        
        black_bg = np.zeros_like(image, dtype= np.uint8)
        masked_image = np.where(mask_3d == 255, image, black_bg)
        
        return masked_image
    
    except Exception as e:
        logger.error(f"Failed to create segment mask for index {segment_index}: {e}", exc_info=True)
        # return full black if error occurs
        return np.zeros_like(image, dtype=np.uint8)
    
def _calculate_segment_stats(labels: np.ndarray, n_clusters: int) -> List[Dict[str, Any]]:
    """
    Calculates pixel number and the percentage for each segment
    
    Args:
        labels: Flattened label array (H*W)
        n_clusters: Number of clusters/segments
        
    Returns:
        List of dicts with 'count' and 'percentage' for each segment
    """
    if labels is None or len(labels) == 0:
        return []
    
    labels_flat = labels.flatten()
    total_pixels = len(labels_flat)
    
    unique, counts = np.unique(labels_flat, return_counts = True)
    
    stats = []
    for i in range(n_clusters):
        if i in unique:
            idx = np.where(unique == i)[0][0]
            count = counts[idx]
        else:
            count = 0
        
        percentage = (count / total_pixels) * 100 if total_pixels > 0 else 0
        stats.append({
            'cluster_id': i,
            'pixel_count': int(count),
            'percentage': float(percentage)
        })
        
    return stats

def _create_overlay_visualization(
    original_image: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Adds the transparent, colorful segments on original image
    
    Args: 
        original_image: Original BGR image
        labels: Flattened label array
        n_clusters: number of clusters
        alpha: Transperency (0= transparent, 1 = opaque)
        
    Returns:
        BGR overlay image
    """
    overlay = original_image.copy()
    labels_2d = labels.reshape(original_image.shape[:2])
    
    colors = SEGMENT_COLORS[:n_clusters]
    
    for i in range(n_clusters):
        mask = labels_2d == i
        if np.any(mask):
            # RGB to BGR conversion for OpenCV
            color_bgr = (colors[i][2] * 255, colors[i][1] * 255, colors[i][0] * 255)
            color_bgr = np.array(color_bgr, dtype=np.uint8)
            
            # Blend the color with original image
            overlay[mask] = cv2.addWeighted(
                overlay[mask], 1 - alpha,
                np.full_like(overlay[mask], color_bgr), alpha, 0
            )
    
    return overlay

def _create_palette_image(
    avg_colors: List[Tuple[float, float, float]],
    height: int = 80,
    width_per_color: int = 150
) -> np.ndarray:
    """
    Creates the color palette as a seperate image
    
    Args:
        avg_colors: List of (R, G, B) tuples
        height: Height of palette strip
        width_per_color: Width allocated per color
        
    Returns:
        RGB image array
    """
    n_colors = len(avg_colors)
    palette_img = np.zeros((height, width_per_color * n_colors, 3), dtype=np.uint8)
    
    for i, color_rgb in enumerate(avg_colors):
        x_start = i * width_per_color
        x_end = (i + 1) * width_per_color
        palette_img[:, x_start:x_end] = np.clip(color_rgb, 0, 255)
        
    return palette_img


def _calculate_delta_e_for_plot(
    result: SegmentationResult,
    target_colors_lab: np.ndarray,
    dbn_model: Optional['DBN'],
    scalers: Optional[List['MinMaxScaler']]
    ) -> Tuple[float, float]:
    """
    Private helper to calculate the avg Delta E scores needed for plot titles.
    
    This isolates the calculation logic from the plotting logic.
    Returns (NaN, NaN) on any failure.

    Args:
        result: The SegmentationResult to analyze.
        target_colors_lab: The "ground truth" LAB colors.
        dbn_model: The DBN model.
        scalers: The DBN scalers.

    Returns:
        A tuple of (avg_delta_e_trad, avg_delta_e_dbn).
    """
    avg_delta_e_trad = float('nan')
    avg_delta_e_dbn = float('nan')

    # Cannot calculate if we have no targets or no segmented colors
    if target_colors_lab is None or target_colors_lab.size == 0 or not result.avg_colors:
        return avg_delta_e_trad, avg_delta_e_dbn

    try:
        calculator = ColorMetricCalculator(target_colors_lab)
        rgb_array = np.clip(np.array(result.avg_colors, dtype=np.float32), 0, 255)
        
        # --- Standard LAB Delta E ---
        avg_lab_traditional = convert_colors_to_cielab(rgb_array)
        if avg_lab_traditional.size > 0:
            delta_e_trad_list = calculator.compute_all_delta_e(avg_lab_traditional)
            # Calculate mean, ignoring 'inf' values (which mean no match was found)
            avg_delta_e_trad = np.mean([d for d in delta_e_trad_list if d != float('inf')])
            if np.isnan(avg_delta_e_trad): avg_delta_e_trad = float('inf')

        # --- DBN LAB Delta E ---
        if dbn_model and scalers and len(scalers) == 3:
             avg_lab_dbn_list = convert_colors_to_cielab_dbn(
                 dbn_model, scalers[0], scalers[1], scalers[2], rgb_array
             )
             avg_lab_dbn = np.array(avg_lab_dbn_list)
             if avg_lab_dbn.size > 0:
                  delta_e_dbn_list = calculator.compute_all_delta_e(avg_lab_dbn)
                  avg_delta_e_dbn = np.mean([d for d in delta_e_dbn_list if d != float('inf')])
                  if np.isnan(avg_delta_e_dbn): avg_delta_e_dbn = float('inf')
                  
    except Exception as e:
         logger.error(f"Error calculating Delta E for plot ({result.method_name}): {e}", exc_info=True)
         
    return avg_delta_e_trad, avg_delta_e_dbn



def _get_lab_strings_for_plot(
    rgb_color_list: List[Tuple[float, float, float]], # Expects a list containing a *single* color
    dbn_model: Optional['DBN'],
    scalers: Optional[List['MinMaxScaler']]
    ) -> Tuple[str, str]:
    """
    Private helper to get formatted LAB value strings for a single color patch.
    
    This runs both standard and DBN conversion for one color and formats
    the result as a string (e.g., "(50.1, -1.2, 3.4)").

    Args:
        rgb_color_list: A list containing the single (R, G, B) tuple to convert.
        dbn_model: The DBN model.
        scalers: The DBN scalers.

    Returns:
        A tuple of (lab_trad_str, lab_dbn_str). Returns ("N/A", "N/A") on failure.
    """
    lab_trad_str = "N/A"
    lab_dbn_str = "N/A"
    
    try:
        if not rgb_color_list:
             return lab_trad_str, lab_dbn_str
             
        rgb_array = np.clip(np.array(rgb_color_list, dtype=np.float32), 0, 255)
        if rgb_array.shape[0] != 1:
             logger.warning(f"_get_lab_strings_for_plot expected 1 color, got {rgb_array.shape[0]}. Using first.")
             
        # --- Standard LAB ---
        avg_lab_traditional = convert_colors_to_cielab(rgb_array)
        if avg_lab_traditional.size > 0:
            l, a, b = avg_lab_traditional[0]
            lab_trad_str = f'({l:.1f}, {a:.1f}, {b:.1f})'

        # --- DBN LAB ---
        if dbn_model and scalers and len(scalers) == 3:
             avg_lab_dbn_list = convert_colors_to_cielab_dbn(
                 dbn_model, scalers[0], scalers[1], scalers[2], rgb_array
             )
             avg_lab_dbn = np.array(avg_lab_dbn_list)
             if avg_lab_dbn.size > 0:
                  l, a, b = avg_lab_dbn[0]
                  lab_dbn_str = f'({l:.1f}, {a:.1f}, {b:.1f})'
                  
    except Exception as e:
         logger.warning(f"Could not get LAB strings for plot: {e}")
         pass # Return "N/A"
         
    return lab_trad_str, lab_dbn_str


# --- Main Visualization Functions ---

def plot_reference_summary(
    kmeans_result: Optional[SegmentationResult],
    som_result: Optional[SegmentationResult],
    original_image: Optional[np.ndarray],
    target_colors_lab: Optional[np.ndarray],
    output_path: Path
    ):
    """
    Creates and saves a comprehensive summary plot for the reference image.
    
    This plot is crucial as it visualizes the *source of truth* for the entire
    analysis. It shows the original image, the results of the two primary
    segmentation methods (K-Means and SOM), and clearly displays the
    final "Target Color Palette" derived from the K-Means results.
    
    The layout uses a 2-row GridSpec:
    - Top Row: Displays the three main images (Original, K-Means, SOM).
    - Bottom Row: Displays the extracted color palette.

    Args:
        kmeans_result: The segmentation data from the K-Means run.
        som_result: The segmentation data from the SOM run.
        original_image: The original BGR image (before preprocessing).
        target_colors_lab: The final (k, 3) array of "ground truth" LAB colors.
        output_path: The full file path where the plot will be saved.
    """
    if original_image is None:
        logger.error("Cannot plot reference summary: Original image is None.")
        return

    # Check if we have all the necessary data to build the color palette
    palette_ready = (
        kmeans_result is not None and
        kmeans_result.is_valid() and
        target_colors_lab is not None and
        kmeans_result.n_clusters == len(kmeans_result.avg_colors) and
        kmeans_result.n_clusters == target_colors_lab.shape[0]
    )

    if not palette_ready:
        logger.warning(f"K-Means result data is incomplete or inconsistent. "
                       f"Reference summary palette cannot be generated accurately.")
        num_colors = 0
    else:
        num_colors = kmeans_result.n_clusters
        # calculate stats
        stats = _calculate_segment_stats(kmeans_result.labels, num_colors)

    ncols = max(3, num_colors) 
    # Create a 2-row figure. The top row (for images) is 4x taller than the bottom row (for palette).
    fig = plt.figure(figsize=(max(16, 4 * ncols), 10)) # Taller figure
    gs = gridspec.GridSpec(2, ncols, figure=fig, height_ratios=[4, 1]) 
    
    fig.suptitle('Reference Image Processing Summary', fontsize=18, weight= 'bold', y=1)
    
    # --- Top Row: Images ---
    ax_orig = fig.add_subplot(gs[0, 0])
    try:
        # Convert BGR (OpenCV) to RGB (Matplotlib)
        ax_orig.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        ax_orig.set_title('Original Reference', fontsize= 13, weight='bold')
    except Exception as e:
        logger.error(f"Error displaying original reference image: {e}")
        ax_orig.text(0.5, 0.5, 'Error Original', ha='center', va='center')
    ax_orig.axis('off')

    ax_kmeans = fig.add_subplot(gs[0, 1])
    if kmeans_result and kmeans_result.segmented_image is not None:
        try:
            k_seg_display = cv2.cvtColor(kmeans_result.segmented_image, cv2.COLOR_BGR2RGB)
            ax_kmeans.imshow(k_seg_display)
            ax_kmeans.set_title(f'K-Means Seg. (k={kmeans_result.n_clusters})', fontsize=13, weight='bold')
        except Exception as e:
            logger.error(f"Error displaying K-Means segmented image: {e}")
            ax_kmeans.text(0.5, 0.5, 'Error K-Means Seg', ha='center', va='center')
    else:
        ax_kmeans.text(0.5, 0.5, 'K-Means Seg.\nNot Available', ha='center', va='center')
    ax_kmeans.axis('off')

    ax_som = fig.add_subplot(gs[0, 2])
    if som_result and som_result.segmented_image is not None:
         try:
            s_seg_display = cv2.cvtColor(som_result.segmented_image, cv2.COLOR_BGR2RGB)
            ax_som.imshow(s_seg_display)
            ax_som.set_title(f'SOM Seg. (k={som_result.n_clusters})', fontsize=13, weight='bold')
         except Exception as e:
            logger.error(f"Error displaying SOM segmented image: {e}")
            ax_som.text(0.5, 0.5, 'Error SOM Seg', ha='center', va='center')
    else:
        ax_som.text(0.5, 0.5, 'SOM Seg.\nNot Available', ha='center', va='center', fontsize= 11)
    ax_som.axis('off')

    # --- Bottom Row: Target Color Palette (from K-Means) ---
    if palette_ready and num_colors > 0:
        for i in range(num_colors):
            ax_patch = fig.add_subplot(gs[1, i]) # Add patch to the bottom row
            
            color_rgb_tuple = kmeans_result.avg_colors[i]
            l_val, a_val, b_val = target_colors_lab[i]
            
            percentage = stats[i]['percentage'] if i < len(stats) else None
            
            # (Assuming avg_colors is (R, G, B) as per SegmentationResult spec)
            title = (f'Target {i+1}\n'
                     f'RGB: ({int(color_rgb_tuple[0])},{int(color_rgb_tuple[1])},{int(color_rgb_tuple[2])})\n'
                     f'LAB: ({l_val:.1f},{a_val:.1f},{b_val:.1f})')
            
            _plot_color_patch(ax_patch, color_rgb_tuple, title, 
                                      percentage=percentage, fontsize=9)
    
    # --- Save ---
    plt.tight_layout(rect=[0, 0.02, 1, 0.97]) # Adjust layout to make room for title
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
        logger.info(f"Reference summary plot saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save reference summary plot to {output_path}: {e}", exc_info=True)
    finally:
        plt.close(fig) # Always close the figure to free up memory

# --- Visualization Function 3: Preprocessing Steps ---
def plot_preprocessing_steps(
    original_image: Optional[np.ndarray], 
    preprocessed_image: Optional[np.ndarray],
    output_path: Path
    ):
    """
    Saves a simple side-by-side comparison of the original vs. preprocessed image.
    
    This is useful for debugging the `Preprocessor` and visually confirming
    the effects of steps like denoising, quantization, and resizing.

    Args:
        original_image: The original image (before preprocessing, BGR format).
        preprocessed_image: The image after preprocessing (BGR format).
        output_path: The full file path (e.g., .../preprocessed/block1_preprocessed_steps.png).
    """
    if original_image is None or preprocessed_image is None:
         logger.warning("Cannot save preprocessing plot with None image(s). Skipping.")
         return
         
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    title_prefix = output_path.stem.replace("_preprocessing_steps", "")
    fig.suptitle(f'Preprocessing Comparison: {title_prefix}', fontsize=14)
    
    try:
        display_orig = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        axes[0].imshow(display_orig)
        axes[0].set_title('Original Image')
    except Exception as e:
         logger.error(f"Error displaying original image in plot: {e}")
         axes[0].text(0.5, 0.5, 'Error Displaying\nOriginal Image', ha='center', va='center')
    axes[0].axis('off')
    
    try:
        display_preproc = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB)
        axes[1].imshow(display_preproc)
        axes[1].set_title('Preprocessed Image')
    except Exception as e:
         logger.error(f"Error displaying preprocessed image in plot: {e}")
         axes[1].text(0.5, 0.5, 'Error Displaying\nPreprocessed Image', ha='center', va='center')
    axes[1].axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=150)
        logger.info(f"Preprocessing steps plot saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save preprocessing steps plot to {output_path}: {e}", exc_info=True)
    finally:
        plt.close(fig)


# --- Visualization Function 4: Overall Delta E Summary (Visual 3) ---

def plot_delta_e_summary_bars(
    results_df: pd.DataFrame,
    output_path: Path
    ):
    """
    Creates a grouped bar chart summarizing the final average Delta E scores.
    
    This is the "final result" plot. It directly compares the performance of
    the 'Standard CIELAB' conversion vs. our 'PSO-DBN CIELAB' conversion
    across every segmentation method that was run.

    Args:
        results_df: The complete DataFrame containing all Delta E 
                    results from the pipeline run.
        output_path: The full file path (e.g., .../summaries/delta_e_summary_chart.png).
    """        
    if results_df is None or results_df.empty:
        logger.warning("Cannot plot Delta E summary: DataFrame is empty. Skipping plot.")
        return

    logger.info(f"Generating Delta E summary bar chart from {len(results_df)} results...")

    try:
        # --- 1. Prepare the Data ---
        # Group by method and k_type, then calculate the mean for Delta E columns
        summary_df = results_df.groupby(['method', 'k_type']).agg(
            Traditional_DeltaE=('traditional_avg_delta_e', 'mean'), 
            PSO_DBN_DeltaE=('pso_dbn_avg_delta_e', 'mean')
        ).reset_index()
        
        # Create a combined label for the x-axis (e.g., "kmeans_opt (determined)")
        summary_df['group_label'] = summary_df['method'] + ' (' + summary_df['k_type'] + ')'

        if summary_df.empty:
            logger.warning("No data to plot after grouping. Skipping Delta E summary chart.")
            return
            
        # --- 2. Setup the Plot ---
        num_groups = len(summary_df)
        index = np.arange(num_groups) # The x locations for the groups
        bar_width = 0.35 # The width of each bar
        
        # Make the figure width dynamic based on the number of groups
        fig, ax = plt.subplots(figsize=(max(12, num_groups * 2.5), 8)) 

        # --- 3. Plot the Bars ---
        # Plot "Standard" bars, shifted slightly left
        bars1 = ax.bar(index - bar_width/2, 
                       summary_df['Traditional_DeltaE'], 
                       bar_width, 
                       label='Standard CIELAB (skimage)',
                       color='#3498db',
                       edgecolor='black',
                       linewidth=1.5,
                       alpha=0.85)

        # Plot "PSO-DBN" bars, shifted slightly right
        bars2 = ax.bar(index + bar_width/2, 
                       summary_df['PSO_DBN_DeltaE'], 
                       bar_width, 
                       label='PSO-DBN CIELAB (Ours)',
                       color='#e67e22',
                       edgecolor='black',
                       linewidth=1.5,
                       alpha=0.85)

        # --- 4. Format the Plot ---
        ax.set_ylabel('Average Delta E 2000 Score (Lower is Better)', fontsize= 13, weight= 'bold')
        ax.set_title(f'Average Color Difference (Delta E) Comparison by Method\n(Dataset: {results_df["dataset"].iloc[0]})', fontsize=16, weight='bold', pad=20)
        ax.set_xticks(index)
        ax.set_xticklabels(summary_df['group_label'], rotation=45, ha="right", fontsize= 10 )
        ax.legend(fontsize=12, loc='upper right', framealpha=0.95)
        ax.grid(axis='y', linestyle='--', alpha=0.5, linewidth=0.8)
        
        # Add value labels on top of each bar
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom',
                       fontsize=9, weight='bold',
                       bbox=dict(boxstyle='round,pad=0.3',
                                facecolor='white', alpha=0.8))
        
        fig.tight_layout() # Adjust plot to prevent labels from overlapping

        # --- 5. Save the Plot ---
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        logger.info(f"Delta E summary bar chart saved to: {output_path}")

    except KeyError as e:
        logger.error(f"Missing column in DataFrame: {e}", exc_info=True)
        logger.error(f"Available columns: {list(results_df.columns)}")
    except Exception as e:
        logger.error(f"Failed to generate Delta E summary: {e}", exc_info=True)
    finally:
        if 'fig' in locals():
            plt.close(fig)

def plot_segmentation_summary(
    result: SegmentationResult,
    original_preprocessed_image: np.ndarray,
    target_colors_lab: np.ndarray,
    dbn_model: Optional['DBN'],
    scalers: Optional[List['MinMaxScaler']],
    output_path: Path
    
):
    """
    Individual segmentation summary with stats
    """
    if not result or not result.is_valid():
        logger.warning(f"Cannot plot summary for invalid result: {result.method_name if result else 'Unknown'}")
        return
    if original_preprocessed_image is None:
        logger.warning(f"Original image is None for {result.method_name}")
        return

    method_name = result.method_name
    num_colors = result.n_clusters
    avg_rgb_colors = result.avg_colors
    
    # calculate stats
    stats = _calculate_segment_stats(result.labels, num_colors)
    
    # calculate delta e
    avg_delta_e_trad, avg_delta_e_dbn = _calculate_delta_e_for_plot(
        result,
        target_colors_lab,
        dbn_model,
        scalers
    )
    
    ncols = max(2, num_colors)
    fig = plt.figure(figsize=(max(14, 4 * ncols), 10))
    gs = gridspec.GridSpec(2, ncols, figure=fig, height_ratios=[4, 1.2])

    plot_title = f"Segmentation Summary: {method_name} (k={num_colors})"
    if not np.isnan(avg_delta_e_trad):
        plot_title += f"\nΔE Traditional: {avg_delta_e_trad:.2f}"
    if not np.isnan(avg_delta_e_dbn):
        plot_title += f" | ΔE PSO-DBN: {avg_delta_e_dbn:.2f}"
    fig.suptitle(plot_title, fontsize=15, weight='bold', y=0.98)

    # --- Top Row ---
    ax_input = fig.add_subplot(gs[0, 0])
    try:
        display_orig = cv2.cvtColor(original_preprocessed_image, cv2.COLOR_BGR2RGB)
        ax_input.imshow(display_orig)
        ax_input.set_title("Preprocessed Input", fontsize=13, weight='bold')
    except Exception as e:
        logger.error(f"Error displaying preprocessed: {e}")
        ax_input.text(0.5, 0.5, 'Error', ha='center', va='center')
    ax_input.axis('off')

    ax_seg = fig.add_subplot(gs[0, 1])
    try:
        display_seg = cv2.cvtColor(result.segmented_image, cv2.COLOR_BGR2RGB)
        ax_seg.imshow(display_seg)
        ax_seg.set_title("Segmented Output", fontsize=13, weight='bold')
    except Exception as e:
        logger.error(f"Error displaying segmented: {e}")
        ax_seg.text(0.5, 0.5, 'Error', ha='center', va='center')
    ax_seg.axis('off')

    # --- Bottom Row: Enhanced Palette ---
    if num_colors > 0 and len(avg_rgb_colors) == num_colors:
        for i in range(num_colors):
            ax_patch = fig.add_subplot(gs[1, i])
            
            color_rgb = avg_rgb_colors[i]
            lab_trad_str, lab_dbn_str = _get_lab_strings_for_plot(
                [color_rgb], dbn_model, scalers
            )
            
            percentage = stats[i]['percentage'] if i < len(stats) else None
            
            patch_title = (f'Color {i+1}\n'
                          f'RGB: ({int(color_rgb[0])},{int(color_rgb[1])},{int(color_rgb[2])})\n'
                          f'LAB: {lab_trad_str}\n'
                          f'DBN: {lab_dbn_str}')
            
            _plot_color_patch(ax_patch, color_rgb, patch_title, 
                                      percentage=percentage, fontsize=8)

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        logger.info(f"Enhanced segmentation summary saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save segmentation summary: {e}", exc_info=True)
    finally:
        plt.close(fig)

    
def plot_segment_palette(
    result: SegmentationResult, 
    preprocessed_image: np.ndarray,
    dbn_model: Optional['DBN'],
    scalers: Optional[List['MinMaxScaler']],
    output_path: Path
):
    """"
    Creates a plot showing each segment of a test image side-by-side
    
    For each cluster (segment) found, this plot shows the isolated pixels
    of that segment on a black background, along with its extracted color palette
    and RGB/LAB/DBN values
    
    Args:
        result: The SegmentationResult object to visualize.
        preprocessed_image: The BGR input image that was segmented.
        dbn_model: The trained DBN model (for DBN LAB conversion).
        scalers: The list of scalers [x, y_l, y_ab] (for DBN LAB conversion).
        output_path: The full file path to save the plot.
    """    
    if not result or not result.is_valid():
        logger.warning(f"Cannot plot segment palette for invalid result({result.method_name}). Skipping")
        return
    
    num_colors = result.n_clusters
    if num_colors == 0:
        logger.warning(f"No clusters found for {result.method_name}. Skipping segment palette plot.")
        return
    
    stats = _calculate_segment_stats(result.labels, num_colors)
    
    # For each segment one column
    fig, axes = plt.subplots(2, num_colors, 
                             figsize= (max(10, 4 * num_colors), 10),
                             gridspec_kw={'height_ratios': [3, 1.2]}
                             )
    if num_colors == 1:
        axes = np.array(axes).reshape(2, 1)
        
    fig.suptitle(f"Segment Palette Breakdown: {result.method_name} (k= {num_colors})", fontsize= 16, weight= 'bold', y= 1)
    
    avg_rgb_colors = result.avg_colors # (R, G, B) tuple list
    
    for i in range(num_colors):
        ax_img = axes[0, i]
        ax_patch = axes[1, i]
        
        # --- 1. Plot the masked segmentation ---
        try:
            segment_image = _create_segment_mask(preprocessed_image, result.labels, i)
            ax_img.imshow(cv2.cvtColor(segment_image, cv2.COLOR_BGR2RGB))
            percentage = stats[i]['percentage'] if i < len(stats) else 0
            ax_img.set_title(f"Segment {i+1}\n({percentage:.1f}% of image)",
                           fontsize=12, weight='bold')
            
        except Exception as e:
            logger.error(f"Error displaying segment mask {i}: {e}")
            ax_img.text(0.5, 0.5, 'Error Mask', ha= 'center', va= 'center')
        
        ax_img.axis('off')
        
        # --- 2. Plot the Color Palette and the Values ---
        try:
            color_rgb = avg_rgb_colors[i]
            
            # calculate the color values
            lab_trad_str, lab_dbn_str = _get_lab_strings_for_plot(
                [color_rgb], dbn_model, scalers
            )
           
            patch_title = (f'RGB: ({int(color_rgb[0])},{int(color_rgb[1])},{int(color_rgb[2])})\n'
                          f'LAB: {lab_trad_str}\n'
                          f'DBN: {lab_dbn_str}')
            
            _plot_color_patch(ax_patch, color_rgb, patch_title, fontsize=9)
        except Exception as e:
            logger.error(f"Error plotting patch {i}: {e}")
        ax_patch.axis('off')
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        logger.info(f"Enhanced segment palette saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save segment palette: {e}", exc_info=True)
    finally:
        plt.close(fig)
        
# --- Private Helper functions for plotting ---


def plot_segmentation_overlay(
    result: SegmentationResult,
    original_image: np.ndarray,
    output_path: Path
):
    """
    Original image vs overlay 
    
    Args:
        result: SegmentationResult object
        original_image: Original preprocessed BGR image
        output_path: Path to save the plot
    """
    if not result or not result.is_valid():
        logger.warning("Cannot create overlay for invalid result")
        return 
    
    try:
        fig, axes = plt.subplots(1, 2, figsize= (16,8))
        
        # Leftside: Original
        axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image', fontsize=14, weight='bold')
        axes[0].axis('off')
        
        # Rigthside: Overlay
        overlay = _create_overlay_visualization(
            original_image, result.labels, result.n_clusters, alpha=0.6
        )
        axes[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f'Segmentation Overlay (k={result.n_clusters})', 
                         fontsize=14, weight='bold')
        axes[1].axis('off')
        
        fig.suptitle(f'Overlay Visualization: {result.method_name}', 
                     fontsize=16, weight='bold', y=0.98)
        
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        logger.info(f"Overlay plot saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to create overlay plot: {e}", exc_info=True)
    finally:
        plt.close(fig)
        
def plot_delta_e_heatmap(
    results_df: pd.DataFrame,
    output_path: Path
):
    """
    Delta E skorlarını heatmap olarak görselleştirir.
    
    Args:
        results_df: DataFrame with Delta E results
        output_path: Path to save the heatmap
    """
    if not SEABORN_AVAILABLE:
        logger.warning("Seaborn not available. Skipping heatmap generation.")
        return
    
    if results_df is None or results_df.empty:
        logger.warning("Cannot create heatmap: DataFrame is empty")
        return
    
    try:
        # Pivot tables oluştur
        pivot_trad = results_df.pivot_table(
            values='traditional_avg_delta_e',
            index='method',
            columns='k_type',
            aggfunc='mean'
        )
        
        pivot_dbn = results_df.pivot_table(
            values='pso_dbn_avg_delta_e',
            index='method',
            columns='k_type',
            aggfunc='mean'
        )
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Traditional heatmap
        sns.heatmap(pivot_trad, annot=True, fmt='.2f', cmap='RdYlGn_r',
                    ax=axes[0], cbar_kws={'label': 'Delta E (Lower is Better)'},
                    linewidths=1, linecolor='white')
        axes[0].set_title('Standard CIELAB Delta E', fontsize=14, weight='bold')
        axes[0].set_xlabel('K Type', fontsize=12, weight='bold')
        axes[0].set_ylabel('Method', fontsize=12, weight='bold')
        
        # PSO-DBN heatmap
        sns.heatmap(pivot_dbn, annot=True, fmt='.2f', cmap='RdYlGn_r',
                    ax=axes[1], cbar_kws={'label': 'Delta E (Lower is Better)'},
                    linewidths=1, linecolor='white')
        axes[1].set_title('PSO-DBN CIELAB Delta E', fontsize=14, weight='bold')
        axes[1].set_xlabel('K Type', fontsize=12, weight='bold')
        axes[1].set_ylabel('Method', fontsize=12, weight='bold')
        
        fig.suptitle('Delta E Heatmap Comparison', fontsize=16, weight='bold', y=1.02)
        
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        logger.info(f"Heatmap saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to create heatmap: {e}", exc_info=True)
    finally:
        if 'fig' in locals():
            plt.close(fig)


def plot_improvement_chart(
    results_df: pd.DataFrame,
    output_path: Path
):
    """
    Shows the PSO-DBN improvement over the traditional
    
    Args:
        results_df: DataFrame with Delta E results
        output_path: Path to save the chart
    """
    if results_df is None or results_df.empty:
        logger.warning("Cannot create improvement chart: DataFrame is empty")
        return
    
    try:
        # group average
        summary_df = results_df.groupby(['method', 'k_type']).agg({
            'traditional_avg_delta_e': 'mean',
            'pso_dbn_avg_delta_e': 'mean'
        }).reset_index()
        
        # calculate improvement percentage
        summary_df['improvement'] = (
            (summary_df['traditional_avg_delta_e'] - 
             summary_df['pso_dbn_avg_delta_e']) / 
            summary_df['traditional_avg_delta_e'] * 100
        )
        
        # create label
        summary_df['label'] = summary_df['method'] + '\n(' + summary_df['k_type'] + ')'
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        x = np.arange(len(summary_df))
        colors = ['#2ecc71' if imp > 0 else '#e74c3c' 
                  for imp in summary_df['improvement']]
        
        bars = ax.bar(x, summary_df['improvement'], color=colors, 
                     alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Axis adjustments
        ax.set_ylabel('Improvement (%)', fontsize=13, weight='bold')
        ax.set_title('PSO-DBN Performance Improvement over Standard CIELAB\n(Positive = Better)',
                     fontsize=15, weight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(summary_df['label'], rotation=45, ha='right', fontsize=10)
        
        # Reference line
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
        ax.grid(axis='y', alpha=0.4, linestyle='--')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', 
                    va='bottom' if height > 0 else 'top',
                    fontsize=10, weight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', 
                             facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        logger.info(f"Improvement chart saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to create improvement chart: {e}", exc_info=True)
    finally:
        if 'fig' in locals():
            plt.close(fig)


def plot_all_methods_comparison(
    image_name: str,
    results_dict: Dict[str, SegmentationResult],
    preprocessed_image: np.ndarray,
    output_path: Path
):
    """
    Compares all the methods side to side
    
    Args:
        image_name: Name of the test image
        results_dict: Dictionary mapping method names to SegmentationResults
        preprocessed_image: The preprocessed BGR image
        output_path: Path to save the comparison grid
    """
    if not results_dict:
        logger.warning("No results to compare")
        return
    
    try:
        n_methods = len(results_dict)
        fig, axes = plt.subplots(3, n_methods, 
                                figsize=(5*n_methods, 14),
                                gridspec_kw={'height_ratios': [3, 3, 1]})
        
        # Tek method için axes'i 2D array'e çevir
        if n_methods == 1:
            axes = np.array(axes).reshape(3, 1)
        
        for idx, (method_name, result) in enumerate(results_dict.items()):
            if not result or not result.is_valid():
                continue
            
            # Row 1: Segmented image
            axes[0, idx].imshow(cv2.cvtColor(result.segmented_image, 
                                            cv2.COLOR_BGR2RGB))
            axes[0, idx].set_title(f'{method_name}\n(k={result.n_clusters})',
                                  fontsize=12, weight='bold')
            axes[0, idx].axis('off')
            
            # Row 2: Overlay
            overlay = _create_overlay_visualization(
                preprocessed_image, result.labels, result.n_clusters, alpha=0.6
            )
            axes[1, idx].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            axes[1, idx].set_title('Overlay View', fontsize=11, weight='bold')
            axes[1, idx].axis('off')
            
            # Row 3: Color palette
            palette_img = _create_palette_image(result.avg_colors)
            axes[2, idx].imshow(palette_img)
            axes[2, idx].set_title('Color Palette', fontsize=11, weight='bold')
            axes[2, idx].axis('off')
        
        fig.suptitle(f'All Methods Comparison: {image_name}', 
                     fontsize=17, weight='bold', y=0.995)
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        logger.info(f"Methods comparison grid saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to create methods comparison: {e}", exc_info=True)
    finally:
        if 'fig' in locals():
            plt.close(fig)

