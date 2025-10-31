# src/utils/visualization.py
# This module is responsible for creating all visual outputs (plots, charts)
# for the analysis pipeline. It uses Matplotlib to generate summaries for
# reference image processing, individual test image segmentation, 
# preprocessing steps, and the final Delta E comparison.

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
except ImportError:
    # Log an error if matplotlib isn't installed, as it's critical for this module
    logging.getLogger(__name__).error("Matplotlib not found. Plotting functions will fail. "
                                     "Please install with 'pip install matplotlib'")

logger = logging.getLogger(__name__)

# --- Helper Function ---

def _plot_color_patch(ax, color_rgb: Tuple[float, float, float], title: str, fontsize: int = 8):
    """
    Internal helper to draw a single color swatch on a matplotlib axis.
    
    This function is the workhorse for creating the color palettes seen in
    the summary plots. It takes a color, draws a rectangle of that color,
    and prints a title underneath it.

    Args:
        ax: The matplotlib Axes (subplot) to draw on.
        color_rgb: A tuple of (R, G, B) values, expected in [0, 255] range.
        title: The text to display below the color patch (e.g., RGB/LAB values).
        fontsize: The font size for the title text.
    """
    # Create a small, solid-color image (50px high, 100px wide)
    color_patch_img = np.full((50, 100, 3), np.clip(color_rgb, 0, 255), dtype=np.uint8)
    ax.imshow(color_patch_img)
    ax.set_title(title, fontsize=fontsize, y=-0.25) # Position title below the patch
    ax.axis('off') # Hide the black box/axis lines

# --- Visualization Function 1: Reference Summary (Visual 1) ---

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
        original_image: The original BGR image (before preprocessing) for comparison.
        target_colors_lab: The final (k, 3) np.ndarray of "ground truth" LAB colors
                           derived from the K-Means result.
        output_path: The full file path (e.g., ".../summaries/reference_summary.png")
                     where the final plot will be saved.
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

    # --- Setup the GridSpec Layout ---
    # Use at least 3 columns (for the 3 images) or more if the palette is wider
    ncols = max(3, num_colors) 
    # Create a 2-row figure. The top row (for images) is 4x taller than the bottom row (for palette).
    fig = plt.figure(figsize=(max(15, 3 * ncols), 9)) # Taller figure
    gs = gridspec.GridSpec(2, ncols, figure=fig, height_ratios=[4, 1]) 
    
    fig.suptitle('Reference Image Processing Summary', fontsize=16, y=1.02)
    
    # --- Top Row: Images ---

    # Plot 1: Original Image
    ax_orig = fig.add_subplot(gs[0, 0])
    try:
        # Convert BGR (OpenCV) to RGB (Matplotlib)
        ax_orig.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        ax_orig.set_title('Original Reference')
    except Exception as e:
        logger.error(f"Error displaying original reference image: {e}")
        ax_orig.text(0.5, 0.5, 'Error Original', ha='center', va='center')
    ax_orig.axis('off')

    # Plot 2: K-Means Segmented Image
    ax_kmeans = fig.add_subplot(gs[0, 1])
    if kmeans_result and kmeans_result.segmented_image is not None:
        try:
            k_seg_display = cv2.cvtColor(kmeans_result.segmented_image, cv2.COLOR_BGR2RGB)
            ax_kmeans.imshow(k_seg_display)
            ax_kmeans.set_title(f'K-Means Seg. (k={kmeans_result.n_clusters})')
        except Exception as e:
            logger.error(f"Error displaying K-Means segmented image: {e}")
            ax_kmeans.text(0.5, 0.5, 'Error K-Means Seg', ha='center', va='center')
    else:
        ax_kmeans.text(0.5, 0.5, 'K-Means Seg.\nNot Available', ha='center', va='center')
    ax_kmeans.axis('off')

    # Plot 3: SOM Segmented Image
    ax_som = fig.add_subplot(gs[0, 2])
    if som_result and som_result.segmented_image is not None:
         try:
            s_seg_display = cv2.cvtColor(som_result.segmented_image, cv2.COLOR_BGR2RGB)
            ax_som.imshow(s_seg_display)
            ax_som.set_title(f'SOM Seg. (k={som_result.n_clusters})')
         except Exception as e:
            logger.error(f"Error displaying SOM segmented image: {e}")
            ax_som.text(0.5, 0.5, 'Error SOM Seg', ha='center', va='center')
    else:
        ax_som.text(0.5, 0.5, 'SOM Seg.\nNot Available', ha='center', va='center')
    ax_som.axis('off')

    # --- Bottom Row: Target Color Palette (from K-Means) ---
    if palette_ready:
        for i in range(num_colors):
            ax_patch = fig.add_subplot(gs[1, i]) # Add patch to the bottom row
            
            color_rgb_tuple = kmeans_result.avg_colors[i]
            l_val, a_val, b_val = target_colors_lab[i]
            
            # (Assuming avg_colors is (R, G, B) as per SegmentationResult spec)
            title = (f'Target {i+1}\n'
                     f'RGB: ({int(color_rgb_tuple[0])},{int(color_rgb_tuple[1])},{int(color_rgb_tuple[2])})\n'
                     f'LAB: ({l_val:.1f},{a_val:.1f},{b_val:.1f})')
            
            _plot_color_patch(ax_patch, color_rgb_tuple, title, fontsize=8)
    
    # --- Save ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for title
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
        logger.info(f"Reference summary plot saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save reference summary plot to {output_path}: {e}", exc_info=True)
    finally:
        plt.close(fig) # Always close the figure to free up memory

# --- Visualization Function 2: Individual Segmentation Summary (Visual 2) ---

def plot_segmentation_summary(
    result: SegmentationResult,
    original_preprocessed_image: np.ndarray,
    target_colors_lab: np.ndarray,
    dbn_model: Optional['DBN'],
    scalers: Optional[List['MinMaxScaler']],
    output_path: Path
    ):
    """
    Creates and saves a summary plot for a *single* test image segmentation.
    
    This plot provides a complete one-glance summary of a single experiment.
    It compares the input image to its segmented output and details the
    extracted color palette, along with its corresponding color values
    (RGB, Standard LAB, and DBN-predicted LAB).
    
    The layout uses a 2-row GridSpec:
    - Top Row: Displays the two main images (Input, Segmented Output).
    - Bottom Row: Displays the extracted color palette.

    Args:
        result: The SegmentationResult object (from K-Means, SOM, etc.) to visualize.
        original_preprocessed_image: The BGR input image *before* segmentation
                                   (but *after* preprocessing).
        target_colors_lab: The "ground truth" LAB colors from the reference image,
                           used to calculate the Delta E scores shown in the title.
        dbn_model: The trained DBN model (for DBN LAB conversion).
        scalers: The list of scalers [x, y_l, y_ab] (for DBN LAB conversion).
        output_path: The full file path to save the plot 
                     (e.g., ".../segmented/kmeans_opt/block1_determined_summary.png").
    """
    if not result or not result.is_valid():
        logger.warning(f"Cannot plot summary for invalid SegmentationResult (method: {result.method_name if result else 'Unknown'}). Skipping.")
        return
    if original_preprocessed_image is None:
        logger.warning(f"Cannot plot summary for {result.method_name}: Original preprocessed image is None. Skipping.")
        return

    method_name = result.method_name
    num_colors = result.n_clusters
    avg_rgb_colors = result.avg_colors

    # 1. Calculate Delta E scores to show in the plot title
    avg_delta_e_trad, avg_delta_e_dbn = _calculate_delta_e_for_plot(
        result, target_colors_lab, dbn_model, scalers
    )
    
    # --- Setup the GridSpec Layout ---
    # Use at least 2 columns (for the 2 images) or more if the palette is wider
    ncols = max(2, num_colors) 
    # Create a 2-row figure. Top row (images) is 4x taller than bottom row (palette).
    fig = plt.figure(figsize=(max(12, 3 * ncols), 9)) # Taller figure
    gs = gridspec.GridSpec(2, ncols, figure=fig, height_ratios=[4, 1]) 

    # Create the main title with the results
    plot_title = f"Segmentation Summary: {method_name} (k={num_colors})"
    if not np.isnan(avg_delta_e_trad):
        plot_title += f"\nAvg Delta E (Trad): {avg_delta_e_trad:.2f}"
    if not np.isnan(avg_delta_e_dbn):
        plot_title += f" | Avg Delta E (DBN): {avg_delta_e_dbn:.2f}"
    fig.suptitle(plot_title, fontsize=14, y=1.02)

    # --- Top Row: Images ---

    # Plot 1: Original (Preprocessed) Image
    ax_input = fig.add_subplot(gs[0, 0])
    try:
        display_orig = cv2.cvtColor(original_preprocessed_image, cv2.COLOR_BGR2RGB)
        ax_input.imshow(display_orig)
        ax_input.set_title("Preprocessed Input")
    except Exception as e:
        logger.error(f"Error displaying preprocessed image: {e}")
        ax_input.text(0.5, 0.5, 'Error Displaying Image', ha='center', va='center')
    ax_input.axis('off')

    # Plot 2: Segmented Image
    ax_seg = fig.add_subplot(gs[0, 1])
    try:
        display_seg = cv2.cvtColor(result.segmented_image, cv2.COLOR_BGR2RGB)
        ax_seg.imshow(display_seg)
        ax_seg.set_title("Segmented Output")
    except Exception as e:
        logger.error(f"Error displaying segmented image for {method_name}: {e}")
        ax_seg.text(0.5, 0.5, 'Error Displaying\nSegmented Image', ha='center', va='center')
    ax_seg.axis('off')

    # --- Bottom Row: Extracted Color Palette ---
    if num_colors > 0 and len(avg_rgb_colors) == num_colors:
        for i in range(num_colors):
            ax_patch = fig.add_subplot(gs[1, i]) # Add patch to the bottom row
            
            color_rgb = avg_rgb_colors[i] # (R, G, B) tuple
            
            # Get the formatted LAB strings
            lab_trad_str, lab_dbn_str = _get_lab_strings_for_plot(
                [color_rgb], dbn_model, scalers
            )
            
            # Create a detailed title for the color patch
            patch_title = (f'Color {i+1}\n'
                           f'RGB: ({int(color_rgb[0])},{int(color_rgb[1])},{int(color_rgb[2])})\n'
                           f'LAB: {lab_trad_str}\n'
                           f'DBN: {lab_dbn_str}')
            
            _plot_color_patch(ax_patch, color_rgb, patch_title, fontsize=8)

    # --- Save ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.90]) # Make space for suptitle
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
        logger.info(f"Segmentation summary plot saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save segmentation summary plot to {output_path}: {e}", exc_info=True)
    finally:
        plt.close(fig)


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
            # (Contains the fix for the 'traditional_ang_delta_e' typo)
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
                       color='tab:blue')

        # Plot "PSO-DBN" bars, shifted slightly right
        bars2 = ax.bar(index + bar_width/2, 
                       summary_df['PSO_DBN_DeltaE'], 
                       bar_width, 
                       label='PSO-DBN CIELAB (Ours)',
                       color='tab:orange')

        # --- 4. Format the Plot ---
        ax.set_ylabel('Average Delta E 2000 Score (Lower is Better)')
        ax.set_title(f'Average Color Difference (Delta E) Comparison by Method\n(Dataset: {results_df["dataset"].iloc[0]})')
        ax.set_xticks(index)
        ax.set_xticklabels(summary_df['group_label'], rotation=45, ha="right", fontsize=9)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7) # Add a light grid for readability

        # Add value labels on top of each bar
        ax.bar_label(bars1, padding=3, fmt='%.2f', fontsize=8)
        ax.bar_label(bars2, padding=3, fmt='%.2f', fontsize=8)
        
        fig.tight_layout() # Adjust plot to prevent labels from overlapping

        # --- 5. Save the Plot ---
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
        logger.info(f"Delta E summary bar chart saved to: {output_path}")

    except ImportError:
         logger.warning("Pandas or Matplotlib not found. Cannot generate Delta E summary chart.")
    except KeyError as e:
         logger.error(f"Failed to generate Delta E summary plot: A required column is missing from the DataFrame. Error: {e}", exc_info=True)
         logger.error(f"Available columns are: {list(results_df.columns)}")
    except Exception as e:
        logger.error(f"Failed to generate Delta E summary chart: {e}", exc_info=True)
    finally:
        if 'fig' in locals(): # Check if fig was successfully created
            plt.close(fig) # Always close the figure


# --- Private Helper functions for plotting ---

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