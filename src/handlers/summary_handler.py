import logging
import numpy as np
from typing import Dict, Any, List
from pathlib import Path

# Gerekli proje içi importlar
from src.utils.output_manager import OutputManager
from src.utils.visualization import plot_delta_e_summary_bars

# Pandas'ı import etmeyi dene, yoksa uyarı ver
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    
logger = logging.getLogger(__name__)

class SummaryHandler:
    """
    Handles the final step of saving results to CSV and plotting the
    overall summary chart.
    
    This class encapsulates the logic from `pipeline._save_and_summarize_results`.
    """
    
    def __init__(self, output_manager: OutputManager):
        """
        Initializes the SummaryHandler.

        Args:
            output_manager: The instance of OutputManager to save outputs.
        """
        self.output_manager = output_manager
        logger.debug("SummaryHandler initialized.")
        
        if not PANDAS_AVAILABLE:
            logger.warning("Pandas library is not installed. "
                           "Summary generation (CSV, plots, console tables) will be skipped. "
                           "Install with 'pip install pandas'")

    def execute(self, all_delta_e: List[Dict[str, Any]]):
        """
        Saves all collected Delta E results to a CSV file, prints summaries
        to the console, and generates the final summary bar chart.

        Args:
            all_delta_e: A list containing all Delta E result dictionaries.
        """
        
        if not all_delta_e:
            logger.warning("No Delta E results were generated to save or summarize.")
            return

        if not PANDAS_AVAILABLE:
            logger.error("Cannot execute SummaryHandler: Pandas is not installed.")
            return

        # 1. Save to CSV
        logger.info(f"Saving {len(all_delta_e)} total Delta E results entries to CSV...")
        try:
            self.output_manager.save_delta_e_results(all_delta_e)
        except Exception as e:
            # save_delta_e_results zaten kendi hatasını loglar, ama yine de
            logger.error(f"Failed to save Delta E results to CSV: {e}", exc_info=True)
            # CSV kaydı başarısız olsa bile özetlemeye devam etmeyi deneyebiliriz
            
        # 2. Print Console Summaries
        try:
            pd.options.mode.chained_assignment = None 
            df = pd.DataFrame(all_delta_e)
            
            if df.empty:
                logger.warning("DataFrame is empty. Cannot summarize.")
                return

            # --- Overall Summary ---
            logger.info("--- Overall Results Summary (Averaged across images and k_types) ---")
            summary = df.groupby('method').agg(
                avg_traditional_delta_e=('traditional_avg_delta_e', 'mean'),
                avg_pso_dbn_delta_e=('pso_dbn_avg_delta_e', 'mean'),
                avg_processing_time=('processing_time', 'mean')
            ).reset_index()
            logger.info("\n" + summary.to_string(float_format="%.3f"))

            # --- Detailed Summary ---
            logger.info("--- Detailed Results by Method and k_type (Averaged across images) ---")
            detailed_summary = df.groupby(['method', 'k_type']).agg(
                avg_traditional_delta_e=('traditional_avg_delta_e', 'mean'),
                avg_pso_dbn_delta_e=('pso_dbn_avg_delta_e', 'mean'),
                avg_processing_time=('processing_time', 'mean'),
                avg_n_clusters=('n_clusters', 'mean')
            ).reset_index()
            logger.info("\n" + detailed_summary.to_string(float_format="%.3f"))
            logger.info("--- End of Summary ---")
            
            # 3. Plot Summary Bar Chart
            try:
                plot_filename = f"{self.output_manager.dataset_name}_delta_e_summary.png"
                plot_output_path = self.output_manager.dataset_dir / "summaries" / plot_filename

                from src.utils.visualization import plot_delta_e_summary_bars
                plot_delta_e_summary_bars(
                    results_df=df,
                    output_path=plot_output_path
                )
            except Exception as plot_err:
                logger.error(f"Failed to generate Delta E summary plot: {plot_err}", exc_info=True)
            
            # 4. NEW: Heatmap Visualization
            try:
                heatmap_filename = f"{self.output_manager.dataset_name}_delta_e_heatmap.png"
                heatmap_path = self.output_manager.dataset_dir / "summaries" / heatmap_filename
                
                from src.utils.visualization import plot_delta_e_heatmap
                plot_delta_e_heatmap(
                    results_df=df,
                    output_path=heatmap_path
                )
            except Exception as heatmap_err:
                logger.error(f"Failed to generate heatmap: {heatmap_err}", exc_info=True)
            
            # 5. NEW: Improvement Chart
            try:
                improvement_filename = f"{self.output_manager.dataset_name}_improvement_chart.png"
                improvement_path = self.output_manager.dataset_dir / "summaries" / improvement_filename
                
                from src.utils.visualization import plot_improvement_chart
                plot_improvement_chart(
                    results_df=df,
                    output_path=improvement_path
                )
            except Exception as imp_err:
                logger.error(f"Failed to generate improvement chart: {imp_err}", exc_info=True)

        except Exception as e:
            logger.error(f"Failed to generate summaries: {e}", exc_info=True)