# python src/main.py --config configurations/pattern_configs/block_config.yaml
import sys
import os
import logging 
import cProfile 
import pstats   
import time
import traceback 
from pathlib import Path
from typing import Optional

# --- Step 1: Add Project Root to Python Path ---
try:
    SCRIPT_DIR = Path(__file__).parent.absolute()
    # Assume main.py is in 'src', so parent is the project root ('prints/')
    PROJECT_ROOT = SCRIPT_DIR.parent
    sys.path.insert(0, str(PROJECT_ROOT))
    # print(f"DEBUG: Project root added to sys.path: {PROJECT_ROOT}") # Uncomment for debugging path issues
except Exception as e:
    # If we can't even determine the paths, something is fundamentally wrong.
    print(f"FATAL ERROR: Could not determine project paths. Cannot continue. Error: {e}")
    sys.exit(1)

# --- Step 2: Import Core Application Components ---
# Why? main.py only needs the main pipeline class and the logging setup function.
# All other heavy lifting (numpy, cv2, tensorflow, etc.) is handled within the pipeline or its dependencies.
try:
    from src.pipeline import ProcessingPipeline # The main class orchestrating the workflow
    from src.utils.setup import setup_logging   # Function to configure logging
    from src.utils.app_setup import AppSetup, AppConfig
    from src.utils.setup import setup_logging, setup_mathplotlib_style
except ImportError as e:
    print(f"FATAL ERROR: Could not import core modules (ProcessingPipeline or setup_logging).")
    print("Please ensure 'src/pipeline.py', 'src/utils/app_setup.py' and 'src/utils/setup.py' exist and are correct.")
    traceback.print_exc() # Print detailed import error traceback
    sys.exit(1)
except Exception as e:
    # Catch any other unexpected import errors
    print(f"FATAL ERROR: An unexpected error occurred during initial imports: {e}")
    traceback.print_exc()
    sys.exit(1)


# --- Environment Settings (e.g., TensorFlow) ---
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Reduce TensorFlow's informational messages (1=INFO, 2=WARNING, 3=ERROR).
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# --- Main Application Function ---
def main(config: AppConfig):
    """
    Sets up logging, initializes and runs the main processing pipeline.
    This function is the primary controller. It assumes all setup (paths, args)
    
    Args:
        config: An AppConfig object containing all startup settings
    """
    profiler: Optional[cProfile.Profile] = None
    if config.profile:
        print("INFO: Performance profiling enabled.") # Simple print as logging might not be fully set yet
        profiler = cProfile.Profile()
        profiler.enable() # Start profiling

    start_time = time.perf_counter() # Start overall timer

    try:
        # 1. Setup Logging: Initialize file and console logging.
        setup_logging(config.output_dir, config.log_level)
    
        # Setup Mathplotlib Style
        setup_mathplotlib_style(config.project_root)
        
        logger = logging.getLogger(__name__) # Get logger instance after setup

        logger.info("Main function started. Initializing processing pipeline...")

        # 2. Create Pipeline Instance: Passing into from AppConfig
        pipeline = ProcessingPipeline(
            config_path = str(config.config_path),
            output_dir = config.output_dir,
            project_root = config.project_root)

        # 3. Run Pipeline: Execute the main workflow.
        logger.info(f"Starting pipeline run for dataset specified in: {Path(config.config_path).name}")
        pipeline.run()

        logger.info("Main execution finished successfully.")

    except Exception as e:
        # Catch any unexpected errors during pipeline execution.
        # Use critical level as these are usually fatal for the run.
        # Ensure logging is set up before trying to log.
        try:
            logging.getLogger(__name__).critical(f"A critical error occurred in main execution: {e}", exc_info=True)
        except Exception as log_e: # If logging itself fails
             print(f"CRITICAL ERROR (Logging failed): {e}")
             print(f"Logging Error: {log_e}")
             traceback.print_exc() 
        raise e
             
    finally:
        # This block always runs, even if errors occurred.
        total_time = time.perf_counter() - start_time
        completion_message = "=" * 80 + f"\nPROCESSING COMPLETED IN {total_time:.2f} SECONDS\n" + "=" * 80
        try:
             logging.getLogger(__name__).info(completion_message)
        except:
             print(completion_message)

        # Save profiling results if enabled.
        if config.profile and profiler:
            profiler.disable() # Stop profiling
            try:
                stats = pstats.Stats(profiler).sort_stats('cumtime') # Sort by cumulative time
                profile_path = config.output_dir / 'profile_stats.txt'
                with open(profile_path, 'w', encoding='utf-8') as f: # Specify encoding
                    stats.stream = f # Redirect output to file
                    stats.print_stats(30) # Print top 30 functions by cumulative time
                logging.getLogger(__name__).info(f"Profiling results saved to: {profile_path}")
            except Exception as e:
                logging.getLogger(__name__).warning(f"Failed to save profiling results: {e}", exc_info=True)


# --- Script Entry Point ---
if __name__ == "__main__":
    app_config: Optional[AppConfig] = None
    success = False
    try:
        # 1. Handle all application setup logic
        setup = AppSetup(PROJECT_ROOT)
        app_config = setup.parse_args()
        
        # 2. Print startup info
        setup.print_startup_info(app_config)
        
        # 3. Call the main function
        main(app_config)
        
        success = True
        
        # 4. Print success message
    except Exception as e:
        # Catch any critical error during setup or the main() call itself.
        print(f"\n❌ A CRITICAL ERROR OCCURRED: {e}")
        traceback.print_exc()
        if app_config:
             print(f"\nCheck log file for details: {app_config.output_dir / 'processing.log'}")
        sys.exit(1) # Hata durumunda 1 ile çık
        
    finally:
        if success and app_config:
            print("\n" + "=" * 60)
            print(f"✅ Processing completed successfully!")
            try:
                display_output = app_config.output_dir.relative_to(PROJECT_ROOT)
            except ValueError:
                display_output = app_config.output_dir
            print(f"📁 Results saved relative to: {display_output}")
            print("=" * 60)
        elif not success:
            print("\n" + "=" * 60)
            print("❌ Processing FAILED. Check logs for details.")
            print("=" * 60)