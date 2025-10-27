# src/main.py
# FINALIZED WITH DOCSTRINGS AND CLEANED COMMENTS

import sys
import os
import argparse
import logging # For logging critical errors before full setup
import cProfile # For performance profiling
import pstats   # For saving profiling results
import time
import traceback # For printing detailed error information
from pathlib import Path
from typing import Optional

# --- Step 1: Add Project Root to Python Path ---
# Why? This allows Python to find our 'src' module (and submodules like
# 'src.pipeline', 'src.utils') correctly, regardless of where the
# script is run from, as long as the directory structure is maintained.
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
except ImportError as e:
    print(f"FATAL ERROR: Could not import core modules (ProcessingPipeline or setup_logging).")
    print("Please ensure 'src/pipeline.py' and 'src/utils/setup.py' exist and are correct.")
    traceback.print_exc() # Print detailed import error traceback
    sys.exit(1)
except Exception as e:
    # Catch any other unexpected import errors
    print(f"FATAL ERROR: An unexpected error occurred during initial imports: {e}")
    traceback.print_exc()
    sys.exit(1)


# --- Environment Settings (e.g., TensorFlow) ---
# Why? These often need to be set *before* TensorFlow is heavily used.
# Force TensorFlow to use CPU only (no GPU). Remove '-1' to enable GPU if available.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Reduce TensorFlow's informational messages (1=INFO, 2=WARNING, 3=ERROR).
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# --- Main Application Function ---

def main(config_path: str, log_level: str, output_dir: Path, profile: bool):
    """
    Sets up logging, initializes and runs the main processing pipeline.

    This function acts as the primary controller after command-line arguments
    are parsed. It handles the overall execution flow, including optional
    performance profiling and top-level error catching.

    Args:
        config_path: Absolute path to the configuration YAML file.
        log_level: The desired logging level string (e.g., 'INFO', 'DEBUG').
        output_dir: The absolute path to the base directory for saving outputs.
        profile: Boolean flag indicating whether to enable cProfile for performance analysis.
    """
    profiler: Optional[cProfile.Profile] = None
    if profile:
        print("INFO: Performance profiling enabled.") # Simple print as logging might not be fully set yet
        profiler = cProfile.Profile()
        profiler.enable() # Start profiling

    start_time = time.perf_counter() # Start overall timer

    try:
        # 1. Setup Logging: Initialize file and console logging.
        #    This should be one of the first steps so subsequent messages are captured.
        setup_logging(output_dir, log_level)
        logger = logging.getLogger(__name__) # Get logger instance *after* setup

        logger.info("Main function started. Initializing processing pipeline...")

        # 2. Create Pipeline Instance: Pass necessary configurations.
        #    All the complex logic is encapsulated within this class.
        pipeline = ProcessingPipeline(
            config_path=config_path,
            output_dir=output_dir,
            project_root=PROJECT_ROOT # Pass the determined project root
        )

        # 3. Run Pipeline: Execute the main workflow.
        logger.info(f"Starting pipeline run for dataset specified in: {Path(config_path).name}")
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
             traceback.print_exc() # Print traceback directly if logging fails

    finally:
        # This block always runs, even if errors occurred.
        total_time = time.perf_counter() - start_time
        completion_message = "=" * 80 + f"\nPROCESSING COMPLETED IN {total_time:.2f} SECONDS\n" + "=" * 80
        # Try logging the final message, but fall back to print if logging failed.
        try:
             logging.getLogger(__name__).info(completion_message)
        except:
             print(completion_message)

        # Save profiling results if enabled.
        if profile and profiler:
            profiler.disable() # Stop profiling
            try:
                stats = pstats.Stats(profiler).sort_stats('cumtime') # Sort by cumulative time
                profile_path = output_dir / 'profile_stats.txt'
                with open(profile_path, 'w', encoding='utf-8') as f: # Specify encoding
                    stats.stream = f # Redirect output to file
                    stats.print_stats(30) # Print top 30 functions by cumulative time
                logging.getLogger(__name__).info(f"Profiling results saved to: {profile_path}")
            except Exception as e:
                logging.getLogger(__name__).warning(f"Failed to save profiling results: {e}", exc_info=True)


# --- Script Entry Point ---
# Why `if __name__ == "__main__":`?
# This standard Python construct ensures that the code inside this block
# only runs when the script is executed directly (e.g., `python src/main.py`),
# and *not* when the script is imported as a module into another script.
# Its primary responsibilities here are:
# 1. Parsing command-line arguments.
# 2. Setting up essential paths (like OUTPUT_DIR).
# 3. Calling the main() function with the parsed arguments.
# 4. Handling critical errors that might occur *before* logging is set up.
if __name__ == "__main__":

    # Use argparse to define and parse command-line arguments.
    parser = argparse.ArgumentParser(
        description="Textile Color Analysis System using PSO-optimized DBN"
        # Add epilog for examples if desired
        # epilog="""Example:\n python src/main.py --config configs/block_config.yaml --log-level DEBUG"""
    )
    # Configuration file path (now required)
    parser.add_argument(
        '--config',
        type=str,
        required=True, # Make it mandatory to provide a config file
        help='Path to the specific pattern configuration YAML file (relative to project root or absolute).'
    )
    # Logging level (optional, defaults to INFO)
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set the minimum logging level for console output.'
    )
    # Output directory override (optional)
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None, # Default is handled below based on PROJECT_ROOT
        help='Override the default output directory (which is PROJECT_ROOT/output).'
    )
    # Profiling flag (optional)
    parser.add_argument(
        '--profile',
        action='store_true', # Makes it a flag: presence means True
        help='Enable detailed performance profiling using cProfile.'
    )

    args = parser.parse_args() # Parse the arguments provided by the user

    # Determine the final output directory.
    if args.output_dir:
        # Use the user-provided directory. Resolve makes it absolute.
        OUTPUT_DIR = Path(args.output_dir).resolve()
    else:
        # Default to 'output' directory inside the project root.
        OUTPUT_DIR = (PROJECT_ROOT / "output").resolve()
    # Ensure the output directory exists.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Resolve the configuration file path (handles relative paths).
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()

    # Check if the configuration file actually exists before proceeding.
    if not config_path.exists():
         # Use print here because logging isn't set up yet.
         print(f"❌ CRITICAL ERROR: Configuration file not found at the resolved path: {config_path}")
         sys.exit(1) # Exit immediately if config is missing

    # --- Run the main application ---
    try:
        # Print initial startup messages (before logging takes over).
        print(f"Starting Textile Color Analysis System...")
        try: # Display relative paths if possible, looks cleaner
             display_config = config_path.relative_to(PROJECT_ROOT)
             display_output = OUTPUT_DIR.relative_to(PROJECT_ROOT)
        except ValueError: # Handle if paths are on different drives etc.
             display_config = config_path
             display_output = OUTPUT_DIR
        print(f"Using Config : {display_config}")
        print(f"Output Dir   : {display_output}")
        print(f"Log Level    : {args.log_level}")
        print(f"Profiling    : {'Enabled' if args.profile else 'Disabled'}")
        print("-" * 60)

        # Call the main function with resolved paths and arguments.
        main(
            config_path=str(config_path),
            log_level=args.log_level,
            output_dir=OUTPUT_DIR,
            profile=args.profile
        )

        print("\n" + "=" * 60)
        print(f"✅ Processing completed successfully!")
        print(f"📁 Results saved relative to: {display_output}")
        print("=" * 60)

    except Exception as e:
        # Catch any critical error during setup or the main() call itself.
        print(f"\n❌ A CRITICAL ERROR OCCURRED: {e}")
        # Print detailed traceback for debugging.
        traceback.print_exc()
        # Exit with a non-zero code to indicate failure.
        sys.exit(1)