import argparse
import sys
from pathlib import Path
from typing import NamedTuple

class AppConfig(NamedTuple):
    """
    A simple data container to hold the app's startup config
    This makes it easy to passs all startup settings (paths, log level etc.)
    as a single, clean object
    """
    config_path : Path
    log_level: str
    output_dir: Path
    profile: bool
    project_root: Path
    
class AppSetup:
    """
    Handles the application's initial setup and configuration
    This class is responsible for:
        1. Defining and parsing command-line arguments
        2. Resolving all necessary paths (project root, config file, output dir)
        3. Creating the output directory
        4. Validating that the config file exits
        5. Printing a summary of the startup settings
    """
    def __init__(self, project_root: Path):
        """
        Initializes the setup ability
        
        Args:
            project_root: The absolute path to the project's root directory
        """
        self.project_root = project_root
        self.parser = self._create_parser()
        
    def _create_parser(self) -> argparse.ArgumentParser:
        """Defines all expected command-line arguments"""
        parser = argparse.ArgumentParser(
            description= "Textile Color Analysis System using PSO-optimized DBN"
        )
        parser.add_argument(
            '--config', type=str, required=True,
            help='Path to the specific pattern configuration YAML file (relative to project root or absolute).'
        )
        parser.add_argument(
            '--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
            default='INFO', help='Set the minimum logging level.'
        )
        parser.add_argument(
            '--output-dir', type=str, default=None,
            help='Override the default output directory (which is PROJECT_ROOT/output).'
        )
        parser.add_argument(
            '--profile', action='store_true',
            help='Enable detailed performance profiling.'
        )
        return parser

    def parse_args(self) -> AppConfig:
        """
        Parses args, resolves paths, creates dirs, and returns an AppConfig.
        
        This is the main method to run the setup logic.
        
        Returns:
            An AppConfig object containing all resolved settings.
            
        Raises:
            SystemExit: If the specified configuration file is not found.
        """
        args = self.parser.parse_args()

        # 1. Determine and create the output directory
        if args.output_dir:
            output_dir = Path(args.output_dir).resolve()
        else:
            output_dir = (self.project_root / "output").resolve()
        
        output_dir.mkdir(parents=True, exist_ok=True)

        # 2. Resolve the configuration file path
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = (self.project_root / config_path).resolve()

        # 3. Validate that the config file exists
        if not config_path.exists():
             # Use print() because logging is not set up yet.
             print(f"❌ CRITICAL ERROR: Configuration file not found at: {config_path}")
             sys.exit(1) # Exit immediately

        # 4. Return the clean configuration object
        return AppConfig(
            config_path=config_path,
            log_level=args.log_level,
            output_dir=output_dir,
            profile=args.profile,
            project_root=self.project_root
        )

    def print_startup_info(self, config: AppConfig):
        """Prints the initial startup message to the console."""
        print(f"Starting Textile Color Analysis System...")
        try:
             # Try to show relative paths for a cleaner log
             display_config = config.config_path.relative_to(self.project_root)
             display_output = config.output_dir.relative_to(self.project_root)
        except ValueError:
             # Fallback if paths are on different drives, etc.
             display_config = config.config_path
             display_output = config.output_dir
        
        print(f"Using Config : {display_config}")
        print(f"Output Dir   : {display_output}")
        print(f"Log Level    : {config.log_level}")
        print(f"Profiling    : {'Enabled' if config.profile else 'Disabled'}")
        print("-" * 60)