import yaml
import logging

def load_config(config_path='config.yaml'):
    try:
        logging.info(f"Loading configuration from {config_path}...")
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logging.info("Configuration loaded successfully.")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file {config_path} not found.")
        return None
    except yaml.YAMLError as exc:
        logging.error(f"Error parsing YAML file: {exc}")
        return None
