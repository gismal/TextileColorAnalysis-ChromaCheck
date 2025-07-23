import pandas as pd
import os
import logging

def save_output(dataset_name, output_type, file_name, content, output_dir="output"):
    """Save output data to a file."""
    os.makedirs(os.path.join(output_dir, dataset_name, output_type), exist_ok=True)
    file_path = os.path.join(output_dir, dataset_name, output_type, file_name)
    try:
        if isinstance(content, dict):
            content_df = pd.DataFrame([content])
            content_df.to_csv(file_path, index=False)
        elif isinstance(content, pd.DataFrame):
            content.to_csv(file_path, index=False)
        else:
            with open(file_path, 'wb') as f:
                f.write(content)  # For image data
        logging.info(f"Saved {file_name} to {file_path}")
    except Exception as e:
        logging.error(f"Error saving {file_name}: {str(e)}")