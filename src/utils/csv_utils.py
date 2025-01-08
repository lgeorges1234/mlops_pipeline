from pathlib import Path
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def import_csv(file_path: Path, **kwargs):
    """
    Import data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        **kwargs: Additional arguments to pass to pd.read_csv
    
    Returns:
        pd.DataFrame: Imported data
    """
    try:
        data = pd.read_csv(file_path, **kwargs)
        logger.debug(f"Successfully imported CSV with shape {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Failed to import CSV: {e}")
        raise

def save_csv(file, file_path: Path, **kwargs):
    """
    Save data to a CSV file.
    
    Args:
        file: DataFrame to save
        file_path: Path where to save the CSV
        **kwargs: Additional arguments to pass to to_csv
    """
    try:
        file.to_csv(file_path, **kwargs)
        logger.debug(f"Successfully saved CSV to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save CSV: {e}")
        raise

def save_processed_data(datasets: dict, output_dir):
    """
    Save all processed datasets.
    
    Args:
        datasets: Dictionary of dataframes to save
        output_dir: Directory where to save the processed data
    """
    logger.debug(f"Starting to save {len(datasets)} processed datasets to {output_dir}")
    try:
        for name, df in datasets.items():
            output_path = output_dir / f"{name}.csv"
            save_csv(df, output_path, index=False)
        logger.debug("Successfully saved all processed datasets")
            
    except Exception as e:
        logger.error(f"Error saving processed data: {e}")
        raise

def validate_csv(file_path: Path) -> bool:
    """
    Validate the downloaded CSV file.
    
    Args:
        file_path: Path to the CSV file
    Returns:
        bool: True if file is valid
    """
    logger.debug(f"Starting CSV validation for {file_path}")
    try:
        # Check file size
        if file_path.stat().st_size == 0:
            raise ValueError("File is empty")
        logger.debug("File size validation passed")
            
        # Check file extension
        if file_path.suffix.lower() != '.csv':
            raise ValueError("Invalid file extension")
        logger.debug("File extension validation passed")
            
        # Try reading first few lines to verify CSV format
        with open(file_path, 'r') as f:
            header = f.readline()
            if ',' not in header:
                raise ValueError("File does not appear to be CSV format")
        logger.debug("CSV format validation passed")
                
        logger.debug("All CSV validations passed successfully")
        return True
        
    except Exception as e:
        logger.error(f"File validation failed: {e}")
        raise