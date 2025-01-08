from typing import List
from pathlib import Path
import sys
import logging

# Setup project root directory and import settings
ROOT_DIR = Path(__file__).parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
from src.config.model_settings import settings

logger = logging.getLogger(__name__)

def validate_settings(required_settings: List[str]) -> dict:
    """
    Validate that required settings exist and are not empty.
    
    Args:
        required_settings: List of setting names that must exist
    
    Returns:
        dict: Dictionary containing the validated settings values
    
    Raises:
        ValueError: If any required setting is missing or empty
    """
    logger.debug("Starting settings validation")
    logger.debug(f"Required settings to validate: {required_settings}")
    
    validated = {}
    logger.debug("Initialized empty validation dictionary")

    for setting in required_settings:
        logger.debug(f"Validating setting: {setting}")
        
        # Skip internal attributes and methods
        if setting.startswith('_') or callable(getattr(settings, setting, None)):
            logger.debug(f"Skipping internal/callable setting: {setting}")
            continue
        
        # Log the current setting value
        current_value = getattr(settings, setting, None)
        logger.debug(f"Current value for {setting}: {current_value}")

        # Check if setting exists
        if not hasattr(settings, setting):
            error_msg = f"Missing required setting: {setting}"
            logger.error(error_msg)
            logger.debug(f"Available settings: {dir(settings)}")
            raise ValueError(error_msg)
        
        # Get setting value and check if empty
        value = getattr(settings, setting)
        logger.debug(f"Retrieved value for {setting}: {value} (type: {type(value)})")
        
        if value is None or value == "":
            error_msg = f"Required setting is empty: {setting}"
            logger.error(error_msg)
            logger.debug(f"Value type: {type(value)}, Value: {value}")
            raise ValueError(error_msg)
        
        # Store the validated value
        validated[setting] = value
        logger.debug(f"Successfully validated and stored {setting}")

    logger.debug(f"Final validated settings: {validated}")
    logger.debug(f"Successfully validated {len(validated)} settings")
    
    return validated