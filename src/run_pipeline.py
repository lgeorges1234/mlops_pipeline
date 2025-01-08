import logging
import subprocess
from pathlib import Path
import sys
from typing import List, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_script(script_path: Path) -> Tuple[bool, str]:
    """
    Execute a Python script and wait for completion.
    
    Args:
        script_path: Path to the script to execute
        
    Returns:
        Tuple of (success boolean, error message if any)
    """
    logger.debug(f"Executing script: {script_path}")
    try:
        # Run script and capture output
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=True,
            text=True
        )
        logger.debug(f"Script completed successfully: {script_path}")
        return True, ""
        
    except subprocess.CalledProcessError as e:
        error_msg = f"Script failed with exit code {e.returncode}\nOutput: {e.output}\nError: {e.stderr}"
        logger.error(error_msg)
        return False, error_msg
        
    except Exception as e:
        error_msg = f"Unexpected error running script: {e}"
        logger.error(error_msg)
        return False, error_msg

def main():
    """Run the complete data processing and modeling pipeline"""
    logger.info("Starting pipeline execution")
    
    # Define script execution order
    scripts = [
        Path("src/data/import_raw_data.py"),
        Path("src/data/make_dataset.py"),
        Path("src/models/train_model.py"),
        # Path("src/models/predict_model.py"),
        Path("src/models/evaluate_model.py")
    ]
    
    # Validate all scripts exist
    for script in scripts:
        if not script.is_file():
            logger.error(f"Script not found: {script}")
            sys.exit(1)
    
    # Execute scripts in sequence
    for script in scripts:
        logger.info(f"Starting execution of: {script.name}")
        
        success, error_msg = run_script(script)
        
        if not success:
            logger.error(f"Pipeline failed at {script.name}")
            logger.error(f"Error details: {error_msg}")
            sys.exit(1)
            
        logger.info(f"Successfully completed: {script.name}")
    
    logger.info("Pipeline completed successfully")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("Pipeline failed with unexpected error", exc_info=True)
        sys.exit(1)