import logging
from pathlib import Path
import sys

# Setup project root directory and import settings
ROOT_DIR = Path(__file__).parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.config.model_settings import settings
from src.utils.model_utils import (
    load_model,
    load_test_data,
    calculate_regression_metrics,
    save_metrics
)
from src.utils.validation_utils import validate_settings

logger = logging.getLogger(__name__)

def main():
    """Evaluate trained model performance"""
    try:
        logger.info("Starting model evaluation pipeline")
        
        # Validate settings
        required_settings = [
            'processed_data_path',
            'model_path',
            'metrics_path',
            'model_filename',
            'metrics_filename'
        ]
        
        validated = validate_settings(required_settings)
        logger.info("Settings validation completed successfully")
        
        # Define paths
        data_dir = Path(validated['processed_data_path'])
        model_file = Path(validated['model_path']) / validated['model_filename']
        metrics_file = Path(validated['metrics_path']) / validated['metrics_filename']
        
        # Load test data
        X_test, y_test = load_test_data(data_dir)
        logger.info("Test data loaded successfully")
        
        # Load model
        model = load_model(model_file)
        logger.info("Model loaded successfully")
        
        # Generate predictions
        logger.debug("Generating predictions")
        predictions = model.predict(X_test)
        logger.info("Predictions generated successfully")
        
        # Calculate metrics
        metrics = calculate_regression_metrics(y_test, predictions)
        logger.info(f"Model evaluation completed - Metrics: {metrics}")
        
        # Save metrics
        save_metrics(metrics, metrics_file)
        logger.info(f"Metrics saved to: {metrics_file}")
        
        logger.info("Model evaluation pipeline completed successfully")
        
    except Exception as e:
        logger.error("Model evaluation pipeline failed", exc_info=True)
        raise

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        main()
    except Exception:
        logger.error("Program failed", exc_info=True)
        sys.exit(1)