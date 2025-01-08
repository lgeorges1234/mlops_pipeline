import json
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
    predict,
    load_feature_names,
    get_feature_values_manually
)
from src.utils.validation_utils import validate_settings

logger = logging.getLogger(__name__)

def main():
    """Run model prediction pipeline"""
    try:
        logger.info("Starting prediction pipeline")
        
        # Validate settings
        required_settings = [
            'model_path',
            'processed_data_path',
            'model_filename'
        ]
        
        validated = validate_settings(required_settings)
        logger.info("Settings validation completed successfully")
        
        # Load model
        model_path = Path(validated['model_path']) / validated['model_filename']
        model = load_model(model_path)
        logger.info("Model loaded successfully")
        
        # Get features either from JSON file or manual input
        if len(sys.argv) == 2:
            json_file = Path(sys.argv[1])
            logger.info(f"Loading features from JSON file: {json_file}")
            
            try:
                with open(json_file, 'r') as file:
                    features = json.load(file)
            except Exception as e:
                logger.error(f"Failed to load JSON file: {e}")
                raise
                
        else:
            logger.info("No JSON file provided, collecting features manually")
            data_path = Path(validated['processed_data_path']) / 'X_train.csv'
            feature_names = load_feature_names(data_path)
            features = get_feature_values_manually(feature_names)
        
        # Generate prediction
        prediction = predict(model, features)
        logger.info(f"Prediction generated successfully: {prediction}")
        print(f"Prediction: {prediction}")
        
        logger.info("Prediction pipeline completed successfully")
        
    except Exception as e:
        logger.error("Prediction pipeline failed", exc_info=True)
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