from pathlib import Path
import logging
import sys
import pandas as pd
import numpy as np

# Setup project root directory and import settings
ROOT_DIR = Path(__file__).parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.config.model_settings import settings
from src.utils.csv_utils import import_csv
from src.utils.model_utils import train_random_forest, evaluate_model, save_model
from src.utils.validation_utils import validate_settings

logger = logging.getLogger(__name__)

def main():
    """Train and evaluate Random Forest model"""
    try:
        logger.info("Starting model training pipeline")
        
        # Settings Validation
        required_settings = [
            'processed_data_path',
            'model_path',
            'random_state'
        ]
        
        validated = validate_settings(required_settings)
        logger.info("Settings validation completed successfully")
        
        # Load processed data
        data_path = Path(validated['processed_data_path'])
        logger.debug(f"Loading processed data from: {data_path}")
        
        X_train_scaled = pd.read_csv(data_path / 'X_train_scaled.csv')
        X_test_scaled = pd.read_csv(data_path / 'X_test_scaled.csv')
        y_train = pd.read_csv(data_path / 'y_train.csv').values.ravel()
        y_test = pd.read_csv(data_path / 'y_test.csv').values.ravel()
        logger.info("Data loading completed successfully")
        
        # Define hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        logger.debug(f"Hyperparameter grid defined: {param_grid}")
        
        # Train model
        best_model, best_params = train_random_forest(
            X_train_scaled,
            y_train,
            param_grid,
            random_state=validated['random_state']
        )
        logger.info(f"Model training completed successfully with parameters: {best_params}")
        
        # Evaluate model
        rmse, r2, _ = evaluate_model(best_model, X_test_scaled, y_test)
        logger.info(f"Model evaluation completed - RMSE: {rmse:.4f}, R2: {r2:.4f}")
        
        # Save model
        model_path = Path(validated['model_path']) / 'trained_model.pkl'
        save_model(best_model, model_path)
        logger.info("Model saved successfully")
        
        logger.info("Model training pipeline completed successfully")
        
    except Exception as e:
        logger.error("Model training pipeline failed", exc_info=True)
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
        logging.error("Program failed", exc_info=True)
        sys.exit(1)