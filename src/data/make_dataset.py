from pathlib import Path
import logging
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

# Setup project root directory and import settings
ROOT_DIR = Path(__file__).parent.parent.parent
import sys
if str(ROOT_DIR) not in sys.path:
   sys.path.append(str(ROOT_DIR))

from src.config.model_settings import settings
from src.utils.csv_utils import save_processed_data, validate_csv, import_csv, save_csv
from src.utils.data_processing import prepare_features, scale_features
from src.utils.validation_utils import validate_settings

logger = logging.getLogger(__name__)


def main():
   """Process raw data into training and test sets"""
   logger = logging.getLogger(__name__)
   
   try:
       logger.info("Starting data processing pipeline")
       
       # Settings Validation
       required_settings = [
           'raw_data_path',
           'raw_data_filename',
           'processed_data_path', 
           'target_column',
           'test_size',
           'random_state',
           'columns_to_drop'
       ]
       
       validated = validate_settings(required_settings)
       logger.info("Settings validation completed successfully")

       # Define input raw data file path
       input_file = Path(validated['raw_data_path']) / validated['raw_data_filename']
       logger.debug(f"Input file path: {input_file}")

       # Define output datasets directory and check if exists
       output_dir = Path(validated['processed_data_path'])
       output_dir.mkdir(parents=True, exist_ok=True)
       logger.debug(f"Output directory created/verified: {output_dir}")

       target_column = validated['target_column']
       test_size = validated['test_size']
       random_state = validated['random_state']
       columns_to_drop = validated['columns_to_drop']
       logger.debug(f"Parameters extracted - target: {target_column}, test_size: {test_size}, random_state: {random_state}")

       # Validate and import raw data
       validate_csv(input_file)
       df = import_csv(input_file)
       logger.info(f"Data validation and import completed successfully. Data shape: {df.shape}")
       
       # Data processing
       features, target = prepare_features(
           df, 
           target_column,
           columns_to_drop
       )
       logger.info(f"Feature preparation completed successfully. Features shape: {features.shape}, Target shape: {target.shape}")
       
       # Data splitting
       X_train, X_test, y_train, y_test = train_test_split(
           features, 
           target,
           test_size=test_size,
           random_state=random_state
       )
       logger.info(f"Train-test split completed successfully. Training set: {X_train.shape}, Test set: {X_test.shape}")
       
       # Features Scaling
       X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)
       logger.info("Feature scaling completed successfully")
           
       # Processed datasets saving
       datasets = {
           'X_train': X_train,
           'X_test': X_test,
           'X_train_scaled': X_train_scaled,
           'X_test_scaled': X_test_scaled,
           'y_train': y_train,
           'y_test': y_test
       }
       save_processed_data(datasets, output_dir)
       logger.info("Processed datasets saved successfully")
       
       logger.info("Data processing pipeline completed successfully")
       
   except Exception as e:
       logger.error("Data processing pipeline failed", exc_info=True)
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