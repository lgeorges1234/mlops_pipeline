import sys
import logging
from pathlib import Path

# Setup project root directory and import settings
ROOT_DIR = Path(__file__).parent.parent.parent
if str(ROOT_DIR) not in sys.path:
   sys.path.append(str(ROOT_DIR))
# from src.config.model_settings import settings

from src.utils.file_operations import download_file, secure_file_permissions, write_file
from src.utils.csv_utils import validate_csv
from src.utils.validation_utils import validate_settings
from src.utils.security_utils import verify_file_hash


def main():
   """
   Main function to download data from AWS S3 and save to local raw data directory.
   
   Raises:
       Exception: If either download or save operations fail
   """
   logger = logging.getLogger(__name__)
   
   try:
       logger.info("Starting raw dataset creation pipeline")
       logger.debug("Initializing pipeline steps")

       # Verify settings variables exists
       required_settings = ['bucket_folder_url', 'raw_data_path', 'raw_data_filename']
       logger.debug(f"Required settings to validate: {required_settings}")

       validated = validate_settings(required_settings)
       logger.info("Settings validation completed successfully")
       
       # Extract validated settings
       url_path = validated['bucket_folder_url']
       raw_datafile = Path(validated['raw_data_path']) / validated['raw_data_filename']
       logger.debug(f"URL path: {url_path}")
       logger.debug(f"Raw data file path: {raw_datafile}")

       # Download the file from the url
       response = download_file(url_path)
       logger.info("File download completed successfully")
       logger.debug(f"Download response received with type: {type(response)}")

       # Write the CSV file into the raw_data folder
       write_file(response, raw_datafile)
       logger.info("File written successfully")
       logger.debug(f"File size: {raw_datafile.stat().st_size} bytes")

       # Validate CSV format of the raw data file
       validate_csv(raw_datafile)
       logger.info("CSV validation completed successfully")
       
       # Ensure file and folder right permissions
       secure_file_permissions(raw_datafile)
       logger.info("File permissions secured successfully")

       logger.info("Raw dataset creation pipeline completed successfully")
       
   except Exception as e:
       logger.error("Raw dataset creation pipeline failed", exc_info=True)
       raise

if __name__ == '__main__':
    # Configure logging
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    try:
        logger.debug("Starting main program execution")
        main()
        logger.debug("Main program execution completed")
    except Exception:
        logging.error("Program failed", exc_info=True)
        sys.exit(1)
