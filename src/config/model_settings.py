from typing import List
from pydantic_settings import BaseSettings
from src.config.setup import ROOT_DIR

class Settings(BaseSettings):
    # Data paths
    bucket_folder_url: str = 'https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv'
    raw_data_path: str = str(ROOT_DIR / 'data/raw_data')
    processed_data_path: str = str(ROOT_DIR / 'data/processed_data')
    model_path: str = str(ROOT_DIR / 'models')
    metrics_path: str = str(ROOT_DIR / 'metrics')
    
    # File names
    raw_data_filename: str = 'raw.csv'
    model_filename: str = 'trained_model.pkl'
    metrics_filename: str = 'scores.json'


    # Model parameters
    target_column: str = 'silica_concentrate'
    columns_to_drop: List = ['date']
    test_size: float = 0.3
    random_state: int = 42

    class Config:
        env_file = '.env'

settings = Settings()
