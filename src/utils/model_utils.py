import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json

logger = logging.getLogger(__name__)

def train_random_forest(
    X_train: pd.DataFrame, 
    y_train: np.ndarray,
    param_grid: Dict[str, Any],
    cv: int = 5,
    random_state: int = 42,
    n_jobs: int = -1
) -> Tuple[RandomForestRegressor, Dict[str, Any]]:
    """
    Train Random Forest model using GridSearchCV for hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training target
        param_grid: Dictionary of parameters for GridSearchCV
        cv: Number of cross-validation folds
        random_state: Random state for reproducibility
        n_jobs: Number of jobs for parallel processing
    
    Returns:
        Tuple containing best model and best parameters
    """
    logger.debug("Initializing RandomForestRegressor and GridSearchCV")
    rf = RandomForestRegressor(random_state=random_state)
    
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=cv,
        scoring='neg_root_mean_squared_error',
        n_jobs=n_jobs,
        verbose=1
    )
    
    logger.debug("Starting GridSearchCV fitting")
    grid_search.fit(X_train, y_train)
    
    logger.debug(f"Best parameters found: {grid_search.best_params_}")
    return grid_search.best_estimator_, grid_search.best_params_

def evaluate_model(
    model: RandomForestRegressor,
    X_test: pd.DataFrame,
    y_test: np.ndarray
) -> Tuple[float, float, np.ndarray]:
    """
    Evaluate model performance on test set.
    
    Args:
        model: Trained RandomForestRegressor
        X_test: Test features
        y_test: Test target
    
    Returns:
        Tuple containing RMSE, R2 score, and predictions
    """
    logger.debug("Generating predictions on test set")
    y_pred = model.predict(X_test)
    
    logger.debug("Calculating performance metrics")
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    logger.debug(f"Model performance - RMSE: {rmse:.4f}, R2: {r2:.4f}")
    return rmse, r2, y_pred


def predict(model: Any, features: Dict[str, float]) -> float:
    """
    Make prediction using loaded model.
    
    Args:
        model: Loaded model object
        features: Dictionary of feature names and values
        
    Returns:
        Model prediction
        
    Raises:
        ValueError: If features are invalid
    """
    logger.debug("Converting features to DataFrame")
    try:
        input_df = pd.DataFrame([features])
        logger.debug(f"Input features: {input_df}")
        
        prediction = model.predict(input_df)
        logger.debug(f"Generated prediction: {prediction[0]}")
        
        return prediction[0]
        
    except ValueError as e:
        logger.error(f"Invalid feature values: {e}")
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise

def load_feature_names(data_path: Path) -> list:
    """
    Load feature names from training data.
    
    Args:
        data_path: Path to training data file
        
    Returns:
        List of feature names
    """
    logger.debug(f"Loading feature names from: {data_path}")
    try:
        X_train = pd.read_csv(data_path)
        feature_names = X_train.columns.tolist()
        logger.debug(f"Loaded {len(feature_names)} feature names")
        return feature_names
        
    except Exception as e:
        logger.error(f"Failed to load feature names: {e}")
        raise

def get_feature_values_manually(feature_names: list) -> Dict[str, float]:
    """
    Collect feature values through user input.
    
    Args:
        feature_names: List of feature names to collect
        
    Returns:
        Dictionary of feature names and values
    """
    logger.debug("Collecting feature values manually")
    features = {}
    try:
        for feature_name in feature_names:
            while True:
                try:
                    value = input(f"Enter value for {feature_name}: ")
                    features[feature_name] = float(value)
                    break
                except ValueError:
                    print(f"Invalid input. Please enter a numeric value for {feature_name}")
        
        logger.debug(f"Collected feature values: {features}")
        return features
        
    except Exception as e:
        logger.error(f"Error collecting feature values: {e}")
        raise


def load_test_data(data_dir: Path) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load test data from processed data directory.
    
    Args:
        data_dir: Path to directory containing test data files
        
    Returns:
        Tuple of (X_test, y_test)
        
    Raises:
        FileNotFoundError: If data files don't exist
    """
    logger.debug(f"Loading test data from: {data_dir}")
    try:
        X_test = pd.read_csv(data_dir / 'X_test.csv')
        y_test = pd.read_csv(data_dir / 'y_test.csv')
        
        logger.debug(f"Test data loaded - X_test shape: {X_test.shape}")
        return X_test, np.ravel(y_test)
        
    except FileNotFoundError as e:
        logger.error(f"Test data files not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        raise

def calculate_regression_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate regression performance metrics.
    
    Args:
        y_true: Actual target values
        y_pred: Predicted target values
        
    Returns:
        Dictionary of metrics
    """
    logger.debug("Calculating regression metrics")
    try:
        mse = mean_squared_error(y_true, y_pred)
        metrics = {
            'mse': float(mse),
            'rmse': float(np.sqrt(mse)),
            'r2': float(r2_score(y_true, y_pred)),
            'mae': float(mean_absolute_error(y_true, y_pred))
        }
        logger.debug(f"Calculated metrics: {metrics}")
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        raise

def save_metrics(metrics: Dict[str, float], output_file: Path) -> None:
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Dictionary of metric names and values
        output_file: Path where metrics will be saved
    """
    logger.debug(f"Saving metrics to: {output_file}")
    try:
        # Ensure directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=4)
            
        logger.debug("Metrics saved successfully")
        
    except Exception as e:
        logger.error(f"Failed to save metrics: {e}")
        raise

def save_model(model: RandomForestRegressor, model_path: Path) -> None:
    """
    Save trained model to disk.
    
    Args:
        model: Trained model to save
        model_path: Path where model will be saved
    """
    logger.debug(f"Saving model to: {model_path}")
    try:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.debug("Model saved successfully")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

def load_model(model_path: Path) -> Any:
    """
    Load trained model from disk.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Loaded model object
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        pickle.UnpicklingError: If model file is corrupted
    """
    logger.debug(f"Loading model from: {model_path}")
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.debug("Model loaded successfully")
        return model
        
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise
    except pickle.UnpicklingError as e:
        logger.error(f"Error loading model - file may be corrupted: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading model: {e}")
        raise