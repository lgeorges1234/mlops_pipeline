import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)

def prepare_features(df: pd.DataFrame, 
                    target_column: str,
                    columns_to_drop: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target from dataframe
    
    Args:
        df: Input dataframe
        target_column: Name of target column
        columns_to_drop: List of columns to drop
    
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    try:
        logger.debug("Starting feature preparation")
        logger.debug(f"Input dataframe shape: {df.shape}")
        
        # Create a copy to avoid modifying original
        df_copy = df.copy()
        logger.debug("Created copy of input dataframe")
        
        # Drop specified columns
        if columns_to_drop:
            logger.debug(f"Dropping columns: {columns_to_drop}")
            df_copy = df_copy.drop(columns_to_drop, axis=1)
            logger.debug(f"Shape after dropping columns: {df_copy.shape}")
            
        # Separate features and target
        logger.debug(f"Extracting target column: {target_column}")
        target = df_copy[target_column]
        features = df_copy.drop(target_column, axis=1)
        
        logger.debug(f"Features shape: {features.shape}")
        logger.debug(f"Target shape: {target.shape}")
        logger.debug("Feature preparation completed successfully")
        
        return features, target
        
    except Exception as e:
        logger.error(f"Error preparing features: {e}")
        raise

def scale_features(X_train: pd.DataFrame, 
                  X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scale features using StandardScaler
    
    Args:
        X_train: Training features DataFrame
        X_test: Testing features DataFrame
    
    Returns:
        Tuple of (scaled train features, scaled test features, fitted scaler)
    """
    try:
        logger.debug("Starting feature scaling")
        logger.debug(f"Training data shape: {X_train.shape}")
        logger.debug(f"Testing data shape: {X_test.shape}")
        
        scaler = StandardScaler()
        logger.debug("Initialized StandardScaler")
        
        # Fit and transform training data
        logger.debug("Fitting and transforming training data")
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        logger.debug("Training data scaled successfully")
        
        # Transform test data
        logger.debug("Transforming test data")
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        logger.debug("Test data scaled successfully")
        
        # Verify scaling results
        logger.debug(f"Scaled training data shape: {X_train_scaled.shape}")
        logger.debug(f"Scaled testing data shape: {X_test_scaled.shape}")
        
        # Log some basic statistics to verify scaling worked correctly
        logger.debug("Verifying scaling statistics...")
        train_mean = X_train_scaled.mean().mean()
        train_std = X_train_scaled.std().mean()
        logger.debug(f"Training data - Mean: {train_mean:.3f}, Std: {train_std:.3f}")
        
        logger.debug("Feature scaling completed successfully")
        
        return X_train_scaled, X_test_scaled, scaler
        
    except Exception as e:
        logger.error(f"Error scaling features: {e}")
        raise