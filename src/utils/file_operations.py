import os
import requests
from requests import Response
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple, Union, BinaryIO

logger = logging.getLogger(__name__)

def secure_file_permissions(file_path: Path) -> None:
    """
    Set secure permissions on created files.
    
    Args:
        file_path: Path to the file to secure
        
    Raises:
        OSError: If setting permissions fails
        Exception: For any other unexpected errors
    """
    logger.debug(f"Setting secure permissions for: {file_path}")
    try:
        # Set file permissions to 644 (rw-r--r--)
        os.chmod(file_path, 0o644)
        # Set directory permissions to 755 (rwxr-xr-x)
        os.chmod(file_path.parent, 0o755)
        logger.debug("File permissions set successfully")
        
    except OSError as e:
        logger.error(f"Failed to set file permissions: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error setting file permissions: {e}")
        raise

def cleanup_on_failure(file_path: Path) -> None:
    """
    Remove incomplete/invalid files on failure.
    
    Args:
        file_path: Path to the file to clean up
    """
    logger.debug(f"Attempting to clean up file: {file_path}")
    try:
        if file_path.exists():
            file_path.unlink()
            logger.debug(f"Cleaned up incomplete file: {file_path}")
        else:
            logger.debug(f"No file to clean up at: {file_path}")
            
    except OSError as e:
        logger.error(f"Failed to clean up file {file_path}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during cleanup: {e}")

def write_file(
    file: Union[Response, BinaryIO], 
    file_path: Union[str, Path]
) -> bool:
    """
    Save file to local storage with secure permissions.
    
    Args:
        file: Response object or file-like object containing the data
        file_path: Path where the file should be saved
    
    Returns:
        bool: True if successful
    
    Raises:
        IOError: If file writing fails
        OSError: If setting permissions fails
        Exception: For any other unexpected errors
    """
    logger = logging.getLogger(__name__)
    file_path = Path(file_path)  # Convert to Path object if string
    logger.debug(f"Creating raw dataset at: {file_path}")
    
    try:
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file in local storage
        with open(file_path, 'wb') as f:
            for chunk in file.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive chunks
                    f.write(chunk)
        
        # Set secure permissions after successful write
        secure_file_permissions(file_path)
        
        logger.debug("Raw dataset created successfully with secure permissions")
        return True

    except (IOError, OSError) as e:
        logger.error(f"Error while writing file {file_path}: {e}")
        cleanup_on_failure(file_path)
        raise
    except Exception as e:
        logger.error(f"Unexpected error while creating dataset: {e}")
        cleanup_on_failure(file_path)
        raise

def download_file(
    url: str, 
    headers: Optional[Dict[str, str]] = None, 
    timeout: Union[float, Tuple[float, float]] = (5, 30)
) -> Response:
    """
    Download file from the provided URL.
    
    Args:
        url: URL to download from
        headers: Optional dictionary of HTTP headers
        timeout: Request timeout in seconds (connect timeout, read timeout)
    
    Returns:
        requests.Response: Response object containing data
    
    Raises:
        requests.HTTPError: If the download fails due to HTTP error
        Exception: For any other unexpected errors
    """
    logger.debug(f"Downloading raw file from: {url}")
    try:
        response = requests.get(
            url,
            stream=True,
            verify=True,
            timeout=timeout,
            headers=headers or {}
        )
        response.raise_for_status()
        logger.debug("Successfully downloaded raw file")
        return response
    
    except requests.HTTPError as e:
        logger.error(f"HTTP error occurred during download: {e}")
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error during download: {e}")
        raise