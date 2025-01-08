from pathlib import Path
import sys

# Define project root directory
ROOT_DIR = Path(__file__).parent.parent.parent

# Add project root to PYTHONPATH if needed
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# Export ROOT_DIR
__all__ = ['ROOT_DIR']