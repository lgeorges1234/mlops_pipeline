import hashlib
from pathlib import Path

def verify_file_hash(file_path: Path, expected_hash: str) -> bool:
    """Verify file integrity using SHA-256"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest() == expected_hash