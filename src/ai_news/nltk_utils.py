"""NLTK utilities with persistent caching and lazy loading."""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Set
import nltk
from nltk.data import find

logger = logging.getLogger(__name__)

# NLTK data packages required for text processing
NLTK_PACKAGES = {
    'punkt': 'tokenizers/punkt',
    'stopwords': 'corpora/stopwords', 
    'wordnet': 'corpora/wordnet',
    'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger'
}

# Persistent cache file location
def get_nltk_cache_file() -> Path:
    """Get the path to the NLTK cache file."""
    home_dir = Path.home()
    cache_dir = home_dir / '.ai_news_cache'
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / 'nltk_status.json'

def load_nltk_cache() -> Dict[str, bool]:
    """Load NLTK download status from persistent cache."""
    cache_file = get_nltk_cache_file()
    
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load NLTK cache: {e}")
    
    return {}

def save_nltk_cache(cache: Dict[str, bool]) -> None:
    """Save NLTK download status to persistent cache."""
    cache_file = get_nltk_cache_file()
    
    try:
        with open(cache_file, 'w') as f:
            json.dump(cache, f, indent=2)
    except IOError as e:
        logger.warning(f"Failed to save NLTK cache: {e}")

def is_nltk_data_available(package_id: str, resource_path: str, cache: Optional[Dict[str, bool]] = None) -> bool:
    """Check if NLTK data is available without downloading.
    
    Args:
        package_id: NLTK package identifier (e.g., 'punkt')
        resource_path: NLTK resource path (e.g., 'tokenizers/punkt')
        cache: Optional cache dictionary to use
    
    Returns:
        True if data is available, False otherwise
    """
    # Use provided cache or load from disk
    if cache is None:
        cache = load_nltk_cache()
    
    # Check cache first
    if package_id in cache:
        return cache[package_id]
    
    # Check filesystem
    try:
        find(resource_path)
        cache[package_id] = True
        save_nltk_cache(cache)
        return True
    except LookupError:
        cache[package_id] = False
        save_nltk_cache(cache)
        return False

def download_nltk_package(package_id: str, resource_path: str, force: bool = False) -> bool:
    """Download NLTK package if needed.
    
    Args:
        package_id: NLTK package identifier
        resource_path: NLTK resource path
        force: Force download even if already available
    
    Returns:
        True if download successful or already available
    """
    if not force and is_nltk_data_available(package_id, resource_path):
        logger.debug(f"NLTK data '{package_id}' already available")
        return True
    
    logger.info(f"Downloading NLTK data: {package_id}")
    
    try:
        # Download with progress indication
        nltk.download(package_id, quiet=False)
        
        # Verify download worked
        find(resource_path)
        
        # Update cache
        cache = load_nltk_cache()
        cache[package_id] = True
        save_nltk_cache(cache)
        
        logger.info(f"Successfully downloaded NLTK data: {package_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download NLTK data '{package_id}': {e}")
        
        # Update cache to mark as failed
        cache = load_nltk_cache()
        cache[package_id] = False
        save_nltk_cache(cache)
        
        return False

def setup_nltk_data(force: bool = False, show_progress: bool = True) -> bool:
    """Setup all required NLTK data.
    
    Args:
        force: Force download even if already available
        show_progress: Whether to show progress messages
    
    Returns:
        True if all packages are available
    """
    if show_progress:
        print("Setting up NLTK data...")
    
    success = True
    cache = load_nltk_cache()
    
    for package_id, resource_path in NLTK_PACKAGES.items():
        if show_progress:
            print(f"  Checking {package_id}...")
        
        if not download_nltk_package(package_id, resource_path, force=force):
            success = False
            if show_progress:
                print(f"  ❌ Failed to setup {package_id}")
        else:
            if show_progress:
                print(f"  ✅ {package_id} ready")
    
    if show_progress:
        if success:
            print("\n✅ NLTK data setup complete!")
        else:
            print("\n❌ Some NLTK data setup failed. Check logs for details.")
    
    return success

def check_nltk_data() -> Dict[str, bool]:
    """Check availability of all required NLTK packages.
    
    Returns:
        Dictionary mapping package names to availability status
    """
    cache = load_nltk_cache()
    status = {}
    
    for package_id, resource_path in NLTK_PACKAGES.items():
        status[package_id] = is_nltk_data_available(package_id, resource_path, cache)
    
    return status

def get_missing_nltk_packages() -> Set[str]:
    """Get list of missing NLTK packages.
    
    Returns:
        Set of missing package identifiers
    """
    status = check_nltk_data()
    return {pkg for pkg, available in status.items() if not available}

def ensure_nltk_data_lazy(package_id: str, resource_path: str) -> bool:
    """Ensure NLTK data is available, downloading if necessary.
    
    This function is designed to be called lazily when NLTK features
    are actually needed, not at import time.
    
    Args:
        package_id: NLTK package identifier
        resource_path: NLTK resource path
    
    Returns:
        True if data is available
    """
    if is_nltk_data_available(package_id, resource_path):
        return True
    
    # Data is missing, try to download it
    logger.warning(f"NLTK data '{package_id}' is missing. Attempting to download...")
    
    if download_nltk_package(package_id, resource_path):
        logger.info(f"NLTK data '{package_id}' downloaded successfully")
        return True
    else:
        logger.error(f"Failed to download NLTK data '{package_id}'. Please run 'uv run ai-news setup-nltk'")
        return False

def clear_nltk_cache() -> None:
    """Clear the NLTK status cache."""
    cache_file = get_nltk_cache_file()
    if cache_file.exists():
        cache_file.unlink()
        logger.info("NLTK cache cleared")

def get_nltk_info() -> Dict[str, any]:
    """Get information about NLTK setup.
    
    Returns:
        Dictionary with NLTK configuration and status info
    """
    return {
        'nltk_version': nltk.__version__,
        'nltk_data_paths': nltk.data.path,
        'cache_file': str(get_nltk_cache_file()),
        'cache_exists': get_nltk_cache_file().exists(),
        'package_status': check_nltk_data(),
        'missing_packages': list(get_missing_nltk_packages())
    }