"""SpaCy model utilities for AI News system."""

import sys
import subprocess
from typing import Optional, Any
import logging
import importlib.util

logger = logging.getLogger(__name__)

def is_spacy_available() -> bool:
    """Check if spaCy is installed."""
    try:
        importlib.util.find_spec('spacy')
        return True
    except ImportError:
        return False

def is_model_available(model_name: str = "en_core_web_md") -> bool:
    """Check if specific spaCy model is available."""
    if not is_spacy_available():
        return False
    try:
        import spacy
        try:
            spacy.load(model_name)
            return True
        except OSError:
            return False
    except ImportError:
        return False

def download_spacy_model(model_name: str = "en_core_web_md", user_prompt: bool = True) -> bool:
    """Download spaCy model with optional user prompt."""
    if not is_spacy_available():
        print("❌ spaCy is not installed. Please install it first:")
        print("   uv add spacy")
        return False
    
    if is_model_available(model_name):
        print(f"✅ SpaCy model '{model_name}' is already available")
        return True
    
    if user_prompt:
        print(f"🤖 SpaCy model '{model_name}' is required for advanced NLP features.")
        response = input("Would you like to download it now? (y/n): ").strip().lower()
        if response not in ['y', 'yes']:
            print("❌ Model download cancelled. Advanced NLP features will be disabled.")
            return False
    
    print(f"📥 Downloading spaCy model '{model_name}'...")
    print("   This may take a few minutes depending on your internet connection.")
    
    try:
        # Use uv pip to install the model (most reliable for this project)
        model_version = "3.8.0"  # Latest compatible version
        model_url = f"https://github.com/explosion/spacy-models/releases/download/{model_name}-{model_version}/{model_name}-{model_version}.tar.gz"
        
        result = subprocess.run([
            "uv", "pip", "install", model_url
        ], capture_output=True, text=True, check=True)
        
        if result.returncode == 0:
            print(f"✅ Successfully downloaded spaCy model '{model_name}'")
            return True
        else:
            print(f"❌ Failed to download model: {result.stderr}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading spaCy model: {e}")
        print(f"   Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print("❌ spaCy command not found. Please ensure spaCy is properly installed.")
        return False

def load_spacy_model(model_name: str = "en_core_web_md", auto_download: bool = True):
    """Load spaCy model with graceful fallback."""
    if not is_spacy_available():
        print("⚠️  spaCy is not available. Using basic text processing only.")
        print("   To enable advanced NLP features: uv add spacy && uv run ai-news setup-spacy")
        return None
    
    if not is_model_available(model_name):
        if auto_download:
            print(f"🤖 SpaCy model '{model_name}' not found. Attempting to download...")
            if download_spacy_model(model_name, user_prompt=False):
                try:
                    import spacy
                    return spacy.load(model_name)
                except Exception as e:
                    print(f"❌ Failed to load model after download: {e}")
                    return None
            else:
                print("❌ Model download failed. Using basic processing only.")
                return None
        else:
            print(f"⚠️  SpaCy model '{model_name}' not available. Using basic processing only.")
            print("   Run 'uv run ai-news setup-spacy' to enable advanced NLP features.")
            return None
    
    try:
        import spacy
        return spacy.load(model_name)
    except Exception as e:
        print(f"❌ Error loading spaCy model: {e}")
        return None

def setup_spacy_interactive():
    """Interactive setup for spaCy models."""
    print("🤖 SpaCy Model Setup")
    print("=" * 50)
    
    if not is_spacy_available():
        print("❌ spaCy is not installed.")
        print("   Install it with: uv add spacy")
        print("   Then run this command again.")
        return False
    
    print(f"✅ spaCy is installed")
    
    model_name = "en_core_web_md"
    if is_model_available(model_name):
        print(f"✅ Model '{model_name}' is already available")
        print("   You're all set for advanced NLP features!")
        return True
    
    print(f"📥 Model '{model_name}' is available for download")
    print(f"   Size: ~40 MB")
    print(f"   Features: Named entity recognition, part-of-speech tagging, dependency parsing")
    print()
    
    response = input("Download spaCy model now? (y/n): ").strip().lower()
    if response in ['y', 'yes']:
        success = download_spacy_model(model_name, user_prompt=False)
        if success:
            print("🎉 Setup complete! You can now use advanced NLP features.")
            return True
        else:
            print("❌ Setup failed. You can try again later with: uv run ai-news setup-spacy")
            return False
    else:
        print("❌ Setup cancelled.")
        return False

def get_spacy_status():
    """Get current spaCy and model status."""
    status = {
        "spacy_available": is_spacy_available(),
        "model_available": is_model_available(),
        "model_name": "en_core_web_md"
    }
    return status