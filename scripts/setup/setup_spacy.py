#!/usr/bin/env python3
"""Setup script for spaCy models - easy one-command installation."""

import sys
import subprocess
import importlib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_spacy_installed():
    """Check if spaCy is installed."""
    try:
        importlib.import_module('spacy')
        return True
    except ImportError:
        return False


def check_model_installed(model_name):
    """Check if a specific spaCy model is installed."""
    try:
        import spacy
        spacy.load(model_name)
        return True
    except OSError:
        return False


def install_spacy():
    """Install spaCy if not already installed."""
    if not check_spacy_installed():
        logger.info("Installing spaCy...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy"])
            logger.info("✅ spaCy installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install spaCy: {e}")
            return False
    else:
        logger.info("✅ spaCy is already installed")
        return True


def install_model(model_name):
    """Install a specific spaCy model."""
    if not check_model_installed(model_name):
        logger.info(f"Installing spaCy model '{model_name}'...")
        logger.info("This may take a few minutes as it downloads a large model...")
        try:
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
            logger.info(f"✅ Model '{model_name}' installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install model '{model_name}': {e}")
            return False
    else:
        logger.info(f"✅ Model '{model_name}' is already installed")
        return True


def setup_ai_news_models():
    """Setup all required models for AI News."""
    required_models = [
        "en_core_web_md",  # Main English model (medium)
    ]
    
    optional_models = [
        "en_core_web_lg",  # Large English model (optional, better accuracy)
        "en_core_web_trf",  # Transformer model (optional, best accuracy)
    ]
    
    logger.info("🐶 AI News - spaCy Model Setup")
    logger.info("=" * 40)
    
    # Install spaCy first
    if not install_spacy():
        logger.error("❌ Cannot proceed without spaCy")
        return False
    
    # Install required models
    success = True
    for model in required_models:
        if not install_model(model):
            success = False
    
    # Ask about optional models
    if success:
        logger.info("\n🎯 Optional models available for better accuracy:")
        for model in optional_models:
            logger.info(f"  - {model}")
        
        try:
            response = input("\nWould you like to install optional models? (y/N): ").strip().lower()
            if response in ['y', 'yes']:
                for model in optional_models:
                    install_model(model)
        except KeyboardInterrupt:
            logger.info("\nSkipping optional models...")
    
    # Final verification
    logger.info("\n🔍 Verifying installation...")
    all_good = True
    for model in required_models:
        if check_model_installed(model):
            logger.info(f"✅ {model} - Ready")
        else:
            logger.error(f"❌ {model} - Not available")
            all_good = False
    
    if all_good:
        logger.info("\n🎉 All required models are installed and ready!")
        logger.info("You can now run AI News with full NLP capabilities.")
        return True
    else:
        logger.error("\n❌ Some required models are missing. Please check the errors above.")
        return False


def main():
    """Main setup function."""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--check":
            logger.info("Checking spaCy installation...")
            if check_spacy_installed():
                logger.info("✅ spaCy is installed")
                if check_model_installed("en_core_web_md"):
                    logger.info("✅ en_core_web_md is installed")
                else:
                    logger.info("❌ en_core_web_md is not installed")
            else:
                logger.info("❌ spaCy is not installed")
            return
        elif sys.argv[1] == "--model":
            if len(sys.argv) > 2:
                install_model(sys.argv[2])
                return
            else:
                logger.error("Please specify a model name")
                return
    
    # Run full setup
    success = setup_ai_news_models()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()