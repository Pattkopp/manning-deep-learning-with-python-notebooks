# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Repository Overview

This repository contains companion Jupyter notebooks for "Deep Learning with Python" by François Chollet and Matthew Watson. It includes notebooks for three editions:

- **Third Edition (2025)**: Root-level notebooks using Keras 3 with JAX/TensorFlow/PyTorch backends
- **Second Edition (2021)**: Located in `second_edition/` using tf.keras with TensorFlow 2.16
- **First Edition (2017)**: Located in `first_edition/` using older TensorFlow/Keras APIs

## Development Environment Setup

### Python Environment
The project uses Python 3.13+ with `uv` for dependency management:

```bash
# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### Key Dependencies
- `jupyter>=1.1.1` - Notebook environment
- `keras>=3.11.3` - Deep learning framework (3rd edition)
- `keras-hub>=0.19.0` - Pre-trained models and utilities
- `tensorflow>=2.20.0` - Backend for Keras
- `matplotlib>=3.10.6` - Plotting and visualization

## Running Notebooks

### Local Development
```bash
# Start Jupyter Lab
jupyter lab

# Or start classic Jupyter Notebook
jupyter notebook

# Run specific notebook
jupyter nbconvert --execute --to notebook chapter08_image-classification.ipynb
```

### Google Colab (Recommended)
All notebooks are designed to run on Google Colab with free GPU access:
- Each notebook contains install commands: `!pip install keras keras-hub --upgrade -q`
- GPU runtime recommended for chapters 8-18
- Change runtime: **Runtime → Change runtime type** in Colab

### Backend Configuration
Third edition notebooks support multiple backends. Set the backend in the second cell:

```python
import os
os.environ["KERAS_BACKEND"] = "jax"  # or "tensorflow" or "pytorch"
```

**Important**: Backend must be set before importing Keras and requires session restart if changed mid-notebook.

## Repository Structure

### Notebook Organization
- `chapter##_*.ipynb` - Third edition notebooks (root level)
- `second_edition/chapter##_*.ipynb` - Second edition notebooks
- `first_edition/##.*-*.ipynb` - First edition notebooks

### Chapter Topics
- **Chapters 2-5**: Fundamentals (mathematical building blocks, ML basics)
- **Chapters 7-8**: Keras deep dive and image classification
- **Chapters 9-12**: Computer vision (ConvNets, segmentation, object detection)
- **Chapter 13**: Time series forecasting
- **Chapters 14-16**: NLP (text classification, transformers, text generation)
- **Chapter 17**: Image generation
- **Chapter 18**: Production best practices

## Kaggle Integration

Many advanced chapters require Kaggle datasets:

### Setup Kaggle Access
1. Create account at https://www.kaggle.com/
2. Generate API key at https://www.kaggle.com/settings
3. Set up authentication:
   - **Colab**: Add `KAGGLE_USERNAME` and `KAGGLE_KEY` as secrets
   - **Local**: Place `kaggle.json` in `~/.kaggle/`

### Usage in Notebooks
```python
import kagglehub
kagglehub.login()  # Interactive login
# Or automatic with pre-configured credentials
```

## Notebook Structure Patterns

### Standard Notebook Setup
All third edition notebooks follow this pattern:
```python
# 1. Install dependencies
!pip install keras keras-hub --upgrade -q

# 2. Set backend
import os
os.environ["KERAS_BACKEND"] = "jax"

# 3. Backend magic for conditional execution
@register_cell_magic
def backend(line, cell):
    # Allows backend-specific code blocks
```

### Common Import Patterns
```python
import keras
from keras import layers
from keras.datasets import mnist, cifar10, imdb
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
```

## Backend-Specific Considerations

### JAX Backend
- Fastest for training large models
- Excellent for research and experimentation
- Best GPU/TPU utilization

### TensorFlow Backend
- Most stable and well-tested
- Best ecosystem support
- Recommended for production

### PyTorch Backend
- Good for users familiar with PyTorch
- Growing ecosystem support
- Flexible debugging

## Development Workflow

### Testing Notebook Changes
```bash
# Check notebook syntax
jupyter nbconvert --execute --to notebook --inplace notebook.ipynb

# Clear outputs for clean commits
jupyter nbconvert --clear-output --inplace *.ipynb
```

### Working with GPU-Intensive Chapters
- Use Colab Pro for faster GPUs on chapters 8-18
- Monitor GPU memory usage in training loops
- Consider reducing batch sizes for local development

## Environment Variables

The repository uses various API keys for external services (stored in `.env`):
- `ANTHROPIC_API_KEY` - For Claude AI integration
- `OPENAI_API_KEY` - For GPT model access  
- `HF_TOKEN` - Hugging Face Hub access
- `LANGSMITH_*` - Language model monitoring
- Various other service tokens

**Note**: Never commit API keys to version control.

## Architecture Notes

### Notebook Design Philosophy
- Code-only format (no explanatory text)
- Designed to accompany the book chapters
- Focus on runnable examples rather than tutorials
- Consistent structure across all editions

### Version Compatibility
- **Third Edition**: Modern Keras 3 with multiple backend support
- **Second Edition**: tf.keras specific implementation
- **First Edition**: Legacy TensorFlow 1.x/early 2.x patterns

Choose the appropriate edition based on your learning goals and framework preferences.