# Voice Command Classifier 🎙️

A deep learning system that recognises voice commands (yes/no/stop/go) 
from audio clips using a fine-tuned Wav2Vec2 transformer model.

## Results

| Model | Test Accuracy |
|-------|--------------|
| Random Forest (baseline) | 61.8% |
| Neural Network (PyTorch) | 69.2% |
| **Wav2Vec2 (final model)** | **92.0%** |

## Demo
> Clone the repo and run `demo.py` to test the model on your own voice.

## Project Overview

This project explores audio classification using transfer learning. 
Starting from a hand-crafted MFCC feature pipeline, I progressively 
improved accuracy by 30% by fine-tuning a pretrained Wav2Vec2 transformer.

### Key Concepts Used
- **MFCC feature extraction** — converting raw audio into 93 meaningful features
- **Transfer learning** — fine-tuning only 197K of 94M parameters
- **Overfitting analysis** — identified and fixed 26% train/test accuracy gap
- **PyTorch** — custom Dataset, DataLoader, training loop from scratch

## Architecture
