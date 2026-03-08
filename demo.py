"""
Voice Command Classifier — Demo
================================
Run this to test the model on a sample audio file.

Usage:
    python demo.py --file path/to/audio.wav
"""

import torch
import librosa
import numpy as np
import argparse
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

# Configuration
WORDS       = ["yes", "no", "stop", "go"]
MODEL_PATH  = "wav2vec2_best.pt"
SAMPLE_RATE = 16000

def load_audio(file_path):
    """Load and preprocess audio file."""
    print(f"Loading audio: {file_path}")
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=1.0)

    # Pad or trim to exactly 1 second
    if len(audio) < SAMPLE_RATE:
        audio = np.pad(audio, (0, SAMPLE_RATE - len(audio)))
    else:
        audio = audio[:SAMPLE_RATE]

    print(f"Audio loaded — duration: {len(audio)/SAMPLE_RATE:.2f}s, sample rate: {SAMPLE_RATE}Hz")
    return audio.astype(np.float32)

def predict(audio, model, processor, device):
    """Run inference and return predicted word + confidence."""
    # Process audio
    inputs = processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding="max_length",
        max_length=SAMPLE_RATE,
        truncation=True
    )

    input_values = inputs.input_values.to(device)

    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(input_values=input_values)
        logits  = outputs.logits

    # Convert to probabilities
    probs      = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    pred_idx   = np.argmax(probs)
    pred_word  = WORDS[pred_idx]
    confidence = probs[pred_idx] * 100

    return pred_word, confidence, probs

def main():
    parser = argparse.ArgumentParser(description="Voice Command Classifier")
    parser.add_argument("--file", type=str, required=True, 
                        help="Path to .wav audio file")
    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print("\nLoading model...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model     = Wav2Vec2ForSequenceClassification.from_pretrained(
        "facebook/wav2vec2-base",
        num_labels=4,
        ignore_mismatched_sizes=True
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model     = model.to(device)
    print("Model loaded ✅")

    # Load and predict
    audio                     = load_audio(args.file)
    pred_word, confidence, probs = predict(audio, model, processor, device)

    # Display results
    print("\n" + "=" * 40)
    print("PREDICTION RESULTS")
    print("=" * 40)
    print(f"Predicted word : {pred_word.upper()}")
    print(f"Confidence     : {confidence:.1f}%")
    print("\nAll class probabilities:")
    for word, prob in zip(WORDS, probs):
        bar    = "█" * int(prob * 30)
        marker = " ← predicted" if word == pred_word else ""
        print(f"  {word:5s} {bar:30s} {prob*100:.1f}%{marker}")
    print("=" * 40)

if __name__ == "__main__":
    main()
