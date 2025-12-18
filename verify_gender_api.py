
import sys
import os
import torch
from kotogram.analysis import gender, style
from kotogram.model import load_model
from kotogram.model import ModelConfig, Tokenizer

# Mock loading logic if needed, or point to trained model
MODEL_PATH = "models/test_style"

def test_api():
    print(f"Loading model from {MODEL_PATH}...")
    try:
        load_model(MODEL_PATH)
    except Exception as e:
        print(f"Failed to load model: {e}")
        # If model loading fails (maybe due to path), we can't test fully.
        # But we can test if the function signatures are correct if they don't strictly require model to be loaded for import.
        # Actually gender() requires a loaded model.
        return

    test_sentences = [
        "私は元気です", # Likely neutral/polite
        "俺は元気だ",   # Likely masculine
        "あたしは元気よ", # Likely feminine
        "本日は晴天なり", # Likely formal/unpragmatic?
    ]

    print("\nTesting gender() API:")
    for sent in test_sentences:
        g = gender(sent)
        print(f"Sentence: {sent}")
        print(f"  gender() result: {g} (Type: {type(g)})")
        
        # Verify type
        if g is not None and not isinstance(g, float):
             print("  FAIL: Expected float or None")
        else:
             print("  PASS type check")

    print("\nTesting style() API:")
    for sent in test_sentences:
        s = style(sent)
        print(f"Sentence: {sent}")
        print(f"  style() result: {s}")
        
        # Verify gender in style
        if s[1] is not None and not isinstance(s[1], float):
             print("  FAIL: Expected float or None for style.gender")
        else:
             print("  PASS type check")

if __name__ == "__main__":
    test_api()
