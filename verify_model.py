#!/usr/bin/env python3
"""
Verify FINAL_MODEL.h5 can predict all 6 classes correctly.
Tests one sample image from each class.
"""
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image

# Configuration
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "FINAL_MODEL.h5"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"
DATASET_DIR = BASE_DIR / "dataset"
IMAGE_SIZE = (224, 224)

def load_class_names():
    with open(CLASS_NAMES_PATH, 'r') as f:
        return json.load(f)

def load_model():
    return tf.keras.models.load_model(str(MODEL_PATH), compile=False)

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE, Image.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)

def verify_model():
    print("\n" + "="*80)
    print("FINAL MODEL VERIFICATION")
    print("="*80)
    
    # Load model and class names
    class_names = load_class_names()
    model = load_model()
    
    print(f"\n✓ Model loaded: {MODEL_PATH}")
    print(f"✓ Classes: {class_names}")
    print(f"✓ Model output shape: {model.output_shape}")
    
    # Test predictions
    print(f"\n{'CLASS':<30} | {'PREDICTED':<30} | {'CONFIDENCE':<12} | {'STATUS'}")
    print("-" * 90)
    
    all_correct = True
    for class_name in class_names:
        class_dir = DATASET_DIR / class_name
        if not class_dir.exists():
            print(f"{class_name:<30} | {'N/A':<30} | {'N/A':<12} | ❌ NO SAMPLES")
            continue
        
        # Get first image
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))
        if not images:
            print(f"{class_name:<30} | {'N/A':<30} | {'N/A':<12} | ❌ NO IMAGES")
            continue
        
        # Predict
        image_path = images[0]
        image_array = preprocess_image(image_path)
        predictions = model.predict(image_array, verbose=0)[0]
        
        best_idx = int(np.argmax(predictions))
        best_confidence = float(predictions[best_idx])
        predicted_class = class_names[best_idx]
        
        # Check if correct
        is_correct = predicted_class == class_name
        status = "✓ CORRECT" if is_correct else f"✗ WRONG (predicted {predicted_class})"
        
        print(f"{class_name:<30} | {predicted_class:<30} | {best_confidence*100:>9.2f}% | {status}")
        
        if not is_correct:
            all_correct = False
    
    print("\n" + "="*80)
    if all_correct:
        print("✅ ALL CLASSES PREDICTED CORRECTLY!")
    else:
        print("⚠️  SOME MISCLASSIFICATIONS DETECTED - See above")
    print("="*80 + "\n")

if __name__ == "__main__":
    verify_model()
