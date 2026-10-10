import tensorflow as tf
import json
from pathlib import Path

# Load model
MODEL_PATH = Path('FINAL_MODEL.h5')
model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)

print("Model layers:")
for i, layer in enumerate(model.layers):
    print(f"{i}: {layer.name} - {layer.__class__.__name__}")

print("\nLooking for conv layers in outer model:")
for layer in reversed(model.layers):
    if 'conv' in layer.name.lower():
        print(f"Found conv layer: {layer.name} - {layer.__class__.__name__}")
        break
else:
    print("No conv layers found in outer model!")

# Check if there's a functional layer that contains conv layers
print("\nChecking functional layers:")
for layer in model.layers:
    if hasattr(layer, 'layers'):  # Check if it's a functional/model layer
        print(f"Functional layer: {layer.name}")
        print("Inner layers:")
        for i, inner_layer in enumerate(layer.layers[-10:]):  # Show last 10 layers
            print(f"  {len(layer.layers)-10+i}: {inner_layer.name} - {inner_layer.__class__.__name__}")
            if 'conv' in inner_layer.name.lower():
                print(f"    *** Found conv layer: {inner_layer.name} ***")

print("\nLooking for conv layers in functional layer:")
for layer in model.layers:
    if hasattr(layer, 'layers'):
        for inner_layer in reversed(layer.layers):
            if 'conv' in inner_layer.name.lower():
                print(f"Found conv layer: {inner_layer.name} - {inner_layer.__class__.__name__}")
                break
        else:
            print("No conv layers found in functional layer!")
        break