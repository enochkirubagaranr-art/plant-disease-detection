import math
from pathlib import Path
import numpy as np
import tensorflow as tf
import json
import train

root = Path(__file__).resolve().parent
class_names = json.loads((root / 'class_names.json').read_text(encoding='utf-8'))

# Load model
model_path = root / 'plant_model.h5'
weights_path = root / 'plant_model_weights.h5'
keras_path = root / 'plant_model.keras'
if model_path.exists():
    model = tf.keras.models.load_model(model_path, compile=False)
    print(f'Loaded model from: {model_path}')
elif weights_path.exists():
    model, _ = train.build_model(len(class_names))
    model.load_weights(str(weights_path))
    print(f'Loaded model weights from: {weights_path}')
elif keras_path.exists():
    model = tf.keras.models.load_model(keras_path, compile=False)
    print(f'Loaded model from: {keras_path}')
else:
    raise FileNotFoundError('No model file found: plant_model.h5, plant_model_weights.h5, or plant_model.keras')

# Dataset and generator
dataset_dir = train.get_dataset_directory()
train_gen, valid_gen, generator_class_names = train.build_generators(dataset_dir)

print('\n--- Class index alignment ---')
print('valid_gen.class_indices:')
for label, idx in sorted(valid_gen.class_indices.items(), key=lambda kv: kv[1]):
    print(f'  {idx}: {label}')
print('\nclass_names.json contents:')
for idx, label in enumerate(class_names):
    print(f'  {idx}: {label}')

index_order = [label for label, _ in sorted(valid_gen.class_indices.items(), key=lambda kv: kv[1])]
if index_order == class_names:
    print('\nOK: class_names.json order matches valid_gen.class_indices order.')
else:
    print('\nMISMATCH: class_names.json does not match valid_gen.class_indices.')
    print('  class_indices order:', index_order)
    print('  class_names order:', class_names)

print(f'\nModel output shape: {model.output_shape}')
model_output_dim = model.output_shape[-1]
print(f'Number of classes in class_names.json: {len(class_names)}')
if model_output_dim != len(class_names):
    print('MISMATCH: model output size does not equal number of class names.')

# Run validation predictions
val_steps = int(math.ceil(valid_gen.samples / valid_gen.batch_size))
preds = model.predict(valid_gen, steps=val_steps, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = valid_gen.classes
accuracy = float(np.mean(y_pred == y_true))
print(f'\nValidation accuracy: {accuracy:.4f}')
print(f'Validation samples: {valid_gen.samples}')

# Confusion matrix and per-class accuracy
num_classes = len(class_names)
cm = np.zeros((num_classes, num_classes), dtype=int)
for actual, pred in zip(y_true, y_pred):
    cm[actual, pred] += 1

print('\nConfusion matrix (rows=true class, cols=predicted class):')
print(cm)

print('\nPer-class accuracy:')
for idx, label in enumerate(class_names):
    total = cm[idx].sum()
    correct = cm[idx, idx]
    acc = float(correct) / total if total else 0.0
    print(f'  {label}: {correct}/{total} = {acc:.4f}')
    if 'Potato' in label:
        mis = cm[idx].copy()
        mis[idx] = 0
        if mis.sum() > 0:
            top = sorted(
                [(class_names[j], int(mis[j])) for j in range(num_classes) if j != idx],
                key=lambda item: item[1],
                reverse=True,
            )[:3]
            print(f'    Top confusions: {top}')

# Detect potato co-confusion patterns
print('\nPotato misclassification summary:')
for idx, label in enumerate(class_names):
    if 'Potato' not in label:
        continue
    total = cm[idx].sum()
    if total == 0:
        continue
    potato_errors = sum(cm[idx, j] for j in range(num_classes) if 'Potato' in class_names[j] and j != idx)
    tomato_errors = sum(cm[idx, j] for j in range(num_classes) if 'Tomato' in class_names[j])
    print(f'  {label}: total={total}, potato_errors={potato_errors}, tomato_errors={tomato_errors}')
    if tomato_errors > 0:
        print('    WARNING: potato class is being predicted as tomato classes.')
    if potato_errors > 0 and potato_errors / total > 0.1:
        print('    WARNING: potato classes are being confused with other potato classes.')
