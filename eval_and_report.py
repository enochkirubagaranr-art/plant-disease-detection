import math
from pathlib import Path
import numpy as np
import tensorflow as tf
import json
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'FINAL_MODEL.h5'
CLASS_NAMES_PATH = ROOT / 'class_names.json'
REPORT_MD = ROOT / 'PROJECT_REPORT.md'

# Load class names
class_names = json.loads(CLASS_NAMES_PATH.read_text(encoding='utf-8'))

# Load model
print(f'Loading model from: {MODEL_PATH}')
model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
print('Model loaded.')

# Minimal local config (avoid importing train which pulls heavy deps)
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 123

# Locate dataset directory
dataset_dir = ROOT / 'dataset'
if not dataset_dir.exists():
    raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

# Build validation generator (no augmentation)
valid_datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2)

train_gen = valid_datagen.flow_from_directory(
    directory=str(dataset_dir),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=SEED,
    interpolation="bicubic",
)

valid_gen = valid_datagen.flow_from_directory(
    directory=str(dataset_dir),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    seed=SEED,
    interpolation="bicubic",
)

gen_class_names = [label for label, _ in sorted(train_gen.class_indices.items(), key=lambda kv: kv[1])]
print('\nClass name order from generator:', gen_class_names)
print('Class names from file:', class_names)

# Run validation predictions
val_steps = int(math.ceil(valid_gen.samples / valid_gen.batch_size))
print(f'Running predictions on validation set ({valid_gen.samples} samples, {val_steps} steps)')
preds = model.predict(valid_gen, steps=val_steps, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = valid_gen.classes
accuracy = float(np.mean(y_pred == y_true))

# Confusion matrix
num_classes = len(class_names)
cm = np.zeros((num_classes, num_classes), dtype=int)
for actual, pred in zip(y_true, y_pred):
    cm[actual, pred] += 1

# Prepare sample predictions (first 5 validation images)
sample_info = []
for idx in range(min(5, len(valid_gen.filepaths))):
    img_path = Path(valid_gen.filepaths[idx])
    img = Image.open(img_path).convert('RGB')
    img = img.resize(IMAGE_SIZE, Image.BICUBIC)
    arr = np.array(img, dtype='float32') / 255.0
    arr = np.expand_dims(arr, axis=0)
    pred = model.predict(arr)
    p_idx = int(np.argmax(pred[0]))
    p_conf = float(np.max(pred[0]))
    sample_info.append({
        'filepath': str(img_path),
        'true_label': class_names[y_true[idx]],
        'pred_label': class_names[p_idx],
        'confidence': p_conf,
    })

# Write markdown report
with open(REPORT_MD, 'w', encoding='utf-8') as f:
    f.write('# Project Evaluation Report\n\n')
    f.write('## 1. Code Execution and Prediction Result\n\n')
    f.write('The model was loaded and evaluated using the project evaluation script `eval_and_report.py`. Below are sample prediction results on validation images.\n\n')
    for s in sample_info:
        f.write(f"- File: {s['filepath']}\\n")
        f.write(f"  - True label: **{s['true_label']}**\\n")
        f.write(f"  - Predicted: **{s['pred_label']}** (confidence: {s['confidence']*100:.2f}%)\\n\n")

    f.write('## 2. Testing Accuracy Output\n\n')
    f.write(f'- Validation samples: {valid_gen.samples}\\n')
    f.write(f'- Validation accuracy: **{accuracy*100:.2f}%**\\n\n')

    f.write('### Confusion Matrix (rows=true class, cols=predicted class)\n\n')
    f.write('```\n')
    for row in cm.tolist():
        f.write(' '.join(str(x) for x in row) + '\n')
    f.write('```\n\n')

    f.write('### Per-class accuracy\n\n')
    for idx, label in enumerate(class_names):
        total = cm[idx].sum()
        correct = cm[idx, idx]
        acc = float(correct) / total if total else 0.0
        f.write(f'- {label}: {correct}/{total} = **{acc*100:.2f}%**\\n')

print('\nReport written to', REPORT_MD)
print('Now converting to PDF using convert_to_pdf.py')

# Convert to PDF
import subprocess, sys
subprocess.run([sys.executable, 'convert_to_pdf.py'], check=True)
print('PDF conversion complete.')
print('Done.')
