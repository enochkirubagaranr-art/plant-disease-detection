# Quick Start: Running Improved Training Pipeline

## Prerequisites

```bash
# Ensure venv is activated
cd c:\Users\Enoch\Desktop\FINAL
venv\Scripts\activate

# Verify required packages (should already be installed)
pip list | findstr tensorflow sklearn scikit-learn pillow numpy
```

## Running Training

### Step 1: Start Training
```bash
python train.py
```

### Step 2: Monitor Training Output

The script will output:

```
====================================================================
DATASET BALANCE ANALYSIS
====================================================================
  Potato_Early_Blight:      1200 images
  Potato_Late_Blight:       1150 images
  Potato_Healthy:           1100 images
  Tomato_Early_Blight:      1000 images
  Tomato_Late_Blight:       950 images
  Tomato_Healthy:           1050 images

  Total images: 6450
  Mean per class: 1075.0
  Min: 950, Max: 1200
  Imbalance ratio (max/min): 1.26x

  Potato classes total: 3450 images
  Tomato classes total: 3000 images
  Potato/Tomato ratio: 1.15x
====================================================================

CLASS WEIGHTS (for imbalance correction)
====================================================================
  Potato_Early_Blight:       0.9231
  Potato_Late_Blight:        0.9565
  Potato_Healthy:            0.9818
  Tomato_Early_Blight:       1.0588
  Tomato_Late_Blight:        1.1111
  Tomato_Healthy:            1.0769
====================================================================

PHASE 1: TRAINING WITH FROZEN BASE MODEL
====================================================================
  (Learning the disease classification head while keeping ImageNet features)

Epoch 1/100
24/24 [==============================] - 45s 2s/step - loss: 2.1234 - accuracy: 0.3421 - val_loss: 1.8765 - val_accuracy: 0.4512
Epoch 2/100
24/24 [==============================] - 42s 2s/step - loss: 1.6543 - accuracy: 0.5123 - val_loss: 1.4321 - val_accuracy: 0.6234

... (training continues)

PHASE 2: FINE-TUNING WITH UNFROZEN BASE MODEL LAYERS
====================================================================
  (Fine-tuning MobileNetV2 layers for plant disease features)

Epoch 51/100
24/24 [==============================] - 52s 2s/step - loss: 0.8234 - accuracy: 0.7845 - val_loss: 0.7123 - val_accuracy: 0.8012

... (fine-tuning continues)

====================================================================
COMPREHENSIVE MODEL EVALUATION
====================================================================

Overall Validation Accuracy: 0.8234 (82.34%)

PER-CLASS ACCURACY (RECALL)
====================================================================
  Potato_Early_Blight:       0.8901 (89.01%)
  Potato_Late_Blight:        0.7654 (76.54%)
  Potato_Healthy:            0.9123 (91.23%)
  Tomato_Early_Blight:       0.8765 (87.65%)
  Tomato_Late_Blight:        0.9012 (90.12%)
  Tomato_Healthy:            0.8234 (82.34%)

CROSS-PLANT CONFUSION ANALYSIS
====================================================================
  Potato misclassified as Tomato: 12
  Tomato misclassified as Potato: 8
  Total cross-plant errors: 20

POTATO CLASSES ANALYSIS (Most Critical)
====================================================================
  Potato overall accuracy: 0.8559 (85.59%)
  Potato samples evaluated: 487

TOMATO CLASSES ANALYSIS
====================================================================
  Tomato overall accuracy: 0.8670 (86.70%)
  Tomato samples evaluated: 413

SAMPLE PREDICTIONS (One per class)
====================================================================
✓ [Potato_Early_Blight     ] predicted=Potato_Early_Blight     confidence=0.9234
✓ [Potato_Late_Blight      ] predicted=Potato_Late_Blight      confidence=0.8765
✓ [Potato_Healthy          ] predicted=Potato_Healthy          confidence=0.9512
✓ [Tomato_Early_Blight     ] predicted=Tomato_Early_Blight     confidence=0.8901
✓ [Tomato_Late_Blight      ] predicted=Tomato_Late_Blight      confidence=0.9345
✓ [Tomato_Healthy          ] predicted=Tomato_Healthy          confidence=0.8634

====================================================================
TRAINING COMPLETE
====================================================================
  Model saved: c:\Users\Enoch\Desktop\FINAL\plant_model.h5
  Classes: 6
  Final validation accuracy: 0.8234
====================================================================
```

### Step 3: Check Outputs

After training, verify these files were created:
```bash
dir c:\Users\Enoch\Desktop\FINAL

# Should see:
# plant_model.h5                (Updated trained model)
# class_names.json              (Class mapping)
# confusion_matrix.png          (Visualization)
# plant_model_checkpoint.h5     (Best weights)
```

## Understanding the Output

### Key Metrics

1. **Potato Overall Accuracy**: Should be ≥85%
   - Target: 85-95%
   - This is the main fix goal

2. **Tomato Overall Accuracy**: Should maintain ≥85%
   - Target: 90-95%
   - Should not decrease from previous runs

3. **Cross-Plant Confusion**: Should be <5% of total samples
   - Example: <50 errors out of 900 samples
   - If high: increase augmentation

4. **Per-Class Accuracy**: All should be >75%
   - If one class <75%: check dataset quality or size

### Warnings to Watch

```
WARNING: Early stopping triggered
- Means: No validation improvement for 10 epochs
- Action: Training stopped automatically (good!)
- Result: Model checkpoint restored to best weights

WARNING: Learning rate reduced
- Means: Validation loss plateau detected
- Action: Learning rate halved automatically
- Result: Fine-tuning continues at slower pace

ERROR: Class folder missing
- Means: Dataset cleanup failed
- Action: Check dataset structure
- Result: Training aborted (requires manual fix)
```

## Troubleshooting

### Issue: "Could not find dataset"
```
Solution:
1. Check dataset_balanced_1000/ exists
2. Verify all 6 class folders present
3. Ensure each folder has images
4. Run again
```

### Issue: "CUDA out of memory"
```
Solution (if GPU):
1. Reduce BATCH_SIZE in train.py from 32 to 16
2. Reduce IMAGE_SIZE from 224 to 192
3. Reduce EPOCHS or increase patience values
```

### Issue: Potato accuracy still low (<80%)
```
Solutions (in order):
1. Increase DROPOUT_RATE from 0.5 to 0.7
2. Increase L2_REGULARIZATION from 1e-4 to 5e-4
3. Increase rotation_range from 40 to 60
4. Collect more potato disease images
5. Try EfficientNetB0/B1 instead of MobileNetV2
```

### Issue: Training too slow
```
Solutions:
1. Increase BATCH_SIZE from 32 to 64
2. Reduce EPOCHS from 100 to 50
3. Reduce resolution: IMAGE_SIZE = (192, 192)
4. Use GPU if available (TensorFlow will auto-detect)
```

## Next Steps After Training

### 1. Test with Streamlit App
```bash
streamlit run streamlit_app.py
```

Then upload test images to verify predictions.

### 2. Evaluate on Test Set (if available)
```bash
python eval_accuracy.py
```

### 3. Compare with Previous Model
```bash
# Backup current model
copy plant_model.h5 plant_model_v2.h5

# Run predictions with new model
python predict.py dataset_balanced_1000/Potato_Early_Blight/img1.jpg
```

### 4. Hyperparameter Adjustment (if needed)

If results not satisfactory, edit `train.py`:
```python
# Increase augmentation strength
rotation_range=60              # Was 40
zoom_range=0.40                # Was 0.25
brightness_range=(0.5, 1.5)   # Was (0.7, 1.3)

# Increase regularization
DROPOUT_RATE = 0.7             # Was 0.5
L2_REGULARIZATION = 5e-4       # Was 1e-4

# Longer training
EPOCHS = 150                   # Was 100
EARLY_STOPPING_PATIENCE = 20   # Was 10
```

Then run training again.

## Performance Expectations

### Training Time
- **Phase 1** (frozen): ~40-60 minutes
- **Phase 2** (fine-tune): ~40-60 minutes
- **Total**: ~2-3 hours (CPU), ~30-45 minutes (GPU)

### Storage
- Model file: ~60-100 MB
- Checkpoint: ~60-100 MB
- Confusion matrix PNG: ~100-200 KB

### Hardware Requirements
- **CPU**: Modern multi-core CPU (Intel i5+, AMD Ryzen 5+)
- **RAM**: 8 GB minimum, 16 GB recommended
- **GPU** (optional): CUDA-capable GPU 2-4 GB VRAM
- **Disk**: 5 GB free space (dataset + models)

## Validation Checklist

After training, verify:

- [ ] `plant_model.h5` exists and file size > 50 MB
- [ ] `class_names.json` contains 6 class names
- [ ] `confusion_matrix.png` generated
- [ ] Overall accuracy > 80%
- [ ] Potato accuracy > 85%
- [ ] Tomato accuracy > 85%
- [ ] Cross-plant confusion < 50 errors
- [ ] No cross-domain errors (healthy confused as disease, etc.)
- [ ] Streamlit app runs and predicts correctly
- [ ] Class balance analysis completed

## Advanced: Custom Evaluation

To run custom evaluation on a test set:

```python
from train import evaluate_model_comprehensive
import tensorflow as tf

model = tf.keras.models.load_model('plant_model.h5', compile=False)
class_names = ['Potato_Early_Blight', ..., 'Tomato_Healthy']

# Create a test generator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_generator = test_datagen.flow_from_directory(
    'path/to/test/dataset',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Evaluate
results = evaluate_model_comprehensive(model, test_generator, class_names, Path('.'))
```

## Questions?

Refer to `TRAINING_IMPROVEMENTS.md` for detailed technical explanations of each improvement.
