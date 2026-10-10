import hashlib
import json
import shutil
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# CONFIGURATION
# =============================================================================
# Image and training parameters
IMAGE_SIZE = (224, 224)  # MobileNetV2 input size
BATCH_SIZE = 32  # Increased from 16 for better gradient estimates
SEED = 123
EPOCHS = 100  # Increased; EarlyStopping will stop early if no improvement
MODEL_FILENAME = "plant_model.h5"
CLASS_NAMES_FILENAME = "class_names.json"

# Hyperparameters
INITIAL_LEARNING_RATE = 1e-4  # Lower LR for transfer learning
MIN_LEARNING_RATE = 1e-7
EARLY_STOPPING_PATIENCE = 10  # More patience to allow longer training
LR_REDUCTION_PATIENCE = 5
DROPOUT_RATE = 0.5  # Higher dropout to prevent overfitting
L2_REGULARIZATION = 1e-4  # Add L2 regularization to dense layers

# Class configuration
POTATO_CLASSES = {"Potato_Early_Blight", "Potato_Late_Blight", "Potato_Healthy"}
TOMATO_CLASSES = {"Tomato_Early_Blight", "Tomato_Late_Blight", "Tomato_Healthy"}
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
EXPECTED_CLASS_FOLDERS = {
    "potato_early_blight": "Potato_Early_Blight",
    "potato_late_blight": "Potato_Late_Blight",
    "potato_healthy": "Potato_Healthy",
    "tomato_early_blight": "Tomato_Early_Blight",
    "tomato_late_blight": "Tomato_Late_Blight",
    "tomato_healthy": "Tomato_Healthy",
    "Potato_Early_Blight": "Potato_Early_Blight",
    "Potato_Late_Blight": "Potato_Late_Blight",
    "Potato_Healthy": "Potato_Healthy",
    "Tomato_Early_Blight": "Tomato_Early_Blight",
    "Tomato_Late_Blight": "Tomato_Late_Blight",
    "Tomato_Healthy": "Tomato_Healthy",
}
REQUIRED_CLASS_FOLDERS = [
    "Potato_Early_Blight",
    "Potato_Late_Blight",
    "Potato_Healthy",
    "Tomato_Early_Blight",
    "Tomato_Late_Blight",
    "Tomato_Healthy",
]
BACKUP_CANDIDATES = ["dataset_balanced_1000", "dataset_balanced", "dataset"]

# =============================================================================
# DATASET ANALYSIS UTILITIES
# =============================================================================
def analyze_dataset_balance(dataset_dir: Path) -> dict:
    """
    Analyze class distribution in the dataset.
    
    Returns:
        Dictionary with counts per class and summary statistics.
    """
    class_counts = {}
    print("\n" + "="*70)
    print("DATASET BALANCE ANALYSIS")
    print("="*70)
    
    for class_name in REQUIRED_CLASS_FOLDERS:
        class_dir = dataset_dir / class_name
        if class_dir.is_dir():
            image_count = sum(
                1 for path in class_dir.iterdir() 
                if path.is_file() and path.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
            )
            class_counts[class_name] = image_count
            print(f"  {class_name:30s}: {image_count:4d} images")
        else:
            class_counts[class_name] = 0
            print(f"  {class_name:30s}: MISSING")
    
    # Summary statistics
    total_images = sum(class_counts.values())
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    mean_count = total_images / len(REQUIRED_CLASS_FOLDERS)
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    print("\n" + "-"*70)
    print(f"  Total images: {total_images}")
    print(f"  Mean per class: {mean_count:.1f}")
    print(f"  Min: {min_count}, Max: {max_count}")
    print(f"  Imbalance ratio (max/min): {imbalance_ratio:.2f}x")
    print("-"*70)
    
    # Potato vs Tomato split
    potato_count = sum(
        class_counts[c] for c in class_counts if c.startswith("Potato_")
    )
    tomato_count = sum(
        class_counts[c] for c in class_counts if c.startswith("Tomato_")
    )
    
    print(f"  Potato classes total: {potato_count} images")
    print(f"  Tomato classes total: {tomato_count} images")
    if min(potato_count, tomato_count) > 0:
        print(f"  Potato/Tomato ratio: {potato_count/tomato_count:.2f}x")
    print("="*70 + "\n")
    
    return class_counts

# =============================================================================
# DATASET HELPERS
# =============================================================================

def get_root_directory() -> Path:
    return Path(__file__).resolve().parent


def _has_required_tomato_structure(path: Path) -> bool:
    if not path.is_dir():
        return False
    for folder in REQUIRED_CLASS_FOLDERS:
        candidate = path / folder
        if not candidate.is_dir():
            return False
        if not any(candidate.iterdir()):
            return False
    return True


def _copy_source_folder(source: Path, target: Path) -> None:
    print(f"Rebuilding dataset root from backup source: {source}")
    if target.exists():
        shutil.rmtree(str(target))
    target.mkdir(parents=True, exist_ok=True)
    for source_name, normalized_name in EXPECTED_CLASS_FOLDERS.items():
        source_dir = source / source_name
        if source_dir.is_dir():
            destination = target / normalized_name
            shutil.copytree(str(source_dir), str(destination), dirs_exist_ok=True)
            print(f"  Copied {source_dir.name} -> {destination.name}")


def get_dataset_directory() -> Path:
    dataset_dir = get_root_directory() / "dataset"
    if _has_required_tomato_structure(dataset_dir):
        return dataset_dir

    for candidate_name in BACKUP_CANDIDATES:
        candidate_dir = get_root_directory() / candidate_name
        if _has_required_tomato_structure(candidate_dir):
            _copy_source_folder(candidate_dir, dataset_dir)
            return dataset_dir

    raise FileNotFoundError(
        "Could not find a full six-class dataset in dataset/, dataset_balanced_1000/, or dataset_balanced."
    )


def archive_path() -> Path:
    archive_dir = get_root_directory() / "dataset_ignored"
    archive_dir.mkdir(parents=True, exist_ok=True)
    return archive_dir


def clean_dataset_structure(dataset_dir: Path) -> None:
    print(f"Cleaning dataset structure in {dataset_dir}")
    archive_dir = archive_path()

    for source in sorted(dataset_dir.iterdir()):
        if not source.is_dir():
            continue

        normalized_name = EXPECTED_CLASS_FOLDERS.get(source.name)
        if normalized_name is None:
            destination = archive_dir / source.name
            if destination.exists():
                destination = archive_dir / f"{source.name}_{len(list(archive_dir.iterdir())) + 1}"
            shutil.move(str(source), str(destination))
            print(f"Archived non-tomato folder: {source.name} -> {destination.name}")
            continue

        target_dir = dataset_dir / normalized_name
        if source.name == normalized_name:
            continue

        if target_dir.exists():
            for item in source.iterdir():
                destination = target_dir / item.name
                if destination.exists():
                    if item.is_dir():
                        shutil.copytree(str(item), str(destination), dirs_exist_ok=True)
                        shutil.rmtree(str(item))
                        continue
                    suffix = 1
                    while destination.exists():
                        destination = target_dir / f"{item.stem}_{suffix}{item.suffix}"
                        suffix += 1
                shutil.move(str(item), str(destination))
            shutil.rmtree(str(source))
            print(f"Merged folder {source.name} into {target_dir.name}")
        else:
            if source.name.lower() == normalized_name.lower():
                temp_dir = dataset_dir / f"{normalized_name}_{uuid.uuid4().hex}"
                source.rename(temp_dir)
                temp_dir.rename(target_dir)
            else:
                source.rename(target_dir)
            print(f"Renamed folder {source.name} -> {target_dir.name}")

    removed_duplicates = 0
    seen_hashes: dict[str, Path] = {}
    for class_name in REQUIRED_CLASS_FOLDERS:
        class_dir = dataset_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Required folder missing after cleanup: {class_name}. Expected all six classes in dataset."
            )

        for image_path in sorted(class_dir.rglob("*")):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
                image_path.unlink()
                print(f"Removed non-image file: {image_path}")
                continue

            image_hash = hashlib.md5(image_path.read_bytes()).hexdigest()
            if image_hash in seen_hashes:
                previous_path = seen_hashes[image_hash]
                if image_path.parent != previous_path.parent:
                    print(f"Cross-class duplicate detected: {image_path} and {previous_path}")
                image_path.unlink()
                removed_duplicates += 1
                print(f"Removed duplicate image: {image_path} (duplicate of {previous_path})")
            else:
                seen_hashes[image_hash] = image_path

    for class_name in REQUIRED_CLASS_FOLDERS:
        class_dir = dataset_dir / class_name
        image_count = sum(1 for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS)
        if image_count == 0:
            raise ValueError(f"No images found in required class folder: {class_name}")
        print(f"Class folder {class_name} contains {image_count} images after cleanup.")

    if removed_duplicates:
        print(f"Total duplicate images removed: {removed_duplicates}")


def build_generators(dataset_dir: Path):
    """
    Build data generators with separate augmentation strategies for train and validation.
    
    The training generator uses aggressive augmentation to:
    - Improve generalization
    - Handle potato class confusion through diversity
    - Simulate real-world lighting and angles
    
    The validation generator applies only rescaling (no augmentation) for fair evaluation.
    """
    
    # TRAINING AUGMENTATION
    # Enhanced augmentation parameters to improve robustness, especially for potato detection
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.2,
        # Rotation: simulate different leaf angles and disease spot visibility
        rotation_range=40,
        # Shift: handle off-center leaves and partial spots
        width_shift_range=0.20,
        height_shift_range=0.20,
        # Shear: simulate tilted perspective
        shear_range=0.15,
        # Zoom: focus on disease spots at different scales
        zoom_range=0.25,
        # Brightness: handle varying lighting conditions (important for disease detection)
        brightness_range=(0.7, 1.3),
        # Horizontal flip: leaves can be photographed from any angle
        horizontal_flip=True,
        # Vertical flip: disease spots can appear anywhere on leaf
        vertical_flip=True,
        # Channel shift: subtle color variation in disease presentation
        channel_shift_range=20.0,
        # Fill mode: handle pixels outside transformation boundaries
        fill_mode="reflect",
    )

    # VALIDATION AUGMENTATION (NO AUGMENTATION)
    # Validation data is NOT augmented - only rescaled for fair model evaluation
    valid_datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2)

    # Build training generator with shuffling
    train_generator = train_datagen.flow_from_directory(
        directory=str(dataset_dir),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=SEED,
        interpolation="bicubic",  # Better quality interpolation
    )

    # Build validation generator without shuffling (for consistent evaluation)
    valid_generator = valid_datagen.flow_from_directory(
        directory=str(dataset_dir),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=SEED,
        interpolation="bicubic",
    )

    # Extract class names in index order from the training generator
    class_names = [
        label for label, _ in sorted(
            train_generator.class_indices.items(), key=lambda kv: kv[1]
        )
    ]
    
    print("\n" + "="*70)
    print("DATA GENERATORS CREATED")
    print("="*70)
    print(f"  Training samples: {train_generator.samples}")
    print(f"  Validation samples: {valid_generator.samples}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Steps per epoch: {np.ceil(train_generator.samples / BATCH_SIZE):.0f}")
    print(f"  Image size: {IMAGE_SIZE}")
    print(f"  Class names: {class_names}")
    print("="*70 + "\n")
    
    return train_generator, valid_generator, class_names



def save_class_names(class_names: list[str]) -> None:
    """Save class names to JSON for use in inference and Streamlit app."""
    path = get_root_directory() / CLASS_NAMES_FILENAME
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(class_names, handle, indent=2)
    print(f"Saved class names to {path}")

# =============================================================================
# MODEL ARCHITECTURE - TRANSFER LEARNING WITH IMPROVED REGULARIZATION
# =============================================================================
def build_model(num_classes: int):
    """
    Build a CNN model using MobileNetV2 transfer learning with careful tuning
    for potato class distinction.
    
    Key improvements:
    1. MobileNetV2 pre-trained on ImageNet for strong feature extraction
    2. Global average pooling to reduce spatial dimensions
    3. Dense layers with L2 regularization to prevent overfitting
    4. Higher dropout rate (0.5) for aggressive regularization
    5. Batch normalization for training stability
    6. Separate batch norm layers for better convergence
    
    This architecture balances:
    - Feature richness (pre-trained weights)
    - Generalization (dropout + L2)
    - Training stability (batch norm)
    - Computational efficiency (MobileNetV2)
    """
    
    # Load pre-trained MobileNetV2 backbone without top classification layer
    # This gives us features learned from 1.4M ImageNet images
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    
    # Freeze base model weights initially (we'll fine-tune later)
    base_model.trainable = False

    # Build custom classification head
    inputs = layers.Input(shape=(*IMAGE_SIZE, 3))
    
    # Pass through pre-trained feature extractor
    x = base_model(inputs, training=False)
    
    # Reduce spatial dimensions with global average pooling
    # This is more robust than flattening for image classification
    x = layers.GlobalAveragePooling2D()(x)
    
    # First dense layer with batch normalization and dropout
    x = layers.Dense(
        256, 
        kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION),
        bias_regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    
    # Second dense layer for additional feature processing
    x = layers.Dense(
        128,
        kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION),
        bias_regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    
    # Output layer: softmax for multi-class classification
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    
    print("\n" + "="*70)
    print("MODEL ARCHITECTURE")
    print("="*70)
    print(f"  Base model: MobileNetV2 (pre-trained on ImageNet)")
    print(f"  Input size: {IMAGE_SIZE}")
    print(f"  Output classes: {num_classes}")
    print(f"  Dropout rate: {DROPOUT_RATE}")
    print(f"  L2 regularization: {L2_REGULARIZATION}")
    print(f"  Total parameters: {model.count_params():,}")
    print("="*70 + "\n")
    
    return model, base_model

# =============================================================================
# MODEL EVALUATION & ANALYSIS
# =============================================================================


def evaluate_model_comprehensive(
    model: tf.keras.Model, 
    generator, 
    class_names: list[str],
    dataset_dir: Path
) -> dict:
    """
    Comprehensive model evaluation including:
    - Overall accuracy
    - Per-class accuracy
    - Confusion matrix
    - Cross-plant confusion analysis (potato vs tomato)
    - Classification report with precision/recall/F1
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*70)
    
    # Generate predictions
    generator.reset()
    val_steps = int(np.ceil(generator.samples / generator.batch_size))
    predictions = model.predict(generator, steps=val_steps, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    y_true = generator.classes
    
    # Overall accuracy
    accuracy = np.mean(y_pred == y_true)
    print(f"\nOverall Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Per-class accuracy
    print("\n" + "-"*70)
    print("PER-CLASS ACCURACY (RECALL)")
    print("-"*70)
    per_class_accuracy = {}
    for i, class_name in enumerate(class_names):
        if cm[i].sum() > 0:
            class_acc = cm[i, i] / cm[i].sum()
            per_class_accuracy[class_name] = class_acc
            print(f"  {class_name:30s}: {class_acc:.4f} ({class_acc*100:.2f}%)")
        else:
            per_class_accuracy[class_name] = 0.0
    
    # Detect cross-plant confusion (potato misclassified as tomato and vice versa)
    print("\n" + "-"*70)
    print("CROSS-PLANT CONFUSION ANALYSIS")
    print("-"*70)
    
    potato_to_tomato = 0
    tomato_to_potato = 0
    
    for true_idx, pred_idx in zip(y_true, y_pred):
        true_class = class_names[true_idx]
        pred_class = class_names[pred_idx]
        
        if true_class.startswith("Potato_") and pred_class.startswith("Tomato_"):
            potato_to_tomato += 1
        elif true_class.startswith("Tomato_") and pred_class.startswith("Potato_"):
            tomato_to_potato += 1
    
    print(f"  Potato misclassified as Tomato: {potato_to_tomato}")
    print(f"  Tomato misclassified as Potato: {tomato_to_potato}")
    print(f"  Total cross-plant errors: {potato_to_tomato + tomato_to_potato}")
    
    # Classification report (precision, recall, F1)
    print("\n" + "-"*70)
    print("DETAILED CLASSIFICATION REPORT")
    print("-"*70)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)
    
    # Potato disease accuracy
    print("-"*70)
    print("POTATO CLASSES ANALYSIS (Most Critical)")
    print("-"*70)
    potato_indices = [i for i, name in enumerate(class_names) if name.startswith("Potato_")]
    potato_mask = np.isin(y_true, potato_indices)
    if potato_mask.sum() > 0:
        potato_accuracy = np.mean(y_pred[potato_mask] == y_true[potato_mask])
        print(f"  Potato overall accuracy: {potato_accuracy:.4f} ({potato_accuracy*100:.2f}%)")
        print(f"  Potato samples evaluated: {potato_mask.sum()}")
    
    # Tomato disease accuracy
    print("-"*70)
    print("TOMATO CLASSES ANALYSIS")
    print("-"*70)
    tomato_indices = [i for i, name in enumerate(class_names) if name.startswith("Tomato_")]
    tomato_mask = np.isin(y_true, tomato_indices)
    if tomato_mask.sum() > 0:
        tomato_accuracy = np.mean(y_pred[tomato_mask] == y_true[tomato_mask])
        print(f"  Tomato overall accuracy: {tomato_accuracy:.4f} ({tomato_accuracy*100:.2f}%)")
        print(f"  Tomato samples evaluated: {tomato_mask.sum()}")
    
    print("="*70 + "\n")
    
    return {
        "overall_accuracy": accuracy,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "potato_to_tomato_errors": potato_to_tomato,
        "tomato_to_potato_errors": tomato_to_potato,
    }


def plot_confusion_matrix(cm, class_names):
    """
    Plot and save confusion matrix visualization.
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix - Plant Disease Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save the plot
    plot_path = get_root_directory() / "confusion_matrix.png"
    plt.savefig(str(plot_path), dpi=150)
    print(f"Confusion matrix saved to {plot_path}")
    plt.close()


def run_sample_predictions(model: tf.keras.Model, class_names: list[str], dataset_dir: Path) -> None:
    """
    Run predictions on one sample image from each class.
    Useful for quick sanity check that model can distinguish all classes.
    """
    print("\n" + "="*70)
    print("SAMPLE PREDICTIONS (One per class)")
    print("="*70)
    predictions = []

    for class_name in class_names:
        class_dir = dataset_dir / class_name
        image_files = sorted(
            [path for path in class_dir.iterdir() 
             if path.is_file() and path.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS]
        )
        if not image_files:
            raise ValueError(f"No sample images found in class folder: {class_name}")

        sample_path = image_files[0]
        image = load_img(str(sample_path), target_size=IMAGE_SIZE)
        image_array = img_to_array(image).astype("float32") / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        prediction = model.predict(image_array, verbose=0)
        predicted_index = int(np.argmax(prediction, axis=1)[0])
        predicted_class = class_names[predicted_index]
        confidence = float(prediction[0, predicted_index])
        predictions.append(predicted_class)

        is_correct = "✓" if predicted_class == class_name else "✗"
        print(
            f"{is_correct} [{class_name:30s}] predicted={predicted_class:30s} "
            f"confidence={confidence:.4f}"
        )

    if len(set(predictions)) != len(class_names):
        raise RuntimeError(
            "Model sample predictions are not distinct across classes. "
            "Please inspect the dataset and training output."
        )
    print("="*70 + "\n")




def train() -> None:
    """
    Complete training pipeline with:
    1. Dataset balance analysis
    2. Advanced data augmentation
    3. Transfer learning with MobileNetV2
    4. Comprehensive callbacks for early stopping, LR scheduling, checkpointing
    5. Two-phase training (frozen then fine-tune)
    6. Detailed evaluation with confusion matrix
    """
    
    # Setup
    dataset_dir = get_dataset_directory()
    clean_dataset_structure(dataset_dir)
    print(f"\nUsing dataset: {dataset_dir}\n")
    
    # STEP 1: Analyze dataset balance
    class_counts = analyze_dataset_balance(dataset_dir)
    
    # STEP 2: Build data generators with advanced augmentation
    train_generator, valid_generator, class_names = build_generators(dataset_dir)
    
    # STEP 3: Build model with transfer learning
    model, base_model = build_model(len(class_names))
    
    # STEP 3.5: Load checkpoint if exists (resume training)
    checkpoint_path = get_root_directory() / "plant_model_checkpoint.h5"
    if checkpoint_path.exists():
        print(f"\nLoading checkpoint from {checkpoint_path}")
        model.load_weights(str(checkpoint_path))
        print("Checkpoint loaded successfully - resuming training from saved state")
    else:
        print("\nNo checkpoint found - starting training from scratch")
    
    # STEP 4: Compute class weights to handle imbalance
    # Classes with fewer samples get higher weights
    class_indices = train_generator.class_indices
    generator_class_counts = {idx: class_counts[name] for name, idx in class_indices.items()}
    total_samples = sum(generator_class_counts.values())
    
    class_weights = {
        class_id: float(total_samples / (len(generator_class_counts) * count))
        for class_id, count in generator_class_counts.items()
    }
    
    print("\n" + "="*70)
    print("CLASS WEIGHTS (for imbalance correction)")
    print("="*70)
    for class_id, weight in sorted(class_weights.items()):
        class_name = class_names[class_id]
        print(f"  {class_name:30s}: {weight:.4f}")
    print("="*70 + "\n")
    
    # STEP 5: Compile model with custom learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    
    # STEP 6: Setup callbacks for better training
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(get_root_directory() / "plant_model_checkpoint.h5"),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=LR_REDUCTION_PATIENCE,
        min_lr=MIN_LEARNING_RATE,
        verbose=1,
    )
    
    callbacks = [model_checkpoint, early_stopping, reduce_lr]
    
    # STEP 7: PHASE 1 - Train with frozen base model (fast training, feature tuning)
    print("\n" + "="*70)
    print("PHASE 1: TRAINING WITH FROZEN BASE MODEL")
    print("="*70)
    print("  (Learning the disease classification head while keeping ImageNet features)")
    print("="*70 + "\n")
    
    history_phase1 = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )
    
    # STEP 8: PHASE 2 - Fine-tune with partially unfrozen base model (slow training, feature refinement)
    # Unfreeze last few layers of base model for fine-tuning
    print("\n" + "="*70)
    print("PHASE 2: FINE-TUNING WITH UNFROZEN BASE MODEL LAYERS")
    print("="*70)
    print("  (Fine-tuning MobileNetV2 layers for plant disease features)")
    print("="*70 + "\n")
    
    # Unfreeze the last 50 layers of base model
    base_model.trainable = True
    for layer in base_model.layers[:-50]:
        layer.trainable = False
    
    # Use lower learning rate for fine-tuning
    optimizer_finetune = tf.keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE / 10)
    model.compile(
        optimizer=optimizer_finetune,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    
    history_phase2 = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )
    
    # STEP 9: Save final model
    model_path = get_root_directory() / MODEL_FILENAME
    model.save(str(model_path))
    print(f"\nSaved final trained model to {model_path}")
    
    save_class_names(class_names)
    
    # STEP 10: Comprehensive evaluation
    eval_results = evaluate_model_comprehensive(model, valid_generator, class_names, dataset_dir)
    
    # Plot confusion matrix
    plot_confusion_matrix(eval_results['confusion_matrix'], class_names)
    
    # STEP 11: Sample predictions for sanity check
    run_sample_predictions(model, class_names, dataset_dir)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"  Model saved: {model_path}")
    print(f"  Classes: {len(class_names)}")
    print(f"  Final validation accuracy: {eval_results['overall_accuracy']:.4f}")
    print("="*70 + "\n")




if __name__ == "__main__":
    train()
