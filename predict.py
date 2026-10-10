import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "FINAL_MODEL.h5"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"
IMAGE_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.70
AMBIGUITY_MARGIN = 0.10
TOP_K = 3


def load_class_names() -> list[str]:
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(f"Class names file not found: {CLASS_NAMES_PATH}")
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as handle:
        names = json.load(handle)
    if not isinstance(names, list) or not all(isinstance(x, str) for x in names):
        raise ValueError("class_names.json must contain a list of label strings.")
    return names


def load_model() -> tf.keras.Model:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return tf.keras.models.load_model(str(MODEL_PATH), compile=False)


def preprocess_image(image_path: str | Path) -> np.ndarray:
    """
    Load and preprocess image from file path.
    MUST match training preprocessing for consistency!
    """
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE, Image.BICUBIC)  # High-quality interpolation
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    return np.expand_dims(array, axis=0)


def preprocess_pil_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess PIL Image object.
    MUST match training preprocessing for consistency!
    """
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE, Image.BICUBIC)  # High-quality interpolation
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    return np.expand_dims(array, axis=0)


def predict_probabilities(model: tf.keras.Model, image_array: np.ndarray) -> np.ndarray:
    raw = model.predict(image_array, verbose=0)
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim == 2 and raw.shape[0] == 1:
        return raw[0]
    if raw.ndim == 1:
        return raw
    raise ValueError(f"Unexpected model output shape: {raw.shape}")


def get_top_k(predictions: np.ndarray, class_names: list[str], k: int = TOP_K) -> list[tuple[str, float, int]]:
    indices = np.argsort(predictions)[::-1][:k]
    return [(class_names[i], float(predictions[i]), int(i)) for i in indices]


def is_ambiguous(predictions: np.ndarray, margin: float = AMBIGUITY_MARGIN) -> bool:
    sorted_indices = np.argsort(predictions)[::-1]
    if len(sorted_indices) < 2:
        return False
    top1 = float(predictions[sorted_indices[0]])
    top2 = float(predictions[sorted_indices[1]])
    gap = top1 - top2
    return top1 < CONFIDENCE_THRESHOLD or gap < margin


def detect_cross_plant_confusion(predictions: np.ndarray, class_names: list[str]) -> dict[str, dict[str, Any]]:
    pairs = [
        ("Potato_Early_Blight", "Tomato_Early_Blight"),
        ("Potato_Late_Blight", "Tomato_Late_Blight"),
    ]
    confusion = {}
    for potato_label, tomato_label in pairs:
        if potato_label not in class_names or tomato_label not in class_names:
            continue
        potato_idx = class_names.index(potato_label)
        tomato_idx = class_names.index(tomato_label)
        confusion[f"{potato_label}_vs_{tomato_label}"] = {
            "potato_prob": float(predictions[potato_idx]),
            "tomato_prob": float(predictions[tomato_idx]),
            "gap": float(abs(predictions[potato_idx] - predictions[tomato_idx])),
        }
    return confusion


def apply_tta(image_path: str | Path, model: tf.keras.Model) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    transforms = [
        lambda img: img,
        lambda img: ImageOps.mirror(img),
        lambda img: ImageOps.flip(img),
    ]
    results = []
    for transform in transforms:
        array = preprocess_pil_image(transform(image))
        results.append(predict_probabilities(model, array))
    return np.mean(results, axis=0)


def validate_class_alignment(class_names: list[str], model: tf.keras.Model) -> None:
    output_dim = int(model.output_shape[-1])
    if len(class_names) != output_dim:
        raise ValueError(
            f"Mismatch: model output dim {output_dim} != class_names.json length {len(class_names)}"
        )


def predict_image(image_path: str | Path, use_tta: bool = False) -> dict[str, Any]:
    class_names = load_class_names()
    model = load_model()
    validate_class_alignment(class_names, model)

    predictions = apply_tta(image_path, model) if use_tta else predict_probabilities(model, preprocess_image(image_path))
    top_predictions = get_top_k(predictions, class_names, k=TOP_K)
    ambiguous = is_ambiguous(predictions)
    confusion_details = detect_cross_plant_confusion(predictions, class_names)

    best_label, best_confidence, best_index = top_predictions[0]
    predicted_label = "Ambiguous Prediction" if ambiguous else best_label

    return {
        "predicted_class": predicted_label,
        "predicted_index": best_index,
        "confidence": best_confidence,
        "score": float(best_confidence),
        "ambiguous": ambiguous,
        "confidence_gap": float(top_predictions[0][1] - top_predictions[1][1]) if len(top_predictions) > 1 else 1.0,
        "top_predictions": top_predictions,
        "distribution": {class_names[i]: float(predictions[i]) for i in range(len(class_names))},
        "cross_plant_confusion": confusion_details,
    }


def print_prediction_report(result: dict[str, Any]) -> None:
    print("Prediction report:\n")
    print(f"Predicted class: {result['predicted_class']}")
    print(f"Predicted index: {result['predicted_index']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Ambiguous: {result['ambiguous']}")
    print(f"Top-1 vs Top-2 gap: {result['confidence_gap']:.4f}\n")

    print("Top predictions:")
    for label, score, idx in result["top_predictions"]:
        print(f"  {idx}: {label} -> {score:.4f}")

    print("\nCross-plant confusion candidates:")
    for pair, stats in result["cross_plant_confusion"].items():
        print(
            f"  {pair}: potato_prob={stats['potato_prob']:.4f}, "
            f"tomato_prob={stats['tomato_prob']:.4f}, gap={stats['gap']:.4f}"
        )

    print("\nFull probability distribution:")
    for label, score in sorted(result["distribution"].items(), key=lambda item: item[1], reverse=True):
        print(f"  {label}: {score:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plant disease prediction with top-k, ambiguity detection, and optional TTA."
    )
    parser.add_argument("image_path", help="Path to the leaf image to classify.")
    parser.add_argument("--tta", action="store_true", help="Enable test-time augmentation for prediction stability.")
    parser.add_argument("--topk", type=int, default=TOP_K, help="Number of top predictions to show.")
    args = parser.parse_args()

    result = predict_image(args.image_path, use_tta=args.tta)
    print_prediction_report(result)


if __name__ == "__main__":
    main()
