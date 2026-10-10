import json
import os
from pathlib import Path
from datetime import datetime
from typing import Any

import cv2
import numpy as np
import requests
import streamlit as st
import tensorflow as tf
import tempfile
from gtts import gTTS
from deep_translator import GoogleTranslator
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'plant_model.h5'
CLASS_NAMES_PATH = BASE_DIR / 'class_names.json'
NEW_DATA_DIR = BASE_DIR / 'new_data'
HISTORY_DIR = BASE_DIR / 'history'
HISTORY_IMAGE_DIR = HISTORY_DIR / 'images'
HISTORY_FILE = HISTORY_DIR / 'history.json'
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY', '')
IMAGE_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.7

NEW_DATA_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

LANGUAGE_MAP = {
    'English': 'en',
    'Kannada': 'kn',
    'Tamil': 'ta',
    'Hindi': 'hi',
}

TRANSLATIONS = {
    'en': {
        'app_title': '🌿 Smart Plant Leaf Disease Assistant',
        'app_subtitle': 'Upload a leaf image to diagnose disease, view severity, and receive intelligent treatment advice.',
        'upload_label': 'Upload a leaf image (JPG, JPEG or PNG)',
        'weather_prompt': 'Location (city or leave blank to auto-detect)',
        'weather_button': 'Fetch Weather',
        'weather_title': 'Weather-Based Intelligence',
        'history_title': 'Prediction History',
        'saved_for_learning': 'Saved for future model improvement.',
        'prediction_title': 'Prediction Result',
        'details_title': 'Disease Details',
        'solution_title': 'Smart Treatment System',
        'voice_button': '🔊 Speak Result',
        'audio_language_label': 'Audio Language',
        'audio_language_note': 'Choose the voice language for spoken results.',
        'error_image': '⚠️ Failed to process the uploaded image. Please upload a valid leaf photo.',
        'error_model': '⚠️ Model load failed. Please ensure plant_model.h5 exists.',
        'error_weather_key': 'OpenWeather API key is missing. Set OPENWEATHER_API_KEY in environment variables.',
        'error_weather': 'Unable to fetch weather. Please check your internet connection or location text.',
        'no_history': 'No saved history yet.',
        'healthy_message': 'Healthy plant detected. Maintain balanced care and monitor regularly.',
        'uncertain_message': 'Prediction uncertain. Try a different photo in better light.',
        'weather_condition': 'Condition',
        'weather_temp': 'Temperature',
        'weather_humidity': 'Humidity',
        'weather_advice': 'Weather-aware advice',
        'severity_low': 'Low severity',
        'severity_medium': 'Medium severity',
        'severity_high': 'High severity',
        'heatmap_caption': 'Grad-CAM heatmap highlights infected regions',
    },
    'kn': {
        'app_title': '🌿 ಸ್ಮಾರ್ಟ್ ಸಸ್ಯ ರೋಗ ಸಹಾಯಕ',
        'app_subtitle': 'ರೋಗವನ್ನು ನಿರ್ಣಯಿಸಲು ಮತ್ತು ಚಿಕಿತ್ಸೆ ಸಲಹೆಗಳನ್ನು ಪಡೆಯಲು ಎಲೆ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ.',
        'upload_label': 'ಎಲೆ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ (JPG, JPEG ಅಥವಾ PNG)',
        'weather_prompt': 'ಸ್ಥಳ (ನಗರ ಅಥವಾ ಖಾಲಿ bırakಲು ಸ್ವಯಂ ಪತ್ತೆ)',
        'weather_button': 'ಹವಾಮಾನ ಪಡೆಯಿ',
        'weather_title': 'ಹವಾಮಾನ ಆಧಾರಿತ ಬುದ್ಧಿವಂತಿಕೆ',
        'history_title': 'ಭವಿಷ್ಯವಾಣಿಯ ಇತಿಹಾಸ',
        'saved_for_learning': 'ಭವಿಷ್ಯ retraining ಗಾಗಿ ಉಳಿಸಲಾಗಿದೆ.',
        'prediction_title': 'ಭವಿಷ್ಯವಾಣಿ ಫಲಿತಾಂಶ',
        'details_title': 'ರೋಗದ ವಿವರಗಳು',
        'solution_title': 'ಚಿಕಿತ್ಸಾ ವ್ಯವಸ್ಥೆ',
        'voice_button': '🔊 ಫಲಿತಾಂಶವನ್ನು ಮಾತನಾಡಿ',
        'audio_language_label': 'ಧ್ವನಿಯ ಭಾಷೆ',
        'audio_language_note': 'ಮಾತಿನ ಫಲಿತಾಂಶಕ್ಕಾಗಿ ಧ್ವನಿ ಭಾಷೆಯನ್ನು ಆಯ್ಕೆ ಮಾಡಿ.',
        'error_image': '⚠️ ಅಪ್‌ಲೋಡ್ ಮಾಡಿದ ಚಿತ್ರವನ್ನು ಪ್ರಕ್ರಿಯೆ ಮಾಡಲು ವಿಫಲವಾಯಿತು. ದಯವಿಟ್ಟು ಮಾನ್ಯ ಪಟ್ಟಿ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ.',
        'error_model': '⚠️ ಮಾದರಿ ಲೋಡ್ ಆಗಲಿಲ್ಲ. ದಯವಿಟ್ಟು plant_model.h5 ಇದ್ದೇ ಇರುವುದನ್ನು ಪರಿಶೀಲಿಸಿ.',
        'error_weather_key': 'OpenWeather API ಕೀ ಇಲ್ಲ. OPENWEATHER_API_KEY ಅನ್ನು ವಾತಾವರಣ ವ್ಯತ್ಯಯಗಳಲ್ಲಿ ನೆಡಿಸಿ.',
        'error_weather': 'ಹವಾಮಾನ ಪಡೆಯಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಇಂಟರ್ನೆಟ್ ಸಂಪರ್ಕ ಅಥವಾ ಸ್ಥಳವನ್ನು ಪರಿಶೀಲಿಸಿ.',
        'no_history': 'ಇನ್ನೂ ಇತಿಹಾಸ محفوظವಾಗಿಲ್ಲ.',
        'healthy_message': 'ಆರೋಗ್ಯಕರ ಸಸ್ಯ ಕಂಡುಹಿಡಿದಿದೆ. ಸಮತೋಲಿತ ಕಾಳಜಿ ಕಾಪಾಡಿ.',
        'uncertain_message': 'ಭವಿಷ್ಯವಾಣಿ ಅನಿಶ್ಚಿತವಾಗಿದೆ. ಉತ್ತಮ ಬೆಳಕಿನಲ್ಲಿ ಮತ್ತೊಂದು ಚಿತ್ರವನ್ನು ಪ್ರಯತ್ನಿಸಿ.',
        'weather_condition': 'ಸ್ಥಿತಿ',
        'weather_temp': 'ತಾಪಮಾನ',
        'weather_humidity': 'ಆರ್ಧತೆ',
        'weather_advice': 'ಹವಾಮಾನ ಆಧಾರಿತ ಸಲಹೆ',
        'severity_low': 'ಕಡಿಮೆ ತೀವ್ರತೆ',
        'severity_medium': 'ಮಧ್ಯಮ ತೀವ್ರತೆ',
        'severity_high': 'ಉಚ್ಚ ತೀವ್ರತೆ',
        'heatmap_caption': 'Grad-CAM ಚಿಹ್ನೆಗಳು ರೋಗಿತ ಪ್ರದೇಶಗಳನ್ನು ಹೈಲೈಟ್ ಮಾಡುತ್ತವೆ',
    },
    'ta': {
        'app_title': '🌿 ஸ்மார்ட் செடி நோய் உதவியாளர்',
        'app_subtitle': 'இலையை பதிவேற்றி நோயை கண்டறிந்து சிகிச்சை அறிவுரைகளை பெறுங்கள்.',
        'upload_label': 'இலையை பதிவேற்றவும் (JPG, JPEG அல்லது PNG)',
        'weather_prompt': 'இடம் (நகரம் அல்லது காலியாக விடவும்)',
        'weather_button': 'பயன்மாலை பெறுக',
        'weather_title': 'வானிலை சார்ந்த அறிவு',
        'history_title': 'கணிப்பு வரலாறு',
        'saved_for_learning': 'புத்தம் பயிற்சிக்காக சேமிக்கப்பட்டது.',
        'prediction_title': 'கணிப்பு முடிவு',
        'details_title': 'நோய் விவரங்கள்',
        'solution_title': 'சிகிச்சை அமைப்பு',
        'voice_button': '🔊 முடிவை பேசுக',
        'audio_language_label': 'ஆடியோ மொழி',
        'audio_language_note': 'பேச்சு முடிவிற்கான மொழியைத் தேர்ந்தெடுக்கவும்.',
        'error_image': '⚠️ பதிவேற்றப்பட்ட படத்தை செயலாக்க முடியவில்லை. சரியான படம் பதிவேற்றவும்.',
        'error_model': '⚠️ மாதிரி ஏற்ற முடியவில்லை. plant_model.h5 இருக்கும் என்பதை உறுதிப்படுத்தவும்.',
        'error_weather_key': 'OpenWeather API விசை இல்லை. OPENWEATHER_API_KEY ஐ அமைக்கவும்.',
        'error_weather': 'வானிலை பெற முடியவில்லை. இணைய மற்றும் இடத்தைக் சரிபார்க்கவும்.',
        'no_history': 'இன்னும் வரலாறு இல்லை.',
        'healthy_message': 'ஆரோக்கியமான செடி கண்டறிந்து விட்டது. பராமரிப்பை தொடரவும்.',
        'uncertain_message': 'கணிப்பு உறுதியற்றது. மற்றொரு படத்தை முயற்சிக்கவும்.',
        'weather_condition': 'நிலை',
        'weather_temp': 'வெப்பநிலை',
        'weather_humidity': ' ஈரப்பதம்',
        'weather_advice': 'வானிலை அறிவுரை',
        'severity_low': 'குறைந்த தீவிரம்',
        'severity_medium': 'மத்திய தீவிரம்',
        'severity_high': 'உயர் தீவிரம்',
        'heatmap_caption': 'Grad-CAM வைரலாகக் காண்பிக்கும் பகுதி',
    },
    'hi': {
        'app_title': '🌿 स्मार्ट पौधा रोग सहायक',
        'app_subtitle': 'रोग का निदान करने और उपचार सलाह प्राप्त करने के लिए पत्ती की छवि अपलोड करें।',
        'upload_label': 'पत्ती की छवि अपलोड करें (JPG, JPEG या PNG)',
        'weather_prompt': 'स्थान (शहर या खाली छोड़ें)',
        'weather_button': 'मौसम प्राप्त करें',
        'weather_title': 'मौसम आधारित बुद्धिमत्ता',
        'history_title': 'पूर्वानुमान इतिहास',
        'saved_for_learning': 'भविष्य के मॉडल सुधार के लिए सहेजा गया।',
        'prediction_title': 'पूर्वानुमान परिणाम',
        'details_title': 'रोग विवरण',
        'solution_title': 'उपचार प्रणाली',
        'voice_button': '🔊 परिणाम बोलें',
        'audio_language_label': 'ऑडियो भाषा',
        'audio_language_note': 'बोली जाने वाली भाषा चुनें।',
        'error_image': '⚠️ अपलोड की गई छवि को संसाधित करने में विफल। कृपया मान्य पत्ती फोटो अपलोड करें।',
        'error_model': '⚠️ मॉडल लोड विफल। कृपया सुनिश्चित करें कि plant_model.h5 मौजूद है।',
        'error_weather_key': 'OpenWeather API कुंजी गायब है। OPENWEATHER_API_KEY सेट करें।',
        'error_weather': 'मौसम प्राप्त करने में असमर्थ। कृपया इंटरनेट अथवा स्थान जाँचें।',
        'no_history': 'कोई इतिहास सहेजा नहीं गया।',
        'healthy_message': 'स्वस्थ पौधा पाया गया। देखभाल जारी रखें।',
        'uncertain_message': 'पूर्वानुमान अनिश्चित है। एक अलग फ़ोटो आज़माएँ।',
        'weather_condition': 'स्थिति',
        'weather_temp': 'तापमान',
        'weather_humidity': 'आर्द्रता',
        'weather_advice': 'मौसम सूचित सलाह',
        'severity_low': 'कम गंभीरता',
        'severity_medium': 'मध्यम गंभीरता',
        'severity_high': 'उच्च गंभीरता',
        'heatmap_caption': 'Grad-CAM संक्रमित क्षेत्रों को हाइलाइट करता है',
    },
}

DISEASE_INFO = {
    'potato_early_blight': {
        'description': 'Early blight is a common fungal disease that causes dark spots and concentric rings on potato leaves.',
        'cause': 'Moisture-assisted fungal infection spreading through debris and soil.',
        'symptoms': 'Brown spots, yellowing leaves, and circular lesions with target-like rings.',
        'prevention': 'Practice crop rotation, remove debris, and space plants for airflow.',
        'organic': 'Use neem oil spray, copper-based organic fungicides, and compost tea to support plant immunity.',
        'chemical': 'Apply copper hydroxide or mancozeb at first signs of early blight.',
    },
    'potato_late_blight': {
        'description': 'Late blight is a fast-spreading disease that can destroy potato foliage and tubers in wet weather.',
        'cause': 'Cool, humid weather that supports spore propagation.',
        'symptoms': 'Dark brown patches on leaves, white fungal growth on undersides, and rotting tubers.',
        'prevention': 'Avoid overhead irrigation, keep foliage dry, and remove infected plants quickly.',
        'organic': 'Use potassium bicarbonate spray and maintain a dry canopy to slow spread.',
        'chemical': 'Treat with fluopicolide or metalaxyl products immediately.',
    },
    'tomato_early_blight': {
        'description': 'Early blight attacks tomato leaves and stems, causing leaf death and reduced fruit yield.',
        'cause': 'Fungal spores carried by wind and splashing water.',
        'symptoms': 'Dark spots with concentric rings on leaves, yellowing, and leaf drop.',
        'prevention': 'Rotate crops, mulch soil, and prune dense foliage.',
        'organic': 'Spray baking soda solution, garlic extract, or neem mixtures.',
        'chemical': 'Use chlorothalonil or copper-based fungicide as needed.',
    },
    'tomato_late_blight': {
        'description': 'Late blight is a destructive disease for tomatoes in cool, damp conditions.',
        'cause': 'Persistent wet weather and poor air circulation.',
        'symptoms': 'Water-soaked spots, fuzzy white growth, and rapid leaf collapse.',
        'prevention': 'Stake plants, remove lower leaves, and improve drainage.',
        'organic': 'Use sulfur dust and homemade neem spray to reduce spore spread.',
        'chemical': 'Apply a systemic fungicide like Ridomil Gold promptly.',
    },
    'potato_healthy': {
        'description': 'Healthy potato foliage with no visible signs of disease.',
        'cause': 'No disease detected.',
        'symptoms': 'Uniform green leaves, strong stems, and no lesions.',
        'prevention': 'Maintain balanced watering, nutrients, and crop rotation.',
        'organic': 'Continue compost mulching and beneficial insect support.',
        'chemical': 'No chemical treatment needed.',
    },
    'tomato_healthy': {
        'description': 'Healthy tomato plants with vibrant leaves and strong fruit set.',
        'cause': 'No disease detected.',
        'symptoms': 'Uniform leaf color, active growth, and no leaf spots.',
        'prevention': 'Keep soil moist, prune regularly, and monitor daily.',
        'organic': 'Use compost tea and natural predators for pest control.',
        'chemical': 'No chemical treatment needed.',
    },
    'Uncertain': {
        'description': 'The model could not confidently classify this leaf image.',
        'cause': 'Low confidence due to image clarity or lighting.',
        'symptoms': 'Blurry or poorly lit leaves, partial leaf coverage, or unusual angles.',
        'prevention': 'Upload a sharper image with uniform lighting.',
        'organic': 'Capture another photo in daylight conditions.',
        'chemical': 'N/A',
    },
}

STYLE_CSS = '''
body {
    background: linear-gradient(135deg, #1e8f5f, #2263b8);
    color: #f4f7f4;
    font-family: 'Segoe UI', sans-serif;
}
.stApp {
    background: transparent;
}
.css-1y4p8pa, .css-1d391kg, .css-18e3th9 {
    background: transparent;
}
.white-card {
    background: #ffffff;
    color: #1f3d2d;
    border-radius: 28px;
    padding: 28px;
    box-shadow: 0 24px 70px rgba(0, 0, 0, 0.18);
    margin-bottom: 24px;
}
.top-card {
    background: rgba(255, 255, 255, 0.14);
    border: 1px solid rgba(255, 255, 255, 0.18);
    border-radius: 28px;
    padding: 32px;
    box-shadow: 0 24px 70px rgba(0, 0, 0, 0.22);
    margin-bottom: 24px;
}
.section-title {
    color: #0e422b;
    font-size: 2rem;
    font-weight: 900;
    margin-bottom: 0.75rem;
}
.card-title {
    color: #134534;
    font-size: 1.5rem;
    font-weight: 800;
    margin-bottom: 0.75rem;
}
.card-subtitle {
    color: #3d5f4f;
    font-size: 1rem;
    margin-bottom: 18px;
}
.metric-box {
    background: #f3f9f4;
    border: 1px solid rgba(32, 84, 42, 0.16);
    border-radius: 22px;
    padding: 20px;
    text-align: left;
    margin-bottom: 16px;
}
.metric-title {
    color: #1a472f;
    font-size: 1rem;
    font-weight: 800;
    margin-bottom: 8px;
}
.metric-value {
    color: #17442e;
    font-size: 1.8rem;
    font-weight: 900;
}
.small-note {
    color: #4d6e5a;
    font-size: 0.95rem;
    line-height: 1.6;
}
.sidebar-title {
    color: #ffffff;
    font-size: 1.2rem;
    font-weight: 800;
    margin-bottom: 12px;
}
.history-card {
    background: rgba(255, 255, 255, 0.14);
    border: 1px solid rgba(255, 255, 255, 0.18);
    border-radius: 22px;
    padding: 16px;
    margin-bottom: 14px;
}
.button-card button {
    background: linear-gradient(135deg, #47b17a, #205dba) !important;
    color: #ffffff !important;
    font-weight: 800 !important;
}
'''  


def set_custom_styles():
    st.markdown(f'<style>{STYLE_CSS}</style>', unsafe_allow_html=True)


def translate_phrase(key: str, lang_code: str) -> str:
    return TRANSLATIONS.get(lang_code, TRANSLATIONS['en']).get(key, TRANSLATIONS['en'].get(key, key))


def translate_text(text: str, lang_code: str) -> str:
    if lang_code == 'en' or not text:
        return text
    try:
        return GoogleTranslator(source='en', target=lang_code).translate(text)
    except Exception:
        return text


@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    except Exception:
        return None


@st.cache_data
def load_class_names():
    if not CLASS_NAMES_PATH.exists():
        return []
    with open(CLASS_NAMES_PATH, 'r', encoding='utf-8') as file:
        return json.load(file)


def preprocess_image(_image: Image.Image) -> np.ndarray:
    image = _image.convert('RGB')
    image = image.resize(IMAGE_SIZE)
    img = np.array(image)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def predict(image_array: np.ndarray) -> tuple[str, float]:
    model = load_model()
    class_names = load_class_names()
    if model is None:
        raise RuntimeError('Model failed to load.')
    if not class_names:
        raise RuntimeError('Class names failed to load.')
    predictions = model.predict(image_array, verbose=0)
    if predictions is None or predictions.size == 0:
        raise RuntimeError('Model returned no prediction.')
    print("=" * 60)
    print("PREDICTION DEBUG INFO")
    print("=" * 60)
    print(f"Raw prediction vector shape: {predictions.shape}")
    print(f"Raw prediction vector: {predictions}")
    print(f"Class names loaded: {class_names}")
    predicted_index = int(np.argmax(predictions[0]))
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[0][predicted_index])
    print(f"Predicted index: {predicted_index}")
    print(f"Predicted class (raw): {predicted_class}")
    print(f"Confidence (0-1): {confidence:.4f}")
    # Normalize to lowercase with underscores for DISEASE_INFO lookup
    normalized_class = predicted_class.lower()
    print(f"Predicted class (normalized): {normalized_class}")
    print("=" * 60)
    return normalized_class, confidence


@st.cache_resource
def load_translator():
    return GoogleTranslator(source='en', target='en')


def get_severity(confidence: float) -> tuple[str, str]:
    score = confidence * 100
    if score > 90:
        return 'High', '#ff4c4c'
    if score >= 70:
        return 'Medium', '#f5c644'
    return 'Low', '#b0b0b0'


def find_last_conv_layer(model: tf.keras.Model) -> str | None:
    for layer in reversed(model.layers):
        if 'conv' in layer.name and len(layer.output_shape) == 4:
            return layer.name
    return None


def generate_gradcam(image_array: np.ndarray, model: tf.keras.Model, class_index: int) -> np.ndarray | None:
    last_conv_name = find_last_conv_layer(model)
    if last_conv_name is None:
        return None
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array)
        loss = predictions[:, class_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1).numpy()
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) == 0:
        return None
    heatmap /= np.max(heatmap)
    return heatmap


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray) -> Image.Image:
    heatmap = np.uint8(255 * heatmap)
    heatmap = np.dstack([heatmap, np.zeros_like(heatmap), np.zeros_like(heatmap), heatmap])
    heatmap_img = Image.fromarray(heatmap, mode='RGBA').resize(image.size)
    overlay = Image.alpha_composite(image.convert('RGBA'), heatmap_img)
    return overlay


def detect_infected_region(image: Image.Image) -> tuple[float, str]:
    rgb = np.array(image)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask_yellow = cv2.inRange(hsv, np.array([5, 50, 50]), np.array([50, 255, 255]))
    mask_brown = cv2.inRange(hsv, np.array([0, 20, 20]), np.array([20, 255, 200]))
    mask = cv2.bitwise_or(mask_yellow, mask_brown)
    ratio = float(np.count_nonzero(mask)) / float(mask.size)
    if ratio > 0.18:
        return ratio, 'High'
    if ratio > 0.08:
        return ratio, 'Medium'
    return ratio, 'Low'


def get_weather(city: str) -> dict[str, Any] | None:
    if not OPENWEATHER_API_KEY:
        return None
    try:
        params = {'q': city, 'appid': OPENWEATHER_API_KEY, 'units': 'metric'}
        response = requests.get('https://api.openweathermap.org/data/2.5/weather', params=params, timeout=8)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def auto_detect_location() -> str | None:
    try:
        response = requests.get('https://ipinfo.io/json', timeout=6)
        response.raise_for_status()
        return response.json().get('city')
    except Exception:
        return None


def weather_advice(class_name: str, weather: dict[str, Any], lang_code: str) -> str:
    condition = weather.get('weather', [{}])[0].get('main', '')
    temp = weather.get('main', {}).get('temp', 0)
    humidity = weather.get('main', {}).get('humidity', 0)
    advice = []
    if condition.lower() in ['rain', 'drizzle', 'thunderstorm']:
        advice.append('High moisture supports fungal growth; reduce overhead irrigation.')
    if humidity and humidity > 80:
        advice.append('Humidity is high; improve airflow around leaves.')
    if temp and temp > 30:
        advice.append('Warm weather detected; shade tender foliage and keep soil evenly moist.')
    if 'late_blight' in class_name:
        advice.append('Late blight is aggressive; remove infected tissue immediately.')
    if 'early_blight' in class_name:
        advice.append('Early blight can spread quickly; apply preventive sprays.')
    if 'healthy' in class_name.lower():
        advice.append('Plant looks healthy; continue weather-aware maintenance.')
    return translate_text(' '.join(advice) or 'No weather-driven action is needed at this time.', lang_code)


def save_history(record: dict[str, Any]) -> None:
    history = []
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as file:
                history = json.load(file)
        except Exception:
            history = []
    history.append(record)
    with open(HISTORY_FILE, 'w', encoding='utf-8') as file:
        json.dump(history[-20:], file, ensure_ascii=False, indent=2)


def load_history() -> list[dict[str, Any]]:
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception:
        return []


def save_training_image(image: Image.Image, class_name: str, filename: str) -> Path:
    target_dir = NEW_DATA_DIR / class_name
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / filename
    image.save(path)
    return path


def save_history_image(image: Image.Image, filename: str) -> Path:
    path = HISTORY_IMAGE_DIR / filename
    image.save(path)
    return path


def speak_text(text: str, lang_code: str) -> str | None:
    try:
        tts = gTTS(text=text, lang=lang_code)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as audio_file:
            tts.save(audio_file.name)
            return audio_file.name
    except Exception:
        return None


def main():
    st.set_page_config(page_title='Plant Leaf Disease Detection', page_icon='🌿', layout='wide')
    set_custom_styles()

    language = st.selectbox('Language', list(LANGUAGE_MAP.keys()), index=0)
    lang_code = LANGUAGE_MAP.get(language, 'en')
    audio_language = st.selectbox(translate_phrase('audio_language_label', lang_code), list(LANGUAGE_MAP.keys()), index=list(LANGUAGE_MAP.keys()).index(language))
    audio_lang_code = LANGUAGE_MAP.get(audio_language, 'en')
    st.caption(translate_phrase('audio_language_note', lang_code))

    title = translate_phrase('app_title', lang_code)
    subtitle = translate_phrase('app_subtitle', lang_code)
    upload_label = translate_phrase('upload_label', lang_code)
    weather_prompt = translate_phrase('weather_prompt', lang_code)
    weather_button_label = translate_phrase('weather_button', lang_code)
    weather_title = translate_phrase('weather_title', lang_code)
    history_title = translate_phrase('history_title', lang_code)
    prediction_title = translate_phrase('prediction_title', lang_code)
    details_title = translate_phrase('details_title', lang_code)
    solution_title = translate_phrase('solution_title', lang_code)
    voice_button_label = translate_phrase('voice_button', lang_code)
    saved_message = translate_phrase('saved_for_learning', lang_code)
    error_image = translate_phrase('error_image', lang_code)
    error_model = translate_phrase('error_model', lang_code)
    error_weather_key = translate_phrase('error_weather_key', lang_code)
    error_weather = translate_phrase('error_weather', lang_code)
    no_history = translate_phrase('no_history', lang_code)
    heatmap_caption = translate_phrase('heatmap_caption', lang_code)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(f'<h1 class="section-title">{title}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p>{subtitle}</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    history = load_history()

    sidebar = st.sidebar
    sidebar.markdown(f'<div class="sidebar-title">{history_title}</div>', unsafe_allow_html=True)
    if history:
        for record in reversed(history[-5:]):
            sidebar.markdown(f'**{record["class_name"].replace("_", " ").title()}**')
            sidebar.write(f'{record["confidence"]}% • {record["severity"]}')
            sidebar.write(record['timestamp'])
            sidebar.markdown('---')
    else:
        sidebar.write(no_history)
    sidebar.markdown(f'<div class="sidebar-title">📌 {weather_title}</div>', unsafe_allow_html=True)

    with sidebar.expander(translate_phrase('weather_title', lang_code), expanded=True):
        city_input = sidebar.text_input(weather_prompt)
        if sidebar.button(weather_button_label):
            if not OPENWEATHER_API_KEY:
                sidebar.warning(error_weather_key)
                weather_info = None
            else:
                city = city_input.strip() or auto_detect_location() or ''
                weather_info = get_weather(city) if city else None
                if weather_info is None:
                    sidebar.error(error_weather)
                else:
                    sidebar.success(f'{translate_phrase("weather_condition", lang_code)}: {weather_info.get("weather", [{}])[0].get("description", "—")}')
                    sidebar.write(f'{translate_phrase("weather_temp", lang_code)}: {weather_info.get("main", {}).get("temp", "—")} °C')
                    sidebar.write(f'{translate_phrase("weather_humidity", lang_code)}: {weather_info.get("main", {}).get("humidity", "—")} %')
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(upload_label, type=['jpg', 'jpeg', 'png'])
    st.markdown('</div>', unsafe_allow_html=True)

    weather_info = None
    if sidebar.button(translate_phrase('weather_button', lang_code), key='weather_main'):
        if not OPENWEATHER_API_KEY:
            sidebar.warning(error_weather_key)
        else:
            city = city_input.strip() or auto_detect_location() or ''
            weather_info = get_weather(city) if city else None
            if weather_info is None:
                sidebar.error(error_weather)

    if uploaded_file is not None:
        model = load_model()
        if model is None:
            st.error(error_model)
            return
        try:
            image = Image.open(uploaded_file).convert('RGB')
            image_array = preprocess_image(image)
            class_name, confidence = predict(image_array)
            score = confidence * 100
            details = DISEASE_INFO.get(class_name, DISEASE_INFO['Uncertain'])
            if confidence < 0.5:
                details = {
                    'description': 'Low confidence prediction. Please upload a clearer image for accurate diagnosis.',
                    'symptoms': 'Unable to determine symptoms due to low confidence.',
                    'cause': 'Cause unknown due to low confidence.',
                    'prevention': 'Upload a better image for prevention advice.',
                    'organic': 'Organic treatment unavailable.',
                    'chemical': 'Chemical treatment unavailable.'
                }
            filename = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{Path(uploaded_file.name).stem}.png'
            history_path = save_history_image(image, filename)
            save_training_image(image, 'awaiting_label', filename)
            st.success(saved_message)
            infected_ratio, infected_label = detect_infected_region(image)
            severity_label, severity_color = get_severity(confidence)
            advice_text = weather_advice(class_name, weather_info, lang_code) if weather_info else translate_text('No weather-driven action is needed at this time.', lang_code)
            translated_class = translate_text(class_name.replace('_', ' ').title(), lang_code)
            translated_description = translate_text(details.get('description', ''), lang_code)
            translated_symptoms = translate_text(details.get('symptoms', ''), lang_code)
            translated_cause = translate_text(details.get('cause', ''), lang_code)
            translated_prevention = translate_text(details.get('prevention', ''), lang_code)
            translated_organic = translate_text(details.get('organic', ''), lang_code)
            translated_chemical = translate_text(details.get('chemical', ''), lang_code)

            st.markdown('<div class="white-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="section-title">🌿 {prediction_title}</div>', unsafe_allow_html=True)
            st.markdown(f'<p class="card-subtitle">{translate_text("Clear plant health signals, severity, and treatment guidance in one place.", lang_code)}</p>', unsafe_allow_html=True)
            cols = st.columns([1, 1])
            with cols[0]:
                st.image(image, caption=translate_text('Original Image', lang_code), use_column_width=True)
            with cols[1]:
                st.markdown(f'<div class="metric-box">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-title">{translate_text("Predicted Class", lang_code)}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{translated_class}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-box">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-title">{translate_text("Confidence", lang_code)}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{score:.1f}%</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-box">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-title">⚠️ {translate_text("Severity", lang_code)}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value" style="color:{severity_color};">{translate_phrase(f"severity_{severity_label.lower()}", lang_code)}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-box">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-title">🛡️ {translate_text("Infected Coverage", lang_code)}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{infected_ratio * 100:.1f}%</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="small-note">{translate_text("This score is based on the model confidence and detected leaf condition.", lang_code)}</div>', unsafe_allow_html=True)
                if st.button(voice_button_label):
                    voice_text_en = f'{class_name.replace("_", " ").title()} detected with {score:.1f} percent confidence. {details["description"]} Symptoms: {details["symptoms"]}. Prevention: {details["prevention"]}. Treatment: {details["organic"]}. {details["chemical"]}.'
                    voice_text = translate_text(voice_text_en, audio_lang_code) if audio_lang_code != 'en' else voice_text_en
                    spoken_file = speak_text(voice_text, audio_lang_code)
                    if spoken_file:
                        st.audio(spoken_file, format='audio/mp3')
                    else:
                        st.error(translate_text('Unable to generate voice at this time.', lang_code))
            st.markdown('</div>', unsafe_allow_html=True)

            show_heatmap = st.checkbox(translate_text('Show heatmap visualization', lang_code), value=False)
            if show_heatmap:
                heatmap = generate_gradcam(image_array, model, int(np.argmax(model.predict(image_array, verbose=0)[0])))
                if heatmap is not None:
                    overlay = overlay_heatmap(image, heatmap)
                    st.markdown('<div class="white-card">', unsafe_allow_html=True)
                    st.markdown(f'<div class="card-title">{translate_text("Heatmap Visualization", lang_code)}</div>', unsafe_allow_html=True)
                    st.image(image, caption=translate_text('Original Image', lang_code), use_column_width=True)
                    st.image(overlay, caption=heatmap_caption, use_column_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info(translate_text('Heatmap is unavailable for this model.', lang_code))

            info_cols = st.columns(2)
            with info_cols[0]:
                st.markdown('<div class="white-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="card-title">🪴 {translate_text("About Disease", lang_code)}</div>', unsafe_allow_html=True)
                st.markdown(f'<p class="small-note"><strong>{translate_text("Description:", lang_code)}</strong> {translated_description}</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="small-note"><strong>{translate_text("Cause:", lang_code)}</strong> {translated_cause}</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('<div class="white-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="card-title">⚠️ {translate_text("Symptoms", lang_code)}</div>', unsafe_allow_html=True)
                st.markdown(f'<p class="small-note">{translated_symptoms}</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with info_cols[1]:
                st.markdown('<div class="white-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="card-title">🛡️ {translate_text("Prevention", lang_code)}</div>', unsafe_allow_html=True)
                st.markdown(f'<p class="small-note">{translated_prevention}</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('<div class="white-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="card-title">💊 {translate_text("Treatment", lang_code)}</div>', unsafe_allow_html=True)
                st.markdown(f'<p class="small-note"><strong>{translate_text("Organic:", lang_code)}</strong> {translated_organic}</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="small-note"><strong>{translate_text("Chemical:", lang_code)}</strong> {translated_chemical}</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="white-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="card-title">💡 {solution_title}</div>', unsafe_allow_html=True)
            st.markdown(f'<p class="small-note">{advice_text}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            record = {
                'timestamp': datetime.now().isoformat(),
                'filename': filename,
                'class_name': translated_class,
                'confidence': round(score, 2),
                'severity': translate_phrase(f'severity_{severity_label.lower()}', lang_code),
                'image_path': str(history_path.relative_to(BASE_DIR)),
            }
            save_history(record)

        except Exception as e:
            st.error(f"{error_image} {e}")
            st.exception(e)

if __name__ == '__main__':
    main()
