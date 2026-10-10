import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from gtts import gTTS
import os
import tempfile

IMAGE_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.70
AMBIGUITY_MARGIN = 0.10
TOP_K = 3

# Configuration
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'FINAL_MODEL.h5'
CLASS_NAMES_PATH = BASE_DIR / 'class_names.json'
OUTPUT_AUDIO_PATH = BASE_DIR / 'output.mp3'
IMAGE_SIZE = (224, 224)

# Disease Info
DISEASE_INFO = {
    'potato_early_blight': {
        'description': 'Early blight on potato causes brown spots and concentric rings on leaves.',
        'symptoms': 'Brown lesions, yellowing leaf edges, and leaf drop.',
        'cause': 'Fungal spores spread by water splash and infected plant debris.',
        'prevention': 'Rotate crops, remove infected foliage, and avoid overhead watering.',
        'treatment': 'Use copper-based fungicides or neem oil and improve air circulation.',
    },
    'potato_late_blight': {
        'description': 'Late blight on potato is a fast-moving disease that leads to blighted foliage and tuber rot.',
        'symptoms': 'Dark watery spots, white mold beneath leaves, and rapid plant collapse.',
        'cause': 'Cool, humid weather that favors Phytophthora infestans.',
        'prevention': 'Keep foliage dry, remove infected plants, and use resistant varieties.',
        'treatment': 'Apply approved late blight fungicides and destroy infected material.',
    },
    'potato_healthy': {
        'description': 'Healthy potato leaves showing no signs of disease.',
        'symptoms': 'Uniform green foliage, strong stems, and healthy growth.',
        'cause': 'No disease detected.',
        'prevention': 'Maintain balanced nutrition, consistent watering, and good airflow.',
        'treatment': 'No treatment needed.',
    },
    'tomato_early_blight': {
        'description': 'Early blight attacks tomato leaves and stems, causing leaf death and reduced fruit yield.',
        'symptoms': 'Dark spots with concentric rings on leaves, yellowing, and leaf drop.',
        'cause': 'Fungal spores carried by wind and splashing water.',
        'prevention': 'Rotate crops, mulch soil, and prune dense foliage.',
        'treatment': 'Use chlorothalonil or copper-based fungicide as needed.',
    },
    'tomato_healthy': {
        'description': 'Healthy tomato plants with vibrant leaves and strong fruit set.',
        'symptoms': 'Uniform leaf color, active growth, and no leaf spots.',
        'cause': 'No disease detected.',
        'prevention': 'Keep soil moist, prune regularly, and monitor daily.',
        'treatment': 'No treatment needed. Continue regular care.',
    },
    'tomato_late_blight': {
        'description': 'Late blight is a destructive disease for tomatoes in cool, damp conditions.',
        'symptoms': 'Water-soaked spots, fuzzy white growth, and rapid leaf collapse.',
        'cause': 'Persistent wet weather and poor air circulation.',
        'prevention': 'Stake plants, remove lower leaves, and improve drainage.',
        'treatment': 'Apply a systemic fungicide like Ridomil Gold promptly.',
    },
}


# Multi-language translations
TRANSLATIONS = {
    'en': {
        'title': '🌿 Plant Leaf Disease Detection',
        'subtitle': 'Upload a plant leaf image to get instant disease diagnosis and treatment advice.',
        'status': '✅ Model Status: Using FINAL_MODEL.h5 (94.6% Validation Accuracy) | Classes: All 6 Plant Diseases',
        'upload': 'Choose a leaf image',
        'detected_disease': 'Detected Disease',
        'confidence': 'Confidence Score',
        'severity': 'Severity',
        'very_high': '🟢 Very High Confidence',
        'high': '🟡 High Confidence',
        'moderate': '🟠 Moderate Confidence',
        'low': '🔴 Low Confidence',
        'low_confidence_warning': '⚠️ Low confidence! The model is uncertain. Please upload a clearer image.',
        'ambiguous_warning': '⚠️ Ambiguous prediction. Multiple diseases have similar scores. Check top predictions below.',
        'confident': '✅ Prediction confident and clear!',
        'speak_result': '🔊 Speak Result',
        'detailed_analysis': '📊 Detailed Prediction Analysis',
        'top_predictions': 'Top 3 Predictions:',
        'full_distribution': 'All Class Probabilities:',
        'gap': 'Top-1 vs Top-2 Gap:',
        'disease_info': '📋 Disease Information',
        'description': 'Description',
        'symptoms': 'Symptoms',
        'cause': 'Cause',
        'prevention': 'Prevention',
        'treatment': 'Treatment',
        'no_info': 'ℹ️ No detailed information available for this prediction.',
        'classes': {
            'Potato_Early_Blight': 'Potato Early Blight',
            'Potato_Healthy': 'Potato Healthy',
            'Potato_Late_Blight': 'Potato Late Blight',
            'Tomato_Early_Blight': 'Tomato Early Blight',
            'Tomato_Healthy': 'Tomato Healthy',
            'Tomato_Late_Blight': 'Tomato Late Blight'
        },
        'diseases': {
            'potato_early_blight': {
                'description': 'Early blight on potato causes brown spots and concentric rings on leaves.',
                'symptoms': 'Brown lesions, yellowing leaf edges, and leaf drop.',
                'cause': 'Fungal spores spread by water splash and infected plant debris.',
                'prevention': 'Rotate crops, remove infected foliage, and avoid overhead watering.',
                'treatment': 'Use copper-based fungicides or neem oil and improve air circulation.',
            },
            'potato_late_blight': {
                'description': 'Late blight on potato is a fast-moving disease that leads to blighted foliage and tuber rot.',
                'symptoms': 'Dark watery spots, white mold beneath leaves, and rapid plant collapse.',
                'cause': 'Cool, humid weather that favors Phytophthora infestans.',
                'prevention': 'Keep foliage dry, remove infected plants, and use resistant varieties.',
                'treatment': 'Apply approved late blight fungicides and destroy infected material.',
            },
            'potato_healthy': {
                'description': 'Healthy potato leaves showing no signs of disease.',
                'symptoms': 'Uniform green foliage, strong stems, and healthy growth.',
                'cause': 'No disease detected.',
                'prevention': 'Maintain balanced nutrition, consistent watering, and good airflow.',
                'treatment': 'No treatment needed.',
            },
            'tomato_early_blight': {
                'description': 'Early blight attacks tomato leaves and stems, causing leaf death and reduced fruit yield.',
                'symptoms': 'Dark spots with concentric rings on leaves, yellowing, and leaf drop.',
                'cause': 'Fungal spores carried by wind and splashing water.',
                'prevention': 'Rotate crops, mulch soil, and prune dense foliage.',
                'treatment': 'Use chlorothalonil or copper-based fungicide as needed.',
            },
            'tomato_healthy': {
                'description': 'Healthy tomato plants with vibrant leaves and strong fruit set.',
                'symptoms': 'Uniform leaf color, active growth, and no leaf spots.',
                'cause': 'No disease detected.',
                'prevention': 'Keep soil moist, prune regularly, and monitor daily.',
                'treatment': 'No treatment needed. Continue regular care.',
            },
            'tomato_late_blight': {
                'description': 'Late blight is a destructive disease for tomatoes in cool, damp conditions.',
                'symptoms': 'Water-soaked spots, fuzzy white growth, and rapid leaf collapse.',
                'cause': 'Persistent wet weather and poor air circulation.',
                'prevention': 'Stake plants, remove lower leaves, and improve drainage.',
                'treatment': 'Apply a systemic fungicide like Ridomil Gold promptly.',
            },
        }
    },
    'ta': {  # Tamil
        'title': '🌿 தாவர இலை நோய் கண்டறிதல்',
        'subtitle': 'தாவர இலையின் படத்தை பதிவேற்றி உடனடியாக நோய் கண்டறிதல் மற்றும் சிகிச்சை ஆலோசனை பெறுங்கள்.',
        'status': '✅ மாடல் நிலை: FINAL_MODEL.h5 (94.6% சரிபார்ப்பு துல்லியம்) | வகுப்புகள்: அனைத்து 6 தாவர நோய்கள்',
        'upload': 'இலை படத்தை தேர்ந்தெடுக்கவும்',
        'detected_disease': 'கண்டறியப்பட்ட நோய்',
        'confidence': 'நம்பிக்கை மதிப்பு',
        'severity': 'தீவிரம்',
        'very_high': '🟢 மிகவும் அதிக நம்பிக்கை',
        'high': '🟡 அதிக நம்பிக்கை',
        'moderate': '🟠 நடுத்தர நம்பிக்கை',
        'low': '🔴 குறைந்த நம்பிக்கை',
        'low_confidence_warning': '⚠️ குறைந்த நம்பிக்கை! மாடல் நிச்சயமற்றது. தெளிவான படத்தை பதிவேற்றவும்.',
        'ambiguous_warning': '⚠️ தெளிவில்லாத கணிப்பு. பல நோய்களுக்கு ஒத்த மதிப்புகள் உள்ளன. மேல் கணிப்புகளை சரிபார்க்கவும்.',
        'confident': '✅ கணிப்பு நம்பிக்கையானது மற்றும் தெளிவானது!',
        'speak_result': '🔊 முடிவை பேசு',
        'detailed_analysis': '📊 விரிவான கணிப்பு பகுப்பாய்வு',
        'top_predictions': 'மேல் 3 கணிப்புகள்:',
        'full_distribution': 'அனைத்து வகுப்பு நிகழ்தகவுகள்:',
        'gap': 'மேல்-1 vs மேல்-2 இடைவெளி:',
        'disease_info': '📋 நோய் தகவல்',
        'description': 'விளக்கம்',
        'symptoms': 'அறிகுறிகள்',
        'cause': 'காரணம்',
        'prevention': 'தடுப்பு',
        'treatment': 'சிகிச்சை',
        'no_info': 'ℹ️ இந்த கணிப்புக்கு விரிவான தகவல் இல்லை.',
        'classes': {
            'Potato_Early_Blight': 'உருளைக்கிழங்கு ஆரம்ப நோய்',
            'Potato_Healthy': 'உருளைக்கிழங்கு ஆரோக்கியமானது',
            'Potato_Late_Blight': 'உருளைக்கிழங்கு தாமத நோய்',
            'Tomato_Early_Blight': 'தக்காளி ஆரம்ப நோய்',
            'Tomato_Healthy': 'தக்காளி ஆரோக்கியமானது',
            'Tomato_Late_Blight': 'தக்காளி தாமத நோய்'
        },
        'diseases': {
            'potato_early_blight': {
                'description': 'உருளைக்கிழங்கில் ஆரம்ப நோய் இலைகளில் பழுப்பு நிற புள்ளிகள் மற்றும் ஒத்த வளையங்களை ஏற்படுத்துகிறது.',
                'symptoms': 'பழுப்பு நிற காயங்கள், மஞ்சள் இலை விளிம்புகள், மற்றும் இலை விழுதல்.',
                'cause': 'பூஞ்சை ஸ்போர்கள் தண்ணீர் தெளிப்பு மற்றும் பாதிக்கப்பட்ட தாவர கழிவுகளால் பரவுகின்றன.',
                'prevention': 'பயிர்களை சுழற்றவும், பாதிக்கப்பட்ட இலைகளை அகற்றவும், மேல் தண்ணீர் தெளிப்பைத் தவிர்க்கவும்.',
                'treatment': 'காப்பர் அடிப்படையிலான பூஞ்சைக்கொல்லிகள் அல்லது நீம் எண்ணெய் பயன்படுத்தி காற்று சுழற்சியை மேம்படுத்தவும்.',
            },
            'potato_late_blight': {
                'description': 'உருளைக்கிழங்கில் தாமத நோய் விரைவாக பாதிப்பை ஏற்படுத்தி இலை பாதிப்பு மற்றும் கிழங்கு அழுகலை உருவாக்குகிறது.',
                'symptoms': 'இருண்ட தண்ணீர் புள்ளிகள், இலைகளின் கீழ் வெள்ளை பூஞ்சை, மற்றும் விரைவான தாவர சரிவு.',
                'cause': 'குளிர்ச்சியான, ஈரமான வானிலை Phytophthora infestans-ஐ ஆதரிக்கிறது.',
                'prevention': 'இலைகளை உலர்ந்து வைக்கவும், பாதிக்கப்பட்ட தாவரங்களை அகற்றவும், எதிர்ப்பு வகைகளை பயன்படுத்தவும்.',
                'treatment': 'அங்கீகரிக்கப்பட்ட தாமத நோய் பூஞ்சைக்கொல்லிகளை பயன்படுத்தி பாதிக்கப்பட்ட பொருட்களை அழிக்கவும்.',
            },
            'potato_healthy': {
                'description': 'நோய் இல்லாத ஆரோக்கியமான உருளைக்கிழங்கு இலைகள்.',
                'symptoms': 'ஒரே மாதிரியான பச்சை இலை நிறம், வலிமையான தண்டுகள், மற்றும் ஆரோக்கியமான வளர்ச்சி.',
                'cause': 'நோய் கண்டறியப்படவில்லை.',
                'prevention': 'சமநிலை உணவு, தொடர்ந்த தண்ணீர் தெளிப்பு, மற்றும் நல்ல காற்று ஓட்டத்தை பராமரிக்கவும்.',
                'treatment': 'சிகிச்சை தேவையில்லை.',
            },
            'tomato_early_blight': {
                'description': 'தக்காளியில் ஆரம்ப நோய் இலைகள் மற்றும் தண்டுகளை தாக்கி இலை இறப்பு மற்றும் குறைந்த பழன் விளைச்சலை ஏற்படுத்துகிறது.',
                'symptoms': 'இலைகளில் ஒத்த வளையங்களுடன் இருண்ட புள்ளிகள், மஞ்சள் நிறம், மற்றும் இலை விழுதல்.',
                'cause': 'பூஞ்சை ஸ்போர்கள் காற்று மற்றும் தெளித்த தண்ணீரால் எடுத்துச் செல்லப்படுகின்றன.',
                'prevention': 'பயிர்களை சுழற்றவும், மண்ணை மூடவும், மற்றும் அடர்ந்த இலைகளை கத்தரிக்கவும்.',
                'treatment': 'குளோரோதாலோனில் அல்லது காப்பர் அடிப்படையிலான பூஞ்சைக்கொல்லி தேவைப்படும்போது பயன்படுத்தவும்.',
            },
            'tomato_healthy': {
                'description': 'ஆரோக்கியமான தக்காளி தாவரங்கள் வண்ணமயமான இலைகள் மற்றும் வலிமையான பழன் அமைப்புடன்.',
                'symptoms': 'ஒரே மாதிரியான இலை நிறம், செயலில் வளர்ச்சி, மற்றும் இலை புள்ளிகள் இல்லை.',
                'cause': 'நோய் கண்டறியப்படவில்லை.',
                'prevention': 'மண்ணை ஈரமாக வைக்கவும், தொடர்ந்து கத்தரிக்கவும், மற்றும் தினமும் கண்காணிக்கவும்.',
                'treatment': 'சிகிச்சை தேவையில்லை. வழக்கமான பராமரிப்பை தொடரவும்.',
            },
            'tomato_late_blight': {
                'description': 'தக்காளியில் தாமத நோய் குளிர்ச்சியான, ஈரமான நிலைமைகளில் அழிவு தரும் நோய்.',
                'symptoms': 'தண்ணீர் நிறைந்த புள்ளிகள், மென்மையான வெள்ளை வளர்ச்சி, மற்றும் விரைவான இலை சரிவு.',
                'cause': 'தொடர்ந்த ஈரமான வானிலை மற்றும் மோசமான காற்று சுழற்சி.',
                'prevention': 'தாவரங்களை கட்டவும், கீழ் இலைகளை அகற்றவும், மற்றும் வடிகட்டலை மேம்படுத்தவும்.',
                'treatment': 'ரிடோமில் கோல்ட் போன்ற ஒரு அமைப்பு பூஞ்சைக்கொல்லியை உடனடியாக பயன்படுத்தவும்.',
            },
        }
    },
    'kn': {  # Kannada
        'title': '🌿 ಸಸ್ಯದ ಎಲೆ ರೋಗ ಪತ್ತೆ',
        'subtitle': 'ಸಸ್ಯದ ಎಲೆಯ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ ತಕ್ಷಣ ರೋಗ ಪತ್ತೆ ಮತ್ತು ಚಿಕಿತ್ಸಾ ಸಲಹೆ ಪಡೆಯಿರಿ.',
        'status': '✅ ಮಾಡೆಲ್ ಸ್ಥಿತಿ: FINAL_MODEL.h5 (94.6% ಮೌಲ್ಯೀಕರಣ ನಿಖರತೆ) | ವರ್ಗಗಳು: ಎಲ್ಲಾ 6 ಸಸ್ಯ ರೋಗಗಳು',
        'upload': 'ಎಲೆ ಚಿತ್ರವನ್ನು ಆಯ್ಕೆ ಮಾಡಿ',
        'detected_disease': 'ಪತ್ತೆಯಾದ ರೋಗ',
        'confidence': 'ನಂಬಿಕೆ ಸ್ಕೋರ್',
        'severity': 'ತೀವ್ರತೆ',
        'very_high': '🟢 ಅತ್ಯಂತ ಹೆಚ್ಚಿನ ನಂಬಿಕೆ',
        'high': '🟡 ಹೆಚ್ಚಿನ ನಂಬಿಕೆ',
        'moderate': '🟠 ಮಧ್ಯಮ ನಂಬಿಕೆ',
        'low': '🔴 ಕಡಿಮೆ ನಂಬಿಕೆ',
        'low_confidence_warning': '⚠️ ಕಡಿಮೆ ನಂಬಿಕೆ! ಮಾಡೆಲ್ ಅನಿಶ್ಚಿತವಾಗಿದೆ. ಸ್ಪಷ್ಟ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ.',
        'ambiguous_warning': '⚠️ ಅಸ್ಪಷ್ಟ ಪೂರ್ವಾಭಾಸ. ಅನೇಕ ರೋಗಗಳಿಗೆ ಹೋಲುವ ಸ್ಕೋರ್‌ಗಳಿವೆ. ಮೇಲಿನ ಪೂರ್ವಾಭಾಸಗಳನ್ನು ಪರಿಶೀಲಿಸಿ.',
        'confident': '✅ ಪೂರ್ವಾಭಾಸ ನಂಬಿಕೆಯುತವಾಗಿದೆ ಮತ್ತು ಸ್ಪಷ್ಟವಾಗಿದೆ!',
        'speak_result': '🔊 ಫಲಿತಾಂಶವನ್ನು ಮಾತನಾಡಿ',
        'detailed_analysis': '📊 ವಿವರವಾದ ಪೂರ್ವಾಭಾಸ ವಿಶ್ಲೇಷಣೆ',
        'top_predictions': 'ಮೇಲಿನ 3 ಪೂರ್ವಾಭಾಸಗಳು:',
        'full_distribution': 'ಎಲ್ಲಾ ವರ್ಗ ಸಂಭಾವನೆಗಳು:',
        'gap': 'ಮೇಲಿನ-1 vs ಮೇಲಿನ-2 ಅಂತರ:',
        'disease_info': '📋 ರೋಗ ಮಾಹಿತಿ',
        'description': 'ವಿವರಣೆ',
        'symptoms': 'ಲಕ್ಷಣಗಳು',
        'cause': 'ಕಾರಣ',
        'prevention': 'ತಡೆಗಟ್ಟುವಿಕೆ',
        'treatment': 'ಚಿಕಿತ್ಸೆ',
        'no_info': 'ℹ️ ಈ ಪೂರ್ವಾಭಾಸಕ್ಕೆ ವಿವರವಾದ ಮಾಹಿತಿ ಲಭ್ಯವಿಲ್ಲ.',
        'classes': {
            'Potato_Early_Blight': 'ಆಲೂಗಡ್ಡೆ ಆರಂಭಿಕ ರೋಗ',
            'Potato_Healthy': 'ಆಲೂಗಡ್ಡೆ ಆರೋಗ್ಯಕರ',
            'Potato_Late_Blight': 'ಆಲೂಗಡ್ಡೆ ತಡಮಾಡಿದ ರೋಗ',
            'Tomato_Early_Blight': 'ಟೊಮೆಟೊ ಆರಂಭಿಕ ರೋಗ',
            'Tomato_Healthy': 'ಟೊಮೆಟೊ ಆರೋಗ್ಯಕರ',
            'Tomato_Late_Blight': 'ಟೊಮೆಟೊ ತಡಮಾಡಿದ ರೋಗ'
        },
        'diseases': {
            'potato_early_blight': {
                'description': 'ಆಲೂಗಡ್ಡೆಯಲ್ಲಿ ಆರಂಭಿಕ ರೋಗ ಎಲೆಗಳಲ್ಲಿ ಕಂದು ಬಣ್ಣದ ಚುಕ್ಕೆಗಳು ಮತ್ತು ಸಮಕೇಂದ್ರಿತ ವಲಯಗಳನ್ನು ಉಂಟುಮಾಡುತ್ತದೆ.',
                'symptoms': 'ಕಂದು ಗಾಯಗಳು, ಹಳದಿ ಎಲೆ ಅಂಚುಗಳು, ಮತ್ತು ಎಲೆ ಬೀಳುವಿಕೆ.',
                'cause': 'ಬೂಜು ಸ್ಪೋರ್‌ಗಳು ನೀರಿನ ಸಿಂಪಡನೆ ಮತ್ತು ಸೋಂಕಿತ ಸಸ್ಯ ಅವಶೇಷಗಳಿಂದ ಹರಡುತ್ತವೆ.',
                'prevention': 'ಬೆಳೆಗಳನ್ನು ತಿರುಗಿಸಿ, ಸೋಂಕಿತ ಎಲೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ, ಮತ್ತು ಮೇಲಿನ ನೀರಿನ ಸಿಂಪಡನೆಯನ್ನು ತಪ್ಪಿಸಿ.',
                'treatment': 'ಕಾಪರ್ ಆಧಾರಿತ ಬೂಜುನಾಶಕಗಳು ಅಥವಾ ನೀಮ್ ಎಣ್ಣೆಯನ್ನು ಬಳಸಿ ಮತ್ತು ಗಾಳಿ ಸಂಚಲನವನ್ನು ಸುಧಾರಿಸಿ.',
            },
            'potato_late_blight': {
                'description': 'ಆಲೂಗಡ್ಡೆಯಲ್ಲಿ ತಡಮಾಡಿದ ರೋಗ ಎಲೆ ಸೋಂಕು ಮತ್ತು ಕಂದು ಅಳುಕನ್ನು ಉಂಟುಮಾಡುವ ವೇಗವಾದ ರೋಗ.',
                'symptoms': 'ಇರುಳು ನೀರಿನ ಚುಕ್ಕೆಗಳು, ಎಲೆಗಳ ಕೆಳಗೆ ಬಿಳಿ ಬೂಜು, ಮತ್ತು ವೇಗವಾದ ಸಸ್ಯ ಕುಸಿತ.',
                'cause': 'ತಣ್ಣನೆಯ, ತೇವಾಂಶಯುತ ಹವಾಮಾನ Phytophthora infestans ಅನ್ನು ಬೆಂಬಲಿಸುತ್ತದೆ.',
                'prevention': 'ಎಲೆಗಳನ್ನು ಒಣಗಿಸಿ, ಸೋಂಕಿತ ಸಸ್ಯಗಳನ್ನು ತೆಗೆದುಹಾಕಿ, ಮತ್ತು ಪ್ರತಿರೋಧಕ ವೈವಿಧ್ಯಗಳನ್ನು ಬಳಸಿ.',
                'treatment': 'ಅನುಮೋದಿತ ತಡಮಾಡಿದ ರೋಗ ಬೂಜುನಾಶಕಗಳನ್ನು ಅನ್ವಯಿಸಿ ಮತ್ತು ಸೋಂಕಿತ ವಸ್ತುಗಳನ್ನು ನಾಶ ಮಾಡಿ.',
            },
            'potato_healthy': {
                'description': 'ರೋಗವಿಲ್ಲದ ಆರೋಗ್ಯಕರ ಆಲೂಗಡ್ಡೆ ಎಲೆಗಳು.',
                'symptoms': 'ಸಮಾನ ಕಂದು ಎಲೆ ಬಣ್ಣ, ಬಲವಾದ ಕಾಂಡಗಳು, ಮತ್ತು ಆರೋಗ್ಯಕರ ಬೆಳವಣಿಗೆ.',
                'cause': 'ರೋಗ ಪತ್ತೆಯಾಗಿಲ್ಲ.',
                'prevention': 'ಸಮತೋಲನಗೊಳಿಸಿದ ಪೋಷಣೆ, ನಿರಂತರ ನೀರಿನ ಸಿಂಪಡನೆ, ಮತ್ತು ಉತ್ತಮ ಗಾಳಿ ಹರಿವನ್ನು ಕಾಪಾಡಿ.',
                'treatment': 'ಚಿಕಿತ್ಸೆ ಅಗತ್ಯವಿಲ್ಲ.',
            },
            'tomato_early_blight': {
                'description': 'ಟೊಮೆಟೊ ಎಲೆಗಳು ಮತ್ತು ಕಾಂಡಗಳನ್ನು ಆಕ್ರಮಿಸುವ ಆರಂಭಿಕ ರೋಗ ಎಲೆ ಸಾವು ಮತ್ತು ಕಡಿಮೆ ಹಣ್ಣು ಉತ್ಪಾದನೆಯನ್ನು ಉಂಟುಮಾಡುತ್ತದೆ.',
                'symptoms': 'ಎಲೆಗಳಲ್ಲಿ ಸಮಕೇಂದ್ರಿತ ವಲಯಗಳೊಂದಿಗೆ ಇರುಳು ಚುಕ್ಕೆಗಳು, ಹಳದಿ, ಮತ್ತು ಎಲೆ ಬೀಳುವಿಕೆ.',
                'cause': 'ಬೂಜು ಸ್ಪೋರ್‌ಗಳು ಗಾಳಿ ಮತ್ತು ಸಿಂಪಡಿಸಿದ ನೀರಿನಿಂದ ಎಡುಗುತ್ತವೆ.',
                'prevention': 'ಬೆಳೆಗಳನ್ನು ತಿರುಗಿಸಿ, ಮಣ್ಣನ್ನು ಮುಚ್ಚಿ, ಮತ್ತು ಅಡಕವಾದ ಎಲೆಗಳನ್ನು ಕತ್ತರಿಸಿ.',
                'treatment': 'ಕ್ಲೋರೋತಾಲೋನಿಲ್ ಅಥವಾ ಕಾಪರ್ ಆಧಾರಿತ ಬೂಜುನಾಶಕವನ್ನು ಅಗತ್ಯವಿದ್ದಾಗ ಬಳಸಿ.',
            },
            'tomato_healthy': {
                'description': 'ವೈಭವದ ಎಲೆಗಳು ಮತ್ತು ಬಲವಾದ ಹಣ್ಣು ಸೆಟ್‌ನೊಂದಿಗೆ ಆರೋಗ್ಯಕರ ಟೊಮೆಟೊ ಸಸ್ಯಗಳು.',
                'symptoms': 'ಸಮಾನ ಎಲೆ ಬಣ್ಣ, ಸಕ್ರಿಯ ಬೆಳವಣಿಗೆ, ಮತ್ತು ಎಲೆ ಚುಕ್ಕೆಗಳಿಲ್ಲ.',
                'cause': 'ರೋಗ ಪತ್ತೆಯಾಗಿಲ್ಲ.',
                'prevention': 'ಮಣ್ಣನ್ನು ತೇವಗೊಳಿಸಿ, ನಿಯಮಿತವಾಗಿ ಕತ್ತರಿಸಿ, ಮತ್ತು ದೈನಂದಿನವಾಗಿ ಮೇಲ್ವಿಚಾರಣೆ ಮಾಡಿ.',
                'treatment': 'ಚಿಕಿತ್ಸೆ ಅಗತ್ಯವಿಲ್ಲ. ನಿಯಮಿತ ಪರಿಪಾಲನೆಯನ್ನು ಮುಂದುವರಿಸಿ.',
            },
            'tomato_late_blight': {
                'description': 'ಟೊಮೆಟೊಗಳಿಗೆ ತಣ್ಣನೆಯ, ತೇವಾಂಶಯುತ ಪರಿಸ್ಥಿತಿಗಳಲ್ಲಿ ನಾಶಕಾರಿ ರೋಗ.',
                'symptoms': 'ನೀರಿನಿಂದ ತುಂಬಿದ ಚುಕ್ಕೆಗಳು, ಮೃದುವಾದ ಬಿಳಿ ಬೆಳವಣಿಗೆ, ಮತ್ತು ವೇಗವಾದ ಎಲೆ ಕುಸಿತ.',
                'cause': 'ನಿರಂತರ ತೇವಾಂಶಯುತ ಹವಾಮಾನ ಮತ್ತು ಕೆಟ್ಟ ಗಾಳಿ ಸಂಚಲನ.',
                'prevention': 'ಸಸ್ಯಗಳನ್ನು ಬೆಂಬಲಿಸಿ, ಕೆಳಗಿನ ಎಲೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ, ಮತ್ತು ಡ್ರೇನೇಜ್ ಅನ್ನು ಸುಧಾರಿಸಿ.',
                'treatment': 'ರಿದೋಮಿಲ್ ಗೋಲ್ಡ್‌ನಂತಹ ಸಿಸ್ಟಮಿಕ್ ಬೂಜುನಾಶಕವನ್ನು ತಕ್ಷಣವೇ ಅನ್ವಯಿಸಿ.',
            },
        }
    },
    'hi': {  # Hindi
        'title': '🌿 पौधे की पत्ती रोग का पता लगाना',
        'subtitle': 'पौधे की पत्ती की छवि अपलोड करें तत्काल रोग निदान और उपचार सलाह प्राप्त करें।',
        'status': '✅ मॉडल स्थिति: FINAL_MODEL.h5 (94.6% सत्यापन सटीकता) | वर्ग: सभी 6 पौधे रोग',
        'upload': 'पत्ती छवि चुनें',
        'detected_disease': 'पता लगाया गया रोग',
        'confidence': 'विश्वास स्कोर',
        'severity': 'तीव्रता',
        'very_high': '🟢 बहुत उच्च विश्वास',
        'high': '🟡 उच्च विश्वास',
        'moderate': '🟠 मध्यम विश्वास',
        'low': '🔴 कम विश्वास',
        'low_confidence_warning': '⚠️ कम विश्वास! मॉडल अनिश्चित है। कृपया एक स्पष्ट छवि अपलोड करें।',
        'ambiguous_warning': '⚠️ अस्पष्ट भविष्यवाणी। कई रोगों के समान स्कोर हैं। नीचे शीर्ष भविष्यवाणियाँ जांचें।',
        'confident': '✅ भविष्यवाणी विश्वासपूर्ण और स्पष्ट है!',
        'speak_result': '🔊 परिणाम बोलें',
        'detailed_analysis': '📊 विस्तृत भविष्यवाणी विश्लेषण',
        'top_predictions': 'शीर्ष 3 भविष्यवाणियाँ:',
        'full_distribution': 'सभी वर्ग संभावनाएँ:',
        'gap': 'शीर्ष-1 बनाम शीर्ष-2 अंतर:',
        'disease_info': '📋 रोग जानकारी',
        'description': 'विवरण',
        'symptoms': 'लक्षण',
        'cause': 'कारण',
        'prevention': 'रोकथाम',
        'treatment': 'उपचार',
        'no_info': 'ℹ️ इस भविष्यवाणी के लिए विस्तृत जानकारी उपलब्ध नहीं है।',
        'classes': {
            'Potato_Early_Blight': 'आलू प्रारंभिक रोग',
            'Potato_Healthy': 'आलू स्वस्थ',
            'Potato_Late_Blight': 'आलू देर से रोग',
            'Tomato_Early_Blight': 'टमाटर प्रारंभिक रोग',
            'Tomato_Healthy': 'टमाटर स्वस्थ',
            'Tomato_Late_Blight': 'टमाटर देर से रोग'
        },
        'diseases': {
            'potato_early_blight': {
                'description': 'आलू पर प्रारंभिक रोग पत्तियों पर भूरे धब्बे और समकेंद्रीय छल्ले पैदा करता है।',
                'symptoms': 'भूरे घाव, पीले पत्ती किनारे, और पत्ती गिरना।',
                'cause': 'फंगल बीजाणु पानी की छींट और संक्रमित पौधे अवशेषों द्वारा फैलते हैं।',
                'prevention': 'फसलें घुमाएँ, संक्रमित पत्तियाँ हटाएँ, और ऊपरी पानी छिड़काव से बचें।',
                'treatment': 'कॉपर-आधारित फंगीसाइड या नीम का तेल उपयोग करें और हवा परिसंचरण में सुधार करें।',
            },
            'potato_late_blight': {
                'description': 'आलू पर देर से रोग एक तेज़ी से आगे बढ़ने वाला रोग है जो पत्तियों को खराब करता है और कंद सड़न पैदा करता है।',
                'symptoms': 'गहरे पानी के धब्बे, पत्तियों के नीचे सफेद फफूंदी, और पौधे की तेज़ी से गिरावट।',
                'cause': 'ठंडा, नम मौसम जो Phytophthora infestans को पसंद करता है।',
                'prevention': 'पत्तियों को सूखा रखें, संक्रमित पौधे हटाएँ, और प्रतिरोधी किस्में उपयोग करें।',
                'treatment': 'अनुमोदित देर से रोग फंगीसाइड लागू करें और संक्रमित सामग्री को नष्ट करें।',
            },
            'potato_healthy': {
                'description': 'कोई रोग के बिना स्वस्थ आलू पत्तियाँ।',
                'symptoms': 'समान हरे पत्ती रंग, मजबूत तने, और स्वस्थ वृद्धि।',
                'cause': 'कोई रोग नहीं पाया गया।',
                'prevention': 'संतुलित पोषण, निरंतर पानी छिड़काव, और अच्छा हवा प्रवाह बनाए रखें।',
                'treatment': 'कोई उपचार आवश्यक नहीं।',
            },
            'tomato_early_blight': {
                'description': 'टमाटर पर प्रारंभिक रोग पत्तियों और तनों पर हमला करता है, पत्ती की मौत और कम फल उत्पादन पैदा करता है।',
                'symptoms': 'पत्तियों पर समकेंद्रीय छल्लों के साथ गहरे धब्बे, पीला होना, और पत्ती गिरना।',
                'cause': 'फंगल बीजाणु हवा और छिड़के गए पानी द्वारा ले जाए जाते हैं।',
                'prevention': 'फसलें घुमाएँ, मिट्टी को ढकें, और घनी पत्तियाँ काटें।',
                'treatment': 'क्लोरोथालोनिल या कॉपर-आधारित फंगीसाइड आवश्यकतानुसार उपयोग करें।',
            },
            'tomato_healthy': {
                'description': 'विविध पत्तियाँ और मजबूत फल सेट के साथ स्वस्थ टमाटर पौधे।',
                'symptoms': 'समान पत्ती रंग, सक्रिय वृद्धि, और कोई पत्ती धब्बे नहीं।',
                'cause': 'कोई रोग नहीं पाया गया।',
                'prevention': 'मिट्टी को गीला रखें, नियमित रूप से काटें, और दैनिक निगरानी करें।',
                'treatment': 'कोई उपचार आवश्यक नहीं। नियमित देखभाल जारी रखें।',
            },
            'tomato_late_blight': {
                'description': 'ठंडी, नम स्थितियों में टमाटर के लिए विनाशकारी रोग।',
                'symptoms': 'पानी से भरे धब्बे, मुलायम सफेद वृद्धि, और तेज़ी से पत्ती गिरावट।',
                'cause': 'लगातार नम मौसम और खराब हवा परिसंचरण।',
                'prevention': 'पौधों को सहारा दें, निचली पत्तियाँ हटाएँ, और जल निकासी में सुधार करें।',
                'treatment': 'रिडोमिल गोल्ड जैसे सिस्टमिक फंगीसाइड तुरंत लागू करें।',
            },
        }
    }
}


def save_voice_file(text, language, output_path=OUTPUT_AUDIO_PATH):
    """Save speech audio to a local MP3 file using gTTS."""
    try:
        lang_codes = {
            'en': 'en',  # English
            'hi': 'hi',  # Hindi
            'ta': 'ta',  # Tamil
            'kn': 'kn'   # Kannada
        }
        lang_code = lang_codes.get(language, 'en')

        tts = gTTS(text=text, lang=lang_code, slow=False)
        tts.save(str(output_path))
        return output_path
    except Exception as e:
        st.error(f"Voice synthesis failed: {e}. Make sure you have internet connection for Google TTS.")
        return None


def create_full_voice_text(display_class, confidence, severity, disease_info, trans):
    """Create comprehensive voice text combining all output information."""
    confidence_pct = confidence * 100
    voice_text = (
        f"{trans['detected_disease']}: {display_class}. "
        f"{trans['confidence']}: {confidence_pct:.1f}%. "
        f"{trans['severity']}: {severity}. "
    )

    if disease_info:
        if disease_info.get('description'):
            voice_text += f"{trans['description']}: {disease_info['description']}. "
        if disease_info.get('cause'):
            voice_text += f"{trans['cause']}: {disease_info['cause']}. "
        if disease_info.get('prevention'):
            voice_text += f"{trans['prevention']}: {disease_info['prevention']}. "
        if disease_info.get('treatment'):
            voice_text += f"{trans['treatment']}: {disease_info['treatment']}. "

    return voice_text


def load_model():
    try:
        print(f"Loading model from: {MODEL_PATH}")
        model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


def load_class_names():
    try:
        with open(CLASS_NAMES_PATH, 'r') as f:
            names = json.load(f)
        print(f"Loaded class names: {names}")
        return names
    except Exception as e:
        st.error(f"Failed to load class names: {e}")
        return []


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess image for model inference.
    MUST match the preprocessing used during training for consistency!
    
    Steps:
    1. Convert to RGB (handle RGBA, grayscale)
    2. Resize to 224×224 with bicubic interpolation (high quality)
    3. Normalize to [0, 1] range
    4. Add batch dimension
    """
    # Ensure RGB
    image = image.convert('RGB')
    
    # Resize with high-quality interpolation (matching training)
    image = image.resize(IMAGE_SIZE, Image.BICUBIC)
    
    # Convert to array and normalize
    img = np.array(image, dtype='float32') / 255.0
    
    # Handle edge cases (shouldn't happen after RGB conversion, but be safe)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    
    # Add batch dimension for model input
    img = np.expand_dims(img, axis=0)
    
    return img


def predict(image_array: np.ndarray, model, class_names):
    """
    Predict disease class from preprocessed image array.
    
    Args:
        image_array: Preprocessed image (1, 224, 224, 3) normalized [0, 1]
        model: Loaded Keras model
        class_names: List of class names in correct order
    
    Returns:
        (predicted_class, confidence, ambiguous, top_predictions, predictions)
    """
    # Get raw predictions
    predictions = model.predict(image_array, verbose=0)[0]
    
    # Validate output shape
    if len(class_names) != predictions.shape[-1]:
        st.error(
            f"Model output size {predictions.shape[-1]} does not match class names length {len(class_names)}. "
            "Check class_names.json for the correct mapping."
        )
        return 'Uncertain', 0.0, False, [], predictions

    # Get best prediction index
    best_idx = int(np.argmax(predictions))
    best_confidence = float(predictions[best_idx])
    
    # Get top-K predictions
    sorted_indices = np.argsort(predictions)[::-1][:TOP_K]
    top_predictions = [(class_names[i], float(predictions[i]), int(i)) for i in sorted_indices]
    
    # Get second-best for ambiguity check
    top2_score = top_predictions[1][1] if len(top_predictions) > 1 else 0.0
    
    # Determine if prediction is ambiguous
    is_low_confidence = best_confidence < CONFIDENCE_THRESHOLD
    is_close_competitors = (best_confidence - top2_score) < AMBIGUITY_MARGIN
    ambiguous = is_low_confidence or is_close_competitors
    
    # Get predicted class name
    predicted_class = class_names[best_idx]
    
    return predicted_class, best_confidence, ambiguous, top_predictions, predictions


def main():
    st.set_page_config(page_title='Plant Leaf Disease Detection', page_icon='🌿', layout='wide')
    # Advanced Modern Colorful UI with Blue-Purple Gradient and Glassmorphism
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #6a11cb 45%, #2575fc 100%);
        color: #111;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        min-height: 100vh;
    }
    .main-header {
        text-align: center;
        color: white;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.4);
        margin-bottom: 40px;
        padding: 30px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 25px;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .main-title {
        font-size: 3.8em;
        font-weight: 900;
        margin-bottom: 15px;
        background: linear-gradient(45deg, #8be9fd, #6272a4, #bd93f9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 2px 2px 6px rgba(0,0,0,0.15);
    }
    .main-subtitle {
        font-size: 1.4em;
        color: #f0f8ff;
        font-weight: 400;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }

    /* Unique Icon Block Cards */
    .prediction-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 25px;
        padding: 30px;
        margin: 20px 0;
        border: 3px solid #667eea;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.2), 0 5px 15px rgba(0,0,0,0.1);
        backdrop-filter: blur(20px);
        transition: all 0.4s ease;
    }
    .prediction-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 25px 50px rgba(102, 126, 234, 0.3), 0 10px 25px rgba(0,0,0,0.15);
    }

    .confidence-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 25px;
        padding: 30px;
        margin: 20px 0;
        border: 3px solid #764ba2;
        box-shadow: 0 15px 35px rgba(118, 75, 162, 0.2), 0 5px 15px rgba(0,0,0,0.1);
        backdrop-filter: blur(20px);
        transition: all 0.4s ease;
    }
    .confidence-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 25px 50px rgba(118, 75, 162, 0.3), 0 10px 25px rgba(0,0,0,0.15);
    }

    .severity-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 25px;
        padding: 30px;
        margin: 20px 0;
        border: 3px solid #f093fb;
        box-shadow: 0 15px 35px rgba(240, 147, 251, 0.2), 0 5px 15px rgba(0,0,0,0.1);
        backdrop-filter: blur(20px);
        transition: all 0.4s ease;
    }
    .severity-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 25px 50px rgba(240, 147, 251, 0.3), 0 10px 25px rgba(0,0,0,0.15);
    }

    .description-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 25px;
        padding: 30px;
        margin: 20px 0;
        border: 3px solid #f5576c;
        box-shadow: 0 15px 35px rgba(245, 87, 108, 0.2), 0 5px 15px rgba(0,0,0,0.1);
        backdrop-filter: blur(20px);
        transition: all 0.4s ease;
    }
    .description-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 25px 50px rgba(245, 87, 108, 0.3), 0 10px 25px rgba(0,0,0,0.15);
    }

    .treatment-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 25px;
        padding: 30px;
        margin: 20px 0;
        border: 3px solid #4facfe;
        box-shadow: 0 15px 35px rgba(79, 172, 254, 0.2), 0 5px 15px rgba(0,0,0,0.1);
        backdrop-filter: blur(20px);
        transition: all 0.4s ease;
    }
    .treatment-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 25px 50px rgba(79, 172, 254, 0.3), 0 10px 25px rgba(0,0,0,0.15);
    }

    .disease-name {
        font-size: 2.8em;
        font-weight: 900;
        color: #111;
        text-align: center;
        margin: 20px 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    /* Enhanced Confidence Meter */
    .confidence-section {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(240, 248, 255, 0.9));
        border-radius: 25px;
        padding: 30px;
        margin: 20px 0;
        border: 3px solid #667eea;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.15);
        backdrop-filter: blur(15px);
    }
    .confidence-label {
        font-size: 1.6em;
        font-weight: 700;
        color: #111;
        margin-bottom: 20px;
        text-align: center;
    }
    .confidence-value {
        font-size: 2.5em;
        font-weight: 900;
        color: #111;
        text-align: center;
        margin: 15px 0;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .progress-bar-container {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 50px;
        padding: 8px;
        margin: 20px 0;
        box-shadow: inset 0 3px 6px rgba(0,0,0,0.1);
        border: 2px solid rgba(102, 126, 234, 0.3);
    }

    /* Color-coded severity levels */
    .severity-high {
        background: linear-gradient(45deg, #4caf50, #388e3c, #2e7d32);
        color: white;
        padding: 20px 40px;
        border-radius: 35px;
        text-align: center;
        font-weight: 800;
        font-size: 1.3em;
        box-shadow: 0 8px 25px rgba(76, 175, 80, 0.4);
        animation: pulse 2s infinite;
        margin: 20px 0;
    }
    .severity-medium {
        background: linear-gradient(45deg, #ff9800, #f57c00, #ef6c00);
        color: white;
        padding: 20px 40px;
        border-radius: 35px;
        text-align: center;
        font-weight: 800;
        font-size: 1.3em;
        box-shadow: 0 8px 25px rgba(255, 152, 0, 0.4);
        animation: pulse 2s infinite;
        margin: 20px 0;
    }
    .severity-low {
        background: linear-gradient(45deg, #f44336, #d32f2f, #c62828);
        color: white;
        padding: 20px 40px;
        border-radius: 35px;
        text-align: center;
        font-weight: 800;
        font-size: 1.3em;
        box-shadow: 0 8px 25px rgba(244, 67, 54, 0.4);
        animation: pulse 2s infinite;
        margin: 20px 0;
    }

    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }

    /* Enhanced info cards with better visibility */
    .info-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 25px;
        margin: 15px 0;
        border-left: 6px solid #667eea;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        backdrop-filter: blur(15px);
        color: #111;
        font-size: 1.1em;
        line-height: 1.6;
    }
    .info-card strong {
        color: #333;
        font-weight: 700;
        font-size: 1.2em;
    }

    /* Status cards with enhanced styling */
    .warning-card {
        background: rgba(255, 152, 0, 0.1);
        border-radius: 20px;
        padding: 25px;
        margin: 15px 0;
        border-left: 6px solid #ff9800;
        box-shadow: 0 8px 25px rgba(255,152,0,0.2);
        color: #111;
        font-weight: 600;
    }
    .error-card {
        background: rgba(244, 67, 54, 0.1);
        border-radius: 20px;
        padding: 25px;
        margin: 15px 0;
        border-left: 6px solid #f44336;
        box-shadow: 0 8px 25px rgba(244,67,54,0.2);
        color: #111;
        font-weight: 600;
    }
    .success-card {
        background: rgba(76, 175, 80, 0.1);
        border-radius: 20px;
        padding: 25px;
        margin: 15px 0;
        border-left: 6px solid #4caf50;
        box-shadow: 0 8px 25px rgba(76,175,80,0.2);
        color: #111;
        font-weight: 600;
    }

    /* Enhanced voice button */
    .voice-button {
        background: linear-gradient(45deg, #667eea, #764ba2, #f093fb);
        color: white;
        border: none;
        padding: 18px 40px;
        border-radius: 40px;
        font-weight: 800;
        font-size: 1.2em;
        cursor: pointer;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        transition: all 0.4s ease;
        margin: 25px auto;
        display: block;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .voice-button:hover {
        transform: translateY(-5px) scale(1.05);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6);
    }

    /* Sidebar enhancements */
    .sidebar-header {
        color: #111;
        font-weight: 800;
        font-size: 1.4em;
        margin-bottom: 20px;
        text-align: center;
        background: rgba(255, 255, 255, 0.9);
        padding: 15px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .language-selector {
        margin: 25px 0;
        padding: 18px;
        background: rgba(15, 32, 39, 0.95);
        border-radius: 20px;
        border: 2px solid #6a11cb;
        box-shadow: 0 6px 25px rgba(106, 17, 203, 0.25);
        color: white;
    }
    .language-selector select,
    .language-selector label {
        color: white !important;
    }
    .language-selector select {
        background: rgba(15, 32, 39, 0.9) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 12px !important;
    }
    .status-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        border: 2px solid #667eea;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.2);
        color: #111;
        font-weight: 600;
    }

    /* Section titles */
    .section-title {
        font-size: 2.0em;
        font-weight: 800;
        color: #111;
        margin: 35px 0 25px 0;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(255,255,255,0.5);
        background: rgba(255, 255, 255, 0.8);
        padding: 15px 30px;
        border-radius: 25px;
        display: inline-block;
        width: 100%;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    }

    /* Detail cards */
    .detail-card {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 20px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        color: #111;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Language selector in sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-header">🌐 Language Settings</div>', unsafe_allow_html=True)
        st.markdown('<div class="language-selector">', unsafe_allow_html=True)
        language = st.selectbox(
            "Select Language / भाषा चुनें / மொழியை தேர்ந்தெடுக்கவும் / ಭಾಷೆಯನ್ನು ಆಯ್ಕೆ ಮಾಡಿ",
            options=['en', 'hi', 'ta', 'kn'],
            format_func=lambda x: {'en': 'English', 'hi': 'हिंदी', 'ta': 'தமிழ்', 'kn': 'ಕನ್ನಡ'}[x],
            key="language_selector"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("**📊 Model Status:**")
        st.markdown("✅ FINAL_MODEL.h5 (94.6% accuracy)")
        st.markdown("✅ All 6 plant diseases supported")
    
    # Get translations for selected language
    trans = TRANSLATIONS[language]

    if 'audio_path' not in st.session_state:
        st.session_state.audio_path = None
        st.session_state.audio_paused = False

    # Main header with plant theme (translated)
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.markdown(f'<div class="main-title">🌿 {trans["title"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="main-subtitle">{trans["subtitle"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    model = load_model()
    class_names = load_class_names()

    if model is None or not class_names:
        st.error("❌ Model or class names failed to load. Check your files.")
        return

    uploaded_file = st.file_uploader(trans['upload'], type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.markdown('<h3 style="color: #1b5e20; text-align: center;">📷 Uploaded Image</h3>', unsafe_allow_html=True)
            image = Image.open(uploaded_file)
            st.image(image, width=380)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            # ===== PREDICTION LOGIC (DO NOT MODIFY) =====
            image_array = preprocess_image(image)
            predicted_class, confidence, ambiguous, top_predictions, predictions = predict(image_array, model, class_names)
            # ============================================
            
            # Format class name for display (translate only for UI)
            display_class = trans['classes'].get(predicted_class, predicted_class.replace('_', ' ').title())
            
            # Determine severity and color
            if confidence >= 0.90:
                severity = trans['very_high']
                severity_class = "severity-high"
            elif confidence >= 0.75:
                severity = trans['high']
                severity_class = "severity-high"
            elif confidence >= 0.50:
                severity = trans['moderate']
                severity_class = "severity-medium"
            else:
                severity = trans['low']
                severity_class = "severity-low"
            
            # 🌿 PREDICTION RESULT CARD
            st.markdown('<div class="section-title">🌿 Prediction Result</div>', unsafe_allow_html=True)
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="disease-name">{display_class}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # 📊 CONFIDENCE METER CARD
            st.markdown('<div class="section-title">📊 Confidence Analysis</div>', unsafe_allow_html=True)
            st.markdown('<div class="confidence-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="confidence-label">📈 {trans["confidence"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="confidence-value">{confidence * 100:.1f}%</div>', unsafe_allow_html=True)
            st.markdown('<div class="progress-bar-container">', unsafe_allow_html=True)
            st.progress(confidence)
            st.markdown('</div>', unsafe_allow_html=True)

            # Color-coded confidence label
            if confidence >= 0.80:
                confidence_label = f'<span style="color: #4caf50; font-weight: 700;">🟢 High Confidence (80-100%)</span>'
            elif confidence >= 0.50:
                confidence_label = f'<span style="color: #ff9800; font-weight: 700;">🟠 Medium Confidence (50-80%)</span>'
            else:
                confidence_label = f'<span style="color: #f44336; font-weight: 700;">🔴 Low Confidence (0-50%)</span>'

            st.markdown(f'<div style="text-align: center; font-size: 1.2em; margin: 15px 0;">{confidence_label}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # ⚠️ SEVERITY LEVEL CARD
            st.markdown('<div class="section-title">⚠️ Severity Level</div>', unsafe_allow_html=True)
            st.markdown('<div class="severity-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="{severity_class}"><strong>{trans["severity"]}: {severity}</strong></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Status messages with enhanced styling
            st.markdown('<br>', unsafe_allow_html=True)
            if confidence < 0.50:
                st.markdown(f'<div class="error-card"><h4>⚠️ {trans["low_confidence_warning"]}</h4></div>', unsafe_allow_html=True)
            elif ambiguous:
                st.markdown(f'<div class="warning-card"><h4>⚠️ {trans["ambiguous_warning"]}</h4></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="success-card"><h4>✅ {trans["confident"]}</h4></div>', unsafe_allow_html=True)
            
            # 🔊 FULL VOICE OUTPUT (ALL TEXT)
            if confidence >= 0.50:
                st.markdown('<br>', unsafe_allow_html=True)
                lookup_key = predicted_class.lower()
                disease_info = DISEASE_INFO.get(lookup_key, {})
                disease_info_translated = trans.get('diseases', {}).get(lookup_key, disease_info)

                # Create comprehensive voice text with translations
                voice_text = create_full_voice_text(display_class, confidence, severity, disease_info_translated, trans)

                col_speak, col_pause = st.columns([1, 1])
                with col_speak:
                    if st.button("🔊 " + trans['speak_result'], key="speak", help="Generate full diagnosis audio"):
                        audio_file = save_voice_file(voice_text, language)
                        if audio_file:
                            st.session_state.audio_path = str(audio_file)
                            st.session_state.audio_paused = False
                            st.success("🔊 Full diagnosis audio generated.")
                with col_pause:
                    if st.button("⏸️ Stop", key="pause", help="Stop current audio playback"):
                        st.session_state.audio_paused = True
                        st.session_state.audio_path = None
                        st.warning("⏸️ Playback stopped.")

                if st.session_state.audio_path and not st.session_state.audio_paused:
                    st.audio(st.session_state.audio_path, format='audio/mp3')

            # 📊 DETAILED ANALYSIS SECTION
            st.markdown('<div class="section-title">📊 Detailed Analysis</div>', unsafe_allow_html=True)
            with st.expander("🔍 " + trans['detailed_analysis']):
                st.markdown('<div class="detail-card">', unsafe_allow_html=True)
                st.write(f"**{trans['top_predictions']}**")
                for label, score, idx in top_predictions:
                    display_label = trans['classes'].get(label, label.replace('_', ' ').title())
                    bar_length = int(score * 50)
                    bar = "█" * bar_length
                    st.write(f"{idx}: {display_label:30s} | {bar:50s} {score*100:5.1f}%")
                
                st.write(f"\n**{trans['full_distribution']}**")
                prob_dict = {trans['classes'].get(class_names[i], class_names[i]): f"{predictions[i]*100:.2f}%" for i in range(len(class_names))}
                st.write(prob_dict)
                
                if len(top_predictions) > 1:
                    gap = top_predictions[0][1] - top_predictions[1][1]
                    st.write(f"**{trans['gap']}** {gap:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # 📖 DISEASE DESCRIPTION CARD
            if confidence >= 0.50:
                lookup_key = predicted_class.lower()
                disease_info = DISEASE_INFO.get(lookup_key, {})
                disease_info_translated = trans.get('diseases', {}).get(lookup_key, disease_info)

                if disease_info_translated:
                    st.markdown('<div class="section-title">📖 Disease Information</div>', unsafe_allow_html=True)
                    st.markdown('<div class="description-card">', unsafe_allow_html=True)

                    st.markdown(f'<div class="info-card"><strong>📝 {trans["description"]}:</strong><br>{disease_info_translated.get("description", disease_info.get("description", "N/A"))}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="info-card"><strong>🔍 {trans["symptoms"]}:</strong><br>{disease_info_translated.get("symptoms", disease_info.get("symptoms", "N/A"))}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="info-card"><strong>⚡ {trans["cause"]}:</strong><br>{disease_info_translated.get("cause", disease_info.get("cause", "N/A"))}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="info-card"><strong>🛡️ {trans["prevention"]}:</strong><br>{disease_info_translated.get("prevention", disease_info.get("prevention", "N/A"))}</div>', unsafe_allow_html=True)

                    st.markdown('</div>', unsafe_allow_html=True)

                    # 💊 TREATMENT CARD
                    st.markdown('<div class="section-title">💊 Treatment & Care</div>', unsafe_allow_html=True)
                    st.markdown('<div class="treatment-card">', unsafe_allow_html=True)
                    st.markdown(f'<div class="info-card"><strong>💊 {trans["treatment"]}:</strong><br>{disease_info_translated.get("treatment", disease_info.get("treatment", "N/A"))}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("ℹ️ " + trans['no_info'])

    # ========== 🤖 AI CHAT ASSISTANT SECTION ==========
    st.markdown('---')
    st.markdown('<div class="section-title">🤖 AI Assistant</div>', unsafe_allow_html=True)
    
    # Plant Knowledge Database
    PLANT_KNOWLEDGE = {
        'early_blight': {
            'keywords': ['early blight', 'blight', 'brown spots', 'lesions'],
            'response': '''
**Early Blight - Comprehensive Guide**

📋 **Description:**
Early blight is a fungal disease affecting potatoes and tomatoes, characterized by brown spots with concentric rings on leaves.

🔍 **Causes:**
- Fungal spores spread through water splash
- Contaminated plant debris in soil
- Warm, humid weather conditions
- Poor air circulation

💊 **Treatment Options:**
- Apply copper-based fungicides weekly
- Use neem oil spray (organic option)
- Remove infected leaves immediately
- Improve air circulation around plants
- Space plants properly to reduce humidity

🛡️ **Prevention Tips:**
- Rotate crops annually
- Remove plant debris after harvest
- Avoid overhead watering (use drip irrigation)
- Mulch soil to prevent spore splash
- Choose resistant varieties
- Monitor plants regularly
- Remove lower infected leaves early
- Ensure good drainage
            '''
        },
        'late_blight': {
            'keywords': ['late blight', 'phytophthora', 'watery spots', 'white mold', 'tuber rot'],
            'response': '''
**Late Blight - Emergency Guide**

📋 **Description:**
Late blight is a rapid, destructive fungal disease caused by Phytophthora infestans. It can destroy an entire crop in days.

🔍 **Causes:**
- Phytophthora infestans pathogen
- Cool (50-65°F), wet weather
- High humidity and leaf wetness
- Infected seed potatoes
- Airborne spore dispersal

💊 **Immediate Treatment:**
- Apply systemic fungicides immediately
- Remove and destroy infected plants
- Do NOT compost infected material (burn it)
- Use approved late blight fungicides
- Spray every 7-10 days during outbreak
- Consider copper-sulfur combinations

🛡️ **Prevention Strategies:**
- Use disease-free seed potatoes
- Plant resistant varieties
- Maintain proper spacing
- Ensure excellent drainage
- Avoid overhead watering
- Eliminate infected debris
- Monitor weather forecasts
- Start preventive spraying before outbreak
            '''
        },
        'healthy_plant': {
            'keywords': ['healthy', 'no disease', 'green', 'normal', 'good'],
            'response': '''
**Healthy Plant Maintenance**

✅ **Your plant is disease-free! Here's how to keep it healthy:**

🌱 **Care Requirements:**
- Consistent watering (1-2 inches per week)
- Balanced NPK fertilization
- Good soil drainage (pH 6.0-7.0)
- 6-8 hours of sunlight daily
- Proper spacing for air circulation

🛡️ **Prevention Measures:**
- Monitor plants regularly for early signs
- Remove any damaged leaves immediately
- Maintain clean gardening tools
- Rotate crops yearly
- Mulch soil (2-3 inches)
- Use disease-resistant varieties
- Practice companion planting

📊 **Monitoring Tips:**
- Check leaves weekly (both sides)
- Look for discoloration or spots
- Inspect stems for lesions
- Note any unusual leaf behavior
- Keep records of observations

💪 **Boost Plant Immunity:**
- Use balanced fertilizers
- Add organic compost
- Avoid over-watering
- Ensure good air flow
- Plant at correct depth
            '''
        },
        'treatment': {
            'keywords': ['treatment', 'cure', 'fix', 'remedy', 'help', 'fungicide'],
            'response': '''
**Plant Disease Treatment Guide**

🏥 **Treatment Categories:**

**Chemical Fungicides:**
- Copper-based: Safe, broad-spectrum
- Sulfur-based: Preventive, organic
- Systemic: Absorbed by plant (late blight)
- Contact: Surface protection only

**Organic Solutions:**
- Neem oil: Natural insecticide & fungicide
- Baking soda spray: Preventive
- Milk spray: Antifungal properties
- Sulfur: Traditional fungicide

**Application Tips:**
- Spray in early morning/late evening
- Ensure complete leaf coverage
- Spray both leaf surfaces
- Follow label directions carefully
- Repeat every 7-14 days
- Stop 7-14 days before harvest

⚠️ **Important:**
- Alternate fungicides to prevent resistance
- Test on small area first
- Wear protective equipment
- Never over-apply
- Check local regulations
            '''
        },
        'prevention': {
            'keywords': ['prevention', 'prevent', 'avoid', 'protect', 'protection'],
            'response': '''
**Plant Disease Prevention**

🛡️ **Top Prevention Strategies:**

**Cultural Practices:**
- Crop rotation (3-4 year cycle)
- Remove infected debris
- Sanitize tools between plants
- Proper spacing (air circulation)
- Avoid overhead watering
- Mulch soil surface

**Resistant Varieties:**
- Choose disease-resistant cultivars
- Check seed packet information
- Ask local nurseries
- Consider heirloom vs. modern

**Environmental Control:**
- Maintain optimal temperature
- Control humidity levels
- Ensure good drainage
- Provide adequate sunlight
- Stake/support plants
- Prune for air flow

**Regular Monitoring:**
- Weekly plant inspections
- Early disease detection
- Quick action on symptoms
- Keep detailed records
- Document problems

**Soil Health:**
- Test soil annually
- Maintain pH 6.0-7.0
- Add organic matter
- Use balanced fertilizer
- Avoid soil compaction
            '''
        }
    }
    
    # Planet Knowledge Database
    PLANET_KNOWLEDGE = {
        'mercury': {
            'keywords': ['mercury'],
            'response': '''
**Mercury - The Smallest Planet**

🌍 **Basic Facts:**
- Closest planet to the Sun
- Smallest planet in our solar system
- Named after Roman messenger god
- Only 4,879 km in diameter

🔥 **Key Features:**
- Surface temperature: 430°C (day), -180°C (night)
- No atmosphere to retain heat
- Covered with craters (Moon-like)
- Fast orbit: 88 Earth days
- Slow rotation: 59 Earth days (spins backward!)

🌙 **Fun Facts:**
- A day on Mercury is longer than its year!
- Only 2 sunrises per Mercury year
- No moons (smallest terrestrial planet)
- Visible from Earth without telescope
- Ancient Romans thought it was two planets
            '''
        },
        'venus': {
            'keywords': ['venus'],
            'response': '''
**Venus - Earth's Twin**

🌍 **Basic Facts:**
- Second planet from the Sun
- Almost same size as Earth (12,104 km)
- Brightest object in night sky
- Named after Roman goddess of love

🔥 **Key Features:**
- Hottest planet: 465°C (hottest than Mercury!)
- Thick atmosphere (sulfuric acid clouds)
- Pressure 92x Earth's atmosphere
- Rotates backwards (retrograde rotation)
- Slow rotation: 243 Earth days

🌙 **Fun Facts:**
- Venus rotates slower than it orbits!
- A Venus day is longer than Venus year
- Surface never explored by humans (too harsh)
- No moons
- Appears as "morning star" or "evening star"
            '''
        },
        'earth': {
            'keywords': ['earth'],
            'response': '''
**Earth - Our Home Planet**

🌍 **Basic Facts:**
- Third planet from the Sun
- Only known planet with life
- Diameter: 12,742 km
- Age: 4.54 billion years

🌊 **Key Features:**
- 71% water coverage
- One natural satellite (The Moon)
- Optimal temperature for life
- Magnetic field protects from solar wind
- Orbits Sun in 365.25 days

🌱 **Fun Facts:**
- Only planet known to have life
- Water is unique (exists in all 3 states)
- Oxygen-rich atmosphere (21% O2)
- Rotates at 1,674 km/hour
- Magnetic north/south poles
- Home to 8.7 million species
            '''
        },
        'mars': {
            'keywords': ['mars'],
            'response': '''
**Mars - The Red Planet**

🔴 **Basic Facts:**
- Fourth planet from the Sun
- Red color from iron oxide (rust)
- Half Earth's diameter: 6,779 km
- Named after Roman god of war

🌋 **Key Features:**
- Two small moons: Phobos & Deimos
- Largest volcano: Olympus Mons
- Grand Canyon equivalent: Valles Marineris
- Thin CO2 atmosphere
- Temperature: -125°C to 20°C

🚀 **Fun Facts:**
- Most explored planet after Earth
- Water ice at polar caps
- Day length: 24.6 hours (like Earth!)
- Evidence of ancient water flows
- Primary target for human colonization
- Dust storms can cover entire planet
            '''
        },
        'jupiter': {
            'keywords': ['jupiter'],
            'response': '''
**Jupiter - The Gas Giant**

🌀 **Basic Facts:**
- Fifth planet from the Sun
- Largest planet in solar system
- Diameter: 139,820 km (11x Earth)
- Named after Roman king of gods

⚡ **Key Features:**
- 79 known moons (largest: Ganymede)
- Great Red Spot (storm 3x Earth size)
- Strong magnetic field
- No solid surface (gas giant)
- Rapid rotation: 10 hours per day

🎆 **Fun Facts:**
- Radiates more heat than it receives
- Wind speeds: 360 km/hour
- Beautiful band patterns
- Ring system (faint)
- Four large Galilean moons
- Could fit 1,300 Earths inside
            '''
        },
        'saturn': {
            'keywords': ['saturn'],
            'response': '''
**Saturn - The Ringed Beauty**

💍 **Basic Facts:**
- Sixth planet from the Sun
- Famous for spectacular ring system
- Diameter: 116,460 km (9x Earth)
- Named after Roman god of wealth

👑 **Key Features:**
- 146 known moons (largest: Titan)
- Rings made of ice and rock particles
- Lowest density: would float in water!
- Colorful cloud bands
- 10.7 hour rotation period

🌌 **Fun Facts:**
- Rings span 282,000 km but very thin
- Titan has liquid methane lakes
- 62 confirmed moons (growing list)
- Beautiful pale yellow color
- Faint ring system too
- Visible to naked eye
            '''
        },
        'uranus': {
            'keywords': ['uranus'],
            'response': '''
**Uranus - The Tilted Planet**

❄️ **Basic Facts:**
- Seventh planet from the Sun
- Rotates on its side (98° tilt)
- Diameter: 50,724 km (4x Earth)
- Named after ancient god of sky

🌬️ **Key Features:**
- Ice giant (not gas giant)
- 27 known moons
- Faint ring system (11 rings)
- Rotates opposite to most planets
- Temperature: -224°C (coldest atmosphere)

🔷 **Fun Facts:**
- Extreme axial tilt (92° from orbit plane)
- Discovered in 1781 (first with telescope)
- Pale blue-green color (methane)
- Wind speeds: 900 km/hour
- One rotation takes 17 hours
- Extreme seasonal changes
            '''
        },
        'neptune': {
            'keywords': ['neptune'],
            'response': '''
**Neptune - The Windy Blue Planet**

🌊 **Basic Facts:**
- Eighth and farthest planet
- Diameter: 49,244 km (4x Earth)
- Named after Roman god of sea
- Deep blue color from methane

💨 **Key Features:**
- Strongest winds: 2,100 km/hour!
- 14 known moons (largest: Triton)
- Faint ring system
- Great Dark Spot (like Jupiter)
- Rotates in 16 hours

🪐 **Fun Facts:**
- Furthest from Sun (30 AU)
- Discovered in 1846
- Triton orbits backwards
- Coldest planetary atmosphere
- Takes 165 years to orbit Sun
- Last visited by Voyager 2 (1989)
            '''
        }
    }
    
    def get_ai_response(user_query):
        """Generate AI response based on user query"""
        query_lower = user_query.lower()
        
        # Check plant knowledge
        for disease, data in PLANT_KNOWLEDGE.items():
            for keyword in data['keywords']:
                if keyword in query_lower:
                    return data['response']
        
        # Check planet knowledge
        for planet, data in PLANET_KNOWLEDGE.items():
            for keyword in data['keywords']:
                if keyword in query_lower:
                    return data['response']
        
        # Default response
        return '''
**Hello! 👋**

I can help you with:

🌿 **Plant Knowledge:**
- Early blight, Late blight
- Healthy plant care
- Treatment & prevention tips

🪐 **Space Knowledge:**
- Mercury, Venus, Earth, Mars
- Jupiter, Saturn, Uranus, Neptune
- Planets facts & features

**Try asking about:**
- "What is early blight?"
- "How to prevent late blight?"
- "Tell me about Mars"
- "What's on Saturn's rings?"

What would you like to know? 😊
        '''
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat input
    user_input = st.text_input("💬 Ask anything about plants or planets...", key="chat_input")
    
    if user_input:
        # Add user message
        st.session_state.chat_history.append({
            'role': 'user',
            'message': user_input
        })
        
        # Generate bot response
        bot_response = get_ai_response(user_input)
        st.session_state.chat_history.append({
            'role': 'bot',
            'message': bot_response
        })
    
    # Display chat history with styling
    st.markdown('''
    <style>
        .chat-container {
            display: flex;
            flex-direction: column;
            gap: 10px;
            max-height: 500px;
            overflow-y: auto;
            padding: 15px;
            background: rgba(15, 32, 39, 0.05);
            border-radius: 10px;
        }
        .user-message {
            display: flex;
            justify-content: flex-end;
            margin-bottom: 8px;
        }
        .user-bubble {
            background: linear-gradient(135deg, #2575fc, #0f2027);
            color: white;
            padding: 12px 16px;
            border-radius: 18px 18px 4px 18px;
            max-width: 70%;
            word-wrap: break-word;
            font-size: 14px;
            box-shadow: 0 2px 8px rgba(37, 117, 252, 0.3);
        }
        .bot-message {
            display: flex;
            justify-content: flex-start;
            margin-bottom: 8px;
        }
        .bot-bubble {
            background: linear-gradient(135deg, #6a11cb, #2575fc);
            color: white;
            padding: 12px 16px;
            border-radius: 18px 18px 18px 4px;
            max-width: 85%;
            word-wrap: break-word;
            font-size: 14px;
            box-shadow: 0 2px 8px rgba(106, 17, 203, 0.3);
        }
        .chat-emoji {
            margin-right: 8px;
            font-size: 18px;
        }
    </style>
    ''', unsafe_allow_html=True)
    
    # Display chat messages
    if st.session_state.chat_history:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg in st.session_state.chat_history:
            if msg['role'] == 'user':
                st.markdown(
                    f'<div class="user-message"><div class="user-bubble"><span class="chat-emoji">👤</span>{msg["message"]}</div></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="bot-message"><div class="bot-bubble"><span class="chat-emoji">🤖</span>{msg["message"]}</div></div>',
                    unsafe_allow_html=True
                )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()


def run():
    """Entry point for executing the app from helpers or IDE run commands."""
    main()

if __name__ == '__main__':
    run()
