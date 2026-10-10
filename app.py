import json
import os
from pathlib import Path
from datetime import datetime
from typing import Any

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'plant_model.h5'
CLASS_NAMES_PATH = BASE_DIR / 'class_names.json'
IMAGE_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.7

DISEASE_INFO = {
    'potato_early_blight': {
        'description': 'Early blight on potato causes brown spots, yellowing, and leaf drop.',
    },
    'potato_late_blight': {
        'description': 'Late blight produces dark, water-soaked lesions that spread quickly in cool, humid conditions.',
    },
    'potato_healthy': {
        'description': 'Healthy potato leaves with no visible disease signs.',
    },
    'tomato_early_blight': {
        'description': 'Early blight on tomato causes dark concentric spots and yellowing.',
    },
    'tomato_late_blight': {
        'description': 'Late blight on tomato causes rapidly spreading brown lesions and leaf collapse.',
    },
    'tomato_healthy': {
        'description': 'Healthy tomato leaves with bright green coloration and intact tissue.',
    },
    'Uncertain': {
        'description': 'The model could not make a confident prediction from this image.',
    },
}

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

@st.cache_data
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert('RGB')
    image = image.resize(IMAGE_SIZE)
    array = np.array(image).astype('float32') / 255.0
    array = np.expand_dims(array, axis=0)
    return array


def predict(image_array: np.ndarray) -> tuple[str, float]:
    model = load_model()
    class_names = load_class_names()
    if model is None or not class_names:
        return 'Uncertain', 0.0
    probabilities = model.predict(image_array, verbose=0)[0]
    top_index = int(np.argmax(probabilities))
    confidence = float(probabilities[top_index])
    class_name = class_names[top_index]
    return class_name, confidence


def main():
    st.set_page_config(page_title='Plant Leaf Disease Detection', page_icon='🌿', layout='centered')
    st.title('Plant Leaf Disease Detection')
    st.write('Upload a leaf image to get a disease prediction and confidence score.')

    uploaded_file = st.file_uploader('Upload a leaf image (JPG, JPEG or PNG)', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        model = load_model()
        if model is None:
            st.error('Model load failed. Please ensure plant_model.h5 exists.')
            return

        try:
            image = Image.open(uploaded_file).convert('RGB')
            image_array = preprocess_image(image)
            class_name, confidence = predict(image_array)
            score = confidence * 100

            st.image(image, caption='Uploaded Image', use_column_width=True)
            st.subheader('Prediction')
            st.write(f'**Class:** {class_name.replace("_", " ").title()}')
            st.write(f'**Confidence:** {score:.1f}%')

            description = DISEASE_INFO.get(class_name, DISEASE_INFO['Uncertain'])['description']
            st.subheader('Description')
            st.write(description)
        except Exception as e:
            st.error(f'Error processing image: {e}')
    else:
        st.info('Please upload an image to get started.')


if __name__ == '__main__':
    main()
