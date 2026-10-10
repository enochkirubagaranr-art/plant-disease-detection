🌿 Plant Disease Detection App

An AI-powered web application that helps users identify plant diseases from leaf images using a trained deep learning model.

🚀 Live Demo

Try the app here: "Plant Disease Detector" (https://plant-disease-detector-pdd.streamlit.app/)

Upload a plant leaf image to the app and explore its disease classification prediction.

✨ Features

- 🌱 Plant Disease Classification — predicts plant disease classes from uploaded leaf images.
- 🧠 Deep Learning Model — uses a trained TensorFlow/Keras model.
- 🖼️ Image Upload — allows users to submit leaf images for analysis.
- 🌐 Web-Based Interface — built with Streamlit and accessible through a public URL.
- 🔊 Text-to-Speech Support — includes gTTS integration for audio output, where implemented in the app.

🛠️ Technologies Used

- Python
- Streamlit
- TensorFlow / Keras
- NumPy
- Pillow (PIL)
- gTTS (Google Text-to-Speech)

📂 Main Project Files

File| Purpose
"streamlit_app_FINAL.py"| Main Streamlit application
"FINAL_MODEL.h5"| Trained plant disease classification model
"class_names.json"| Class labels used by the model
"requirements.txt"| Python dependencies

💻 Run Locally

1. Clone this repository:
   
   git clone https://github.com/enochkirubagaranr-art/plant-disease-detection.git

2. Open the project folder:
   
   cd plant-disease-detection

3. Create and activate a virtual environment (optional but recommended):
   
   python -m venv venv
   
   Windows PowerShell:
   
   .\venv\Scripts\Activate.ps1

4. Install the dependencies:
   
   pip install -r requirements.txt

5. Start the application:
   
   streamlit run streamlit_app_FINAL.py

6. Open the local URL displayed in your terminal.

🎯 Project Objective

The goal of this project is to demonstrate how deep learning can support plant disease identification through image classification and provide an accessible web interface for users.

⚠️ Disclaimer

Predictions are model-generated estimates and may not always be accurate. For important agricultural decisions, confirm results with a qualified agricultural expert.

👨‍💻 Author

Enoch Kirubagaran R

🔗 Links

- Live Application: https://plant-disease-detector-pdd.streamlit.app/
- GitHub Repository: https://github.com/enochkirubagaranr-art/plant-disease-detection
