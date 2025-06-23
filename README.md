#  Speech Emotion Recognition (SER) Using CNN

This project builds a **Speech Emotion Recognition (SER)** system using **Mel spectrograms** and a **Convolutional Neural Network (CNN)** model. The goal is to classify human emotions like *happy*, *sad*, *angry*, etc., from `.wav` audio clips. The final model is deployed via a user-friendly **Streamlit web app**.

---

## Project Overview

- **Dataset**: [RAVDESS](https://zenodo.org/record/1188976) (Ryerson Audio-Visual Database of Emotional Speech and Song)
- **Classes**: `['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']`
- **Tech Stack**:
  - Feature Extraction: `librosa`
  - Model: CNN using `TensorFlow/Keras`
  - Deployment: `Streamlit` app for interactive prediction
- **Approach**:
  - Audio preprocessing
  - Mel spectrogram extraction
  - Model training (80/20 split)
  - Accuracy/loss visualization
  - Real-time audio file prediction via UI

---

## 🗂️ Folder Structure

MARS/
├── app.py # Streamlit app for deployment
├── train_model.py # Code to train CNN model
├── model/
│ └── emotion_model.h5 # Trained Keras model
├── utils/
│ └── feature_extraction.py # Mel-spectrogram feature function
├── data/
│ └── Audio_Speech_Actors/ # Extracted RAVDESS speech audio files
├── requirements.txt
└── README.md


---

## 🔧 Setup & Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/Payal-iitr/speech-emotion-recognition.git
   cd speech-emotion-recognition
2. Install required packages
pip install -r requirements.txt
3. Download and prepare data

**Download the two RAVDESS zip files:**
Audio_Speech_Actors_01-24.zip
Audio_Song_Actors_01-24.zip
Extract into a single directory (e.g., data/Audio_Speech_Actors)

**Model Evaluation**

Model Evaluation
Train/Validation Split: 80/20

Model Architecture: Convolutional Neural Network (CNN) trained on Mel spectrograms

Evaluation Metrics:

Test Accuracy:63 %

Macro F1-score: 0.62

Weighted F1-score: 0.63

Performance by Emotion:

Neutral: F1-score 0.75

Angry: F1-score 0.75

Happy: F1-score 0.64

Disgust and Fearful show lower performance due to fewer examples and emotional overlap

**Observations:**

High precision in Neutral and Surprised, though recall varies

Balanced macro and weighted averages show moderate generalization

Useful foundation for improving class balance, data augmentation, or deeper CNNs

**Features**

Upload .wav file via drag-and-drop

Real-time prediction of emotion

Clean and interactive UI with Streamlit

Playback of uploaded audio

**Libraries Used**

Python

NumPy, Pandas, Matplotlib, Seaborn

TensorFlow/Keras

scikit-learn

Librosa (for audio processing)

Streamlit (for deployment)

**Key Functions**

extract_mel_spec() – Extracts and pads Mel spectrogram features

train_model() – Builds, trains, and saves CNN model

predict_emotion() – Loads model & predicts emotion from new audio

app.py – Streamlit frontend
