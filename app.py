import streamlit as st
import tensorflow as tf
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os


# Load model
MODEL_PATH = "emotion_model.h5"
assert os.path.exists(MODEL_PATH), f"Model file not found at {MODEL_PATH}"
model = tf.keras.models.load_model(MODEL_PATH)

# Emotion labels (update if different)
labels = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']


st.title("🎵 Speech Emotion Recognition")
uploaded_file = st.file_uploader("Upload a .wav audio file", type=["wav"])

def extract_mel_spec(file):
    y, sr = librosa.load(file, duration=3, sr=22050)
    
    y = librosa.util.fix_length(y, size=sr * 3)


    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = mel_db[:, :128]
    return mel_db[np.newaxis, ..., np.newaxis]  # shape: (1, 128, 128, 1)

if uploaded_file is not None:
    st.audio(uploaded_file)
    features = extract_mel_spec(uploaded_file)
    prediction = model.predict(features)
    emotion = labels[np.argmax(prediction)]
    st.success(f"Predicted Emotion: **{emotion.upper()}**")
