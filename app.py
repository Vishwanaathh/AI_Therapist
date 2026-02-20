import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os
from dotenv import load_dotenv
from pathlib import Path
import google.generativeai as genai

# =======================
# Fix SpeechBrain torchaudio backend
# =======================
import torchaudio
torchaudio.set_audio_backend("sox_io")

from speechbrain.pretrained import EncoderClassifier
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# =============================
# LOAD ENV
# =============================
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY not found in .env file")
    st.stop()

genai.configure(api_key=api_key)
llm = genai.GenerativeModel("models/gemini-2.5-flash")

# =============================
# LOAD MODELS
# =============================
# Facial emotion model
emotion_model = YOLO("models/best.pt")
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Voice emotion model
voice_model = EncoderClassifier.from_hparams(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    savedir="pretrained_models/emotion_recognition"
)

# =============================
# PAGE CONFIG + STYLING
# =============================
st.set_page_config(page_title="VishwAI Therapist", layout="centered")

st.markdown("""
<style>
.stApp { background-color: #ffe4ec; color: #6a0dad; font-family: Arial, sans-serif; }
.main-title { text-align: center; font-size: 42px; font-weight: bold; color: #6a0dad; margin-bottom: 20px; }
.helpline-box { background-color: #ffccd9; padding: 20px; border-radius: 15px; border: 1px solid #ff99bb; margin-bottom: 25px; color: #6a0dad; font-weight: 500; line-height: 1.6; }
.helpline-text { color: #6a0dad; font-size: 16px; }
[data-testid="stChatMessage"] { background-color: #ffb6c1; border-radius: 12px; padding: 10px; color: #6a0dad; }
.emotion-box { background-color: #ff99bb; padding: 12px; border-radius: 12px; margin-top: 15px; text-align: center; font-weight: bold; color: #6a0dad; }
body, p, span, li { color: #6a0dad; }
button, .stButton>button { background-color: #ff99bb; color: #6a0dad; border-radius: 10px; border: none; padding: 8px 16px; font-weight: bold; }
button:hover, .stButton>button:hover { background-color: #ff85aa; cursor: pointer; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">💗 VishwAI – Emotion Aware Therapist</div>', unsafe_allow_html=True)

# =============================
# HELPLINE
# =============================
st.markdown("""
<div class="helpline-box">
<div class="helpline-text">
🚨 <b>If you are in immediate danger, please call your local emergency number.</b><br><br>
🇺🇸 USA: Call or Text <b>988</b><br>
🇮🇳 India: Kiran Mental Health Helpline <b>1800-599-0019</b><br>
🇬🇧 UK: Samaritans <b>116 123</b><br>
🇨🇦 Canada: Talk Suicide <b>1-833-456-4566</b><br>
🌍 International Help:
<a href="https://www.opencounseling.com/suicide-hotlines" target="_blank">
Find your country here
</a>
</div>
</div>
""", unsafe_allow_html=True)

# =============================
# SESSION STATE
# =============================
if "detected_emotion" not in st.session_state:
    st.session_state.detected_emotion = "Unknown"
if "processed_image" not in st.session_state:
    st.session_state.processed_image = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "detected_voice_emotion" not in st.session_state:
    st.session_state.detected_voice_emotion = "Unknown"

# =============================
# CAMERA SECTION
# =============================
st.subheader("📷 Capture Your Emotion")
camera_image = st.camera_input("")

if camera_image is not None:
    file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    detected = "No face detected"

    for (x, y, w, h) in faces:
        face = cv2.resize(frame[y:y+h, x:x+w], (224, 224))
        results = emotion_model(face, verbose=False)
        probs = results[0].probs
        label = emotion_model.names[probs.top1]
        confidence = float(probs.top1conf)

        detected = f"{label} ({confidence:.2f})"
        st.session_state.detected_emotion = label

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, detected, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    st.session_state.processed_image = frame

if st.session_state.processed_image is not None:
    st.image(st.session_state.processed_image, channels="BGR")
    st.markdown(f'<div class="emotion-box">Detected Facial Emotion: {st.session_state.detected_emotion}</div>', unsafe_allow_html=True)

# =============================
# VOICE SECTION
# =============================
st.subheader("🎤 Speak Your Feelings")
audio_input = st.audio_input("Record your voice (wav, max 10s)")

if audio_input:
    audio_bytes = audio_input.read()
    with open("user_audio.wav", "wb") as f:
        f.write(audio_bytes)

    tone_probs = voice_model.classify_file("user_audio.wav")
    voice_label = tone_probs[3]
    st.session_state.detected_voice_emotion = voice_label

    st.markdown(f'<div class="emotion-box">Detected Voice Emotion: {voice_label}</div>', unsafe_allow_html=True)

# =============================
# CHAT SECTION
# =============================
st.subheader("💬 Talk to VishwAI")
user_input = st.chat_input("How are you feeling today?")

if user_input:
    # TEXT SENTIMENT
    scores = sia.polarity_scores(user_input)
    compound = scores["compound"]
    if compound >= 0.05:
        sentiment_label = "Positive 😊"
    elif compound <= -0.05:
        sentiment_label = "Negative 😞"
    else:
        sentiment_label = "Neutral 😐"

    # Display sentiment
    st.chat_message("user")
    st.write(user_input)
    st.markdown(f"**Sentiment:** {sentiment_label} | **Compound Score:** {compound:.2f}")

    # GEMINI PROMPT
    prompt = f"""
You are VishwAI, a calm, emotionally intelligent AI therapist.

Detected facial emotion: {st.session_state.detected_emotion}
Detected voice emotion: {st.session_state.get('detected_voice_emotion', 'Unknown')}
User message sentiment: {sentiment_label} (compound: {compound:.2f})

User message: {user_input}

Respond empathetically.
Acknowledge facial, voice, and text emotions naturally.
Do not give medical diagnosis.
"""

    response = llm.generate_content(prompt)

    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("assistant", response.text))

# DISPLAY CHAT HISTORY
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(message)