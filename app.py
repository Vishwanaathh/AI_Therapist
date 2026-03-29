
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os
from dotenv import load_dotenv
from pathlib import Path
import google.generativeai as genai
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import yagmail
from gtts import gTTS
import io

# =============================
# INIT
# =============================
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# =============================
# EMAIL CONFIG (HARDCODED)
# =============================
EMAIL_USER = "csfinancialservices4@gmail.com"
EMAIL_PASS = "ckvv hidk ikxq ugmf"

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
emotion_model = YOLO("models/best.pt")

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =============================
# PAGE CONFIG + STYLING
# =============================
st.set_page_config(page_title="VishwAI Therapist", layout="centered")

st.markdown("""
<style>
.stApp {
    background-color: #ffe4ec;
    color: #6a0dad;
    font-family: Arial;
}
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #6a0dad;
}
.helpline-box {
    background-color: #ffccd9;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 25px;
}
.emotion-box {
    background-color: #ff99bb;
    padding: 12px;
    border-radius: 12px;
    text-align: center;
}
[data-testid="stChatMessage"] {
    background-color: #ffb6c1;
    border-radius: 12px;
    padding: 10px;
}
button, .stButton>button {
    background-color: #ff99bb;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">💗 VishwAI – Emotion Aware Therapist</div>', unsafe_allow_html=True)

# =============================
# HELPLINE
# =============================
st.markdown("""
<div class="helpline-box">
🚨 <b>If you are in immediate danger, call emergency services.</b><br><br>
🇮🇳 India: 1800-599-0019
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

# =============================
# CAMERA
# =============================
st.subheader("📷 Capture Your Emotion")

camera_image = st.camera_input("")

if camera_image is not None:
    file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))

        results = emotion_model(face, verbose=False)
        probs = results[0].probs

        label = emotion_model.names[probs.top1]
        confidence = float(probs.top1conf)

        st.session_state.detected_emotion = label

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, f"{label} ({confidence:.2f})",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0,255,0), 2)

    st.session_state.processed_image = frame

if st.session_state.processed_image is not None:
    st.image(st.session_state.processed_image, channels="BGR")
    st.markdown(
        f'<div class="emotion-box">Detected Emotion: {st.session_state.detected_emotion}</div>',
        unsafe_allow_html=True
    )

# =============================
# CHAT
# =============================
st.subheader("💬 Talk to VishwAI")

user_input = st.chat_input("How are you feeling today?")

if user_input:
    scores = sia.polarity_scores(user_input)
    compound = scores["compound"]

    if compound >= 0.05:
        sentiment_label = "Positive 😊"
    elif compound <= -0.05:
        sentiment_label = "Negative 😞"
    else:
        sentiment_label = "Neutral 😐"

    with st.chat_message("user"):
        st.write(user_input)

    st.markdown(f"**Sentiment:** {sentiment_label} | Score: {compound:.2f}")

    prompt = f"""
    You are VishwAI, a calm AI therapist.

    Detected emotion: {st.session_state.detected_emotion}
    User message: {user_input}

    Respond empathetically.
    """

    response = llm.generate_content(prompt)
    assistant_text = response.text

    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("assistant", assistant_text))

    with st.chat_message("assistant"):
        st.write(assistant_text)

    # =============================
    # TEXT TO SPEECH
    # =============================
    tts = gTTS(text=assistant_text, lang="en")
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)

    st.audio(mp3_fp, format="audio/mp3")

# =============================
# DISPLAY HISTORY
# =============================
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(message)

# =============================
# DOWNLOAD CHAT
# =============================
if st.session_state.chat_history:
    chat_text = ""
    for role, message in st.session_state.chat_history:
        chat_text += f"{role.upper()}: {message}\n\n"

    st.download_button(
        label="⬇️ Download Chat",
        data=chat_text,
        file_name="vishwai_chat.txt",
        mime="text/plain"
    )

# =============================
# EMAIL CHAT
# =============================
st.subheader("📧 Send Chat via Email")

receiver_email = st.text_input("Enter recipient email")

if st.button("Send Email"):
    if receiver_email and st.session_state.chat_history:
        try:
            yag = yagmail.SMTP(EMAIL_USER, EMAIL_PASS)

            chat_text = ""
            for role, message in st.session_state.chat_history:
                chat_text += f"{role.upper()}: {message}\n\n"

            yag.send(
                to=receiver_email,
                subject="VishwAI Chat Conversation",
                contents=chat_text
            )

            st.success("✅ Email sent successfully!")

        except Exception as e:
            st.error(f"❌ Failed to send email: {e}")
    else:
        st.warning("⚠️ Enter email and have at least one message")
