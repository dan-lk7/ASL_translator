import streamlit as st
import tempfile
import os
import cv2
import time
from backend.predict import predict_video
import os
from translations import TRANSLATIONS
def speak(text):
    os.system(
        f'powershell -c "Add-Type -AssemblyName System.Speech; '
        f'(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{text}\')"'
    )
# WEBCAM RECORDING
def record_video(output_path="temp_webcam.mp4",duration=3):
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 20
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path,fourcc,fps,(width, height))
    start_time = time.time()
    while (time.time() - start_time) < duration:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        remaining = duration - int(time.time() - start_time)
        cv2.putText(frame,f"Recording: {remaining}s",(20, 40),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0),2)
        cv2.putText(frame,"Make your sign",(20, 80),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2)
        cv2.imshow("ASL Webcam",frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return output_path
# STREAMLIT UI
st.set_page_config(page_title="ASL Translator",layout="centered")
st.title("ASL Translator")
language = st.selectbox(
    "Language",
    ["English","Français","Swahili"])
LANG_MAP = {
    "English": "en",
    "Français": "fr",
    "Swahili": "sw"
}
# WEBCAM SECTION
st.subheader("🎥 Webcam")
if st.button("Capturer depuis Webcam"):
    with st.spinner("Enregistrement..."):
        video_path = record_video()
    st.success("Vidéo enregistrée")
    st.video(video_path)
    prediction, confidence = (predict_video(video_path))
    translated_word = TRANSLATIONS[prediction][LANG_MAP[language]]
    st.success(f"Translation : {translated_word}")
    st.info(f"Confidence : {confidence * 100:.2f}%")
    translations = {
        "hello": "Hello",
        "yes": "Yes",
        "no": "No",
        "drink": "Drink",
        "help": "Help"
    }
    speak(translated_word)
    if os.path.exists(video_path):
        os.remove(video_path)
# FILE UPLOAD SECTION
st.subheader("Upload vidéo")
uploaded_file = st.file_uploader("Choisir une vidéo",type=["mp4", "avi", "mov"],key="video_upload")
if uploaded_file is not None:
    st.video(uploaded_file)
    if st.button("Traduire"):
        with tempfile.NamedTemporaryFile(delete=False,suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name
        try:
            prediction, confidence = (predict_video(temp_path))
            st.success(f"Prediction : {prediction}")
            st.info(f"Confidence : {confidence * 100:.2f}%")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
