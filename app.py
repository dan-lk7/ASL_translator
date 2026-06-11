import streamlit as st
import tempfile
import os

from backend.predict import predict_video
from translations import TRANSLATIONS

# =====================================
# CONFIG
# =====================================

st.set_page_config(
    page_title="ASL Translator",
    layout="centered"
)

st.title("🤟 ASL Translator")

st.markdown(
    """
    Traduction de la langue des signes américaine (ASL)
    """
)

# =====================================
# LANGUE
# =====================================

language = st.selectbox(
    "Choisir une langue",
    ["English", "Français", "Swahili"]
)

LANG_MAP = {
    "English": "en",
    "Français": "fr",
    "Swahili": "sw"
}

# =====================================
# UPLOAD VIDEO
# =====================================

st.subheader("📹 Charger une vidéo")

uploaded_file = st.file_uploader(
    "Choisir une vidéo",
    type=["mp4", "avi", "mov"]
)

if uploaded_file is not None:

    st.video(uploaded_file)

    if st.button("🔍 Traduire"):

        with st.spinner("Analyse en cours..."):

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            ) as tmp:

                tmp.write(uploaded_file.read())
                temp_path = tmp.name

            try:

                prediction, confidence = predict_video(
                    temp_path
                )

                translated_word = TRANSLATIONS[
                    prediction
                ][LANG_MAP[language]]

                st.success(
                    f"Signe détecté : {prediction}"
                )

                st.success(
                    f"Traduction : {translated_word}"
                )

                st.info(
                    f"Confiance : {confidence * 100:.2f}%"
                )

            except Exception as e:

                st.error(
                    f"Erreur : {str(e)}"
                )

            finally:

                if os.path.exists(temp_path):
                    os.remove(temp_path)

# =====================================
# INFORMATIONS
# =====================================

st.markdown("---")

st.markdown(
    """
### Signes supportés

- hello
- yes
- no
- drink
- help

### Instructions

1. Enregistrer une vidéo de 2 à 5 secondes.
2. Réaliser un signe.
3. Charger la vidéo.
4. Cliquer sur **Traduire**.
"""
)
