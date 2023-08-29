# app.py
# !pip install streamlit omegaconf scipy
# !pip install torch
from lightning.app import LightningApp
import torch
from io import BytesIO
from functools import partial
from scipy.io.wavfile import write
import streamlit as st


class StreamlitApp(app.components.ServeStreamlit):
    def build_model(self):
        sample_rate = 48000
        model, _ = torch.hub.load('snakers4/silero-models', model='silero_tts',speaker="v3_en")
        return partial(model.apply_tts, sample_rate=sample_rate, speaker="en_0"), sample_rate

    def render(self):
        st.title("Text To Speech")
        text = st.text_input("Text:", "Lightning Apps are the best!")

        if text:
            model, sample_rate = self.model
            audio_numpy = model(text).numpy()
            audio = BytesIO()
            write(audio, sample_rate, audio_numpy)
            audio.seek(0)
            st.audio(audio)

app = LightningApp(StreamlitApp())
