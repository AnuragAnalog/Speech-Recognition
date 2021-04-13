#!/usr/bin/python3.6

import nltk
import streamlit as st
import speech_recognition as sr

from nltk.sentiment.vader import SentimentIntensityAnalyzer

st.set_page_config(page_title="Audio Sentiment Analysis")

@st.cache
def download_nltk_files():
    nltk.download('punkt')
    nltk.download('vader_lexicon')

def transcribe_audio(fname):
    recognizer = sr.Recognizer()
    audio_file = sr.AudioFile(fname)

    with audio_file as src:
        audio_data = recognizer.record(src)

    return recognizer.recognize_google(audio_data)

download_nltk_files()

st.title("Audio Sentiment Analysis")
uploaded_files = st.file_uploader("Upload your audio file", type=['wav', 'mp3'], accept_multiple_files=True)
st.text("Select multiple files to do analysis on bulk")

senti = SentimentIntensityAnalyzer()

if uploaded_files is not None and len(uploaded_files) == 1:
    audio_file = uploaded_files[0]
    text = transcribe_audio(audio_file)
    st.write(f"**Audio Text**: {text}")

    hear_audio = st.checkbox("Want to Hear the audio?")
    if hear_audio:
        st.audio(audio_file)

    polarities = senti.polarity_scores(text)

    st.subheader("Calculated Sentiments")
    st.write(f"*Percentage of Positive:* {polarities['pos']}")
    st.write(f"*Percentage of Negative:* {polarities['neg']}")
    st.write(f"*Percentage of Neutral:* {polarities['neu']}")
    st.write(f"***Overall Sentiment:*** {polarities['compound']}")
elif len(uploaded_files) > 1:
    recognizer = sr.Recognizer()

    for f in uploaded_files:
        text = transcribe_audio(f)
        st.write(f"**{f.name}:** {text}")

        polarities = senti.polarity_scores(text)
        st.write(f"Sentiment: {polarities['compound']}")