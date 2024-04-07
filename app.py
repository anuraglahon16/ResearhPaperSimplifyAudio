import streamlit as st
from tools import extract_text_from_pdf, start_working, text_to_audio

st.title("Research Paper Audio Conversation")

uploaded_file = st.file_uploader("Upload a research paper (PDF)", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Extracting text from PDF..."):
        research_paper_text = extract_text_from_pdf(uploaded_file)
    
    with st.spinner("Generating conversation..."):
        conversation = start_working(research_paper_text)
    
    with st.spinner("Converting conversation to audio..."):
        try:
            audio_files = text_to_audio(conversation)
            st.success("Audio conversation generated!")
            
            for audio_file in audio_files:
                st.audio(audio_file)
        except Exception as e:
            st.error(f"Error generating audio: {str(e)}")
    
    with st.expander("Conversation Transcript"):
        st.write(conversation)