import os
import streamlit as st
from PIL import Image
import tempfile
from pathlib import Path
import utils.sound as speech
import utils.face as face

st.write("# Speech-Splendid")

inputCont = st.container()
with inputCont:
    filename = st.file_uploader("Choose a recorded speech with your face clearly in frame", accept_multiple_files = False)
    st.write("\n\n")

if filename is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file: 
        fp = Path(tmp_file.name)
        fp.write_bytes(filename.getvalue())

    st.write("## Speech Analysis Results")
    speechCont = st.container()
    with speechCont:
        speech.extract_audio(tmp_file)

    st.write("## Face Analysis Results")
    faceCont = st.container()
    with faceCont:
        face.analyze_face(tmp_file)