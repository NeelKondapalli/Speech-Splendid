import cv2
import streamlit as st
from feat import Detector
import math

def load_detector():
    return Detector(
        face_model = "retinaface",
        landmark_model = "mobilefacenet",
        au_model = "svm",
        emotion_model = "resmasknet",
        facepose_model = "img2pose",
    )
detector = load_detector()

def analyze_face(tmp_file):
    cap = cv2.VideoCapture(tmp_file.name)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    try: 
        video_prediction = detector.detect_video(tmp_file.name, skip_frames = math.trunc(frames/3))
        emotions = ['fear', 'happiness', 'sadness', 'surprise', 'neutral', 'anger', 'disgust']
        readings = []
        print(video_prediction.head())
        for x in emotions:
            readings.append(round(sum(video_prediction[x]),2))
        sorted_emotion = [x for x in sorted(zip(readings, emotions), reverse=True)]
        #print(sorted_emotion)
        st.write(f"The top emotions detected were: {sorted_emotion[0][1]}, {sorted_emotion[1][1]}")
        if sorted_emotion[0][1] != "happiness" and sorted_emotion[0][1] != "happiness":
            st.write("Tip: Smile more!")
        with st.expander("Detailed View"):
            c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
            c1.metric("Fear", str(readings[0]))
            c2.metric("Joy", str(readings[1]))
            c3.metric("Sadness", str(readings[2]))
            c4.metric("Surprise", str(readings[3]))
            c5.metric("Neutral", str(readings[4]))
            c6.metric("Anger", str(readings[5]))
            c7.metric("Disgust", str(readings[6]))
            print(video_prediction.head())
            print(video_prediction.shape)
            print(sorted_emotion)
    except:
        st.write("There was a problem processing the video. Make sure you are centered in it and that the quality is high. Otherwise, try selecting a smaller clip.")

        
