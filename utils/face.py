import cv2
import streamlit as st
from feat import Detector
import math
from deepface import DeepFace
import pandas as pd
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
    #st.write(frames)
    current_frame = math.trunc(frames/25)
    inc = math.trunc(frames/25)
    count = 0
    emotions = {"angry": 0, "disgust": 0, "fear": 0, "happy": 0, "sad": 0, "surprise": 0, "neutral": 0}
    rows = []
    crashed = 0 
 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        try:
            results = DeepFace.analyze(frame, actions = ["emotion"])
            #st.write(results)
            rows.append(results["emotion"])
        except:
            print("Row crashed")
            crashed += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        current_frame += inc
        count +=1
    print(count)
    print("Crashed: " + str(crashed))
    try:
        df = pd.DataFrame(rows)
        print(df.head())
        for x in emotions:
            emotions[x] = round(sum(df[x]), 2)
        #st.write(emotions)
        emotions_norm = {}
        for i in emotions:
            emotions_norm[i] = float(emotions[i])/sum(emotions.values())
            #print(str(emotions[i]) + "/" + str(sum(emotions.values())) + "=" + str(float(emotions[i])/sum(emotions.values())))
        c1, c2 = st.columns(2)
        sort_e = dict(sorted(emotions_norm.items(), key=lambda item: item[1], reverse = True))
        max_e = list(sort_e)[0]
        c1.metric("Overall Strongest Sentiment", max_e[0].upper() + max_e[1:])
        with st.expander("View All Subscores"):
            c2, c3, c4, c5, c6, c7, c8 = st.columns(7)
            c2.metric(list(sort_e)[0], str(round(list(sort_e.values())[0], 2)))
            c3.metric(list(sort_e)[1], str(round(list(sort_e.values())[1], 2)))
            c4.metric(list(sort_e)[2], str(round(list(sort_e.values())[2], 2)))
            c5.metric(list(sort_e)[3], str(round(list(sort_e.values())[3], 2)))
            c6.metric(list(sort_e)[4], str(round(list(sort_e.values())[4], 2)))
            c7.metric(list(sort_e)[5], str(round(list(sort_e.values())[5], 2)))
            c8.metric(list(sort_e)[6], str(round(list(sort_e.values())[6], 2)))
    except:
        st.write("Sorry, we're having some trouble processing that video. Try re-recording it or recording it on a different device.")
    #st.write(emotions_norm)
    # video_prediction = detector.detect_video(tmp_file.name, skip_frames = math.trunc(frames/3))
    # emotions = ['fear', 'happiness', 'sadness', 'surprise', 'neutral', 'anger', 'disgust']
    # readings = []
    # print(video_prediction.head())
    # for x in emotions:
    #     readings.append(round(sum(video_prediction[x]),2))
    # sorted_emotion = [x for x in sorted(zip(readings, emotions), reverse=True)]
    # #print(sorted_emotion)
    # st.write(f"The top emotions detected were: {sorted_emotion[0][1]}, {sorted_emotion[1][1]}")
    # if sorted_emotion[0][1] != "happiness" and sorted_emotion[0][1] != "happiness":
    #     st.write("Tip: Smile more!")
    # with st.expander("Detailed View"):
    #     c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
    #     c1.metric("Fear", str(readings[0]))
    #     c2.metric("Joy", str(readings[1]))
    #     c3.metric("Sadness", str(readings[2]))
    #     c4.metric("Surprise", str(readings[3]))
    #     c5.metric("Neutral", str(readings[4]))
    #     c6.metric("Anger", str(readings[5]))
    #     c7.metric("Disgust", str(readings[6]))
    #     print(video_prediction.head())
    #     print(video_prediction.shape)
    #     print(sorted_emotion)
    #except:
        #st.write("There was a problem processing the video. Make sure you are centered in it and that the quality is high. Otherwise, try selecting a smaller clip.")

        
