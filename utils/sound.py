import wave
import contextlib
import streamlit as st
import moviepy.editor as mp
from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer





authenticator = IAMAuthenticator(st.secrets["key"])
speech_to_text = SpeechToTextV1(authenticator=authenticator)
speech_to_text.set_service_url(st.secrets["url"])
def sentiment_vader(sentence):
    sid_obj = SentimentIntensityAnalyzer()

    sentiment_dict = sid_obj.polarity_scores(sentence)
    negative = sentiment_dict['neg'] * 100
    neutral = sentiment_dict['neu'] * 100
    positive = sentiment_dict['pos'] * 100
    compound = sentiment_dict['compound'] * 100

    if compound >= 30 :
        overall_sentiment = "Positive"

    elif compound <= - 30 :
        overall_sentiment = "Negative"

    else :
        overall_sentiment = "Neutral"
  
    return negative, neutral, positive, compound, overall_sentiment


def extract_audio(video_file):
    """
    Extracts .wav audio file from video.

    video_file: file to seperate
    """
    video_name = video_file.name
    clip = mp.VideoFileClip(video_name)
    audio_path = "file_audio.wav"
    clip.audio.write_audiofile(audio_path)
    analyze_speech(audio_path)

def analyze_speech(filename):
    with contextlib.closing(wave.open(filename,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        seconds = frames / float(rate)
    with open(filename, 'rb') as audio_file:
        speech_recognition_results = speech_to_text.recognize(
            audio=audio_file,
            content_type='audio/wav'
        ).get_result()
    transcript = ""
    for x in speech_recognition_results['results']:
        transcript += x["alternatives"][0]["transcript"]
    transcript = transcript.replace("%HESITATION", "uh")
    with st.expander("View Transcription"):
            st.write(transcript)
    negative, neutral, positive, compound, overall_sentiment = sentiment_vader(transcript)
    st.write("### Sentiment Analysis Scores")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Overall Sentiment", overall_sentiment)
    c2.metric("Compound", str(round(compound)) + "%")
    c3.metric("Positive", str(round(positive))+ "%")
    c4.metric("Neutral", str(round(neutral)) + "%")
    c5.metric("Negative", str(round(negative)) + "%")
    c1, c2 = st.columns(2)
    with c1:
        if seconds:
            st.write(f"Words/Minute ≈ {round((len(transcript.split()))*60/seconds)}")
            st.write("(RECOMMENDED): 120-160 WPM")
    fluff = ['um', 'uh', 'ah', 'er', 'oh']
    fluff_c = 0
    hedging = ['i mean', 'i guess', 'i suppose']
    hedge_c = 0
    filler_adverbs = ['very', 'really']
    ad_c = 0
    tlist = transcript.split()
    for x in tlist:
        if x.lower() in fluff:
            fluff_c += 1
        if x.lower() in filler_adverbs:
            ad_c += 1
    for x in range(len(tlist)-1):
        if (tlist[x] + " " + tlist[x+1]) in hedging:
            hedge_c += 1
    with c1:
        st.write(f"Filler word count: {fluff_c} ({round((fluff_c/len(tlist)) * 100)}%)")
        st.write(f"Hedging language word count: {hedge_c} ({round((hedge_c/len(tlist)) * 100)}%)")
        st.write(f"Empty Adverb word count: {ad_c} ({round((ad_c/len(tlist)) * 100)}%)")
    with c2:
        with st.expander("Metric Information"):
                st.write("Overall Sentiment: The general emotion in the text. Calculated via the compound score")
                st.write("Compound: The sum of the positive, neutral, and negative scores which is normalized between -1 and 1.")
                st.write("Filler Word Count: Filler words such as 'uh' and 'um' drastically decrease the confidence of your speech.")
                st.write("Hedging Language: Language such as 'I mean' or 'I guess' indicate a lack of confidence")
                st.write("Empty Adverbs: Adverbs such as 'really' or 'very' should be avoided and replaced with better language.")
        


    

 