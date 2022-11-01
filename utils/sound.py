import os
import wave
import contextlib
import streamlit as st
import moviepy.editor as mp
from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from expertai.nlapi.cloud.client import ExpertAiClient

os.environ["EAI_USERNAME"] = st.secrets["username"]
os.environ["EAI_PASSWORD"] = st.secrets["password"]
client = ExpertAiClient()
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
    feedback = []
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
    st.write("### Linguistic Analysis Scores")
    c1, c2 = st.columns(2)
    with c1:
        if seconds:
            wpm = round((len(transcript.split()))*60/seconds)
            st.write(f"Words/Minute ≈ {wpm}")
    fluff = ['um', 'uh', 'ah', 'er', 'oh']
    fluff_c = 0
    hedging = ['i mean', 'i guess', 'i suppose', 'i think', 'you know']
    hedge_c = 0
    filler_adverbs = ['very', 'really', 'totally', 'actually', 'basically', 'seriously']
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
    fluff_score = round((fluff_c/len(tlist)) * 100)
    hedge_score = round((hedge_c/len(tlist)) * 100)
    adverb_score = round((ad_c/len(tlist)) * 100)
    with c1:
        st.write(f"Filler word count: {fluff_c} ({fluff_score}%)")
        st.write(f"Hedging language word count: {hedge_c} ({hedge_score}%)")
        st.write(f"Empty Adverb word count: {ad_c} ({adverb_score}%)")
    if fluff_score:
        feedback.append("- Use less filler words such as 'uh', 'ah', or 'um'. These words add uncertainty to your speech and indicate a lack of confidence. Practice the speech more so you pause less.")
    if hedge_score:
        feedback.append("- Phrases such as 'I mean', 'I guess', or 'I suppose' are known as hedging language, and they decrease the confidence of your speech. Decrease your usage of them")
    if adverb_score:
        feedback.append("- Adverbs such as 'really', 'very', and 'totally' are unnecessary in most cases and can be eliminated with better word choice. Try searching up words that are equivalent to what you are trying to say")
    if wpm < 120:
        feedback.append(f"- Your words/min rate was {wpm}. For most speeches, the recommended rate is between 120 and 160. Try speaking a bit faster on your next try.")
    if wpm > 160:
        feedback.append(f"- Your words/minute rate was {wpm}. For most speeches, the recommended rate is between 120 and 160. Try slowing down on your next try.")
    with c2:
        with st.expander("Metric Information"):
                st.write("Overall Sentiment: The general emotion in the text. Calculated via the compound score")
                st.write("Compound: The weighted sum of the positive, neutral, and negative scores which is normalized between -1 and 1.")
                st.write("Filler Word Count: Filler words such as 'uh' and 'um' drastically decrease the confidence of your speech.")
                st.write("Hedging Language: Language such as 'I mean' or 'I guess' indicate a lack of confidence")
                st.write("Empty Adverbs: Adverbs such as 'really' or 'very' should be avoided and replaced with better language.")
        with st.expander("Topic & Behavioral Analysis"):
            taxonomy = 'iptc'
            language= 'en'
            output_cats = client.classification(body={"document": {"text": transcript}}, params={'taxonomy': taxonomy, 'language': language})
            cats = []
            for category in output_cats.categories:
                for x in category.hierarchy:
                    cats.append(x)
            st.write("Detected topics:")
            st.write(cats)
            st.write("If these topics do not line up with what the speech was about, you might need to use clearer language or avoid steering off topic.\n")

            st.write("Behavioral Characteristics:")
            output_bev = client.classification(body={"document": {"text": transcript}}, params={'taxonomy': "behavioral-traits", 'language': language})
            behaviors = []
            for bev in output_bev.categories:
                behaviors.append(bev.hierarchy[2])
            st.write(behaviors)
    with st.expander("Feedback"):
        for x in feedback:
            st.write(x)
 


    

 