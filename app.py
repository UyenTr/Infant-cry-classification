
import streamlit as st
import joblib
import base64,time
import logging
import pyaudio, wave, pylab
import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
# from pygame import mixer
from scipy.io.wavfile import write
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, make_scorer
from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV,cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from main import FeatureExtraction 
import pandas as pd

st.set_page_config(page_title="Infants translate", page_icon=":baby:")
new_title = "<h1 style='text-align: center; style=font-size: 42px;'>INFANTS TRANSLATE</h1>"
st.markdown(new_title, unsafe_allow_html=True)
instruction = "<h3 style='text-align: center; style=font-size: 30px;'> Click '1. Start recording' to start </h3>"
st.markdown(instruction, unsafe_allow_html=True)

# st.title('Infants talk')
"""###"""

st.markdown(
    """

    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 500px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 500px;
        margin-left: -500px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
hello= ("<h1 style='text-align: center; style=font-size: 30px;'>Hello, there!</h1>")
st.sidebar.markdown(hello, unsafe_allow_html=True)
greet = "<h4 style='text-align: center; style=font-size: 30px;'>Crying is the only way for your baby communicate!</h4>"
st.sidebar.markdown(greet, unsafe_allow_html=True)
greet1 = "<h3 style='text-align: center; style=font-size: 30px;'> In 3 steps, you will find out what baby trying to tell.</h3>"
st.sidebar.markdown(greet1, unsafe_allow_html=True)
st.sidebar.image("st_img.gif", use_column_width=True)

#####################

##############################
#SETTING RECORDED AUDIO
##############################
feature_extraction = FeatureExtraction()

class_list = ['burp', 'hungry', 'lowerwindpain', 'tired', 'uncomfortable']

@st.cache
def load_model(path):
    model = joblib.load(path)
    return model

pipeline = load_model('svc_final_pipeline.pkl')

def preprocess_input(path):
    audio_data, sr = librosa.load(path, sr=44100, mono=True, duration=3)
    mean_features, label = feature_extraction.feature_extraction(data=audio_data)
    mean_features = np.array(mean_features)
    mean_features  = np.squeeze(mean_features)

    return mean_features

# def plot_prob(prediction,class_list):
#     pred = prediction[0]
#     df = pd.DataFrame(list(zip(class_list, pred)))
#     clrs = ['cornflowerblue' if (x < np.max(pred)) else 'deeppink'for x in pred ]
#     fig, ax = plt.subplots(figsize=(1,1))
#     ax.bar (df[0],df[1],color=clrs,align='center')
#     ax.set_yticks([])

#     return st.pyplot(fig)

def record(duration):
    filename = "recorded.wav"
    chunk = 1024
    FORMAT = pyaudio.paInt16
    channels = 1
    sample_rate = 44100
    record_seconds = duration
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
    channels=channels,
    rate=sample_rate,
    input=True,
    output=True,
    frames_per_buffer=chunk)
    frames = []
    for i in range(int(44100 / chunk * record_seconds)):
        data = stream.read(chunk)
        frames.append(data)
        wf = wave.open(filename, "wb")
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))
        wf.close() 
        audio="recorded.wav"
def burp():
    msg=("<h3 style='text-align: center; style=font-size: 30px;'>BURP me please!</h3>")
    st.markdown(msg, unsafe_allow_html=True)
    msg2=("<h4 style='text-align: center'>Here are few suggestions for how to burp your baby:</h4>")
    st.markdown(msg2, unsafe_allow_html=True)
    col1, col2 = st.beta_columns(2)
    with col1:
        st.write('')
        st.write('')
        st.image('burp_tech.gif')
    with col2:
        st.image('burp_tech1.gif')

def hungry():
    msgh=("<h3 style='text-align: center; style=font-size: 30px;'>A MILK bottle please!</h3>")
    st.markdown(msgh, unsafe_allow_html=True)
    st.image('milk.png')

def pain():
    msgp=("<h3 style='text-align: center; style=font-size: 30px;'>I have BELLY pain!</h3>")
    st.markdown(msgp, unsafe_allow_html=True)
    msgp2=("<h4 style='text-align: center'>A good tummy massage would help. You can refer to below technique</h4>")
    st.markdown(msgp2, unsafe_allow_html=True)
    st.image('Baby_Massage_Gas.png')

def tired():
    msgp=("<h3 style='text-align: center; style=font-size: 30px;'>I need to SLEEP!</h3>")
    st.markdown(msgp, unsafe_allow_html=True)



def discomfort():
    msgd=("<h3 style='text-align: center; style=font-size: 30px;'>Something BOTHERS me!</h3>")
    st.markdown(msgd, unsafe_allow_html=True)
    msgd2=("<h4 style='text-align: center'>Check if the baby is too HOT, too COLD or a NAPKIN change is needed</h4>")
    st.markdown(msgd2, unsafe_allow_html=True)
    msg4=("<h5 style='text-align: center'>After you've done everything and the baby is still crying, here is a well-known technique to stop baby cry </h5>")
    st.markdown(msg4, unsafe_allow_html=True)
    img1 =("<div style='text-align: center'><img src='https://j.gifs.com/m8J9kM.gif' /></div>")
    st.markdown(img1,unsafe_allow_html=True)
    # st.image('nothing.gif')

###################
#Content
###################
col1, col2 = st.beta_columns([0.4,1])
# this will put a button in the middle column
with col1:
        
    # if st.button("Start Recording"):
    #     with st.spinner("Listening"):
    #         record(3)
    #         st.success("Recording completed")
    st.text('')
    start_execution = st.button(' 1. Start recording')
    if start_execution:
        with col2:
            gif_runner = st.image('listen.gif')
            record(3)
            gif_runner.empty()
            s = st.success("Recording completed")   


col1, col2 = st.beta_columns([0.4,1])
with col1:
    st.text('')
    if st.button("2. Check recording"):
        with col2:
            audio_file = open("recorded.wav", 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/wav',start_time=0)
            

st.text('')
start_execution = st.button('3. Translate now')
if start_execution:
    # img2 =("<div style='text-align: center'><img src='https://im3.ezgif.com/tmp/ezgif-3-4cb06ad2d23b.gif' /></div>")
    gif_runner = st.image('thinking.gif')
    X_val = preprocess_input('recorded.wav')
    prediction = pipeline.predict([X_val])
    result = {0:'burp', 1:'hungry',2: 'lowerwindpain',3:'tired', 4:'uncomfortable'}
    # max_pos= prediction.argmax()
    # st.write(max_pos)
    # st.write(prediction)
    for i in result.keys():
        if prediction == i:
            label = result[i]
            if result [i] == 'burp':
                burp()
            elif result[i] == 'hungry':
                hungry()
            elif result[i] == 'lowerwindpain':
                pain()
            elif result[i] == 'tired':
                tired()
            else:
                discomfort()
    gif_runner.empty()
  


