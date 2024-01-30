from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import platform
import numpy as np
import av 
from streamlit_webrtc import webrtc_streamer
import cv2

threshold1 = 100
threshold2 = 200

def normal_webcam():
    webrtc_streamer(key="example", video_transformer_factory=None)

def callback(frame):
    img = frame.to_ndarray(format="bgr24")

    img = cv2.cvtColor(cv2.Canny(img, threshold1, threshold2), cv2.COLOR_GRAY2BGR)

    return av.VideoFrame.from_ndarray(img, format="bgr24")
def canny_webcam():
    webrtc_streamer(key="example",  video_frame_callback=callback)


def chose_webcam_param():
    webrtc_streamer(key="example", fps=30, video_transformer_factory=None)

def sidebars(): 

    st.sidebar.markdown("""
    # Welcome to this Streamlit App!

    This app aim to use external usb camera from your pc and show it in this app.
    """)
    threshold1 = st.sidebar.slider("Threshold1", min_value=0, max_value=1000, step=1, value=100)
    threshold2 = st.sidebar.slider("Threshold2", min_value=0, max_value=1000, step=1, value=200)
    on = st.sidebar.toggle("Turn model on", value=False)   
    


    st.sidebar.markdown('''
    # Here some stats about pc in which is running this app
    ''')

    df = pd.DataFrame({'machine':platform.machine(), 'version':platform.version(), 'platform':platform.platform(),
    'uname': platform.uname(), 'system':platform.system(), 'processor':platform.processor()})

    st.sidebar.dataframe(df)

    return  on

def main():
    st.markdown("""

    # Welcome to this Streamlit App!

    This app aim to use external usb camera from your pc and show it in this app.
    """)
    on = sidebars()
   
    if on:
        st.write("switch is on")
        canny_webcam()    
    else:
        normal_webcam()


    save_file = st.checkbox("Save file")

    if save_file:
        st.write("File is saved")
   

if __name__ == "__main__":
    main()