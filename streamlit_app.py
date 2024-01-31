from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import platform
import numpy as np
import os
import av 
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, ClientSettings #to be installed
import cv2 #to be installed headless
import threading
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore
from PIL import Image
import time
from typing import Union #to be installed
from io import BytesIO
from bulb_detection import detect_bulb

threshold1 = 100
threshold2 = 200

RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
MEDIA_STREAM_CONSTRAINTS = {"audio": False, "video": {"width": {"min": 800, "ideal": 1200, "max": 1920 }, "height": {"min": 600, "ideal": 900, "max": 1080 }}}

WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints=MEDIA_STREAM_CONSTRAINTS
)


class OpenCVVideoProcessor(VideoProcessorBase):
        type: Literal["noop", "cartoon", "edges", "rotate"]
        out_image: np.ndarray = None

        def __init__(self) -> None:
            self.type = "noop"

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")

            if self.type == "noop":
                pass
                self.out_image = img.copy()
            elif self.type == "cartoon":
                # prepare color
                img_color = cv2.pyrDown(cv2.pyrDown(img))
                for _ in range(6):
                    img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
                img_color = cv2.pyrUp(cv2.pyrUp(img_color))

                # prepare edges
                img_edges = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img_edges = cv2.adaptiveThreshold(
                    cv2.medianBlur(img_edges, 7),
                    255,
                    cv2.ADAPTIVE_THRESH_MEAN_C,
                    cv2.THRESH_BINARY,
                    9,
                    2,
                )
                img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)

                # combine color and edges
                img = cv2.bitwise_and(img_color, img_edges)
                self.out_image = img.copy()
            elif self.type == "edges":
                # perform edge detection
                img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)
                self.out_image = img.copy()
            elif self.type == "rotate":
                # rotate image
                rows, cols, _ = img.shape
                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), frame.time * 45, 1)
                img = cv2.warpAffine(img, M, (cols, rows))
                self.out_image = img.copy()
            
                

            return av.VideoFrame.from_ndarray(img, format="bgr24")


def normal_webcam():
    '''
    This function is used to show the webcam in the app.
    
    Returns
    -------
    ctx: webrtc_streamer object'''
    ctx = webrtc_streamer(key="example")
    return ctx

def callback(frame):
    '''
    This function is used to transform the video frame from the webcam.
    
    Parameters
    ----------
    frame: av.VideoFrame
        Video frame from the webcam.
    
    Returns
    -------
    av.VideoFrame
    '''
    img = frame.to_ndarray(format="bgr24")

    img = cv2.cvtColor(cv2.Canny(img, threshold1, threshold2), cv2.COLOR_GRAY2BGR)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

def model_webcam():
    '''
    This function is used to show the webcam in the app and apply the model.
    
    Returns
    -------
    ctx: webrtc_streamer object
    '''
    ctx= webrtc_streamer(key="example", video_frame_callback=callback)
    return ctx

def new_version_webcam():
    
    webrtc_ctx = webrtc_streamer(
        key="opencv-filter",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_processor_factory=OpenCVVideoProcessor,
        async_processing=True,
        #media_stream_constraints={}}
    )

    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.type = st.radio(
            "Select transform type", ("noop", "cartoon", "edges", "rotate")
        )

    return webrtc_ctx

def sidebars(): 
    '''
    This function is used to show the sidebar in the app.
    
    Returns
    -------
    on: bool, model switch on or off'''


    st.sidebar.markdown("""
    # Welcome to this Streamlit App!

    This app aim to use external usb camera from your pc and show it in this app.
    """)
    threshold1 = st.sidebar.slider("Threshold1", min_value=0, max_value=1000, step=1, value=100)
    threshold2 = st.sidebar.slider("Threshold2", min_value=0, max_value=1000, step=1, value=200)
    on = st.sidebar.toggle("Turn model on", value=False)   
    

    #github link to the project

    st.sidebar.markdown("""
                        # Github link to the project 	:card_index_dividers:
                        
                        
                        

                        [Torbidos7/Streamlit](https://github.com/Torbidos7/Streamlit)

                       
                        """)
    return  on

def main():
    st.markdown("""

    # Welcome to this Streamlit App!

    This app aim to use external usb camera from your pc and show it in this app.
    """)
    on = sidebars()
   
    if on:
        ctx = model_webcam()
   
    ctx = new_version_webcam()

    if ctx.video_transformer:

        snap = st.button("Snapshot and Bulb detection")
       # immagine = np.zeros((1,1,1))
        if snap:
            out_image = ctx.video_transformer.out_image
            #immagine = out_image.copy()
            if out_image is not None:
                
                st.write("Output image:")
                st.image(out_image, channels="BGR")
                out_image = cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB)
                result_image = Image.fromarray(out_image.astype('uint8'), 'RGB')
      
                st.write("Bulb detection:")
                bulb_points, img, total_hair, percentage_bulb = detect_bulb(out_image)
                st.write("Total hair: ", total_hair)
                st.write("Total bulb: ", len(bulb_points))
                st.write(f"Percentage bulb: {percentage_bulb:.2f}%")
                st.write("Output image:")
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                st.image(img, channels="BGR")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result_mask = Image.fromarray(img.astype('uint8'), 'RGB')
                # img = Image.open(result)    
                buf = BytesIO()
                result_image.resize((MEDIA_STREAM_CONSTRAINTS["video"]["width"]["max"], MEDIA_STREAM_CONSTRAINTS["video"]["height"]["max"]), Image.BICUBIC)
                result_image.save(buf, format="JPEG")
                result_mask.resize((MEDIA_STREAM_CONSTRAINTS["video"]["width"]["max"], MEDIA_STREAM_CONSTRAINTS["video"]["height"]["max"]), Image.BICUBIC)
                result_mask.save(buf, format="JPEG")
                byte_im = buf.getvalue()  
                Col1, Col2 = st.columns([1,1])
                with Col1:
                    btn = st.download_button(
                        label="Download image",
                        data=byte_im,
                        file_name = "Image_"+time.strftime("%Y-%m-%d-%H:%M:%S")+".png",
                        mime="image/png")
                with Col2:  
                    btn = st.download_button(
                        label="Download mask",
                        data=byte_im,
                        file_name = "Mask_"+time.strftime("%Y-%m-%d-%H:%M:%S")+".png",
                        mime="image/png")    
                
            
        else:
            st.warning("No frames available yet.")


if __name__ == "__main__":
    main()