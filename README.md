# Streamlit

This Python script is a Streamlit application that uses an external USB camera from your PC and displays it in the app. It also provides an option to apply various transformations to the video frames.

## Table of Contents

- [Streamlit](#streamlit)
  - [Table of Contents](#table-of-contents)
  - [About the Code](#about-the-code)
    - [OpenCVVideoProcessor Class](#opencvvideoprocessor-class)
    - [callback function](#callback-function)
    - [new\_version\_webcam function](#new_version_webcam-function)
  - [Online Demo](#online-demo)
  - [Running the Code](#running-the-code)
  - [Bugs and feature requests](#bugs-and-feature-requests)
  - [Authors](#authors)
  - [Thanks](#thanks)
  - [Copyright and license](#copyright-and-license)
## About the Code

The script imports necessary modules and packages. These include streamlit for the web app, cv2 for image processing, streamlit_webrtc for real-time communication (like video streaming), numpy and pandas for data handling, and others.

### OpenCVVideoProcessor Class
```python
class OpenCVVideoProcessor(VideoProcessorBase):
    type: Literal["noop", "cartoon", "edges", "rotate"]
    out_image: np.ndarray = None

    def __init__(self) -> None:
        self.type = "noop"

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        # Various transformations are applied based on the type
        # The transformed image is stored in self.out_image
        return av.VideoFrame.from_ndarray(img, format="bgr24")
```

This class inherits from **VideoProcessorBase** and is used to transform video frames. It has a recv method that applies various transformations to a frame based on the *type* attribute and converts it to a numpy array.

### callback function
```python
def callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.cvtColor(cv2.Canny(img, threshold1, threshold2), cv2.COLOR_GRAY2BGR)
    return av.VideoFrame.from_ndarray(img, format="bgr24")
```

This function is used to apply the Canny edge detection algorithm to the video frames. It takes a frame as input and returns a frame with the edges detected.

### new_version_webcam function
```python
def new_version_webcam():
    webrtc_ctx = webrtc_streamer(
        key="opencv-filter",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_processor_factory=OpenCVVideoProcessor,
        async_processing=True,
        media_stream_constraints={"video": {"width": {"min": 800, "ideal": 1200, "max": 1920 }, "height": {"min": 600, "ideal": 900, "max": 1080 }}}
    )
    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.type = st.radio(
            "Select transform type", ("noop", "cartoon", "edges", "rotate")
        )
    return webrtc_ctx
```
This function sets up a webcam stream with the option to apply various transformations to each frame. The type of transformation is selected using a radio button in the Streamlit app.

## Online Demo

You can find a demo of the web app hosted on Streamlit Cloud: 
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://system-in.streamlit.app/)

## Running the Code

To run the code, navigate to the directory containing `streamlit_app.py` and execute the following command in your terminal:

```bash
streamlit run streamlit_app.py
```
This will start the Streamlit web application.

## Bugs and feature requests

Have a bug or a feature request? Please first read and search for existing and closed issues. If your problem or idea is not addressed yet, [please open a new issue](https://github.com/Torbidos7/Streamlit/issues/new).
## Authors

- [@Torbidos7](https://github.com/Torbidos7)

## Thanks

Thank you for coming :stuck_out_tongue_closed_eyes:

## Copyright and license

This project is licensed under the MIT License.
