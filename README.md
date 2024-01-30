# Streamlit

This Python script is a Streamlit application that uses an external USB camera from your PC and displays it in the app. It also provides an option to apply the Canny edge detection algorithm to the video frames.

## Table of Contents

- [Streamlit](#streamlit)
  - [Table of Contents](#table-of-contents)
  - [About the Code](#about-the-code)
    - [Video Trasformer Class](#video-trasformer-class)
    - [callback function](#callback-function)
    - [main function](#main-function)
  - [Online Demo](#online-demo)
  - [Running the Code](#running-the-code)
  - [Bugs and feature requests](#bugs-and-feature-requests)
  - [Authors](#authors)
  - [Thanks](#thanks)
  - [Copyright and license](#copyright-and-license)
     
## About the Code

The script imports necessary modules and packages. These include streamlit for the web app, cv2 for image processing, streamlit_webrtc for real-time communication (like video streaming), numpy and pandas for data handling, and others.

### Video Trasformer Class
```python
class VideoTransformer(VideoTransformerBase):
        frame_lock: threading.Lock  
        out_image: Union[np.ndarray, None]

        def __init__(self) -> None:
            self.frame_lock = threading.Lock()
            self.out_image = None

        def transform(self, frame: av.VideoFrame) -> np.ndarray:
            out_image = frame.to_ndarray(format="bgr24")

            with self.frame_lock:
                self.out_image = out_image
            return out_image
```

This class inherits from **VideoTransformerBase** and is used to transform video frames. It has a *transform* method that converts a frame to a numpy array. It also has a *frame_lock* attribute that is used to lock the frame and an out_image attribute that stores the converted frame.

### callback function
```python
def callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.cvtColor(cv2.Canny(img, threshold1, threshold2), cv2.COLOR_GRAY2BGR)
    return av.VideoFrame.from_ndarray(img, format="bgr24")
```

This function is used to apply the Canny edge detection algorithm to the video frames. It takes a frame as input and returns a frame with the edges detected.

### main function
```python
def main():
    st.markdown("""
    # Welcome to this Streamlit App!
    This app aim to use external usb camera from your pc and show it in this app.
    """)
    on = sidebars()
   
    if on:
        st.write("switch is on")
        ctx = canny_webcam()    
    else:
        ctx = normal_webcam()

    if ctx.video_transformer:
        snap = st.button("Snapshot")
        if snap:
            with ctx.video_transformer.frame_lock:
                out_image = ctx.video_transformer.out_image

            if out_image is not None:
                st.write("Output image:")
                st.image(out_image, channels="BGR")
                my_path = os.path.abspath(os.path.dirname(__file__))       
                cv2.imwrite(os.path.join(my_path, "../Data/"+"filename.jpg"), out_image)
            else:
                st.warning("No frames available yet.")
```

This function sets up the main part of the Streamlit app. It displays a welcome message, calls the sidebars function to set up the *sidebar*, and sets up the video stream based on the state of the toggle in the sidebar. If the "*Snapshot*" button is clicked, it saves the current frame as an image.
    
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
