from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import platform
#import pycaret
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
# #def chose_camera():
#     st.markdown("""

#     # Choose your camera

#     """)

#     option = st.selectbox(
#         'Which camera do you want to use?',
#         ('Laptop camera', 'External camera'))

#     st.write('You selected:', option)

#     if option == 'Laptop camera':
#         cap = cv2.VideoCapture(0)
#     else:
#         cap = cv2.VideoCapture(1)

#     return cap

# def camera(chose_camera):

#     st.markdown("""

#     # Camera

#     """)

#     cap = chose_camera

#     while True:
#         ret, frame = cap.read()
#         cv2.imshow('frame', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

def webcam():
    webrtc_streamer(key="example", video_transformer_factory=None)

def main():
    st.markdown("""

    # Welcome to My First Streamlit App!

    This first application is try to implement different machine learning dataset and algorithms using [streamlit](https://streamlit.io/) and pycaret.
    """)




    with st.echo(code_location='below'):
        total_points = st.slider("Number of points in spiral", 1, 5000, 2000)
        num_turns = st.slider("Number of turns in spiral", 1, 100, 9)

        Point = namedtuple('Point', 'x y')
        data = []

        points_per_turn = total_points / num_turns

        for curr_point_num in range(total_points):
            curr_turn, i = divmod(curr_point_num, points_per_turn)
            angle = (curr_turn + 1) * 2 * math.pi * i / points_per_turn
            radius = curr_point_num / total_points
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            data.append(Point(x, y))

        st.altair_chart(alt.Chart(pd.DataFrame(data), height=500, width=500)
            .mark_circle(color='#0068c9', opacity=0.5)
            .encode(x='x:Q', y='y:Q'))

    st.markdown('''
    # Here some stats about your pc
    ''')

    df = pd.DataFrame({'machine':platform.machine(), 'version':platform.version(), 'platform':platform.platform(),
    'uname': platform.uname(), 'system':platform.system(), 'processor':platform.processor()})

    st.dataframe(df)

    #show camera and chose camera

    camera(chose_camera())

if __name__ == "__main__":
    main()