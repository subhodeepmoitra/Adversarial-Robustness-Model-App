import streamlit as st
from streamlit_webrtc import webrtc_streamer

def video_frame_callback(frame):
    return frame

webrtc_streamer(key="example", video_frame_callback=video_frame_callback)
