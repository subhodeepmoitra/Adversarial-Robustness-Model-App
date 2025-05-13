import asyncio
import streamlit as st
from streamlit_webrtc import webrtc_streamer

# Ensuring asyncio loop is running
asyncio.get_event_loop().run_until_complete(asyncio.sleep(0))

def video_frame_callback(frame):
    return frame

webrtc_streamer(
    key="example", 
    video_frame_callback=video_frame_callback
)
