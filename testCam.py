import asyncio
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame = None

    def recv(self, frame):
        self.frame = frame.to_ndarray(format="bgr24")
        return frame

# Avoid manually starting an event loop
def run_webrtc():
    webrtc_streamer(
        key="example", 
        video_processor_factory=VideoProcessor
    )

if __name__ == "__main__":
    run_webrtc()
