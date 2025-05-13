from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
import streamlit as st

# Custom STUN/TURN servers
rtc_configuration = {
    "iceServers": [
        {
            "urls": "stun:stun.l.google.com:19302"  # Google's public STUN server
        },
        {
            "urls": "turn:your_turn_server_address",  # Replace with your TURN server address
            "username": "your_username",  # TURN server username
            "credential": "your_password"  # TURN server password
        }
    ]
}

# Video Processor
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame = None

    def recv(self, frame):
        self.frame = frame.to_ndarray(format="bgr24")
        return frame

# Streamlit UI and WebRTC setup
def run_webrtc():
    webrtc_streamer(
        key="example", 
        video_processor_factory=VideoProcessor,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
    )

if __name__ == "__main__":
    run_webrtc()
