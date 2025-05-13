
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import torch
import torch.nn as nn
import numpy as np
import av
import cv2
from PIL import Image
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dummy Model or real autoencoder
class DummyAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

model = DummyAutoencoder().to(device).eval()

# FGSM
def fgsm_attack(model, data, epsilon):
    data = data.clone().detach().requires_grad_(True)
    output = model(data)
    loss = nn.MSELoss()(output, data)
    model.zero_grad()
    loss.backward()
    perturbed_data = data + epsilon * data.grad.sign()
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    return perturbed_data.detach()

# Video transformer
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    def transform_frame(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(device)
        adv = fgsm_attack(model, tensor, epsilon=0.07)
        adv_img = adv.squeeze().detach().cpu().numpy()
        adv_img = np.transpose(adv_img, (1, 2, 0)) * 255
        adv_img = adv_img.astype(np.uint8)
        return av.VideoFrame.from_ndarray(adv_img, format="rgb24")

# Streamlit UI
st.title("Live FGSM Adversarial Attack")
webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
