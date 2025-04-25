import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from model import SkipAutoencoder, fgsm_attack, my_pgd_attack, compute_psnr, compute_ssim  # Import your model and utils

# Load the trained model
@st.cache_resource
def load_model():
    model = SkipAutoencoder()
    model.load_state_dict(torch.load("model_components/FGSM+PGD_Trained_Model.pth", map_location='cpu'))
    model.eval()
    return model

model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((54, 54)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4187, 0.4761, 0.4798], std=[0.2204, 0.2285, 0.2455])
    ])
    return transform(image).unsqueeze(0)

# Reverse normalization for display
def denormalize_image(tensor):
    mean = torch.tensor([0.4187, 0.4761, 0.4798]).view(3, 1, 1)
    std = torch.tensor([0.2204, 0.2285, 0.2455]).view(3, 1, 1)
    return torch.clamp(tensor.cpu() * std + mean, 0, 1)

# App UI
st.title("Adversarial Defense with Multi-Headed Attention Autoencoder")

uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

attack_type = st.selectbox("Choose Adversarial Attack", ["None", "FGSM", "PGD"])

epsilon = st.slider("Epsilon (Perturbation Strength)", 0.0, 1.0, 0.1, step=0.01)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)

    image_tensor = preprocess_image(image).to(device)

    # Generate adversarial example
    if attack_type == "FGSM":
        adv_image = fgsm_attack(model, image_tensor, epsilon).detach()
    elif attack_type == "PGD":
        adv_image = my_pgd_attack(model, image_tensor, epsilon=epsilon, alpha=0.045, num_steps=15).detach()
    else:
        adv_image = image_tensor

    # Reconstruct
    reconstructed = model(adv_image).detach()

    # Metrics
    psnr_val = compute_psnr(image_tensor, reconstructed)
    ssim_val = compute_ssim(image_tensor, reconstructed)

    # Display
    col1, col2, col3 = st.columns(3)
    col1.image(denormalize_image(image_tensor.squeeze()), caption="Original", use_column_width=True)
    col2.image(denormalize_image(adv_image.squeeze()), caption="Adversarial", use_column_width=True)
    col3.image(denormalize_image(reconstructed.squeeze()), caption="Reconstructed", use_column_width=True)

    st.write(f"*PSNR:* {psnr_val:.2f}")
    st.write(f"*SSIM:* {ssim_val:.4f}")