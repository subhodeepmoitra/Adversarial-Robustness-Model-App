import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Define the SkipAutoencoder model
class SkipAutoencoder(nn.Module):
    def __init__(self):
        super(SkipAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Attack Functions
def fgsm_attack(model, image, epsilon):
    image.requires_grad = True
    output = model(image)
    loss = nn.MSELoss()(output, image)
    model.zero_grad()
    loss.backward()
    perturbed_image = image + epsilon * image.grad.sign()
    return torch.clamp(perturbed_image, 0, 1)

def my_pgd_attack(model, image, epsilon, alpha, num_steps):
    perturbed_image = image.clone().detach()
    perturbed_image.requires_grad = True
    for _ in range(num_steps):
        output = model(perturbed_image)
        loss = nn.MSELoss()(output, image)
        model.zero_grad()
        loss.backward()
        perturbed_image = perturbed_image + alpha * perturbed_image.grad.sign()
        perturbed_image = torch.min(torch.max(perturbed_image, image - epsilon), image + epsilon)
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

# Metrics Functions
def compute_psnr(original, reconstructed):
    mse = torch.mean((original - reconstructed) ** 2)
    if mse == 0:
        return 100
    max_pixel = 1.0  # Assuming input images are normalized between [0, 1]
    return 20 * torch.log10(max_pixel / torch.sqrt(mse))

def compute_ssim(original, reconstructed):
    original = original.squeeze().cpu().numpy()
    reconstructed = reconstructed.squeeze().cpu().numpy()
    return ssim(original, reconstructed, multichannel=True)

# Image Preprocessing and Normalization
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((54, 54)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4187, 0.4761, 0.4798], std=[0.2204, 0.2285, 0.2455])
    ])
    return transform(image).unsqueeze(0)

def denormalize_image(tensor):
    mean = torch.tensor([0.4187, 0.4761, 0.4798]).view(3, 1, 1)
    std = torch.tensor([0.2204, 0.2285, 0.2455]).view(3, 1, 1)
    return torch.clamp(tensor.cpu() * std + mean, 0, 1)

# Streamlit UI and Execution
st.title("Adversarial Defense with Multi-Headed Attention Autoencoder")

uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
attack_type = st.selectbox("Choose Adversarial Attack", ["None", "FGSM", "PGD"])
epsilon = st.slider("Epsilon (Perturbation Strength)", 0.0, 1.0, 0.1, step=0.01)

model = SkipAutoencoder()
model.eval()

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)
    
    image_tensor = preprocess_image(image)
    
    # Generate adversarial example
    if attack_type == "FGSM":
        adv_image = fgsm_attack(model, image_tensor, epsilon).detach()
    elif attack_type == "PGD":
        adv_image = my_pgd_attack(model, image_tensor, epsilon, alpha=0.045, num_steps=15).detach()
    else:
        adv_image = image_tensor

    # Reconstruct image
    reconstructed = model(adv_image).detach()

    # Metrics
    psnr_val = compute_psnr(image_tensor, reconstructed)
    ssim_val = compute_ssim(image_tensor, reconstructed)

    # Display Results
    col1, col2, col3 = st.columns(3)
    col1.image(denormalize_image(image_tensor.squeeze()), caption="Original", use_column_width=True)
    col2.image(denormalize_image(adv_image.squeeze()), caption="Adversarial", use_column_width=True)
    col3.image(denormalize_image(reconstructed.squeeze()), caption="Reconstructed", use_column_width=True)

    st.write(f"*PSNR:* {psnr_val:.2f}")
    st.write(f"*SSIM:* {ssim_val:.4f}")
