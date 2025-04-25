import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Assuming you have your SkipAutoencoder model defined already
class SkipAutoencoder(nn.Module):
    def __init__(self, num_heads=4):
        super(SkipAutoencoder, self).__init__()

        # Encoder with multi-headed attention
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.ReLU()
        )

        # Skip connection
        self.conv_skip = nn.Conv2d(16, 16, kernel_size=1)

        # Adding multi-head attention layer after encoding
        self.attention = MultiHeadAttention(32, num_heads=num_heads)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=1, padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=1, padding=1, output_padding=0),
            nn.Sigmoid()
        )

        self.decoder_attention = MultiHeadAttentionDecoder(3, num_heads=1)

    def forward(self, x):
        # Encoder
        x1 = self.encoder[0:2](x)  # First Conv layer and activation
        x2 = self.encoder[2:](x1)  # Second Conv layer and activation

        # Skip connection
        skip = self.conv_skip(x1)

        # Apply multi-head attention on encoded features
        x2 = self.attention(x2, x2, x2)

        # Decoder
        x = self.decoder[0:2](x2)  # First deconv layer and activation
        x = self.decoder[2](x + skip)  # Second deconv layer with skip connection

        # Apply multi-head attention on decoded features
        x = self.decoder_attention(x, x, x)

        return x

# Load your model
model = SkipAutoencoder().cuda()  # Move to GPU if available
model.load_state_dict(torch.load('model_components/FGSM+PGD_Trained_Model.pth'))
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((54, 54)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4187, 0.4761, 0.4798], std=[0.2204, 0.2285, 0.2455])
])

# Streamlit UI for uploading an image
st.title('Image Reconstruction with Adversarial Attacks')
st.write("Upload an image and see the reconstructed version with adversarial noise.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Open the uploaded image and apply transformations
    img = Image.open(uploaded_file)
    img = img.convert("RGB")  # Convert to RGB if not already
    img_tensor = transform(img).unsqueeze(0).cuda()  # Add batch dimension and move to GPU

    # Run the model on the image
    with torch.no_grad():
        reconstructed_image = model(img_tensor)  # Run reconstruction
    
    # Display original and reconstructed images
    st.image(img, caption="Original Image", use_column_width=True)
    st.write("Reconstructed Image:")
    
    # Convert the tensor back to a PIL image for display
    reconstructed_image = reconstructed_image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    reconstructed_image = np.clip(reconstructed_image, 0, 1)
    st.image(reconstructed_image, caption="Reconstructed Image", use_column_width=True)

    # Optionally, display some metrics like PSNR and SSIM
    # You could add your metrics function calls here
