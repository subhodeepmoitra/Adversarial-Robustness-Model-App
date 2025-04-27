import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn

# Define the Multi-Head Attention Layer
class MultiHeadAttention(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        super(MultiHeadAttention, self).__init__()
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        # Define the query, key, and value projections for each head
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Output projection
        self.out_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, query, key, value):
        batch_size, channels, height, width = query.size()

        # Apply projections to get query, key, and value
        query = self.query_conv(query)  # (B, C, H, W)
        key = self.key_conv(key)        # (B, C, H, W)
        value = self.value_conv(value)  # (B, C, H, W)

        # Reshape for multi-head attention
        query = query.view(batch_size, self.num_heads, self.head_dim, height * width)  # (B, heads, head_dim, H*W)
        key = key.view(batch_size, self.num_heads, self.head_dim, height * width)        # (B, heads, head_dim, H*W)
        value = value.view(batch_size, self.num_heads, self.head_dim, height * width)  # (B, heads, head_dim, H*W)

        # Transpose for attention computation
        query = query.permute(0, 1, 3, 2)  # (B, heads, H*W, head_dim)
        key = key.permute(0, 1, 2, 3)      # (B, heads, head_dim, H*W)

        # Compute attention scores
        attention_scores = torch.matmul(query, key) / (self.head_dim ** 0.5)  # (B, heads, H*W, H*W)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply attention weights to value
        attention_output = torch.matmul(attention_weights, value.permute(0, 1, 3, 2))  # (B, heads, H*W, head_dim)

        # Reshape back to original shape
        attention_output = attention_output.permute(0, 1, 3, 2).contiguous()  # (B, heads, head_dim, H*W)
        attention_output = attention_output.view(batch_size, channels, height, width)  # (B, C, H, W)

        # Project back to the original channel size
        out = self.out_conv(attention_output)

        return out
class MultiHeadAttentionDecoder(nn.Module):
    def __init__(self, in_channels, num_heads=1):
        super(MultiHeadAttentionDecoder, self).__init__()
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        # Define the query, key, and value projections for each head
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Output projection
        self.out_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, query, key, value):
        batch_size, channels, height, width = query.size()

        # Apply projections to get query, key, and value
        query = self.query_conv(query)  # (B, C, H, W)
        key = self.key_conv(key)        # (B, C, H, W)
        value = self.value_conv(value)  # (B, C, H, W)

        # Reshape for multi-head attention
        query = query.view(batch_size, self.num_heads, self.head_dim, height * width)  # (B, heads, head_dim, H*W)
        key = key.view(batch_size, self.num_heads, self.head_dim, height * width)        # (B, heads, head_dim, H*W)
        value = value.view(batch_size, self.num_heads, self.head_dim, height * width)  # (B, heads, head_dim, H*W)

        # Transpose for attention computation
        query = query.permute(0, 1, 3, 2)  # (B, heads, H*W, head_dim)
        key = key.permute(0, 1, 2, 3)      # (B, heads, head_dim, H*W)

        # Compute attention scores
        attention_scores = torch.matmul(query, key) / (self.head_dim ** 0.5)  # (B, heads, H*W, H*W)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply attention weights to value
        attention_output = torch.matmul(attention_weights, value.permute(0, 1, 3, 2))  # (B, heads, H*W, head_dim)

        # Reshape back to original shape
        attention_output = attention_output.permute(0, 1, 3, 2).contiguous()  # (B, heads, head_dim, H*W)
        attention_output = attention_output.view(batch_size, channels, height, width)  # (B, C, H, W)

        # Project back to the original channel size (1 channel in the decoder output)
        out = self.out_conv(attention_output)

        return out
        
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SkipAutoencoder().to(device)
#model.load_state_dict(torch.load('model_components/FGSM+PGD_Trained_Model.pth', map_location=torch.device('cpu')))
#model.load_state_dict(torch.load('model_components/SVHM_FGSM_epoch1.pth', map_location=torch.device('cpu')))
model_paths = {
    "FGSM+PGD_Trained_Model": 'model_components/FGSM+PGD_Trained_Model.pth',
    "SVHM_FGSM_epoch1": 'model_components/SVHM_FGSM_epoch1.pth',
    "FGSM_SVHN_parallel_E31_multiheaded_pgd_2(Added on 27/04/2025": 'model_components/FGSM_SVHN_parallel_E31_multiheaded_pgd_2.pth',
    #"PGD Epoch 5": 'model_components/SVHM_PGD_epoch5.pth',
    #"Combined FGSM+PGD": 'model_components/FGSM+PGD_Trained_Model.pth'
}
#Creating a dropdown in Streamlit
selected_model_name = st.selectbox("Choose a model to use for reconstruction:", list(model_paths.keys()))
model_path = model_paths[selected_model_name]
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((40, 40)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        transforms.ToTensor(),
        #transforms.Normalize((0.5,), (0.5,)) # Normalize Gray-scale
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize RGB (Type 1)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for RGB (Type 2)
])

st.markdown(
    """
    <style>
    .stApp {
        background-color: #6B8E23;  /* Olive Green color */
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Streamlit UI for uploading an image
st.title('Original image reconstruction from adversarial inputs (By Cheems Researchers)')
st.write("Upload an image and see the reconstructed version with adversarial noise. (Currently support for only FGSM)")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Open the uploaded image and apply transformations
    img = Image.open(uploaded_file)
    img = img.convert("RGB")  # Convert to RGB if not already
    #img_tensor = transform(img).unsqueeze(0).cuda()  # Add batch dimension and move to GPU
    img_tensor = transform(img).unsqueeze(0).cpu()  # Add batch dimension and move to CPU


    # Run the model on the image
    with torch.no_grad():
        reconstructed_image = model(img_tensor)  # Run reconstruction
    
    # Display original and reconstructed images side by side
    col1, col2 = st.columns(2)  # Create two columns

    with col1:
        st.image(img, caption="Original Image", use_column_width=True)
    
    with col2:
        # Convert the tensor back to a PIL image for display
        reconstructed_image = reconstructed_image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        reconstructed_image = np.clip(reconstructed_image, 0, 1)
        st.image(reconstructed_image, caption="Reconstructed Image", use_column_width=True)


    # Optionally, display some metrics like PSNR and SSIM
    # You could add your metrics function calls here
