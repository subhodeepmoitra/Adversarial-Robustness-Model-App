
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import cv2
from io import BytesIO

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

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SkipAutoencoder().to(device)

# Use the selected model from dropdown
model_paths = {
    "FGSM+PGD_Trained_Model": 'model_components/FGSM+PGD_Trained_Model.pth',
    "SVHM_FGSM_epoch1": 'model_components/SVHM_FGSM_epoch1.pth',
    "FGSM_SVHN_parallel_E31_multiheaded_pgd_2(Added on 27/04/2025": 'model_components/FGSM_SVHN_parallel_E31_multiheaded_pgd_2.pth',
    "traffic_light_FGSM_E21": 'model_components/traffic_light_FGSM_E21.pth',
}

# Streamlit UI for selecting the model
selected_model_name = st.selectbox("Choose a model to use for reconstruction:", list(model_paths.keys()))
model_path = model_paths[selected_model_name]
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Define the image transformations (for video frames)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Streamlit UI for video capture
st.title('Real-time Video Reconstruction from Adversarial Inputs (By Cheems Researchers)')
st.write("This demo processes live video feed frame by frame and shows the reconstructed version with adversarial noise.")

# OpenCV to capture video from webcam
cap = cv2.VideoCapture(0)  # 0 for webcam, or you can put the path for an external video file

if not cap.isOpened():
    st.error("Error: Unable to access the webcam or video file.")
else:
    # Start with an initial 3-second wait before applying the attack
    attack_delay = 3  # Seconds to wait before applying attack
    time.sleep(attack_delay)

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Convert the BGR frame from OpenCV to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        
        # Apply transformations
        img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension and move to device
        
        # Run the model on the frame
        with torch.no_grad():
            reconstructed_image = model(img_tensor)  # Run reconstruction

        # Convert the tensor back to a NumPy array
        reconstructed_image_np = reconstructed_image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        reconstructed_image_np = np.clip(reconstructed_image_np, 0, 1)
        
        # Create a figure for displaying original and reconstructed images
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        
        # Plot the original image
        axs[0].imshow(frame_rgb)
        axs[0].set_title('Original Frame')
        axs[0].axis('off')

        # Plot the reconstructed image
        axs[1].imshow(reconstructed_image_np)
        axs[1].set_title('Reconstructed Frame')
        axs[1].axis('off')

        # Display the figure
        st.pyplot(fig)

        # Optional: add frame rate control to avoid excessive computation
        
        # Exit condition: Stop video when user presses 'q' (you can modify this in Streamlit as needed)
        # Use a key press event or another method to stop capturing.
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
