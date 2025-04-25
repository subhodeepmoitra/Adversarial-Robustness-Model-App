# Adversarial-Robustness-Model-App
This research application aims to enhance the adversarial robustness of autonomous vehicles by reconstructing original images from adversarial inputs. The system utilizes a custom deep learning model to recover original, noise-free images from adversarially perturbed inputs.

The model is trained to handle adversarial attacks like FGSM (Fast Gradient Sign Method) and aims to improve the robustness of autonomous vehicle systems against such attacks.

![image](https://github.com/user-attachments/assets/87fde247-1abb-4738-adc7-d55210d8139f)

URL: https://adversarialrobustnessapp.streamlit.app/
How It Works

 Step 1: Upload Image: Upload an image of a scene that has been adversarially modified (e.g., by FGSM).

 Step 2: Reconstruction: The model processes the uploaded image and reconstructs it by leveraging multi-head attention to focus on significant image features and removing adversarial perturbations.

 Step 3: Result: The original and reconstructed images are displayed side by side for comparison.

    
