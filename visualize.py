import torch
from PIL import Image
from matplotlib import pyplot as plt
from model import Autoencoder
from torchvision import transforms
import numpy as np
# Load the model
image_size = 416

model = Autoencoder()
model.load_state_dict(torch.load('autoencoder.pth'))

# Set the model to evaluation mode
model.eval()

# Preprocess the input data
color_transform = transforms.Compose([
    transforms.Resize((416,416)),
    transforms.ToTensor()
])

# Load and preprocess the input image
input_image = Image.open('template.png').convert('L') # Convert image to grayscale
input_image1 = color_transform(input_image).unsqueeze(0)  # Add batch dimension

# Move the model and input to the same device (CPU or GPU)

# Get the colorized output
with torch.no_grad():
    output_image = model(input_image1)

# Postprocess the output image
output_image = output_image.squeeze(0).cpu().numpy()
output_image = np.transpose(output_image, (1, 2, 0))  # Change from (C, H, W) to (H, W, C)

# Convert to range [0, 255] and uint8
output_image = (output_image * 255).astype(np.uint32)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Display the input grayscale image
ax[0].imshow(input_image, cmap='gray')
ax[0].set_title("Original Image")
ax[0].axis('off')

# Display the colorized image
ax[1].imshow(output_image)
ax[1].set_title("Processed Image")
ax[1].axis('off')

plt.show()
