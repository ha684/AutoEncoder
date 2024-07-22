import torch.nn as nn
import torch
from torch.utils.data import  Dataset
import os
from PIL import Image
import numpy as np
# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),   # 1 input channel, 64 output channels
            nn.ReLU(),
            nn.MaxPool2d(2, padding=0),  # Ensure downsampling keeps dimensions correct
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 64 input channels, 128 output channels
            nn.ReLU(),
            nn.MaxPool2d(2, padding=0)   # Ensure downsampling keeps dimensions correct
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 128 input channels, 128 output channels
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 128 input channels, 64 output channels
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),  # 64 input channels, 1 output channel
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Noisy(Dataset):
    def __init__(self, noisy_images, clean_images):
        self.noisy_images = noisy_images
        self.clean_images = clean_images

    def __len__(self):
        return len(self.noisy_images)

    def __getitem__(self, idx):
        noisy_image = self.noisy_images[idx]
        clean_image = self.clean_images[idx]
        return torch.tensor(noisy_image, dtype=torch.float32), torch.tensor(clean_image, dtype=torch.float32)
    
class icdar2015(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.image_paths = [f for f in os.listdir(data_path) if f.endswith('.jpg')]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, self.image_paths[idx])
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image

# EarlyStopping class
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
def make_noise(normal_image):
    mean = 0
    sigma = 1
    gauss = np.random.normal(mean, sigma, normal_image.shape)
    noise_image = normal_image + gauss * 0.08
    return noise_image