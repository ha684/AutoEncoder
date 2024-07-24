import torch.nn as nn
import torch
from torch.utils.data import  Dataset
import os
from PIL import Image
import numpy as np
import torchvision.models as models

# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        resnet = models.resnet18(pretrained=True)
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class icdar2015(Dataset):
    def __init__(self, data_path, transform=None,color_transform=None):
        self.data_path = data_path
        self.image_paths = [f for f in os.listdir(data_path) if f.endswith(('.jpg', '.png', '.gif'))]
        self.transform = transform
        self.color_transform = color_transform
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, self.image_paths[idx])
        grayscale_image = Image.open(img_path).convert("L")  # Load as grayscale
        color_image = Image.open(img_path).convert("RGB")  # Load as color

        if self.transform:
            grayscale_image = self.transform(grayscale_image)
            gray_np = grayscale_image.squeeze().numpy()
            # Add noise
            noisy_gray_np = make_noise(gray_np)
            # Convert back to tensor
            grayscale_image = torch.from_numpy(noisy_gray_np).unsqueeze(0)
        if self.color_transform:
            color_image = self.color_transform(color_image)
        return grayscale_image.float(), color_image.float()

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
    return np.clip(noise_image, 0, 1)