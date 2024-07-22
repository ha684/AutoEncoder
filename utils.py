import numpy as np
from torchvision import transforms
from model import icdar2015, make_noise
import os
import pickle

image_size = 224
n_batchsize = 32
train_path = './icdar2015/train'
val_path = './icdar2015/test'

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])
#load and convert train and val data to numpy arrays
train_data = icdar2015(train_path, transform=transform)
train_images = np.array([train_data[i].numpy() for i in range(len(train_data))])

val_data = icdar2015(val_path, transform=transform)
val_images = np.array([val_data[i].numpy() for i in range(len(val_data))])

#make noise
noise_train_images = np.array([make_noise(img) for img in train_images])
noise_val_images = np.array([make_noise(img) for img in val_images])

#save to pickle file
if not os.path.exists('data.dat'):
    with open("data.dat", "wb") as f:
        pickle.dump([noise_train_images, noise_val_images, train_images, val_images], f)
