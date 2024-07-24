import torch.nn as nn
import torch
import torch.optim as optim
from model import Autoencoder,EarlyStopping
from utils import load_data
from torch.utils.data import DataLoader

n_epochs = 2000
n_batchsize = 16

train_dataset, val_dataset = load_data()
train_loader = DataLoader(train_dataset, batch_size=n_batchsize, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=n_batchsize, shuffle=False)

model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
stop = EarlyStopping()

for epoch in range(n_epochs):
    model.train()
    train_loss = 0
    for noisy_imgs, clean_imgs in train_loader:
        noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)

        optimizer.zero_grad()
        outputs = model(noisy_imgs)
        loss = criterion(outputs, clean_imgs)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for noisy_imgs, clean_imgs in val_loader:
            noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
            outputs = model(noisy_imgs)
            loss = criterion(outputs, clean_imgs)
            val_loss += loss.item()

    print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss/len(train_loader)}, Validation Loss: {val_loss/len(val_loader)}")
    stop(val_loss/len(val_loader))
    if stop.early_stop:
        print("Early stopping")
        break
    
torch.save(model.state_dict(), "autoencoder.pth")