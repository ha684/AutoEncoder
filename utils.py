from torchvision import transforms
from model import icdar2015

image_size = 416
n_batchsize = 16
train_path = './icdar2015/train'
val_path = './icdar2015/test'

grayscale_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])

color_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])
def load_data():
    train_data = icdar2015(train_path, transform=grayscale_transform, color_transform=color_transform)
    val_data = icdar2015(val_path, transform=grayscale_transform, color_transform=color_transform)
    return train_data, val_data

# grayscale_train_images = torch.stack([train_data[i][0] for i in range(len(train_data))])
# color_train_images = torch.stack([train_data[i][1] for i in range(len(train_data))])
# grayscale_val_images = torch.stack([val_data[i][0] for i in range(len(val_data))])
# color_val_images = torch.stack([val_data[i][1] for i in range(len(val_data))])

