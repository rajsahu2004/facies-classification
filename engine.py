import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    for images, labels in tqdm(dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)


from unet import UNet
from dataset import SeismicDataset
data = SeismicDataset('data/train/train_seismic.npy', 'data/train/train_labels.npy')
dataloader = DataLoader(data, batch_size=8, shuffle=True)
model = UNet(in_channels=1, num_classes=1)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
train_one_epoch(model, dataloader, nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters()), device)
