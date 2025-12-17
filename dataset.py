import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms

class SeismicDataset(Dataset):
    def __init__(self, train_seismic_path, train_labels_path, transform=None):
        self.seismic_data = np.load(train_seismic_path)  # Shape: (num_samples, height, width)
        self.labels = np.load(train_labels_path)         # Shape: (num_samples, height, width)
        self.transform = transform
        self.seismic_data = np.pad(self.seismic_data, ((0,0), (1,2), (0,1)), mode='constant')
        self.labels = np.pad(self.labels, ((0,0), (1,2), (0,1)), mode='constant')

    def __len__(self):
        return len(self.seismic_data)
    
    def __getitem__(self, idx):
        seismic_section = self.seismic_data[idx]
        label_section = self.labels[idx]

        # Expand dimensions to add channel dimension
        seismic_section = np.expand_dims(seismic_section, axis=0)
        label_section = np.expand_dims(label_section, axis=0)
        
        if self.transform:
            seismic_section = self.transform(seismic_section)
            label_section = self.transform(label_section)
        
        return torch.tensor(seismic_section, dtype=torch.float32), torch.tensor(label_section, dtype=torch.float32)
