import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

DATA_DIR = "raman_blood_dataset"

class RamanDataset(Dataset):
    def __init__(self, root):
        self.cubes = sorted(glob.glob(os.path.join(root, "raman_cubes", "*.npy")))
        self.masks = sorted(glob.glob(os.path.join(root, "ground_truth_masks", "*.png")))
    
    def __len__(self): return len(self.cubes)
    
    def __getitem__(self, idx):
        # Load Cube (128x128x100) -> Transpose to (100x128x128)
        cube = np.load(self.cubes[idx])
        cube = np.transpose(cube, (2, 0, 1))
        
        # Load Mask
        mask = plt.imread(self.masks[idx])
        if mask.ndim == 3: mask = mask[:,:,0] # Take 1 channel if RGB
        mask = (mask > 0).astype(np.int64)
        
        return torch.tensor(cube, dtype=torch.float32), torch.tensor(mask)

class RamanUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Input 100 bands (Raman Spectrum)
        self.enc1 = nn.Sequential(nn.Conv2d(100, 64, 3, 1, 1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU())
        self.out = nn.Conv2d(128, 2, 1) # 2 Classes: Blood vs Plastic

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        return self.out(x)

def train():
    if not os.path.exists(DATA_DIR): return print("Run data_factory.py first!")
    
    loader = DataLoader(RamanDataset(DATA_DIR), batch_size=4, shuffle=True)
    model = RamanUNet()
    opt = optim.Adam(model.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()
    
    print("Training Raman AI Model...")
    for epoch in range(3):
        total_loss = 0
        for x, y in loader:
            opt.zero_grad()
            pred = model(x)
            loss = crit(pred, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(loader):.4f}")
        
    torch.save(model.state_dict(), "raman_model.pth")
    print("Model Saved.")

if __name__ == "__main__":
    train()