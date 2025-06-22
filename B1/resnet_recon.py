import os
import sys
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from pytorch_msssim import SSIM
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
import torch.optim as optim
from torchvision.datasets import FashionMNIST 
from pathlib import Path

A_DIR = str(Path(__file__).resolve().parent.parent)
sys.path.append(A_DIR)
from set_project_path import PROJECT_ROOT, DATA_PATH # Set the public paths for workspace and datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = os.path.join(PROJECT_ROOT, "B1/models")
model_path = os.path.join(model_dir, "fashion_mnist_denoise.pth")


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.residual(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)

class DenoisingResNetSkip(nn.Module): # Symmetric Encoding and Decoding Structure
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = ResBlock(1, 32)           # [B, 32, 28, 28]
        self.pool1 = nn.MaxPool2d(2)            # → 14x14
        self.enc2 = ResBlock(32, 64)          # [B, 64, 14, 14]
        self.pool2 = nn.MaxPool2d(2)            # → 7x7

        # Bottleneck
        self.bottleneck = ResBlock(64, 64)    # [B, 64, 7, 7]

        # Decoder
        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)  # → 14x14 ，Up-sampling operation, transposed convolution
        self.dec1 = ResBlock(64 + 64, 32)     # concat with enc2

        self.up2 = nn.ConvTranspose2d(32, 32, 2, stride=2)  # → 28x28
        self.dec2 = ResBlock(32 + 32, 16)     # concat with enc1

        # Final
        self.final = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)           # [B, 32, 28, 28]
        x2 = self.enc2(self.pool1(x1))  # [B, 64, 14, 14]
        x3 = self.bottleneck(self.pool2(x2))  # [B, 64, 7, 7]

        x = self.up1(x3)            # [B, 64, 14, 14]
        x = torch.cat([x, x2], dim=1)  #  Downsampling will lose the original data, and it is necessary to concatenate the result of upsampling to supplement the lost detail information.
        x = self.dec1(x)

        x = self.up2(x)             # [B, 32, 28, 28]
        x = torch.cat([x, x1], dim=1)
        x = self.dec2(x)

        return torch.sigmoid(self.final(x))

class NoisyMNIST(Dataset):  
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        noise = torch.randn_like(img) * 0.3
        noisy_img = torch.clamp(img + noise, 0., 1.)
        return noisy_img, img

def train(model, train_loader, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()
    ssim_loss = SSIM(data_range=1.0, size_average=True, channel=1)
    os.makedirs(model_dir, exist_ok=True)
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for noisy_imgs, clean_imgs in train_loader:
            noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
            outputs = model(noisy_imgs)

            mse = mse_loss(outputs, clean_imgs)
            ssim = 1 - ssim_loss(outputs, clean_imgs)  # The SSIM value should be as large as possible, so 1 - SSIM is used as the loss function.
            loss = 0.3 * mse + 0.7 * ssim

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_path)
            print("Best model saved.")

def visualize_denoising(model, data_loader, num_images=6):
    model.eval()
    noisy_imgs, clean_imgs = next(iter(data_loader))
    noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
    
    with torch.no_grad():
        outputs = model(noisy_imgs)

    noisy = noisy_imgs[:num_images].cpu()
    clean = clean_imgs[:num_images].cpu()
    output = outputs[:num_images].cpu()

    # Calculate the SSIM for each image
    ssim_scores = []
    ssim = SSIM(data_range=1.0, size_average=True, channel=1)
    for i in range(num_images):
        ssim_val = ssim(
            clean[i].unsqueeze(0),
            output[i].unsqueeze(0),
        )
        ssim_scores.append(ssim_val.item())

    # visualization
    fig, axes = plt.subplots(3, num_images, figsize=(num_images * 2, 6))
    for i in range(num_images):

        axes[0, i].imshow(clean[i].squeeze(), cmap='gray')
        axes[1, i].imshow(noisy[i].squeeze(), cmap='gray')
        axes[2, i].imshow(output[i].squeeze(), cmap='gray')

        axes[0, i].set_title("Clean")
        axes[1, i].set_title("Noisy")
        axes[2, i].set_title(f"Denoised\nSSIM: {ssim_scores[i]:.3f}")
        
        for ax in axes[:, i]:
            ax.axis('off')

    plt.tight_layout()
    
    print(f"Average SSIM: {np.mean(ssim_scores):.4f}")
    plt.savefig("image_reconstruction", bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":

    # Load dataset
    transform = transforms.ToTensor()
    train_dataset = FashionMNIST(root=DATA_PATH, train=True, download=True, transform=transform)
    test_dataset = FashionMNIST(root=DATA_PATH, train=False, download=True, transform=transform)
    print("data init")
    print(f"Number of samples in the training set: {len(train_dataset)}")  # 60000
    print(f"Number of samples in the testing set: {len(test_dataset)}")    # 10000

    train_loader = DataLoader(NoisyMNIST(train_dataset), batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(NoisyMNIST(test_dataset), batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    print("model init")
    model = DenoisingResNetSkip().to(device)

    # Check if the model exists
    if os.path.exists(model_path):
        print("The model already exists. Loading for testing...")
        model.load_state_dict(torch.load(model_path))
        visualize_denoising(model, test_loader, num_images=12)
    else:
        print("Model does not exist. Starting training...")
        train(model, train_loader, epochs=50)
        print("Training completed. Starting the test...")
        visualize_denoising(model, test_loader, num_images=12)


