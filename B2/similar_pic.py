import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import random
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path


A_DIR = str(Path(__file__).resolve().parent.parent)
sys.path.append(A_DIR)
from set_project_path import PROJECT_ROOT, DATA_PATH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using ", device)
model_dir = os.path.join(PROJECT_ROOT, "B2/models")
model_path = os.path.join(model_dir, "siamese_cifar_mnist.pth")
os.makedirs(model_dir, exist_ok=True)

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),  # For MNIST (1 channel)
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward_once(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)

class ContrastiveLoss(nn.Module): # Contrastive Loss 
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        distance = F.pairwise_distance(output1, output2)
        loss = torch.mean((1 - label) * torch.pow(distance, 2) +
                          label * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        return loss

class SiameseMNIST(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.labels = self.dataset.targets.tolist()
        self.label_to_indices = self._build_label_index()

    # Build the mapping from labels to index lists
    def _build_label_index(self):
        label_to_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(idx)
        return label_to_indices

    def __getitem__(self, index):
        img1, label1 = self.dataset[index]
        # Randomly decide whether to generate pairs of the same kind or pairs of different kinds
        if random.random() < 0.5:
            idx2 = random.choice(self.label_to_indices[label1])
        else:
            other_labels = [l for l in self.label_to_indices.keys() if l != label1]
            label2 = random.choice(other_labels)
            idx2 = random.choice(self.label_to_indices[label2])
            
        img2, label2 = self.dataset[idx2]
        return img1, img2, torch.tensor(int(label1 != label2), dtype=torch.float32)

    def __len__(self):
        return len(self.dataset)


def compute_similarity(model, img1, img2):
    img1_tensor = img1.unsqueeze(0).to(device)
    img2_tensor = img2.unsqueeze(0).to(device)
    with torch.no_grad():
        out1, out2 = model(img1_tensor, img2_tensor)
        dist = F.pairwise_distance(out1, out2)
        sim = (1 - dist / 2.0).clamp(0, 1)
        return sim.item()

def train(model, train_loader, epochs=10):
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        for img1, img2, label in train_loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            out1, out2 = model(img1, img2)
            loss = criterion(out1, out2, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # save model
    torch.save(model.state_dict(), model_path)


def test(model, test_set, model_path, save_path='comparison.png'):
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    same_num_imgs = []  # The same number (such as two 2 number)
    diff_num_imgs = []  # Different numbers (such as 4 and 9)
    
    # using number 2
    for img, label in test_set:
        if label == 2:
            same_num_imgs.append(img)
            if len(same_num_imgs) == 2:
                break
    #using number 4 and 9
    for img, label in test_set:
        if label == 2 and len(diff_num_imgs) == 0:
            diff_num_imgs.append(img)
        elif label == 9 and len(diff_num_imgs) == 1:
            diff_num_imgs.append(img)
            if len(diff_num_imgs) == 2:
                break
    
    # Calculate similarity
    same_sim = compute_similarity(model, same_num_imgs[0], same_num_imgs[1])
    diff_sim = compute_similarity(model, diff_num_imgs[0], diff_num_imgs[1])
    
    # Create a single figure with both comparisons
    plt.figure(figsize=(12, 6))
    
    # Adjust layout with gridspec for better control
    gs = plt.GridSpec(2, 3, width_ratios=[1, 1, 0.2], height_ratios=[1, 1])
    
    # Same Digit Comparison
    ax1 = plt.subplot(gs[0, 0])
    ax1.imshow(same_num_imgs[0].squeeze(), cmap='gray')
    ax1.set_title("Digit: 2", fontsize=10)
    ax1.axis('off')
    
    ax2 = plt.subplot(gs[0, 1])
    ax2.imshow(same_num_imgs[1].squeeze(), cmap='gray')
    ax2.set_title("Digit: 2", fontsize=10)
    ax2.axis('off')
    
    # Add similarity text in the third column
    ax_text1 = plt.subplot(gs[0, 2])
    ax_text1.axis('off')
    ax_text1.text(0.1, 0.5, 
                 f"Similarity: {same_sim:.4f}\nSame Digit Comparison",
                 fontsize=10, va='center')
    
    # Different Digits Comparison
    ax3 = plt.subplot(gs[1, 0])
    ax3.imshow(diff_num_imgs[0].squeeze(), cmap='gray')
    ax3.set_title("Digit: 2", fontsize=10)
    ax3.axis('off')
    
    ax4 = plt.subplot(gs[1, 1])
    ax4.imshow(diff_num_imgs[1].squeeze(), cmap='gray')
    ax4.set_title("Digit: 9", fontsize=10)
    ax4.axis('off')
    
    # Add similarity text in the third column
    ax_text2 = plt.subplot(gs[1, 2])
    ax_text2.axis('off')
    ax_text2.text(0.1, 0.5, 
                 f"Similarity: {diff_sim:.4f}\nDifferent Digits Comparison",
                 fontsize=10, va='center')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":

    trian_set = datasets.MNIST(root=DATA_PATH, train=train, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(SiameseMNIST(trian_set), batch_size=64, shuffle=True)

    test_set = datasets.MNIST(root = DATA_PATH, train=False, download=True, transform=transforms.ToTensor())

    model = SiameseNetwork().to(device)

    if os.path.exists(model_path):
        print("The model already exists. Directly load the test...")
        test(model, test_set, model_path)
    else:
        print("The model doesn't exist. Start training...")
        train(model, train_loader, epochs=10)
        print("The training is completed and the test begins...")
        test(model, test_set, model_path)
