import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

A_DIR = str(Path(__file__).resolve().parent.parent)
sys.path.append(A_DIR)
from set_project_path import PROJECT_ROOT, DATA_PATH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using ", device)
model_dir = os.path.join(PROJECT_ROOT, "B3/models")
model_path = os.path.join(model_dir, "B3_model.pth")
os.makedirs(model_dir, exist_ok=True)

class DeeperCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(DeeperCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)   # 1x28x28 → 32x28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)            # 64x28x28
        self.pool1 = nn.MaxPool2d(2, 2)                                     # → 64x14x14

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)           # 128x14x14
        self.pool2 = nn.MaxPool2d(2, 2)                                     # → 128x7x7

        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))       
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.3): # noise ~ N(mean, std^2)
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy_tensor = tensor + noise
        return torch.clamp(noisy_tensor, 0., 1.)

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

class AddSaltPepperNoise(object):
    def __init__(self, amount=0.05, salt_vs_pepper=0.5):
        """
        amount: The proportion of noise added
        salt_vs_pepper: salt:pepper, 1:1 is default
        """
        self.amount = amount
        self.salt_vs_pepper = salt_vs_pepper

    def __call__(self, tensor):
        noisy = tensor.clone()
        num_pixels = tensor.numel()
        num_salt = int(num_pixels * self.amount * self.salt_vs_pepper)
        num_pepper = int(num_pixels * self.amount * (1.0 - self.salt_vs_pepper))

        # salt noise , pixel = 1
        coords = [torch.randint(0, s, (num_salt,)) for s in tensor.shape]
        noisy[coords] = 1.0

        # pepper noise , pixel = 0
        coords = [torch.randint(0, s, (num_pepper,)) for s in tensor.shape]
        noisy[coords] = 0.0

        return noisy

    def __repr__(self):
        return f"{self.__class__.__name__}(amount={self.amount}, salt_vs_pepper={self.salt_vs_pepper})"

def train(model, model_path, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # preprocessing 
    transform = transforms.Compose([
        transforms.ToTensor(), # totensor and normalise to [0~1]
        transforms.Normalize((0.5,), (0.5,)) # map to [-1,1] mean,std = 0.5
    ])

    # MNIST data
    mnist_train = torchvision.datasets.MNIST(root=DATA_PATH, train=True, download=True, transform=transform)

    mnist_loader = torch.utils.data.DataLoader(mnist_train, batch_size=64, shuffle=True)

    # train
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_batches = 0

        for inputs, labels in mnist_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_batches += 1

        avg_loss = running_loss / total_batches
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), model_path)


def test(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()  # （close dropout, BN）

    transform = transforms.Compose([
        transforms.ToTensor(),  #totensor and normalise to [0~1]
        transforms.Normalize((0.5,), (0.5,)) # map to [-1,1] mean,std = 0.5
    ])

    # add noise
    noisy_transform = transforms.Compose([
        transforms.ToTensor(),
        AddGaussianNoise(mean=0., std=0.3),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # add salt noise
    salt_pepper_noisy_transform = transforms.Compose([
        transforms.ToTensor(),
        AddSaltPepperNoise(amount=0.8, salt_vs_pepper=0.5),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    noisy_test_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=False, download=True, transform=noisy_transform)
    noisy_test_loader = torch.utils.data.DataLoader(noisy_test_dataset, batch_size=64, shuffle=False)

    salt_pepper_noisy_test_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=False, download=True, transform=salt_pepper_noisy_transform)
    salt_pepper_noisy_test_loader = torch.utils.data.DataLoader(salt_pepper_noisy_test_dataset, batch_size=64, shuffle=False)

    # Calculate accuracies
    # Clean accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    clean_accuracy = 100 * correct / total
    print(f"Test Accuracy: {clean_accuracy:.2f}%")

    # Gaussian noise accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in noisy_test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    noisy_accuracy = 100 * correct / total
    print(f"Noisy Test Accuracy: {noisy_accuracy:.2f}%")

    # Salt-pepper noise accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in salt_pepper_noisy_test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    salt_pepper_accuracy = 100 * correct / total
    print(f"Salt pepper Noisy Test Accuracy: {salt_pepper_accuracy:.2f}%")

    # show original image and noise image
    orig_img = test_dataset[0][0]
    noisy_img = noisy_test_dataset[0][0]
    salt_img = salt_pepper_noisy_test_dataset[0][0]

    plt.figure(figsize=(10, 3))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(orig_img.squeeze(), cmap='gray')
    plt.title(f"Original\nAccuracy: {clean_accuracy:.2f}%")
    
    # Gaussian noisy image
    plt.subplot(1, 3, 2)
    plt.imshow(noisy_img.squeeze(), cmap='gray')
    plt.title(f"Gaussian Noise\nAccuracy: {noisy_accuracy:.2f}%")
    
    # Salt-pepper noisy image
    plt.subplot(1, 3, 3)
    plt.imshow(salt_img.squeeze(), cmap='gray')
    plt.title(f"Salt & Pepper Noise\nAccuracy: {salt_pepper_accuracy:.2f}%")
    
    plt.tight_layout()
    plt.savefig("compare_noise_vs_clean.png")
    plt.close()

if __name__ == "__main__":
        # load model
    model = DeeperCNN(in_channels=1).to(device)

    if os.path.exists(model_path):
        print("The model already exists. Directly load the test...")
        test(model, model_path)
    else:
        print("The model doesn't exist. Start training...")
        train(model, model_path)
        print("The training is completed and the test begins...")
        test(model, model_path)
