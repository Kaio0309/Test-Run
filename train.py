import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import MNISTModel
from datetime import datetime
import os

def train():
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    model = MNISTModel().to(device)
    print(f"Total parameters: {model.count_parameters():,}")
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    print("Downloading/Loading MNIST dataset...")
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=transform),
        batch_size=64, shuffle=True)
    
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print("Starting training...")
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Training Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    # Save model with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join('models', f'model_mnist_{timestamp}.pth')
    torch.save(model.state_dict(), save_path, _use_new_zipfile_serialization=True)
    print(f"Model saved as {save_path}")
    
if __name__ == "__main__":
    train() 