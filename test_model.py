import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from model import MNISTModel
import glob
import pytest
import os

def get_latest_model():
    model_files = glob.glob(os.path.join('models', 'model_mnist_*.pth'))
    return max(model_files) if model_files else None

def test_model_architecture():
    model = MNISTModel()
    
    # Test 1: Check model parameters count
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 25000, f"Model has {total_params} parameters, should be less than 25000"
    
    # Test 2: Check input shape compatibility
    test_input = torch.randn(1, 1, 28, 28)
    try:
        output = model(test_input)
        assert output.shape == (1, 10), f"Output shape is {output.shape}, should be (1, 10)"
    except Exception as e:
        pytest.fail(f"Model failed to process 28x28 input: {str(e)}")

def test_model_accuracy():
    # Load the latest model
    model_path = get_latest_model()
    assert model_path is not None, "No trained model found"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTModel().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Load test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, download=True, transform=transform),
        batch_size=1000)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    assert accuracy > 95, f"Model accuracy is {accuracy}%, should be > 95%"

if __name__ == "__main__":
    pytest.main([__file__]) 