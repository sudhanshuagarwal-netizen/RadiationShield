import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def get_fashion_mnist_dataloaders(batch_size=128, val_split=0.1):
    """
    Load FashionMNIST dataset and return train, val, and test DataLoaders.
    """
    # Define transforms: convert images to tensors and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),                    # Convert PIL image to tensor (0-1 range)
        transforms.Normalize((0.2860,), (0.3530,))  # FashionMNIST mean and std for grayscale
    ])
    
    # Download and load full training set
    train_dataset = datasets.FashionMNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # Download and load test set
    test_dataset = datasets.FashionMNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Split train into train + validation
    val_size = int(val_split * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Dataset loaded successfully!")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


def visualize_samples(loader, num_samples=10):
    """Visualize random samples from a DataLoader and save the plot."""
    images, labels = next(iter(loader))
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.ravel()
    
    for i in range(num_samples):
        axes[i].imshow(images[i].squeeze(), cmap='gray')
        axes[i].set_title(class_names[labels[i]])
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Create results folder if it doesn't exist
    import os
    os.makedirs('results', exist_ok=True)
    
    save_path = 'results/fashion_mnist_samples.png'
    plt.savefig(save_path)
    print(f"\n✅ Saved sample images to {save_path}")
    
    # Only try to show if running in interactive environment
    try:
        plt.show()
    except:
        print("Plot saved but could not display window (running in terminal).")
    
    plt.close()  # Clean up

# Quick test when running this file directly
if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_fashion_mnist_dataloaders(batch_size=128)
    print("DataLoaders created successfully!")
    
    visualize_samples(train_loader)