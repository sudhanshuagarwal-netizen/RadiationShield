import torch
import numpy as np

print("=== RadAI Shield Setup Test ===")
print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")

# Check device
device = torch.device("cpu")
print(f"Using device: {device}")

# Create a simple tensor and do a basic operation
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
y = x * 2 + 1
print("\nSample tensor operation:")
print(f"Input tensor:\n{x}")
print(f"Result (x * 2 + 1):\n{y}")

print("\n✅ Setup test completed successfully!")