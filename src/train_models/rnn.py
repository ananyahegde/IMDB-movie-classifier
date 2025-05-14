import torch

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.set_default_device(device)
print(f"Using device = {torch.get_default_device()}")

