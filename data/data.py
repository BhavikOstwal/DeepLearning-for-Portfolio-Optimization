import torch
from torch.utils.data import Dataset

class PortfolioDataset(Dataset):
    """Custom Dataset for using it with Dataloader class of pytorhc."""
    def __init__(self, sequences):
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        x, r_next = self.sequences[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(r_next, dtype=torch.float32)