import torch
import numpy as np

device = torch.device('mps' if torch.mps.is_available() else 'cpu')

def evaluate_sharpe(model, sequences):
    """Evaluate Sharpe ratio on a set of sequences."""
    model.eval()
    R_p_list = []
    with torch.no_grad():
        for x, r_next in sequences:
            x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
            weights = model(x)
            R_p = (weights * torch.tensor(r_next, dtype=torch.float32).to(device)).sum().item()
            R_p_list.append(R_p)
    mean_R_p = np.mean(R_p_list)
    std_R_p = np.std(R_p_list, ddof=1)
    return mean_R_p / (std_R_p + 1e-6) if std_R_p > 0 else 0
