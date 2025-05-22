import torch
import torch.nn
import numpy as np
from PF_OPTIM.utils.preprcoess import load_data, standardize_data, calculate_returns, create_sequences
from PF_OPTIM.data.data import PortfolioDataset
from torch.utils.data import DataLoader, Dataset 
from PF_OPTIM.model.lstm_model import PortfolioModel
import os

device = torch.device('mps' if torch.mps.is_available() else 'cpu')

def test_model(model, test_sequences, returns, sigma_tgt=0.10, C=0.0001):
    """Test the model with volatility scaling and transaction costs."""
    model.eval()
    R_p_net = []
    scaled_w_prev = np.zeros(4)
    sigma_i = returns.ewm(span=50).std() * np.sqrt(252)  # Annualized volatility
    
    for t in range(len(test_sequences)):
        x, _ = test_sequences[t]
        r_t = returns.iloc[t + 50].values  # Align with sequence index
        sigma_i_t = sigma_i.iloc[t + 50].values
        with torch.no_grad():
            weights = model(torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)).cpu().numpy()[0]
        scaled_w = (sigma_tgt / sigma_i_t) * weights
        cost = C * np.sum(np.abs(scaled_w - scaled_w_prev)) if t > 0 else 0
        R_p = np.sum(scaled_w * r_t) - cost
        R_p_net.append(R_p)
        scaled_w_prev = scaled_w
    
    # Compute performance metrics
    mean_R_p = np.mean(R_p_net)
    std_R_p = np.std(R_p_net, ddof=1)
    sharpe = mean_R_p / std_R_p if std_R_p > 0 else 0
    
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

    return {
        f'\n{BLUE}Expected Return (Annualized){RESET}': mean_R_p * 252,
        f'{YELLOW}Volatility (Annualized){RESET}': std_R_p * np.sqrt(252),
        f'{GREEN}Sharpe Ratio (Annualized){RESET}': sharpe * np.sqrt(252)
    }


if __name__ == "__main__":
    file_paths = ['VTI.csv', 'AGG.csv', 'DBC.csv', 'VIX.csv']
    data = load_data(file_paths)
    returns = calculate_returns(data)

    test_start = '2019-01-01'
    test_end = '2026-12-31'
    test_data = data[(data.index >= test_start) & (data.index <= test_end)]

    train_start = '2009-01-01'
    train_end = '2019-01-01'
    test_start = '2019-01-01'
    test_end = '2019-12-31'
    train_data = data[(data.index >= train_start) & (data.index < train_end)]
    # Note: Keep the trainig data same(as the model was trained on) as it's mean and std is used for standardizing test data also..
    
    test_data = data[(data.index >= test_start) & (data.index <= test_end)]
    train_prices_std, test_prices_std, train_returns_std, test_returns_std = standardize_data(train_data, test_data)

    model = PortfolioModel().to(device)
    state_dict = torch.load('PF_OPTIM/checkpoint/best_model.pt', weights_only=True)
    model.load_state_dict(state_dict)

    test_returns = returns.loc[test_prices_std.index]
    test_sequences = create_sequences(test_prices_std, test_returns_std)
    results = test_model(model, test_sequences, test_returns, sigma_tgt=0.1, C=0.0001) #selected the hyperparams that gave best results(results tabel in paper)
    
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    print()