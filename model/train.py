import torch
import torch.nn
import numpy as np
from PF_OPTIM.utils.preprcoess import load_data,standardize_data, calculate_returns, create_sequences
from PF_OPTIM.utils.metrics import evaluate_sharpe 
from PF_OPTIM.data.data import PortfolioDataset
from torch.utils.data import DataLoader, Dataset 
from PF_OPTIM.model.lstm_model import PortfolioModel

device = torch.device('mps' if torch.mps.is_available() else 'cpu')


def train_model(model, train_loader, val_sequences, epochs=100, patience=10):
    """Train the model using negative Sharpe ratio as loss."""
    optimizer = torch.optim.Adam(model.parameters())
    best_val_sharpe = -np.inf
    wait = 0
    
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_r_next in train_loader:
            batch_x, batch_r_next = batch_x.to(device), batch_r_next.to(device)
            optimizer.zero_grad()
            weights = model(batch_x)
            R_p = (weights * batch_r_next).sum(dim=1)
            mean_R_p = R_p.mean()
            std_R_p = R_p.std(unbiased=False)
            loss = -mean_R_p / (std_R_p + 1e-6)  # Negative Sharpe ratio
            loss.backward()
            optimizer.step()
        
        # Validation
        val_sharpe = evaluate_sharpe(model, val_sequences)
        print(f"Epoch {epoch+1}, Validation Sharpe: {val_sharpe:.4f}")
        if val_sharpe > best_val_sharpe:
            best_val_sharpe = val_sharpe
            wait = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered")
                break
    
    # model.load_state_dict(torch.load('best_model.pt'))
    return model


if __name__ == '__main__':
    file_paths = ['VTI.csv', 'AGG.csv', 'DBC.csv', 'VIX.csv']
    data = load_data(file_paths)
    returns = calculate_returns(data)
    
    train_start = '2009-01-01'
    train_end = '2019-01-01'  # stop a bit earlier for testing window
    train_data = data[(data.index >= train_start) & (data.index < train_end)]

    train_prices_std, _, train_returns_std, _ = standardize_data(train_data, train_data)
    full_train_sequences = create_sequences(train_prices_std, train_returns_std)
    val_size = int(0.1 * len(full_train_sequences))
    train_sequences = full_train_sequences[:-val_size]
    val_sequences = full_train_sequences[-val_size:]

    train_dataset = PortfolioDataset(train_sequences)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    model = PortfolioModel().to(device)
    model = train_model(model, train_loader, val_sequences,patience=10)
    