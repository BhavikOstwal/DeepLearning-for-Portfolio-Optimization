import torch.nn as nn

class PortfolioModel(nn.Module):
    """LSTM-based model for portfolio weight prediction."""
    def __init__(self, input_size=8, hidden_size=64, output_size=4):
        super(PortfolioModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.linear(h_n[-1])
        weights = self.softmax(out)
        return weights