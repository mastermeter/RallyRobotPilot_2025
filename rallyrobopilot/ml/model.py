# rallyrobopilot/ml/model.py
import torch.nn as nn

class RobopilotMLP(nn.Module):
    """A flexible Multi-Layer Perceptron for the Rally Robopilot."""
    def __init__(self, input_size=15, output_size=4, dropout_rate=0.5):
        super(RobopilotMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)