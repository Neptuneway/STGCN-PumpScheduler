import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np

class GCNLSTMModel(nn.Module):
    def __init__(self, 
                 num_nodes=33,           # Number of monitoring points
                 input_dim=1,            # Input feature dimension per node (pressure value)
                 hidden_dim=64,          # GCN hidden layer dimension
                 lstm_hidden_dim=64,     # LSTM hidden layer dimension
                 lstm_layers=4,          # Number of LSTM layers
                 output_dim=8,           # Output dimension (number of head loss outputs, now 8)
                 dropout=0.5,            # Dropout rate
                 enable_bias=True):      # Whether to use bias
        super(GCNLSTMModel, self).__init__()
        
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        self.output_dim = output_dim
        self.dropout_rate = dropout
        
        # GCN layers (2 layers) - corresponds to stblock_num=2 in STGCN
        self.gcn = nn.ModuleList([
            GCNConv(input_dim, hidden_dim, bias=enable_bias),
            GCNConv(hidden_dim, hidden_dim, bias=enable_bias)
        ])
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=hidden_dim * num_nodes,  # Feature dimension output by GCN
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layers
        self.fc1 = nn.Linear(lstm_hidden_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)
        
        # Activation function
        self.relu = nn.ReLU()
        
    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward propagation - improved version: process by time step
        Args:
            x: Input tensor [batch_size, seq_len, num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Edge weights [num_edges]
        Returns:
            output: Predicted head loss [batch_size, output_dim]
        """
        batch_size, seq_len, num_nodes, input_dim = x.shape
        
        # Improved scheme: process by time step
        gcn_outputs = []
        for t in range(seq_len):
            # Extract current time step [batch, nodes, features]
            xt = x[:, t, :, :]  
            
            # Pass through shared GCN layers
            for i, gcn_layer in enumerate(self.gcn):
                xt = F.relu(gcn_layer(xt, edge_index, edge_weight))
                if i < len(self.gcn) - 1:  # Do not add dropout to the last layer
                    xt = F.dropout(xt, p=self.dropout_rate, training=self.training)
            
            gcn_outputs.append(xt)  # Save current time step result

        # Combine time steps [batch, seq_len, nodes*features]
        lstm_input = torch.stack(gcn_outputs, dim=1)
        lstm_input = lstm_input.view(batch_size, seq_len, -1)  # Flatten node dimension
        
        # Pass through LSTM layer
        lstm_out, _ = self.lstm(lstm_input)
        
        # Take the output of the last time step
        lstm_out = lstm_out[:, -1, :]  # [batch_size, lstm_hidden_dim]
        
        # Pass through fully connected layers
        out = self.relu(self.fc1(lstm_out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

class EarlyStopping:
    """
    Early stopping mechanism
    """
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience 