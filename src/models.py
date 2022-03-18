import torch

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

from torch_geometric.nn import GCNConv,global_mean_pool 
# from torch_geometric.utils import add_self_loops, degree

from dataset import SEQ_TO_TOKEN, TOKEN_TO_SEQ

class FeatureEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.dense1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(hidden_dim, output_dim)
        #self.dropout = nn.Dropout(p=0.1)
    def forward(self, x):
        x = self.dense1(x)
        #x = self.bn1(x)
        x = self.relu(x)
        x = self.dense2(x)
        #x = self.dropout(x)
        return x
    
# class GCNConv(MessagePassing):
#     def __init__(self, in_channels, out_channels):
#         super(GCNConv, self).__init__(aggr='add')
#         self.lin = torch.nn.Linear(in_channels, out_channels)

#     def forward(self, x, edge_index):
#         # Step 1: Add self-loops
#         edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

#         # Step 2: Multiply with weights
#         x = self.lin(x)

#         # Step 3: Calculate the normalization
#         row, col = edge_index
#         deg = degree(row, x.size(0), dtype=x.dtype)
#         deg_inv_sqrt = deg.pow(-0.5)
#         norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

#         # Step 4: Propagate the embeddings to the next layer
#         return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x,
#                               norm=norm)

#     def message(self, x_j, norm):
#         # Normalize node features.
#         return norm.view(-1, 1) * x_j
    
class GraphEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.conv2 = GCNConv(hidden_dim, output_dim)
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
#        x = F.dropout(x, training=self.training)
        x = self.conv2(x)
        x = global_mean_pool(x)
        return x
    
class SequenceEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_lstm_layers):
        super().__init__()
        self.embedding = nn.Embedding(len(SEQ_TO_TOKEN), input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_lstm_layers, batch_first=True)
        
    def forward(self, x, x_len, h):
        x = self.embedding(x)
        x = pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        x, (h, c) = self.lstm(x, (h, h))
        x, x_len = pad_packed_sequence(x, batch_first=True)
        return x, x_len
    
class E2ERegression(nn.Module):
    def __init__(
        self, 
        fe_input_dim, 
        fe_hidden_dim, 
        fe_output_dim,
        se_input_dim,
        se_num_lstm_layers,
        output_dim=2,
    ):
        super().__init__()
        self.fe = FeatureEmbedding(fe_input_dim, fe_hidden_dim, fe_output_dim)
        self.se = SequenceEmbedding(se_input_dim, fe_output_dim, se_num_lstm_layers)
        self.dense = nn.Linear(fe_output_dim, output_dim)
        self.num_lstm_layers = se_num_lstm_layers
        #self.dropout = nn.Dropout(p=0.1)
    
    def forward(self, feature, sequence, sequence_len):
        x = self.fe(feature)
        x = torch.stack([x] * self.num_lstm_layers)
        x, x_len = self.se(sequence, sequence_len, x)
        x = torch.stack([x[i, l - 1, :] for i, l in enumerate(x_len)])
        #x = self.dropout(x)
        x = self.dense(x)
        return x

