import torch

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

from torch_geometric.nn import GCNConv, global_mean_pool 
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


from dataset import SEQ_TO_TOKEN, TOKEN_TO_SEQ

TYPES = [
    '$_NOT_',
    '$_AND_',
    '$_OR_',
    '$_XOR_',
    '$_MUX_',
    '$_DFF_N_',
    '$_DFF_P_',
    '$_DFF_NN0_',
    '$_DFF_NN1_',
    '$_DFF_NP0_',
    '$_DFF_NP1_',
    '$_DFF_PN0_',
    '$_DFF_PN1_',
    '$_DFF_PP0_',
    '$_DFF_PP1_',
    'input',
    'output',
]
TYPES_TO_IDS = {x: i for i, x in enumerate(TYPES)}

LEN_TYPES = len(TYPES)

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
    
class GCN(nn.Module):
    def __init__(self, hidden_dim, output_dim, embedding_dim=None, raw_graph=False):
        super().__init__()
        self.raw_graph = raw_graph
        if raw_graph:
            print('using raw graph')
            input_dim = 2 # TODO: adjust this accordingly
        else:
            if embedding_dim is None:
                embedding_dim = hidden_dim
            print('using preprocessed graph')
            self.embedding_table = torch.nn.Embedding(LEN_TYPES, embedding_dim)
            input_dim = embedding_dim
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        batch = data.batch
        edge_index = data.edge_index
        if self.raw_graph:
            x0, x1, edge_index = data.invert0, data.invert1, data.edge_index
            x = torch.stack([x0, x1])
            x = x.T.type(torch.float)
        else:
            ids = []
            for dt in data.type:
                ids += [TYPES_TO_IDS[x] for x in dt]
            ids = torch.tensor(ids).cuda()
            x = self.embedding_table(ids)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)

        return x

class E2ERegression(nn.Module):
    def __init__(
        self, 
        gcn_hidden_dim, 
        gcn_output_dim,
        se_input_dim,
        se_num_lstm_layers,
        output_dim=2,
        raw_graph=False,
        embedding_dim=None,
    ):
        super().__init__()
        self.gcn = GCN(
            gcn_hidden_dim, 
            gcn_output_dim, 
            embedding_dim=embedding_dim,
            raw_graph=raw_graph,
        )
        self.se = SequenceEmbedding(se_input_dim, gcn_output_dim, se_num_lstm_layers)
        self.dense = nn.Linear(gcn_output_dim, output_dim)
        self.num_lstm_layers = se_num_lstm_layers
        #self.dropout = nn.Dropout(p=0.1)
    
    def forward(self, graph_input, sequence, sequence_len):
        x = self.gcn(graph_input)
        x = torch.stack([x] * self.num_lstm_layers)
        x, x_len = self.se(sequence, sequence_len, x)
        x = torch.stack([x[i, l - 1, :] for i, l in enumerate(x_len)])
        #x = self.dropout(x)
        x = self.dense(x)
        return x

