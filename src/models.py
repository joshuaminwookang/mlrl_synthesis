import torch

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

from torch_geometric.nn import GCNConv, global_mean_pool 
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from dataset_graph import SEQ_TO_TOKEN, NODE_TYPES, TYPES_TO_IDS

LEN_TYPES = len(NODE_TYPES)

class SequenceEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_lstm_layers):
        super().__init__()
        self.embedding = nn.Embedding(len(SEQ_TO_TOKEN), input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_lstm_layers, batch_first=True)
        
    def forward(self, x, x_len, h):
        x = self.embedding(x)
        x = pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        if h is not None:
            x, (h, c) = self.lstm(x, (h, h))
        else:
            x, (h, c) = self.lstm(x)
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
            ids = data.type
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
        mode='sequential',
    ):
        super().__init__()
        assert mode in ['sequential', 'parallel']
        print(f'Run as the {mode} mode')
        self.mode = mode
        self.gcn = GCN(
            gcn_hidden_dim, 
            gcn_output_dim, 
            embedding_dim=embedding_dim,
            raw_graph=raw_graph,
        )
        self.se = SequenceEmbedding(se_input_dim, gcn_output_dim, se_num_lstm_layers)
        dense_dim = gcn_output_dim + se_input_dim if mode == 'parallel' else se_input_dim
        self.dense = nn.Linear(dense_dim, output_dim)
        self.num_lstm_layers = se_num_lstm_layers
        #self.dropout = nn.Dropout(p=0.1)
    
    def forward(self, graph_input, sequence, sequence_len):
        graph_embedding = self.gcn(graph_input)
        if self.mode == 'parallel':
            h = None
        else:
            h = torch.stack([graph_embedding] * self.num_lstm_layers)
        x, x_len = self.se(sequence, sequence_len, h)
        x = torch.stack([x[i, l - 1, :] for i, l in enumerate(x_len)])
        if self.mode == 'parallel':
            x = torch.cat([graph_embedding, x], dim=-1)
        #x = self.dropout(x)
        x = self.dense(x)
        return x

