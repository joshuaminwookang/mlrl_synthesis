import random
import pickle,csv
from tqdm import tqdm
import numpy as np
import torch
import copy
import os, glob
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric.utils.convert as convert
import networkx as nx

SEQ_TO_TOKEN = {
    '&if -W 300 -K 6 -v': 0, 
    '&st': 1, 
    '&synch2': 2, 
    '&dc2': 3, 
    '&if -W 300 -y -K 6': 4, 
    '&syn2': 5, 
    '&sweep': 6, 
    '&mfs': 7, 
    '&scorr': 8, 
    '&if -W 300 -g -K 6': 9, 
    '&b -d': 10, 
    '&if -W 300 -x -K 6': 11, 
    '&dch': 12, 
    '&b': 13, 
    '&syn4': 14, 
    '&dch -f': 15, 
    '&syn3': 16
}

TOKEN_TO_SEQ = {}
for x, y in SEQ_TO_TOKEN.items():
    TOKEN_TO_SEQ[y] = x

def prepare_dataset(data, graphs_dict):
    features = []
    #print('preparing dataset')
    #print(data)
    for _, d in tqdm(data.items()):
        label = {}
        label['Path_Delay'] = d['Path_Delay'] / 1e2
        label['Slice_LUTs'] = d['Slice_LUTs'] / 1e3
        name = d['Benchmark']
        # assert name in graphs_dict
        if name not in graphs_dict:
            continue
        feature = copy.deepcopy(graphs_dict[name])
        feature.labels = list(label.values())
        feature.sequence = preprocess_sequence([d['Sequence']])[0]
        
        features.append(feature)
    return features


def preprocess_sequence(sequences):
    # convert the string representation into a list of tokens
#    print('preprocessing sequences')
    seq_list = []
    for seq in sequences:
        seq = seq.split(';')[2: -3] # remove the redundant parts
        sl = []
        for s in seq:
            if s.startswith('&'):
                sl.append(SEQ_TO_TOKEN[s])
        seq_list.append(np.array(sl))
    return seq_list

def flatten_all(data):
    flattened_data = []
    for d in data:
        fd = list(d.values())
        flattened_data.append(fd)
    return np.array(flattened_data)

def normalize(data):
    eps = 1e-5
    data_t = np.transpose(data)
    for i in range(len(data_t)):
        mean = np.mean(data_t[i])
        std = np.std(data_t[i])
        data_t[i] = (data_t[i] - mean) / (std + eps)
    return np.transpose(data_t)


# Preprocess data (merge GML files and label/sequence data)
# Store processed PyTorch Geoemtric Data in individual pkl files for each graph
def preprocess_data(graph_data_dir, label_seq_data_path, output_dir):
    if not isinstance(label_seq_data_path, list):
        label_seq_data_path = [label_seq_data_path]

    # find all GMLs in graph data dir
    gmls = glob.glob(os.path.normpath(graph_data_dir+"/*.gml"))

    # loop over each graph GML inidividually
    for gml_file in gmls:
        filename = os.path.basename(gml_file)
        name = filename[filename.find('_') + 1 :filename.find('.gml')]
        print(name)

        # read graph from GML and convert to PyG Data object
        G = nx.read_gml(gml_file)
        data = convert.from_networkx(G)
        graph_dict = {}
        graph_dict[name] = data  
        graphs = []

        # itreate over all label/sequence data to produce dataset
        for _data_path in label_seq_data_path:
            with open(_data_path, 'rb') as f:
                data = pickle.load(f)
                for _data in data:
                    _graphs = prepare_dataset(_data, graph_dict)
                    graphs += _graphs
        print(f'length of graphs: {len(graphs)}')
        
        # Store current graphs' data as pkl
        pkl_file = os.path.normpath(os.path.join(output_dir, name + ".pkl"))
        with open(pkl_file, "wb") as handle:
            pickle.dump(graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # features_normalized = np.array(features)
    # Uncomment the following for testing/ablation
    #features_normalized = np.zeros_like(features_normalized)
    #features_normalized = np.random.rand(*features_normalized.shape)

class CustomDataset(Dataset):
    def __init__(self, graphs=None, data_path=None):
        if data_path is not None:
            self.graphs = preprocess_data(data_path)
        else:
            assert graphs is not None 
            self.graphs = graphs
        self.input_dim = self.graphs.shape[-1]

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph_tensor = torch.tensor(self.graphs[idx]).to(torch.float)
        # sequence = torch.tensor(self.sequences[idx]).to(torch.long)
        # label = torch.tensor(self.labels[idx]).to(torch.float)
        #return {'feature': feature, 'sequence': sequence, 'label': label}
        return graph_tensor

# top level caller 
def generate_datasets(graph_data_dir, label_seq_data_path, p_val=0.2):
    # features, sequences, labels = preprocess_data(data_path, feature_dim=feature_dim)
    graphs = preprocess_data(graph_data_dir, label_seq_data_path)

    num_training_sets = int((1 - p_val) * len(graphs))
    indices_all = np.array(list(range(len(graphs))))
    random.seed(100)
    random.shuffle(indices_all)
    print(indices_all)
    train_indices = indices_all[:num_training_sets]
    val_indices = indices_all[num_training_sets:]

    train_dataset = CustomDataset(
        graphs[train_indices]
        # features[train_indices], 
        # sequences[train_indices],
        # labels[train_indices],
    )
    valid_dataset = CustomDataset(
        graphs[train_indices]
        # features[val_indices], 
        # sequences[val_indices],
        # labels[val_indices],
    )

    return train_dataset, valid_dataset

def pad_collate(batch):
    features, sequences, labels, = zip(*batch)
    sequences_len = [len(s) for s in sequences]
    features_pad = pad_sequence(features, batch_first=True, padding_value=0)
    sequences_pad = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels_pad = pad_sequence(labels, batch_first=True, padding_value=0) 
    return features_pad, sequences_pad, sequences_len, labels_pad


if __name__ == '__main__':
    output_dir = "../epfl_gmls"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    graphs = preprocess_data('../epfl_gmls', '../epflBmarks.pkl', output_dir)
