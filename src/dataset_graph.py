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

NODE_TYPES = [
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

TYPES_TO_IDS = {x: i for i, x in enumerate(NODE_TYPES)}
SEQ_TO_TOKEN = ["rewrite", "rewrite -z", "refactor", "refactor -z",
               "balance",  "dc2",
               "if -K 6 -x", "if -K 6 -g",
               "resub -K 8 -N 1",  "resub -K 8 -N 2",
               "resub -K 12 -N 1",  "resub -K 12 -N 2",
               "resub -K 8 -N 1 -z", "resub -K 8 -N 2 -z",
               "resub -K 12 -N 1 -z",  "resub -K 12 -N 2 -z"]
SEQ_TO_TOKEN = {x: i for i,x in enumerate(SEQ_TO_TOKEN)}

# SEQ_TO_TOKEN = {
#     '&if -W 300 -K 6 -v': 0, 
#     '&st': 1, 
#     '&synch2': 2, 
#     '&dc2': 3, 
#     '&if -W 300 -y -K 6': 4, 
#     '&syn2': 5, 
#     '&sweep': 6, 
#     '&mfs': 7, 
#     '&scorr': 8, 
#     '&if -W 300 -g -K 6': 9, 
#     '&b -d': 10, 
#     '&if -W 300 -x -K 6': 11, 
#     '&dch': 12, 
#     '&b': 13, 
#     '&syn4': 14, 
#     '&dch -f': 15, 
#     '&syn3': 16
# }

# for x, y in SEQ_TO_TOKEN.items():
#     TOKEN_TO_SEQ[y] = x

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
#        seq = seq.split(';')[2: -3] # remove the redundant parts
        seq = seq.split(';')[0: -2] # remove the redundant parts
        sl = []
        for s in seq:
#            if s.startswith('&'):
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
def preprocess_data(graph_data_dir, label_seq_data_path, output_dir, debug=False):
    '''
    graph_data_dir: a str of the directory path that contains graph data in gml or pkl
    label_seq_data_path: a str or a list of pkl files that contain labels and sequences
    '''
    if not isinstance(label_seq_data_path, list):
        label_seq_data_path = [label_seq_data_path]

    # find all GMLs in graph data dir
    gmls = glob.glob(os.path.normpath(graph_data_dir+"/*.gml"))

    graphs_all = []
    # loop over each graph GML inidividually
    for i, gml_file in enumerate(gmls):
        filename = os.path.basename(gml_file)
        name = filename[filename.find('_') + 1 :filename.find('.gml')]
        pkl_file = os.path.normpath(os.path.join(output_dir, name + ".pkl"))
        if os.path.exists(pkl_file):
            print(f'{name}.pkl exists, loading from pickle file')
            with open(pkl_file, "rb") as handle:
                graphs = pickle.load(handle)
        else:
            print(f'{name}.pkl does not exist, generating from gml file')

            # read graph from GML and convert to PyG Data object
            G = nx.read_gml(gml_file)
            data = convert.from_networkx(G)
            ids = torch.tensor([TYPES_TO_IDS[x] for x in data.type])
            data.type = ids
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
            # Store current graphs' data as pkl
            with open(pkl_file, "wb") as handle:
                print(f'dumping {name}.pkl')
                pickle.dump(graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f'length of graphs: {len(graphs)}')
        graphs_all += graphs

        if debug:
            break # for fast testing purpose

    print(f'length of all graphs: {len(graphs_all)}')
    return graphs_all

# top level caller 
def generate_dataloaders(
    graph_data_dir, 
    label_seq_data_path, 
    train_batch_size=256, 
    eval_batch_size=256,
    p_val=0.2, 
    seed=100,
    debug=False,
):
    # features, sequences, labels = preprocess_data(data_path, feature_dim=feature_dim)
    graphs = preprocess_data(graph_data_dir, label_seq_data_path, output_dir=graph_data_dir, debug=debug)
    random.seed(seed)
    random.shuffle(graphs)

    num_training_sets = int((1 - p_val) * len(graphs))
    training_dataset = graphs[:num_training_sets]
    valid_dataset = graphs[num_training_sets:]

    train_dataloader = DataLoader(
        training_dataset,
        batch_size=train_batch_size,
        shuffle=True,
    ) 

    valid_datalader = DataLoader(
        valid_dataset,
        batch_size=eval_batch_size,
        shuffle=True,
    )

    return train_dataloader, valid_datalader


if __name__ == '__main__':
    output_dir = "../epfl_gmls"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    graphs = preprocess_data('../epfl_gmls', '../epflBmarks.pkl', output_dir)
