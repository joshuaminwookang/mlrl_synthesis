import random
import pickle,csv
from tqdm import tqdm
import numpy as np
import torch
import copy
import os, glob
from torch.nn.utils.rnn import pad_sequence

from torch_geometric.data import Data, Batch
import torch_geometric.utils.convert as convert
import networkx as nx

NODE_TYPES = [
    '$_NOT_',
    '$_AND_',
    '$_OR_',
    '$_XOR_',
    '$_MUX_',
    'input',
    'output'
]

TYPES_TO_IDS = {x: i for i, x in enumerate(NODE_TYPES)}

def types_to_ids(type_str):
    if type_str not in NODE_TYPES:
        return len(NODE_TYPES)
    return TYPES_TO_IDS[type_str]
    
SEQ_TO_TOKEN = ["rewrite", "rewrite -z", "refactor", "refactor -z",
              "balance",  "dc2", "dch -f",
              "if -K 6 -x", "if -K 6 -g",
              "resub -K 8 -N 1",  "resub -K 8 -N 2",
              "resub -K 12 -N 1",  "resub -K 12 -N 2",
              "resub -K 8 -N 1 -z", "resub -K 8 -N 2 -z",
              "resub -K 12 -N 1 -z",  "resub -K 12 -N 2 -z"]

# restricted case where number of ops = 7
# SEQ_TO_TOKEN = ["rewrite", "rewrite -z", "refactor", "refactor -z",
#                  "balance",  "resub", "resub -z"]
SEQ_TO_TOKEN = {x: i for i,x in enumerate(SEQ_TO_TOKEN)}

class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, data, graph_dict):
        super().__init__()
        self.data = data
        self.graph_dict = graph_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        name, label, sequence = self.data[idx]
        graph = self.graph_dict[name]
        return {'graph': graph, 'sequence': sequence, 'label': label}


def graph_dataset_collate_fn(data_list):
    graphs = [data['graph'] for data in data_list]
    labels = [data['label'] for data in data_list]
    sequences = [data['sequence'] for data in data_list]
    graph_collate =  Batch.from_data_list(graphs, [])

    return {'graph': graph_collate, 'label': labels, 'sequence': sequences}
    

class DataLoader(torch.utils.data.DataLoader):
    def __init__(self,
                 dataset,
                 batch_size=1,
                 shuffle=False,
                 follow_batch=[],
                 **kwargs):
        super(DataLoader, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=graph_dataset_collate_fn,
            **kwargs)


def get_label_seq(data, target_names=None):
    print(target_names)
    label_seq = []
    for d in tqdm(data):
        name = d['Benchmark']
        if target_names is not None and name not in target_names:
            continue
        label = [d['Path_Delay'] / 1e2, d['Slice_LUTs'] / 1e3] # delay, area
        sequence = preprocess_sequence([d['Sequence']])[0]
        label_seq.append([name, label, sequence])
    return label_seq


def prepare_dataset(data, graphs_dict):
    features = []
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
    seq_list = []
    for seq in sequences:
        seq = seq.split(';') # remove the redundant parts
        sl = []
        for s in seq:
            sl.append(SEQ_TO_TOKEN[s])
        seq_list.append(np.array(sl))
    return seq_list

def get_graph(
    graph_data_dir, 
    output_dir,
    train_circuits=None,
    test_circuits=None,
):
    # find all GMLs in graph data dir
    gmls = glob.glob(os.path.normpath(graph_data_dir+"/*.gml"))
    graph_dict = {} # graph name to graph data
    for i, gml_file in enumerate(gmls):
        filename = os.path.basename(gml_file)
        name = filename[filename.find('_') + 1 :filename.find('.gml')]
        pkl_file = os.path.normpath(os.path.join(output_dir, name + ".pkl"))
        #if dataset_names is not None and name not in dataset_names:
        #    continue

        if os.path.exists(pkl_file):
            print(f'{name}.pkl exists, loading from pickle file')
            with open(pkl_file, "rb") as handle:
                data = pickle.load(handle)
        else:
            print(f'{name}.pkl does not exist, generating from gml file')
            G = nx.read_gml(gml_file)
            data = convert.from_networkx(G)

            # convert str type into integer ids
            ids = torch.tensor([TYPES_TO_IDS[x] for x in data.type])
            data.type = ids
            with open(pkl_file, "wb") as handle:
                print(f'dumping {name}.pkl')
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if name == 'mctrl': name = 'mem_ctrl'
        graph_dict[name] = data  

    train_graph = {}
    test_graph = {}
    for name, graph in graph_dict.items():
        if train_circuits is not None and name not in train_circuits:
            pass
        else:
            train_graph[name] = graph
        if test_circuits is not None and name not in test_circuits:
            pass
        else:
            test_graph[name] = graph

    print(f"training circuits: {list(train_graph.keys())}")
    print(f"testing circuits: {list(test_graph.keys())}")
    return train_graph, test_graph

# Preprocess data (merge GML files and label/sequence data)
# Store processed PyTorch Geoemtric Data in individual pkl files for each graph
def preprocess_data(
    data_path, 
    debug=False,
):
    '''
    graph_data_dir: a str of the directory path that contains graph data in gml or pkl
    data_path: a str or a list of pkl files that contain labels and sequences
    '''

    # parse labels and synthesis sequences
    if not isinstance(data_path, list):
        data_path = [data_path]

    data_list = []
    for data_path in data_path:
        with open(data_path, 'rb') as f:
            data = pickle.load(f) # dict or a list of dict
            if not isinstance(data, list):
                data = list(data.values())
            if not isinstance(data, list):
                data = [list(d.values()) for d in data]
                data = [x for d in data for x in d]
            data_list += data

    data = get_label_seq(data_list)
    return data

# top level caller 
def generate_dataloaders(
    graph_data_dir, 
    train_data_path, 
    test_data_path=None,
    train_batch_size=256, 
    eval_batch_size=256,
    train_circuits=None,
    test_circuits=None,
    split_trainset=False,
    train_dataset_portion=None,
    p_val=0.2, 
    seed=100,
    debug=False,
):
    assert train_circuits is not None and test_circuits is not None

    train_graph, test_graph = get_graph(
        graph_data_dir, 
        output_dir=graph_data_dir, 
        train_circuits=train_circuits,
        test_circuits=test_circuits,
    )

    train_data = preprocess_data(
        train_data_path, 
        debug=debug,
    ) # data is a list of name, label and sequence tuples

    random.seed(seed)

    if split_trainset:
        print('running as non-dataset-assigned mode')
        print('randomly splitting the dataset into train/test sets')
        _train_data = []
        for d in train_data:
            name = d[0]
            if name in train_circuits:
                _train_data.append(d)
        train_data = _train_data
        random.shuffle(train_data)

        num_train_sets = int((1 - p_val) * len(train_data))
        train_dataset = train_data[:num_train_sets]
        test_dataset = train_data[num_train_sets:]

    else:
        print('running as dataset-assigned mode')
        if test_data_path is not None:
            test_data = preprocess_data(
                test_data_path, 
                debug=debug,
            ) # data is a list of name, label and sequence tuples
        else:
            test_data = train_data 

        train_dataset = []
        test_dataset = []
        for d in train_data:
            name = d[0]
            if name in train_circuits:
                train_dataset.append(d)

        for d in test_data:
            name = d[0]
            if name in test_circuits:
                test_dataset.append(d)
        random.shuffle(train_dataset)


    if train_dataset_portion is not None:
        train_dataset_len = int(len(train_dataset) * train_dataset_portion)
        print(f'trimming the trainset length: {len(train_dataset)} -> {train_dataset_len}')
        train_dataset = train_dataset[:train_dataset_len]
    train_dataset = GraphDataset(train_dataset, train_graph)
    test_dataset = GraphDataset(test_dataset, test_graph)

    print(f'length train set: {len(train_dataset)}')
    print(f'length test set: {len(test_dataset)}')

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
    ) 

    test_datalader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=True,
    )

    return train_dataloader, test_datalader


if __name__ == '__main__':
    output_dir = "../epfl_gmls"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    graphs = preprocess_data('../epfl_gmls', '../epflBmarks.pkl', output_dir)
