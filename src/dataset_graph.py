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
               "balance",  "dc2", "dch -f",
               "if -K 6 -x", "if -K 6 -g",
               "resub -K 8 -N 1",  "resub -K 8 -N 2",
               "resub -K 12 -N 1",  "resub -K 12 -N 2",
               "resub -K 8 -N 1 -z", "resub -K 8 -N 2 -z",
               "resub -K 12 -N 1 -z",  "resub -K 12 -N 2 -z"]
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
            #collate_fn=lambda data_list: Batch.from_data_list(
            #    data_list, follow_batch),
            collate_fn=graph_dataset_collate_fn,
            **kwargs)


def get_label_seq(data, target_names=None):
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
        seq = seq.split(';')[0: -2] # remove the redundant parts
        sl = []
        for s in seq:
            sl.append(SEQ_TO_TOKEN[s])
        seq_list.append(np.array(sl))
    return seq_list


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
def preprocess_data(graph_data_dir, label_seq_data_path, output_dir, dataset_names=None, debug=False):
    '''
    graph_data_dir: a str of the directory path that contains graph data in gml or pkl
    label_seq_data_path: a str or a list of pkl files that contain labels and sequences
    '''

    # parse labels and synthesis sequences
    if not isinstance(label_seq_data_path, list):
        label_seq_data_path = [label_seq_data_path]

    data_list = []
    for data_path in label_seq_data_path:
        with open(data_path, 'rb') as f:
            data = pickle.load(f) # dict or a list of dict
            if not isinstance(data, list):
                data = list(data.values())
            if not isinstance(data, list):
                data = [list(d.values()) for d in data]
                data = [x for d in data for x in d]
            data_list += data

    label_seq = get_label_seq(data_list, target_names=dataset_names)

    # find all GMLs in graph data dir
    gmls = glob.glob(os.path.normpath(graph_data_dir+"/*.gml"))

    graph_dict = {} # graph name to graph data
    for i, gml_file in enumerate(gmls):
        filename = os.path.basename(gml_file)
        name = filename[filename.find('_') + 1 :filename.find('.gml')]
        pkl_file = os.path.normpath(os.path.join(output_dir, name + ".pkl"))
        if dataset_names is not None and name not in dataset_names:
            continue

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

        graph_dict[name] = data  

    return label_seq, graph_dict

# top level caller 
def generate_dataloaders(
    graph_data_dir, 
    label_seq_data_path, 
    train_batch_size=256, 
    eval_batch_size=256,
    train_dataset_names=None,
    test_dataset_names=None,
    p_val=0.2, 
    seed=100,
    debug=False,
):
    # features, sequences, labels = preprocess_data(data_path, feature_dim=feature_dim)
    dataset_assigned = train_dataset_names is not None
    if dataset_assigned:
        assert test_dataset_names is not None
        dataset_names = train_dataset_names + test_dataset_names
    data, graph_dict = preprocess_data(
        graph_data_dir, 
        label_seq_data_path, 
        output_dir=graph_data_dir, 
        dataset_names=dataset_names if dataset_assigned else None,
        debug=debug,
    ) # data is a list of name, label and sequence tuples

    random.seed(seed)

    if not dataset_assigned:
        print('running as non-dataset-assigned mode')
        print('randomly splitting the dataset into train/test sets')
        random.shuffle(data)

        num_train_sets = int((1 - p_val) * len(graphs))
        train_dataset = data[:num_train_sets]
        valid_dataset = data[num_train_sets:]

    else:
        print('running as dataset-assigned mode')
        train_dataset = []
        valid_dataset = []
        for d in data:
            name = d[0]
            if name in train_dataset_names:
                train_dataset.append(d)
            elif name in test_dataset_names:
                valid_dataset.append(d)
            else:
                raise ValueError
        random.shuffle(train_dataset)

    train_dataset = GraphDataset(train_dataset, graph_dict)
    valid_dataset = GraphDataset(valid_dataset, graph_dict)

    print(f'length train set: len(train_dataset)')
    print(f'length valid set: len(valid_dataset)')

    train_dataloader = DataLoader(
        train_dataset,
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
