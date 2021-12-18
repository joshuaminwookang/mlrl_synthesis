import random
import pickle
from tqdm import tqdm
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

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

def prepare_dataset(data):
    labels = []
    features = []
    sequences = []
    print('preparing dataset')
    for _, d in tqdm(data.items()):
        feature = {}
        label = {}
        label['Path_Delay'] = d['Path_Delay'] / 1e2
        label['Slice_LUTs'] = d['Slice_LUTs'] / 1e3
        for f in ['CI', 'CO', 'level', 'level_avg',
                  'cut', 'xor', 'xor_ratio', 'mux', 'mux_ratio', 'and', 'and_ratio',
                  'obj', 'power', 'LUT', 'fanin', 'fanout', 'mffc', 
                  #'fanin_max', 'fanin_avg', 'slack',
                  'fanout_max', 'fanout_avg', 'mffc_max', 'mffc_avg']:
            feature[f] = d[f]
        labels.append(label)
        features.append(feature)
        sequences.append(d['Sequence'])
    return features, labels, sequences

def preprocess_faninout(data):
    print('preprocessing fanin and fanout')
    for d in tqdm(data):
        fi = d.pop('fanin')
        fo = d.pop('fanout')
        d['fanin_all'] = fi['2'] # assume all fanin's are 2
        
        fo_threshold = 10
        fo_all, fo_large = 0, 0
        
        for k, v in fo.items():
            if k not in [str(x) for x in list(range(fo_threshold))]:
                fo_large += v
            fo_all += v
        
        for i in range(fo_threshold):
            d['fanout_ratio_%d' % i] = fo['%d' % i] / fo_all
        d['fanout_ratio_large'] = fo_large / fo_all
        d['fanout_all'] = fo_all
        
def preprocess_slack(data):
    print('preprocessing slack')
    for d in tqdm(data):
        if 'slack' in d:
            slack = d.pop('slack')
        '''
        slack_threshold = 21 # x10
        slack_all = slack['total_nodes']
        
        keys_skip = ['total_nodes']
        for i in range(slack_threshold):
            lower = 10 * i
            upper = lower + 10
            key = f"{lower}_{upper}"
            name = f"slack_ratio_{key}"
            keys_skip.append(key)
            if key not in slack:
                d[name] = 0.
            else:
                d[name] = slack[key] / slack_all
        
        slack_large = 0
        for k, v in slack.items():
            if k not in keys_skip:
                slack_large += v
        d['slack_ratio_large'] = slack_large / slack_all
        d['slack_all'] = slack_all
        '''
                
def preprocess_mffc(data):
    print('preprocess mffc')
    for d in tqdm(data):
        mffc = d.pop('mffc')
        mffc_threshold = 9
        mffc_all, mffc_large = 0, 0
        
        for k, v in mffc.items():
            if k not in [str(x) for x in list(range(mffc_threshold))]:
                mffc_large += v
            mffc_all += v
        
        for i in range(mffc_threshold):
            d['mffc_ratio_%d' % i] = 0. if mffc_all == 0 else mffc['%d' % i] / mffc_all
        d['mffc_ratio_large'] = 0. if mffc_all == 0 else mffc_large / mffc_all
        d['mffc_all'] = mffc_all
        
def preprocess_LUT(data):
    print('preprocess LUT')
    for d in tqdm(data):
        lut = d.pop('LUT')
        for i in [2, 3, 4, 5, 6]:
            key = f"{i}_LUT_ratio"
            lut_ratio = lut[key] if key in lut else 0.
            d[f"LUT_ratio_{i}"] = lut_ratio / 100 # convert percentage
        d["LUT_level"] = lut["level"]
        d["LUT_level_avg"] = lut["level_avg"]
        d["LUT_size_avg"] = lut["size_avg"]
        d["LUT_total"] = lut["total"]

def preprocess_sequence(sequences):
    # convert the string representation into a list of tokens
    print('preprocessing sequences')
    seq_list = []
    for seq in tqdm(sequences):
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

def preprocess_data(data_path):
    if not isinstance(data_path, list):
        data_path = [data_path]
    features, labels, sequences = [], [], []

    for _data_path in data_path:
        with open(_data_path, 'rb') as f:
            data = pickle.load(f)

        _features, _labels, _sequences = prepare_dataset(data)
        features += _features
        labels += _labels
        sequences += _sequences

    preprocess_mffc(features)
    preprocess_faninout(features)
    # preprocess_slack(features)
    preprocess_LUT(features)

    print(features[0])
    features_flatted = flatten_all(features)
    features_normalized = normalize(features_flatted)

    sequences_list = preprocess_sequence(sequences)
    labels_flattened = flatten_all(labels)

    return features_normalized, np.array(sequences_list), labels_flattened

class CustomDataset(Dataset):
    def __init__(self, features=None, sequences=None, labels=None, data_path=None):
        if data_path is not None:
            self.features, self.sequences, self.labels = preprocess_data(data_path)
        else:
            assert features is not None 
            assert sequences is not None 
            assert labels is not None 
            self.features, self.sequences, self.labels = features, sequences, labels
        self.input_dim = self.features.shape[-1]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx]).to(torch.float)
        sequence = torch.tensor(self.sequences[idx]).to(torch.long)
        label = torch.tensor(self.labels[idx]).to(torch.float)
        #return {'feature': feature, 'sequence': sequence, 'label': label}
        return feature, sequence, label

def generate_datasets(data_path, p_val=0.2):
    features, sequences, labels = preprocess_data(data_path)
    num_training_sets = int((1 - p_val) * len(features))
    indices_all = np.array(list(range(len(features))))
    random.seed(100)
    random.shuffle(indices_all)
    print(indices_all)
    train_indices = indices_all[:num_training_sets]
    val_indices = indices_all[num_training_sets:]

    train_dataset = CustomDataset(
        features[train_indices], 
        sequences[train_indices],
        labels[train_indices],
    )
    valid_dataset = CustomDataset(
        features[val_indices], 
        sequences[val_indices],
        labels[val_indices],
    )
    '''
    print(train_indices)
    print(val_indices)
    print(len(features), len(train_indices), len(val_indices))
    print(len(train_dataset))
    print(len(valid_dataset))
    '''

    return train_dataset, valid_dataset

def pad_collate(batch):
    features, sequences, labels, = zip(*batch)
    sequences_len = [len(s) for s in sequences]
    features_pad = pad_sequence(features, batch_first=True, padding_value=0)
    sequences_pad = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels_pad = pad_sequence(labels, batch_first=True, padding_value=0) 
    return features_pad, sequences_pad, sequences_len, labels_pad


if __name__ == '__main__': 
    features, sequences, labels = preprocess_data('../../epfl_arithmetic.pkl')

