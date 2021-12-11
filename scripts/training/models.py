from tqdm import tqdm 

import argparse

import random
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
from scipy import stats

from preprocessing import SEQ_TO_TOKEN, TOKEN_TO_SEQ, preprocess_data

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

#def train(model, 
if __name__ == '__main__': 
    parser = argparse.ArgumentParser(prog="Regression")
    parser.add_argument("--fe_hidden_dim", type=int, default=128, help="Feature embedding hidden dim")
    parser.add_argument("--fe_output_dim", type=int, default=128, help="Feature embedding output dim")
    parser.add_argument("--se_input_dim", type=int, default=128, help="Sequence embedding input dim")
    parser.add_argument("--se_num_layers", type=int, default=2, help="Sequence embedding num lstm layers")

    args = parser.parse_args()

    data_path = ['epfl_arithmetic.pkl', 'epfl_control.pkl', 'vtr_select.pkl']
    #valid_path = ['vtr_testset_rand.pkl']
    train_dataset, valid_dataset = generate_datasets(data_path)
    #dataset = CustomDataset(data_path=data_path)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, collate_fn=pad_collate)

    #valid_dataset = CustomDataset(data_path=valid_path)
    valid_dataloader = DataLoader(valid_dataset, batch_size=64, num_workers=0, collate_fn=pad_collate)

    input_dim = train_dataset.input_dim
    model = E2ERegression(
        input_dim,
        args.fe_hidden_dim,
        args.fe_output_dim,
        args.se_input_dim,
        args.se_num_layers
    )
    model = model.cuda()

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    #import wandb
    #name = f"fh{args.fe_hidden_dim}_fo{args.fe_output_dim}_si{args.se_input_dim}_sl{args.se_num_layers}"
    #wandb.init(project="synthesis", entity="sehoonkim", name=name)

    for epoch in range(50):
        loss_all = 0
        cnt = 0

        valid_loss_all = 0
        valid_cnt = 0

        for i, batch in tqdm(enumerate(train_dataloader)):
            features, sequences, sequences_len, labels = batch
            features = features.cuda()
            sequences = sequences.cuda()
            labels = labels.cuda()
            x = model(features, sequences, sequences_len)

            loss = loss_fn(x, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cnt += 1
            loss_all += loss

        model.eval()

        x_delay_all = []
        x_area_all = []
        labels_delay_all = []
        labels_area_all = []

        for i, batch in tqdm(enumerate(valid_dataloader)):
            features, sequences, sequences_len, labels = batch
            features = features.cuda()
            sequences = sequences.cuda()
            labels = labels.cuda()
            x = model(features, sequences, sequences_len)

            loss = loss_fn(x, labels)

            x_delay = x[:, 0].cpu().tolist()
            x_area = x[:, 1].cpu().tolist()
            labels_delay = labels[:, 0].cpu().tolist()
            labels_area = labels[:, 1].cpu().tolist()
            x_delay_all += x_delay
            x_area_all += x_area
            labels_delay_all += labels_delay
            labels_area_all += labels_area

            valid_cnt += 1
            valid_loss_all += loss

        model.train()
        corr_delay = stats.spearmanr(x_delay_all, labels_delay_all).correlation
        corr_area = stats.spearmanr(x_area_all, labels_area_all).correlation

        #wandb.log({"loss": loss_all / cnt, "val_loss": valid_loss_all / valid_cnt,
        #    "corr_delay": corr_delay, "corr_area": corr_area})
        print(f"epoch {epoch}, loss {loss_all / cnt}, val_loss {valid_loss_all / valid_cnt}")
        print(f"    corr_delay {corr_delay}, corr_area {corr_area}")
