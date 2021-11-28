from tqdm import tqdm 

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from preprocessing import SEQ_TO_TOKEN, TOKEN_TO_SEQ, preprocess_data

class FeatureEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.dense1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x = self.dense1(x)
        #x = self.bn1(x)
        x = self.relu(x)
        x = self.dense2(x)
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
    
    def forward(self, feature, sequence, sequence_len):
        x = self.fe(feature)
        x = torch.stack([x] * self.num_lstm_layers)
        x, x_len = self.se(sequence, sequence_len, x)
        x = torch.stack([x[i, l - 1, :] for i, l in enumerate(x_len)])
        x = self.dense(x)
        return x

class CustomDataset(Dataset):
    def __init__(self, data_path):
        self.features, self.sequences, self.labels = preprocess_data(data_path)
        self.input_dim = self.features.shape[-1]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx]).to(torch.float)
        sequence = torch.tensor(self.sequences[idx]).to(torch.long)
        label = torch.tensor(self.labels[idx]).to(torch.float)
        #return {'feature': feature, 'sequence': sequence, 'label': label}
        return feature, sequence, label

def pad_collate(batch):
    features, sequences, labels, = zip(*batch)
    sequences_len = [len(s) for s in sequences]
    features_pad = pad_sequence(features, batch_first=True, padding_value=0)
    sequences_pad = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels_pad = pad_sequence(labels, batch_first=True, padding_value=0) 
    return features_pad, sequences_pad, sequences_len, labels_pad

#def train(model, 
if __name__ == '__main__': 
    data_path = '../../../epfl_arithmetic.pkl'
    dataset = CustomDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0, collate_fn=pad_collate)

    input_dim = dataset.input_dim
    model = E2ERegression(input_dim, 128, 128, 128, 2)
    model = model.cuda()

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(50):
        loss_all = 0
        cnt = 0
        for i, batch in tqdm(enumerate(dataloader)):
            features, sequences, sequences_len, labels = batch
            features = features.cuda()
            sequences = sequences.cuda()
            labels = labels.cuda()
            x = model(features, sequences, sequences_len)

            loss = loss_fn(x, labels)
            #print(x)
            #print(labels)
            #print(loss)
            #print(AA)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cnt += 1
            loss_all += loss
        print(f"epoch {epoch}, loss {loss_all / cnt}")


    '''
    features, sequences, labels = preprocess_data(data_path)
    input_dim = features.shape[-1]

    model = E2ERegression(input_dim, 128, 128, 128, 2)
    feature = torch.tensor(features[0]).to(torch.float)
    feature = torch.stack([feature + 1, feature])
    sequence = torch.tensor(sequences[1]).to(torch.long)
    sequence = torch.stack([sequence + 1, sequence])
    print(feature.shape, sequence.shape)
    x = model(feature, sequence, [2, 1])
    print(x.shape)
    '''
