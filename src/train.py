import os
import sys
import argparse
from tqdm import tqdm 

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from scipy import stats
from models import E2ERegression
from dataset_graph import generate_dataloaders

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(prog="Regression")
    parser.add_argument("--bs", type=int, default=64, help="Batch size")
    parser.add_argument("--feature_dim", type=int, default=64, help="Feature graph embedding dim")
    parser.add_argument("--fe_hidden_dim", type=int, default=64, help="Feature embedding hidden dim")
    parser.add_argument("--fe_output_dim", type=int, default=256, help="Feature embedding output dim")
    parser.add_argument("--se_input_dim", type=int, default=64, help="Sequence embedding input dim")
    parser.add_argument("--se_num_layers", type=int, default=4, help="Sequence embedding num lstm layers")
    parser.add_argument("--dump_path", type=str, default=None, help="Save path for the best model")
    parser.add_argument("--label_seq_data_path", type=str, default='../epflBmarks.pkl', help="Dataset paths, separated with comma(,)")
    parser.add_argument("--graph_data_dir", type=str, default='../epfl_gmls', help="graph data dir path")
    parser.add_argument("--debug", action='store_true', default=False, help="debugging mode")

    args = parser.parse_args()

    label_seq_data_path = args.label_seq_data_path.split(',')

    train_dataloader, valid_dataloader = generate_dataloaders(
        graph_data_dir=args.graph_data_dir,
        label_seq_data_path=args.label_seq_data_path,
        train_batch_size=args.bs,
        eval_batch_size=args.bs,
        debug=args.debug,
    )

    sys.exit() # TODO fill out the remaining training pipeline below

    input_dim = train_dataset.input_dim
    model = E2ERegression(
        input_dim,
        args.fe_hidden_dim,
        args.fe_output_dim,
        args.se_input_dim,
        args.se_num_layers
    )

    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(0))
        print("Using GPU id {}".format(0))
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")

    # model = model.cuda()
    model = model.to(device=device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    name = f"fh{args.fe_hidden_dim}_fo{args.fe_output_dim}_si{args.se_input_dim}_sl{args.se_num_layers}"
    #import wandb
    #wandb.init(project="synthesis", entity="sehoonkim", name=name)

    best_metric = 0

    for epoch in range(50):
        loss_all = 0
        cnt = 0

        valid_loss_all = 0
        valid_cnt = 0

        for i, batch in tqdm(enumerate(train_dataloader)):
            features, sequences, sequences_len, labels = batch
            features = features.to(device=device)
            sequences = sequences.to(device=device)
            labels = labels.to(device=device)
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
            features = features.to(device=device)
            sequences = sequences.to(device=device)
            labels = labels.to(device=device)
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

        if args.dump_path is not None and corr_delay > best_metric:
            print('Best Model, saving the model checkpoint')
            torch.save(model.state_dict(), os.path.join(args.dump_path, f"{name}.ckpt"))
            best_metric = corr_delay

