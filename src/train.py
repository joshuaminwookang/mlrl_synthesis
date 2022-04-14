import os
import sys
import argparse
from tqdm import tqdm 

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from scipy import stats
from models import E2ERegression, GCN
from dataset_graph import generate_dataloaders

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(prog="Regression")
    parser.add_argument("--bs", type=int, default=64, help="Batch size")
    #parser.add_argument("--feature_dim", type=int, default=64, help="Feature graph embedding dim")
    #parser.add_argument("--fe_hidden_dim", type=int, default=64, help="Feature embedding hidden dim")
    #parser.add_argument("--fe_output_dim", type=int, default=256, help="Feature embedding output dim")
    parser.add_argument("--gcn_hidden_dim", type=int, default=64, help="GCN hidden dim")
    parser.add_argument("--gcn_output_dim", type=int, default=64, help="GCN output dim")
    parser.add_argument("--se_input_dim", type=int, default=64, help="Sequence embedding input dim")
    parser.add_argument("--se_num_layers", type=int, default=4, help="Sequence embedding num lstm layers")
    parser.add_argument("--dump_path", type=str, default=None, help="Save path for the best model")
    parser.add_argument("--label_seq_data_path", type=str, default='../epflBmarks.pkl', help="Dataset paths, separated with comma(,)")
    parser.add_argument("--graph_data_dir", type=str, default='../epfl_gmls', help="graph data dir path")
    parser.add_argument("--debug", action='store_true', default=False, help="debugging mode")
    parser.add_argument("--wandb_entity", type=str, default=None, help="wandb entity (id) name")
    parser.add_argument("--wandb_project", type=str, default=None, help="wandb project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="wandb project name")

    args = parser.parse_args()

    label_seq_data_path = args.label_seq_data_path.split(',')

    train_dataloader, valid_dataloader = generate_dataloaders(
        graph_data_dir=args.graph_data_dir,
        label_seq_data_path=args.label_seq_data_path,
        train_batch_size=args.bs,
        eval_batch_size=args.bs,
        debug=args.debug,
    )

    #model = GCN(args.gcn_hidden_dim, args.gcn_output_dim)
    model = E2ERegression(
        args.gcn_hidden_dim,
        args.gcn_output_dim,
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

    if args.wandb_entity is not None:
        import wandb
        assert args.wandb_project is not None
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_name)
        wandb.config.update(args)

    #name = f"fh{args.fe_hidden_dim}_fo{args.fe_output_dim}_si{args.se_input_dim}_sl{args.se_num_layers}" #TODO

    best_metric = 0

    for epoch in range(50):
        loss_all = 0
        cnt = 0

        valid_loss_all = 0
        valid_cnt = 0

        for i, inputs in tqdm(enumerate(train_dataloader)):
            inputs = inputs.to(device=device)
            sequence = inputs.sequence
            sequence_len = [len(s) for s in sequence]
            sequence = pad_sequence([torch.tensor(s) for s in sequence], batch_first=True, padding_value=0)
            sequence = sequence.to(device=device)

            labels = torch.tensor(inputs.labels)
            labels = labels.to(device=device)
            outputs = model(inputs, sequence, sequence_len)

            loss = loss_fn(outputs, labels)
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

        with torch.no_grad():
            for i, inputs in tqdm(enumerate(valid_dataloader)):
                inputs = inputs.to(device=device)
                sequence = inputs.sequence
                sequence_len = [len(s) for s in sequence]
                sequence = pad_sequence([torch.tensor(s) for s in sequence], batch_first=True, padding_value=0)
                sequence = sequence.to(device=device)

                labels = torch.tensor(inputs.labels)
                labels = labels.to(device=device)
                outputs = model(inputs, sequence, sequence_len)

                loss = loss_fn(outputs, labels)

                x_delay = outputs[:, 0].cpu().tolist()
                x_area = outputs[:, 1].cpu().tolist()
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


        if args.wandb_entity is not None:
            wandb.log({"loss": loss_all / cnt, "val_loss": valid_loss_all / valid_cnt,
                "corr_delay": corr_delay, "corr_area": corr_area})

        print(f"epoch {epoch}, loss {loss_all / cnt}, val_loss {valid_loss_all / valid_cnt}")
        print(f"    corr_delay {corr_delay}, corr_area {corr_area}")

        if args.dump_path is not None and corr_delay > best_metric:
            print('Best Model, saving the model checkpoint')
            torch.save(model.state_dict(), os.path.join(args.dump_path, f"{name}.ckpt"))
            best_metric = corr_delay

