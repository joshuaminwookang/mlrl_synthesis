import os
import sys
import argparse
from tqdm import tqdm 

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import numpy as np

from scipy import stats
from models import E2ERegression, GCN
from dataset_graph import generate_dataloaders

FULL_DATASET = [
    'adder', 'arbiter', 'bar', 'cavlc', 'ctrl', 'dec', 
    'div', 'i2c', 'int2float', 'log2', 'max', 'mem_ctrl',
    'multiplier', 'priority', 'router', 'sin', 'sqrt', 
    'square', 'voter',
]

FULL_DATASET_STR = ",".join(FULL_DATASET)

SAVE_PATH_DEFAULT = 'checkpoints'

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def parse_batch(data, device):
    inputs = data['graph']
    sequence = data['sequence']
    label = data['label']

    inputs = inputs.to(device=device)
    sequence_len = [len(s) for s in sequence]
    sequence = pad_sequence([torch.tensor(s) for s in sequence], batch_first=True, padding_value=0)
    sequence = sequence.to(device=device)

    label = torch.tensor(label)
    label = label.to(device=device)
    return inputs, sequence, sequence_len, label

def loss_fn(outputs, label):
    loss = ((outputs - label) / label) ** 2
    return loss.mean()

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(prog="Regression")
    parser.add_argument("--bs", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epoch", type=int, default=50, help="Num training epochs")
    parser.add_argument("--train_circuits", type=str, 
        default=FULL_DATASET_STR, help="Training circuit names, comma separated")
    parser.add_argument("--test_circuits", type=str, default=None, help="Testing circuit names, comma separated")
    parser.add_argument("--split_trainset", action='store_true', 
        default=False, help="Split train dataset to generate validation dataset")
    parser.add_argument("--train_dataset_portion", type=float, 
        default=None, help="The portion of dataset to use for training")
    parser.add_argument("--gcn_hidden_dim", type=int, default=64, help="GCN hidden dim")
    parser.add_argument("--gcn_output_dim", type=int, default=64, help="GCN output dim")
    parser.add_argument("--gcn_num_layers", type=int, default=2, help="GCN num_layers")
    parser.add_argument("--se_input_dim", type=int, default=64, help="Sequence embedding input dim")
    parser.add_argument("--se_num_layers", type=int, default=4, help="Sequence embedding num lstm layers")
    parser.add_argument("--parallel", action='store_true', default=False, help="run GCN and RNN in parallel")
    parser.add_argument("--dump_path", type=str, default=None, help="Save path for the best model")
    parser.add_argument("--train_data_path", type=str, 
        default='../datasets/run_restricted_epfl_arith.pkl,../datasets/run_restricted_epfl_control.pkl', 
        help="Dataset paths, comma separated")
    parser.add_argument("--test_data_path", type=str, default=None, help="Dataset paths, comma separated")
    parser.add_argument("--graph_data_dir", type=str, default='../epfl_gatelevel_gmls', help="graph data dir path")
    parser.add_argument("--raw_graph", action='store_true', default=False, help="whether to use raw graph data")
    parser.add_argument("--debug", action='store_true', default=False, help="debugging mode")
    parser.add_argument("--eval_only", action='store_true', default=False, help="run eval only")
    parser.add_argument("--wandb_entity", type=str, default=None, help="wandb entity (id) name")
    parser.add_argument("--wandb_project", type=str, default=None, help="wandb project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="wandb project name")
    parser.add_argument("--save_path", type=str, default=None, help="model checkpoint path")
    parser.add_argument("--load_path", type=str, default=None, help="model checkpoint load path")

    args = parser.parse_args()

    if args.save_path is not None:
        save_path = args.save_path
    elif args.wandb_name is not None:
        save_path = os.path.join(SAVE_PATH_DEFAULT, args.wandb_name)
        print(f'checkpointing into {save_path}')
        if not os.path.exists(save_path):
            os.makedirs(save_path) 
    else:
        print('No checkpointing')
        save_path = None

    train_data_path = args.train_data_path.split(',')
    if args.test_data_path is not None:
        test_data_path = args.test_data_path.split(',')
    else:
        test_data_path = None

    train_circuits_names = args.train_circuits.split(',')
    if args.test_circuits is not None:
        test_circuits_names = args.test_circuits.split(',')
    else:
        test_circuits_names = train_circuits_names

    train_dataloader, valid_dataloader = generate_dataloaders(
        graph_data_dir=args.graph_data_dir,
        train_data_path=train_data_path,
        test_data_path=test_data_path,
        train_batch_size=args.bs,
        eval_batch_size=args.bs,
        train_circuits=train_circuits_names,
        test_circuits=test_circuits_names,
        split_trainset=args.split_trainset,
        train_dataset_portion=args.train_dataset_portion,
        debug=args.debug,
    )

    model = E2ERegression(
        gcn_hidden_dim=args.gcn_hidden_dim,
        gcn_output_dim=args.gcn_output_dim,
        gcn_num_layers=args.gcn_num_layers,
        se_input_dim=args.se_input_dim,
        se_num_lstm_layers=args.se_num_layers,
        raw_graph=args.raw_graph,
        mode='parallel' if args.parallel else 'sequential',
    )

    if args.load_path is not None:
        print('loading model...')
        model.load_state_dict(torch.load(args.load_path))

    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(0))
        print("Using GPU id {}".format(0))
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")

    # model = model.cuda()
    model = model.to(device=device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epoch
    )

    do_wandb = args.wandb_entity is not None and not args.debug

    if do_wandb:
        import wandb
        assert args.wandb_project is not None
        wandb.init(
            project=args.wandb_project, 
            entity=args.wandb_entity, 
            name=args.wandb_name,
            id=args.wandb_name,
            resume='allow',
        )
        wandb.config.update(args)

    best_metric = 1e4

    for epoch in range(args.epoch):
        if not args.eval_only:
            loss_all = 0
            cnt = 0

            for i, data in tqdm(enumerate(train_dataloader)):
                inputs, sequence, sequence_len, label = parse_batch(data, device)
                outputs = model(inputs, sequence, sequence_len)

                loss = loss_fn(outputs, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                cnt += 1
                loss_all += loss

            scheduler.step()

        model.eval()

        valid_loss_all = 0
        valid_cnt = 0

        x_delay_all = []
        x_area_all = []
        label_delay_all = []
        label_area_all = []

        with torch.no_grad():
            for i, data in tqdm(enumerate(valid_dataloader)):
                inputs, sequence, sequence_len, label = parse_batch(data, device)
                outputs = model(inputs, sequence, sequence_len)

                loss = loss_fn(outputs, label)

                x_delay = outputs[:, 0].cpu().tolist()
                x_area = outputs[:, 1].cpu().tolist()
                label_delay = label[:, 0].cpu().tolist()
                label_area = label[:, 1].cpu().tolist()
                x_delay_all += x_delay
                x_area_all += x_area
                label_delay_all += label_delay
                label_area_all += label_area

                valid_cnt += 1
                valid_loss_all += loss

        model.train()
        corr_delay = stats.spearmanr(x_delay_all, label_delay_all).correlation
        corr_area = stats.spearmanr(x_area_all, label_area_all).correlation

        mape_delay = mean_absolute_percentage_error(label_delay_all, x_delay_all)
        mape_area = mean_absolute_percentage_error(label_area_all, x_area_all)

        if do_wandb:
            wandb.log({"loss": loss_all / cnt, "val_loss": valid_loss_all / valid_cnt,
                "corr_delay": corr_delay, "corr_area": corr_area,
                "mape_delay": mape_delay, "mape_area": mape_area,
            })

        if not args.eval_only:
            print(f"epoch {epoch}, loss {loss_all / cnt}, val_loss {valid_loss_all / valid_cnt}")
        else:
            print(f"val_loss {valid_loss_all / valid_cnt}")
        print(f"    corr_delay {corr_delay}, corr_area {corr_area}")
        print(f"    mape_delay {mape_delay}, mape_area {mape_area}")

        if args.eval_only:
            break

        if mape_delay < best_metric and save_path is not None:
            print('Best Model, saving the model checkpoint')
            torch.save(model.state_dict(), os.path.join(save_path, f"best.ckpt"))
            best_metric = mape_delay
