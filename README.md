## Prerequisite
Make sure that your gml files are in `epfl_gatelevel_gmls` and data files including sequences and labels (`run_epfl_arith.pkl`, `run_epfl_control.pkl`, and `run_epfl_extralegnths.pkl`) are in `datasets` under the root folder.

## Training
Below is the default run script:
```
python train.py --wandb_entity [wandb id] --wandb_project [wandb project name] --wandb_name [wandb name] \
  --epoch 50 --bs 32 --gcn_num_layers 2 --se_num_layers 4 --lr 1e-4 
```
You can drop `--wandb_entity` (and other wandb flags) if you don't want to log.
By default, the best model (one with the smallest delay MAPE) is saved under `checkpoints/{wandb_name}`; however, you can change the default checkpoint path by manually assigning `--save_path` flag. 
Later you can reload the checkpoint using `--load_path` flag, e.g., `--load checkpoints/log_folder/best_model.pt`.
`--gcn_num_layers` and `se_num_layers` assigns the number of GCN layers and LSTM layers each.
Please use `-h` to check more model/training hyperparameter flags.

The following running script will train and evaluate on all type of circuits, i.e., train data = test data = all circuits, which is not a desired behavior.
You can assign train/test datasets and circuit types following the examples below.

**1. Split train set into train and test dataset**
```
python train.py --wandb_entity [wandb id] --wandb_project [wandb project name] --wandb_name [wandb name] \
  --epoch 50 --bs 32 --gcn_num_layers 2 --se_num_layers 4 --lr 1e-4 --split
```
`--split` flag will split the train dataset into 8:2 randomly into a new train dataset and a test set.

**2. Train and test on only specific circuit types**
```
python train.py --wandb_entity [wandb id] --wandb_project [wandb project name] --wandb_name [wandb name] \
  --epoch 50 --bs 32 --gcn_num_layers 2 --se_num_layers 4 --lr 1e-4 \
  --train_circuits "multiplier,sin,sqrt,square" --test_circuits "adder,voter"
```
Use `--train_circuits` and `--test_circuits` flags to constraints circuit types for the train and test datasets respectively. If you don't assign `--test_circuits`, it will be initialized the same as `--test_circuits`. If you don't assign `--test_circuits`, it will be assigned all possible circuits.

**3. Manually assign train and test dataset paths**
```
python train.py --wandb_entity [wandb id] --wandb_project [wandb project name] --wandb_name [wandb name] \
  --epoch 50 --bs 32 --gcn_num_layers 2 --se_num_layers 4 --lr 1e-4 \
  --train_data_path '../datasets/run_epfl_arith.pkl' --test_data_path 'run_epfl_extralegnths.pkl'
```
This will use data from `run_epfl_arith.pkl` for training and `run_epfl_extralegnths.pkl` for testing. 
If you don't assign `--test_data_path`, it will be assigned same as `--train_data_path`.
If you don't assign `--train_data_path`, it will use `run_epfl_arith.pkl` and `run_epfl_control.pkl` together.
You can combine these flags with the ones in the step 1 and step 2 as well.

**4. Run using a portion of training datasets**

Add `--train_dataset_portion 0.1` to use e.g., 10% (0.1) of your training dataset. This is compatible with the other flags, i.e., any training dataset paths or circuit types. For the fine-tuning usage, make sure to load checkpoints properly using `--load_path [PATH]`.

**5. Eval only**

Use `--eval_only` flag to evaluate only.
