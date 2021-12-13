## Training

Get prepared for the training datasets in the shared drive.
`epfl_arithmetic.pkl`, `epfl_control.pkl` and `vtr_select.pkl` are currently available datasets.

Run the following command:
```
python train.py --dataset_path {DATASET_PATHS} --dump_path {DUMP_PATH}
```

* `DATASET_PATHS` is the dataset paths separeted with commas(,), i.e., `epfl_arithmetic.pkl,epfl_control.pkl,vtr_select.pkl`.
* `DUMP_PATH` is the directory path to dump checkpoints. The model checkpoint with the highest `corr_delay` will be stored in the path.
* You can also specify the model hyperparameters (i.e., dimensions). Otherwise, the model is initialized with the default values. Please use `-h` argument for detail. 


## Evaluation

Get prepared for the validation dataset and model checkpoint.
You can either train the model and use your own checkpoint, or you can download `fh64_fo256_si64_sl4.ckpt` from the shared drive.
This is the checkpoint file with the default model hyperparameter (i.e., dimension) setting.

Run the following command:
```
python eval.py --dataset_path {DATASET_PATHS} --load_path {LOAD_PATH}
```

* `DATASET_PATHS` is same as above. 
* `LOAD_PATH` is the path to the checkpoint, e.g., `fh64_fo256_si64_sl4.ckpt`. 
* Make sure that all the dimensions of the model match with the checkpoint file. You can set the dimension of the model with additional arguments. Please use `-h` argument for detail. 
