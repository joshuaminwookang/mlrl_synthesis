#!/usr/bin/python3
# 2021-22 Josh Kang (mkang@eecs.berkeley)
# 
import argparse
from multiprocessing import Pool
import os,random, subprocess, glob, json, re
import abc_scripts, synthesis
import numpy as np

# run a single synth run through sbatch on slurm
def gen_sbatch_scripts(**kwargs):
    filename = os.path.basename(kwargs['input_file'])
    ip = filename[filename.find('/') + 1 :filename.find('.v')]
    sbatch_script=os.path.join(kwargs['output_dir'], "{}_{}_{}.sh".format(ip, kwargs['synth_method'], kwargs['random_seq_len']))
    this_file = os.path.abspath( __file__ )
    script = f'''#!/bin/bash
# generated at by synthesis.py on batch mode
# Job name:
#SBATCH --job-name={ip}_{kwargs['random_seq_len']}
#
# Account:
#SBATCH --account=fc_bdmesh
#
# Partition:
#SBATCH --partition=savio2
#
# Quality of Service:
#SBATCH --qos=savio_normal
# Num Cores per Task
#SBATCH --cpus-per-task=24
#
# Wall clock limit:
#SBATCH --time=30:00:00
#
## Command(s) to run:
echo {this_file} --input_file {kwargs['input_file']} -o {kwargs['output_dir']} --synth_method {kwargs['synth_method']} -r {kwargs['random_seq_len']} --batch {kwargs['batch']} 
python {this_file} --input_file {kwargs['input_file']} -o {kwargs['output_dir']} --synth_method {kwargs['synth_method']} -r {kwargs['random_seq_len']} --batch {kwargs['batch']} 
'''
    with open(sbatch_script, "w") as f:
        f.write(script)

def prepare_batch_synthesis(**kwargs):
    params = kwargs
    output_dir = os.path.abspath(kwargs["output_dir"])
    params["output_dir"] = output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    verilogs = glob.glob(os.path.normpath(kwargs['input_dir']+"/*.v"))
    do_slurm = params['slurm']
    
    exp_sizes = define_experiment() if not kwargs['restricted'] else define_experiment_restricted() 
    #exp_sizes = define_experiment_restricted() 
    # create sbatch scripts for each Verilog input, for each random sequence length target
    for v in verilogs:
        params["input_file"] = os.path.abspath(v)
        for r in exp_sizes.keys():
            params["random_seq_len"] = r
            gen_sbatch_scripts(**params)

def define_experiment():
    num_ops  = abc_scripts.get_num_abc_ops()
    exp_sizes = {}
    exp_sizes[3] = abc_scripts.get_index_bounds_abc(3)
#    exp_sizes[4] = 60 # ~1K
    exp_sizes[5] = 300 # ~5K
    #exp_sizes[8] = 60 # ~1K
    exp_sizes[10] = 600 # ~10K
    #exp_sizes[12] = 60 # ~1K
    exp_sizes[15] = 600  # ~10K
    #exp_sizes[16] = 60 # ~1K
    #exp_sizes[20] = 60 # ~1K
    return exp_sizes

def define_experiment_restricted():
    num_ops  = abc_scripts.get_num_abc_ops(restricted=True)
    exp_sizes = {}
    exp_sizes[2] = 4
    exp_sizes[3] = 1 + num_ops + num_ops**2 + num_ops**3 + num_ops**4 # ~2800
    exp_sizes[5] = 400 # ~3K
    exp_sizes[10] = 700 # ~5K
    exp_sizes[15] = 1400  # ~10K
    exp_sizes[20] = 1400 # ~10K
    return exp_sizes

# generate random matrix with rows as random ABC transformation sequences
def gen_sequence_matrix(random_seq_len, samples_per_first_op, restricted):
    num_ops = abc_scripts.get_num_abc_ops(restricted=restricted)
    np.random.seed(1)
    if random_seq_len == 2:
        M = np.random.randint(num_ops, size=(samples_per_first_op, 10))
        return M
    if random_seq_len <= 3:
        seq_list = []
        for idx in range(samples_per_first_op):
            seq_list.append(abc_scripts.parse_index(idx, restricted)) # RESTRICTED
        M = np.array(seq_list, dtype = object)
        return M
    rands = np.random.randint(num_ops, size=(samples_per_first_op, random_seq_len-1))
    samples = samples_per_first_op
    while np.unique(rands, axis=0).shape[0] != samples_per_first_op:
        samples = samples*2
        rands = np.random.randint(num_ops, size=(samples, random_seq_len-1))
        rands = np.unique(rands, axis=0)[0:samples_per_first_op]

    headers = np.tile(np.repeat(np.arange(num_ops), samples_per_first_op), (1,1)).T
    M = np.concatenate((headers, np.tile(rands, (num_ops,1))), axis=1)
    
    assert(M.shape == (num_ops*samples_per_first_op, random_seq_len))
    return M

# run batch synthesis 
def run_batch_synthesis(**kwargs):
    params = kwargs
    exp_sizes = define_experiment() if not kwargs['restricted'] else define_experiment_restricted()
    random_seq_len = kwargs['random_seq_len']
    sequence_matrix = gen_sequence_matrix(random_seq_len, exp_sizes[random_seq_len], kwargs['restricted'])
    index = 0
    end_idx = sequence_matrix.shape[0]

    batch_size = params['batch']
    del params['input_dir']
    del params['batch']
    del params['slurm']
    
    while True:
        if end_idx <= index :
            break
        pool = Pool(processes=batch_size)
        for _ in range(batch_size):
            if end_idx <= index :
                continue
            params['index'] = index
            params['index_list'] = sequence_matrix[index].tolist() if random_seq_len > 3 else sequence_matrix[index]
            done = pool.apply_async(synthesis.run_synthesis, list(params.values()))
            index = index +1
        pool.close()
        pool.join()
    
def main():
    parser = argparse.ArgumentParser(
            description='Single run of Yosys-ABC + Vivado')
    # Single run parameters
    parser.add_argument('--input_file' , type=str, help='Input directory with Verilog files') # TODO Batch mode
    parser.add_argument('-o','--output_dir' , type=str, help='Output directory top level') # TODO Batch mode
    parser.add_argument('--index', type=int, help='Index of current sequence', default=-1)
    parser.add_argument('--synth_method', type=str, help='Target Mapping + synth method', default="fpga-abc")
    parser.add_argument('--clock_period', type=int, help='Target clock rate (in picoseconds) for syntehsis', default=5000)
    parser.add_argument('--grade', type=int, help='Target Xilinx FPGA device grade', default=1)
    parser.add_argument('--device', type=str, help='Target Xilinx FPGA device', default="xc7a200tffv1156-1")
    parser.add_argument('--run_analysis', type=bool, help='Run Vivado or Synopsis post-Synthesis analysis backend', default=True)
    parser.add_argument('-r','--random_seq_len', type=int, help='Length of random sequence to generate; 0 implies do not do random', default=0)
    parser.add_argument('--restricted', action='store_true', help='Synthesis transformations limited to 7 operations: b, rw, rwz, rs, rsz, rf, rfz')

    # Batch run parameters (to be popped before handed over to single synthesis run method 'synthesis.run_synthesis'
    parser.add_argument('-i','--input_dir' , type=str, help='Input directory with Verilog files') # TODO Batch mode
    parser.add_argument('-b', '--batch', type=int, help='Synthesis run batch size', default=0)
    parser.add_argument('-s', '--slurm', action='store_true', help='Run on slurm')

    args = parser.parse_args()
    kwargs = vars(args)
    if args.slurm :
        prepare_batch_synthesis(**kwargs)
    else:
        run_batch_synthesis(**kwargs)
    
if __name__ == '__main__':
    main()    
    
