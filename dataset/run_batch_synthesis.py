#!/usr/bin/python3
# 2021-22 Josh Kang (mkang@eecs.berkeley)
# 
import argparse
from multiprocessing import Pool
import os,random, subprocess, glob, json, re
import abc_scripts, synthesis

# run a single synth run through sbatch on slurm
def gen_sbatch_scripts(input_file=None, output_dir=None, index=0, synth_method='fpga-abc9', clock_period=5000, grade=1, device="",\
                  run_analysis=True, random_seq_len=0):
    output_dir = os.path.abspath(output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    filename = os.path.basename(input_file)
    ip = filename[filename.find('_') + 1 :filename.find('.v')]
    sbatch_script=os.path.join(output_dir, "{}_{}_{}.sh".format(synth_method, ip, index))
    this_file = os.path.abspath( __file__ )

    script = f'''#!/bin/bash
# generated at by synthesis.py on batch mode
# Job name:
#SBATCH --job-name={synth_method}_{ip}_{index}
#
# Account:
#SBATCH --account=fc_bdmesh
#
# Partition:
#SBATCH --partition=savio3
#
# Quality of Service:
#SBATCH --qos=savio_normal
# Num Cores per Task
#SBATCH --cpus-per-task=1
#
# Wall clock limit:
#SBATCH --time=01:00:00
#
## Command(s) to run:
echo {this_file} -i {input_file} -o {output_dir} --synth_method {synth_method}  --index {index} -r {random_seq_len}
python {this_file} -i {input_file} -o {output_dir}  --synth_method {synth_method}  --index {index} -r {random_seq_len}
'''
    with open(sbatch_script, "w") as f:
        f.write(script)

def run_batch_synthesis(**kwargs):
    # assume that input path is a directory of all input Verilogs
    params = kwargs
    batch_size = params['batch']
    max_len = params['max_length']
    do_slurm = params['slurm']
    del params['batch']
    del params['max_length']
    del params['slurm']
    
    verilogs = glob.glob(os.path.normpath(kwargs['input_file']+"/*.v"))
    print(verilogs)
    
    if params['synth_method'] == "fpga-abc9":
        end_idx  = abc_scripts.get_index_bounds_abc9(max_len)
    else:
        end_idx  = abc_scripts.get_index_bounds_abc(max_len)

    # spawn parallel jobs(processes)
    for v in verilogs:
        index = 0
        while True:
            if end_idx <= index :
                break
            pool = Pool(processes=batch_size)
            for i in range(batch_size):
                if end_idx <= index :
                    continue
                params['input_file'] = v
                params['index'] = index
                done = pool.apply_async(run_synthesis, list(params.values()))
                index = index +1
            pool.close()
            pool.join()
                
    # if do_slurm:
    #     for v in verilogs:
    #         index = 0
    #         while True:
    #             if end_idx <= index :
    #                 break
    #             # params['input_file'] = os.path.abspath(v)
    #             # params['index'] = index
    #             # done = run_sbatch(**params)
    #             # index = index+1
    #             pool = Pool(processes=batch_size)
    #             for i in range(batch_size):
    #                 if end_idx <= index :
    #                     continue
    #                 params['input_file'] = os.path.abspath(v)
    #                 params['index'] = index
    #                 done = pool.apply_async(run_sbatch, list(params.values()))
    #                 index = index +1
    #             pool.close()
    #             pool.join()
    # else:
    #     for v in verilogs:
    #         index = 0
    #         while True:
    #             if end_idx <= index :
    #                 break
    #             pool = Pool(processes=batch_size)
    #             for i in range(batch_size):
    #                 if end_idx <= index :
    #                     continue
    #                 params['input_file'] = v
    #                 params['index'] = index
    #                 done = pool.apply_async(run_synthesis, list(params.values()))
    #                 index = index +1
    #             pool.close()
    #             pool.join()
        
def test(**kwargs):
    max_len = kwargs['max_length']
    if kwargs['synth_method'] == "fpga-abc9":
        end_idx  = abc_scripts.get_index_bounds_abc9(max_len)
    else:
        end_idx  = abc_scripts.get_index_bounds_abc(max_len)
    # for i in range(end_idx):
    #     print(abc_scripts.parse_index_abc(i-1))
    print(end_idx)
    
def main():
    parser = argparse.ArgumentParser(
            description='Single run of Yosys-ABC + Vivado')
    # Single run parameters
    parser.add_argument('-i','--input_file' , type=str, help='Input Verilog') # TODO Batch mode
    parser.add_argument('-o','--output_dir' , type=str, help='Output directory top level') # TODO Batch mode
    parser.add_argument('--index', type=int, help='Index of current sequence', default=-1)
    parser.add_argument('--synth_method', type=str, help='Target Mapping + synth method', default="fpga-abc9")
    parser.add_argument('--clock_period', type=int, help='Target clock rate (in picoseconds) for syntehsis', default=5000)
    parser.add_argument('--grade', type=int, help='Target Xilinx FPGA device grade', default=1)
    parser.add_argument('--device', type=str, help='Target Xilinx FPGA device', default="xc7a200tffv1156-1")
    parser.add_argument('--run_analysis', type=bool, help='Run Vivado or Synopsis post-Synthesis analysis backend', default=True)
    #parser.add_argument('--do_random', action='store_true', help='Test random synthesis recipes')
    parser.add_argument('-r','--random_seq_len', type=int, help='Length of random sequence to generate; 0 implies do not do random', default=0)

    # Batch run parameters
    parser.add_argument('-b', '--batch', type=int, help='Synthesis run batch size', default=0)
    parser.add_argument('-s', '--slurm', action='store_true', help='Run on slurm')
    parser.add_argument('-m', '--max_length', type=int, help='maximum length of synthesis recipes to run', default=0)
    #parser.add_argument('-n', '--min_length', type=int, help='minimum length of synthesis recipes to run', default=0)

    #parser.add_argument('--stochastic', type=int, help='Whether to use stochastic synthesis', default=0)
    args = parser.parse_args()
    kwargs = vars(args)
    test(**kwargs)
    run_batch_synthesis(**kwargs)
    
if __name__ == '__main__':
    main()    
    
