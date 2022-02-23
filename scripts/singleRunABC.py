#!/usr/bin/python3

import argparse
import os,random
# from datetime import datetime
    
aig_sweep = "&scorr;&sweep;"
#aig_permute_in = ["", "&dfs", "&dfs -n", "&dfs -r", "&dfs -n -r"]
aig_permute_in = [""]
# aig_zero_cost_replace_ops = ["",
#                "&put;resub -K 8 -N 2 -z;&get -n;", "&put;resub -K 8 -N 3 -z;&get -n;",
#                "&put;resub -K 12 -N 2 -z;&get -n;", "&put;resub -K 12 -N 3 -z;&get -n;",
#                "&put;resub -K 16 -N 2 -z;&get -n;", "&put;resub -K 16 -N 3 -z;&get -n;"]

aig_zero_cost_replace_ops = ["", ""]
# aig_ind_ops = ["&dc2", "&syn2", "&b", "&b -d",
#                "&put;resub -K 8 -N 2;&get", "&put;resub -K 8 -N 3;&get",
#                "&put;resub -K 12 -N 2;&get", "&put;resub -K 12 -N 3;&get",
#                "&put;resub -K 16 -N 2;&get", "&put;resub -K 8; &get",
#                "&if -W 300 -x", "&if -W 300 -g"]


aig_ind_ops = ["&dc2", "&syn2", "&syn3", "&syn4", "&b", "&b -d",
               "&if -W 300 -x -K 6", "&if -W 300 -g -K 6", "&if -W 300 -y -K 6"]
aig_ch_ops = ["&synch2", "&dch", "&dch -f"]
abc9_ops = aig_ind_ops + aig_ch_ops + ["&if -W 300 -K 6 -v;&mfs;&st", "&if -W 300 -K 6 -v;&st"]
abc9_ops_lib = aig_ind_ops + aig_ch_ops + ["&if -W 300 -v;&mfs;&st", "&if -W 300 -v;&st"]

# old stuff
options = ["rewrite", "rewrite -z", "refactor", "refactor -z", "resub -K 8", "resub -K 4", "resub -K 12", "resub -N 0", "resub -N 2", "resub -N 3", "balance",  "dc2"]

opener = "strash;ifraig;scorr;"
# closure_ftune = "strash;ifraig;scorr;dc2;strash;dch -f;if -K 6;mfs2;lutpack -S 1"
closure_whitebox_delay = "strash;ifraig;scorr;strash;dch -f;if -v;mfs2;print_stats -l"
closure = "dretime; strash; dch -f; if -v; mfs2" # LUTPACK or not; dretime or not with -

def parse_index_single_list(idx):
    i = idx
    ind_idx = []
    while i >= 0 :
        num_options = len(abc9_ops)
        remainder = i % num_options
        divisor = i // num_options
        ind_idx.append(remainder)
        if divisor <= 0 : 
            break;
        else : 
            i = divisor-1
    return ind_idx

def parse_index(idx):
    i = idx
    perm_idx = 0
    # Structural choice AIG optimization options
    ch_idx = i % len(aig_ch_ops)
    i = i // len(aig_ch_ops)

    # Tech Indepdent AIG rewriting optimization options
    ind_idx = []
    while i >= 0 :
        num_options = len(aig_ind_ops)
        remainder = i % num_options
        divisor = i // num_options
        ind_idx.append(remainder)
        if divisor <= 0 : 
            break;
        else : 
            i = divisor-1
    return ind_idx, ch_idx, perm_idx

def get_seq_abc9_single_list(idx, lib_num):
    seq = aig_sweep
    i = idx
    ind_idx = parse_index_single_list(idx)
    if lib_num > 0:
        for op in ind_idx:
            seq += abc9_ops_lib[op]  + ";"
        seq +="&if -W 300 -v;&mfs;"
    else:
        for op in ind_idx:
            seq += abc9_ops[op] + ";"
        seq += "&if -W 300 -K 6 -v;&mfs;"
    return seq

def get_abc9_stochastic(idx, lib_num):
    seq = aig_sweep
    i = idx
    ind_idx = parse_index_single_list(idx)
    if lib_num > 0:
        seq += "&if -W 300 -v;&stochsyn -v -I 10 -N 1000 \"&st;"
        for op in ind_idx:
            seq += abc9_ops_lib[op]  + ";"
        seq +="&if -W 300 -v;&mfs\";"
    else:
        seq += "&if -W 300 -K 6 -v;&stochsyn -v -I 10 -N 1000 \"&st;"
        for op in ind_idx:
            seq += abc9_ops[op] + ";"
        seq += "&if -W 300 -K 6 -v;&mfs\";"
    seq += "&ps -l"
    return seq

def get_seq_abc9(idx, lib_num):
    seq = aig_sweep
    i = idx
    ind_idx, ch_idx, perm_idx = parse_index(idx)
    
    # Step 1 and 1.5: Tech Ind AIG rewriting
    seq += aig_permute_in[perm_idx] + ";"
    for op in ind_idx:
        seq += aig_ind_ops[op] + ";"
        
    # Step 2. Structural choice based rewriting
    seq += aig_ch_ops[ch_idx] + ";"
    if lib_num > 0:
        seq +="&if -W 300 -v;&mfs;"
    else:
        seq +="&if -W 300 -K 6 -v;&mfs;"

    return seq

def get_seq(idx): 
    num_options = len(options) 
    seq = opener
    i = idx
    while idx >= 0:
        remainder = i % num_options
        divisor = i // num_options
        seq += options[remainder] + ";"
        if divisor <= 0 : 
            break;
        else : 
            i = divisor
    seq += closure_whitebox_delay
    return seq

def get_rand_seq_abc9(seq_len, lib_num, idx):
    aig_all_ops = aig_ch_ops + aig_ind_ops
    num_options = len(aig_all_ops) 
    seq = aig_sweep

    random.seed(idx)
    n_iter = 1
    while True:
        for i in range(seq_len) :
            r = random.randint(0, 12)
            if r < num_options :
                seq += aig_all_ops[r] + ";"
        if lib_num > 0:
            seq +="&if -W 300 -v;&mfs;"
        else:
            seq +="&if -W 300 -K 6 -v;&mfs;"
        s = random.randint(0, 2**n_iter)
        if s == 0:
            seq +="&st;"
            continue
        else: 
            break
            # iterate again!
            seq +="&ps -l -D abc.txt"
    return seq

def main():
    parser = argparse.ArgumentParser(
            description='Single run of Yosys-ABC + Vivado')
    parser.add_argument('--input', type=str, help='Input Verilog')
    parser.add_argument('--device', type=str, help='Target Xilinx FPGA device', default="xc7a200")
    parser.add_argument('--speed', type=int, help='Target clock rate (in picoseconds) for syntehsis', default=5000)
    parser.add_argument('--grade', type=int, help='Target Xilinx FPGA device grade', default=1)
    parser.add_argument('--abc9', type=int, help='Index of current sequence', default=-1)

    parser.add_argument('--idx', type=int, help='Index of current sequence', default=-1)
    parser.add_argument('--random', type=int, help='Random Sequence Length', default=0)
    parser.add_argument('--vivado', type=int, help='Random Sequence Length', default=0)
    parser.add_argument('--lut_library', type=int, help='Index of LUT Library to use: 0 is default', default=0)
    parser.add_argument('--stochastic', type=int, help='Whether to use stochastic synthesis', default=0)
    args = parser.parse_args()

    do_abc9 = args.abc9 > 0
    do_random = args.random_seq_len > 0
    do_stochastic = args.stochastic > 0
    lut_lib_num = args.lut_library
    
    print("rec_start3 " + os.path.dirname(os.path.abspath(__file__)) + "/include/rec6Lib_final_filtered3_recanon.aig")
    if args.in_idx == -2:
        print("&scorr;&sweep;&dc2;&dch -f;&if -W 300 -K 6 -v;&mfs;")
        return
    if lut_lib_num > 0:
        print("read_lut " + os.path.dirname(os.path.abspath(__file__)) + "/lut_library/LUTLIB_{}.txt".format(lut_lib_num))
    if do_stochastic:
        print(get_abc9_stochastic(args.in_idx, lut_lib_num))
        return
    if do_random:
        #print(get_rand_seq_abc9(args.random_seq_len,lut_lib_num, args.in_idx))
        random_num = random.randint(len(abc9_ops)**(args.random_seq_len-1), len(abc9_ops)**(args.random_seq_len))
        print(get_seq_abc9_single_list(random_num, lut_lib_num))
        print("&ps; &pfeatures stats.json; &pfanstats fanstats.json;&write temp.aig")
    elif do_abc9:
        print(get_seq_abc9_single_list(args.in_idx, lut_lib_num))
        print("&ps; &pfeatures stats.json; &pfanstats fanstats.json;&write temp.aig")
    else :
        print(get_seq(args.in_idx))
    # print("write_blif internal.blif")

if __name__ == '__main__':
    main()    
    
