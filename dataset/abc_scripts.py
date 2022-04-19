
#!/usr/bin/python3

import argparse
import sys,os,random
from datetime import datetime
    
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

# ABC 9 Optimization passes
aig_sweep = "&scorr;&sweep;"
abc9_ind_ops = ["&dc2", "&syn2", "&syn3", "&syn4", "&b", "&b -d",
               "&if -W 300 -x -K 6", "&if -W 300 -g -K 6", "&if -W 300 -y -K 6"]
abc9_ch_ops = ["&synch2", "&dch", "&dch -f"]
abc9_ops = abc9_ind_ops + abc9_ch_ops + ["&if -W 300 -K 6 -v;&mfs;&st", "&if -W 300 -K 6 -v;&st"]

# ABC optimization passes
abc_ind_ops = ["rewrite", "rewrite -z", "refactor", "refactor -z",
               "balance",  "balance -d", "dc2",
               "if -W 300 -K 6; strash", "if -W 300 -K 6 -g", "if -W 300 -K 6 -x","if -W 300 -K 6 -y",
               "resub -K 8 -N 1",  "resub -K 8 -N 3",
               "resub -K 16 -N 1",  "resub -K 16 -N 3",
               "resub -K 8 -N 1 -z", "resub -K 8 -N 3 -z",
               "resub -K 16 -N 1 -z",  "resub -K 16 -N 3 -z"]
abc_ch_ops = ["dch", "dch -f"]
abc_ops = abc_ind_ops + abc_ch_ops

abc_opener = "strash;ifraig;scorr;"
# closure_ftune = "strash;ifraig;scorr;dc2;strash;dch -f;if -K 6;mfs2;lutpack -S 1"
closure_whitebox_delay = "strash;ifraig;scorr;strash;dch -f;if -v;mfs2;print_stats -l"
closure = "dretime; strash; dch -f; if -v; mfs2" # LUTPACK or not; dretime or not with 

def get_num_abc9_ops():
    return len(abc9_ops)
def get_num_abc_ops():
    return len(abc_ops)

def get_index_bounds_abc(max_len):
    max_idx = 1
    for l in range(max_len):
        max_idx += (len(abc_ind_ops) ** (l+1)) * len(abc_ch_ops) 
    return max_idx

def get_index_bounds_abc9(max_len):
    max_idx = 1
    for l in range(max_len):
        max_idx = len(abc9_ops) ** (l+1)
    return max_idx

# Helper function: index (integer) -> list of synth ops/passes
def parse_index_abc9(idx):
    i = idx
    ind_idx = []
    num_options = len(abc9_ops)
    while i >= 0 :
        remainder = i % num_options
        divisor = i // num_options
        ind_idx.append(remainder)
        if divisor <= 0 : 
            break;
        else : 
            i = divisor-1
    seq = ""
    ind_idx.reverse()
    for op in ind_idx:
        seq += abc9_ops[op] + ";"
    return seq

def parse_index_abc(idx):
    i = idx // len(abc_ch_ops)
    ch_ops = idx % len(abc_ch_ops)
    num_options = len(abc_ind_ops)
    ind_idx = []
    while i >= 0 :
        remainder = i % num_options
        divisor = i // num_options
        ind_idx.append(remainder)
        if divisor <= 0 : 
            break;
        else : 
            i = divisor-1
    seq = ""
    ind_idx.reverse()
    for op in ind_idx:
        seq += abc_ind_ops[op] + ";"
    seq += abc_ch_ops[ch_ops] + ";"
    return seq

def get_abc9_sequence(idx, random_seq_len):
    if idx < -1:
        return "&scorr;&sweep;&dc2;&dch -f;&if -W 300 -K 6 -v;&mfs;"
    seq = "rec_start3 " + os.path.dirname(os.path.abspath(__file__)) + "/include/rec6Lib_final_filtered3_recanon.aig\n"
    seq += aig_sweep + "\n"
    custom_seq = ""
    if random_seq_len > 0:
        random_num = random.randint(len(abc9_ops)**(random_seq_len-1), len(abc9_ops)**(random_seq_len))
        custom_seq = parse_index_abc9(random_num-1)
    else:
        custom_seq = parse_index_abc9(idx-1)
    seq += custom_seq + "\n"
    seq += "&if -W 300 -K 6 -v;&mfs;\n"
    return seq

def get_abc_sequence(idx, random_seq_len):
    if idx < -1:
        return "strash; ifraig; scorr; dc2; dretime; strash; dch -f; if; mfs2\n"
    seq = "rec_start3 " + os.path.dirname(os.path.abspath(__file__)) + "/include/rec6Lib_final_filtered3_recanon.aig\n"
    seq += abc_opener + "\n"
    custom_seq = ""
    if random_seq_len > 0:
        random_num = random.randint(get_index_bounds_abc(random_seq_len-1, random_seq_len))
        custom_seq = parse_index_abc(random_num-1)
    else:
        custom_seq = parse_index_abc(idx-1)
    seq += custom_seq + "\n"
    seq += "if -W 300 -K 6 -v;mfs2\n"
    return seq

# def get_abc9_stochastic(idx, lib_num):
#     seq = aig_sweep
#     i = idx
#     ind_idx = parse_index_single_list(idx)
#     if lib_num > 0:
#         seq += "&if -W 300 -v;&stochsyn -v -I 10 -N 1000 \"&st;"
#         for op in ind_idx:
#             seq += abc9_ops_lib[op]  + ";"
#         seq +="&if -W 300 -v;&mfs\";"
#     else:
#         seq += "&if -W 300 -K 6 -v;&stochsyn -v -I 10 -N 1000 \"&st;"
#         for op in ind_idx:
#             seq += abc9_ops[op] + ";"
#         seq += "&if -W 300 -K 6 -v;&mfs\";"
#     seq += "&ps -l"
#     return seq
    
