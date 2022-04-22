#!/usr/bin/python3

import argparse
import sys,os,random
import numpy as np
from datetime import datetime
    
# ABC 9 Optimization passes
aig_sweep = "&scorr;&sweep;"
abc9_ind_ops = ["&dc2", "&syn2", "&syn3", "&syn4", "&b", "&b -d",
               "&if -W 300 -x -K 6", "&if -W 300 -g -K 6", "&if -W 300 -y -K 6"]
abc9_ch_ops = ["&synch2", "&dch", "&dch -f"]
abc9_ops = abc9_ind_ops + abc9_ch_ops + ["&if -W 300 -K 6 -v;&mfs;&st", "&if -W 300 -K 6 -v;&st"]

# ABC optimization passes
abc_ind_ops = ["rewrite", "rewrite -z", "refactor", "refactor -z",
               "balance",  "dc2",
               "if -K 6; strash", "if -K 6 -g",
               "resub -K 8 -N 1",  "resub -K 8 -N 2",
               "resub -K 12 -N 1",  "resub -K 12 -N 2",
               "resub -K 8 -N 1 -z", "resub -K 8 -N 2 -z",
               "resub -K 12 -N 1 -z",  "resub -K 12 -N 2 -z"]
# abc_ind_ops = ["rewrite", "rewrite -z", "rewrite -l",
#                "refactor", "refactor -z", "refactor -l",
#                "balance",  "balance -d", "balance -l", "dc2",
#                "if -W 300 -K 6; strash", "if -W 300 -K 6 -g", "if -W 300 -K 6 -x","if -W 300 -K 6 -y",
#                "resub -K 8 -N 1",  "resub -K 8 -N 2","resub -K 8 -N 3",
#                "resub -K 8 -N 1 -z",  "resub -K 8 -N 2 -z","resub -K 8 -N 3 -z",
#                "resub -K 10 -N 1",  "resub -K 10 -N 2","resub -K 10 -N 3",
#                "resub -K 10 -N 1 -z",  "resub -K 10 -N 2 -z","resub -K 10 -N 3 -z",
#                "resub -K 12 -N 1",  "resub -K 12 -N 2","resub -K 12 -N 3",
#                "resub -K 12 -N 1 -z",  "resub -K 12 -N 2 -z","resub -K 12 -N 3 -z"]
#abc_ch_ops = ["dch", "dch -f"]
abc_ops = abc_ind_ops 
abc_opener = "strash;ifraig;scorr;"

def get_num_abc9_ops():
    return len(abc9_ops)
def get_num_abc_ops():
    return len(abc_ops)

def get_index_bounds_abc(max_len):
    max_idx = 1
    for l in range(max_len):
        max_idx += len(abc_ind_ops) ** (l+1)
    return max_idx

    
# list of integers (0~N) -> sinlge string of synthesis sequence
def get_abc_sequence_from_list (idx_list):
    seq = abc_opener + "\n"
    for idx in idx_list:
        seq += abc_ind_ops[idx] + ";"
    seq += "dch -f;if -K 6 -v;mfs2\n"
    return seq

def parse_index(idx):
    i = idx-1
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
    print(ind_idx)
    return ind_idx

def get_abc_sequence(idx):
    if idx < -1:
        return "strash; ifraig; scorr; dc2; dretime; strash; dch -f; if; mfs2\n"
    #seq = "rec_start3 " + os.path.dirname(os.path.abspath(__file__)) + "/include/rec6Lib_final_filtered3_recanon.aig\n"
    seq = abc_opener + "\n"
    idx_list = parse_index(idx)
    for op in idx_list:
        seq += abc_ind_ops[op] + ";"
    seq += "\n"
    seq += "dch -f;if -K 6 -v;mfs2\n"
    return seq

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

