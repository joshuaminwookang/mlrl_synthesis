import numpy as np
import pandas as pd
import glob, os.path, os
import argparse, re
import pickle, json, sys

def lines_that_contain(string, fp):
    return [line for line in fp if string in line]

def merge_results_from_dir(dirname):
    data = []
    subdirs =  glob.glob(os.path.normpath(dirname + "/tab_*"))
    print("Found {} sub-directories".format(len(subdirs)))
    
    # Get data from Yosys and Vivado logs
    for subdir in subdirs:
        this_data = {}
        json_file = glob.glob(os.path.normpath(subdir + "/*.json"))
        try:
            with open(json_file[0], "r") as fp:
                this_data = json.load(fp)
        except :
            print("Could not open/read file:", subdir)
            continue
        if "Slice_LUTs" is in this_data.keys():
            this_data["Area"] = float(this_data["Slice_LUTs"])
            del this_data["Slice_LUTs"] 
        seq = this_data["Sequence"]
        if this_data["Sequence"] == "dch -f;if -K 6 -v;mfs" or this_data["Sequence"] == "":
            continue
        if seq.find('dch') >= 0:
            this_data["Sequence"] = seq[:seq.find('dch')-1]
        data.append(this_data)
    return data


def main():
    parser = argparse.ArgumentParser(
            description='Create ABC script for each permuted sequence of synthesis transformations')
    parser.add_argument('-i', '--input', type=str, help='Results Directory')
    args = parser.parse_args()
    dir = os.path.abspath(args.input)
    summary_name = os.path.basename(dir)
    data = merge_results_from_dir(dir)
    #sorted_data = {k: v for k, v in sorted(data.items(), key=lambda i: (i[1]["Benchmark"], i[1]["Index"]))}
    sorted_data = sorted(data, key=lambda i: (i["Benchmark"], i["Index"]))

    with open(summary_name+".pkl", "wb") as handle:
        pickle.dump(sorted_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
