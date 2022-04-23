import numpy as np
import pandas as pd
import glob, os.path, os
import argparse, re
import pickle, json, sys

def lines_that_contain(string, fp):
    return [line for line in fp if string in line]

def merge_results_from_dir(dirname):
    data = {}
    subdirs =  glob.glob(os.path.normpath(dirname + "/tab_*"))
    print("Found {} sub-directories".format(len(subdirs)))
    
    # Get data from Yosys and Vivado logs
    i = 0
    for subdir in subdirs:
        this_data = {}
        json_file = glob.glob(os.path.normpath(subdir + "/*.json"))    
        try:
            print(json_file[0])
            with open(json_file[0], "r") as fp:
                this_data = json.load(fp)
            print(this_data)
        except OSError:
            print("Could not open/read file:", json_file[0])
            sys.exit()
#        this_data["Benchmark"] = this_data.pop("Becnhmark")
        data[i] = this_data
        i += 1
    return data


def main():
    parser = argparse.ArgumentParser(
            description='Create ABC script for each permuted sequence of synthesis transformations')
    parser.add_argument('-i', '--input', type=str, help='Results Directory')
    args = parser.parse_args()
    dir = os.path.abspath(args.input)
    summary_name = os.path.basename(dir)
    data = merge_results_from_dir(dir)
    sorted_data = {k: v for k, v in sorted(data.items(), key=lambda i: (i[1]["Benchmark"], i[1]["Index"]))}

    with open(summary_name+".pkl", "wb") as handle:
        pickle.dump(sorted_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
