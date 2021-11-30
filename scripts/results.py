import numpy as np
import pandas as pd
import glob, os.path, os
import argparse, re
import pickle, json

def lines_that_contain(string, fp):
    return [line for line in fp if string in line]

def load_data_from_dir(dirname):
    data = {}
    # logs = glob.glob(os.path.normpath(dirname + "/*/test_5000.log"))
    subdirs =  glob.glob(os.path.normpath(dirname + "/tab*"))
    # scripts = glob.glob(os.path.normpath(dirname + "/*/*.abc.script"))
    print("Found {} directories".format(len(subdirs)))
    # yosys_logs = glob.glob(os.path.normpath(dirname + "/*/yosys.log"))

    # seqs=[]; bmarks=[]; indices=[];
    # path_delay=[]; logic_delay=[]; net_delay=[]; Logic_Delay_Percentage=[]
    # Slice_LUTs=[]; lut_Logic=[]; lut_mem = []; reg_ff = []; reg_latch = []
    # abc_delay = []; abc_area = []

    # Populate index and sequence fields
    # data["Index"] = [ os.path.basename(script).split('.')[1] for script in scripts ]
    # data["Benchmark"] = [ os.path.basename(script).split('.')[0] for script in scripts ]
    # for script in scripts:
    #     with open(script, "r") as fp:
    #          seqs.append(lines_that_contain("&scorr", fp)[0])
    # data["Sequence"] = seqs
    
    # Get data from Yosys and Vivado logs
    i = 0
    for subdir in subdirs:
        this_data = {}
        vivado_log = glob.glob(os.path.normpath(subdir + "/test_5000.log"))
        # edif_file = glob.glob(os.path.normpath(subdir + "/*.edif"))
        yosys_log = glob.glob(os.path.normpath(subdir + "/yosys.log"))
        script = glob.glob(os.path.normpath(subdir + "/*.abc.script"))
        stats1 = glob.glob(os.path.normpath(subdir + "/stats.json"))
        stats2= glob.glob(os.path.normpath(subdir + "/fanstats.json"))
        index = os.path.basename(script[0]).split('.')[1]
        bmark = os.path.basename(script[0]).split('.')[0]

        this_data['Index'] = index
        this_data['Benchmark'] = bmark

        with open(script[0], "r") as fp:
             this_data['Sequence'] = lines_that_contain("&scorr", fp)[0]
        with open(vivado_log[0], "r") as fp:
            try:
                this_data["Path_Delay"] = float(re.findall(r'\d+.\d+', lines_that_contain("Path Delay", fp)[0])[0])
            except:
                print("No Vivado log for {} {}".format(bmark, index))
                continue
                this_data["Path_Delay"] = np.nan
            try:
                fp.seek(0)
                this_data["Logic_Delay"] = float(re.findall(r'\d+.\d+', lines_that_contain("Logic Delay", fp)[0])[0])
            except:
                this_data["Logic_Delay"] = np.nan
            try:
                fp.seek(0)
                this_data["Net_Delay"] = float(re.findall(r'\d+.\d+', lines_that_contain("Net Delay", fp)[0])[0])
            except:
                this_data["Net_Delay"] = np.nan
            try:      
                fp.seek(0)
                percentage = re.findall(r'\d+%', lines_that_contain("Logic Delay", fp)[0])[0]
                this_data["Logic_Delay_Percentage"] = float(percentage[:-1])
            except:
                this_data["Logic_Delay_Percentage"] = np.nan
            try:
                fp.seek(0)
                this_data["Slice_LUTs"] = int(re.findall(r'\d+', lines_that_contain("Slice LUTs", fp)[0])[0])
            except:
                this_data["Slice_LUTs"] = np.nan
            try:
                fp.seek(0)
                this_data["LUT_as_Logic"] = int(re.findall(r'\d+', lines_that_contain("LUT as Logic", fp)[0])[0])
            except:
                this_data["LUT_as_Logic"] = np.nan
            try:
                fp.seek(0)
                this_data["LUT_as_Memory"] = int(re.findall(r'\d+', lines_that_contain("LUT as Memory", fp)[0])[0])
            except:
                this_data["LUT_as_Memory"] = np.nan
            try:
                fp.seek(0)
                this_data["Regs_FF"] = int(re.findall(r'\d+', lines_that_contain("Register as Flip Flop", fp)[0])[0])
            except:
                this_data["Regs_FF"] = np.nan
            try:
                fp.seek(0)
                this_data["Regs_Latch"] = int(re.findall(r'\d+', lines_that_contain("Register as Latch", fp)[0])[0])
            except:
                this_data["Regs_Latch"] = np.nan
            
        # with open(yosys_log[0], "r") as fp:
        #     last_if_line =  lines_that_contain("Del =", fp)[-1]
        #     try:
        #         this_data["ABC_Delay"] = float(re.findall(r'\d+.\d+',last_if_line)[0])
        #         continue
        #     except:
        #         this_data["ABC_Delay"] = np.nan
        #     try:
        #         this_data["ABC_Area"] = int(round(float(re.findall(r'\d+.\d+', last_if_line)[1])))
        #     except:
        #         this_data["ABC_Area"] = np.nan
        try:
            fp_stats1 =  open(stats1[0], "r")
            stats1_data = json.load(fp_stats1)
            fp_stats1.close()
        except:
            print("No stats.json for {} {}".format(bmark,index))
            continue
        try:
            fp_stats2 =  open(stats2[0], "r")
            stats2_data = json.load(fp_stats2)
            fp_stats2.close()
        except:
            print("No fanstats.json for {} {}".format(bmark,index))
            continue
        this_data = {**this_data, **stats1_data, **stats2_data}
        data[i] = this_data
        i += 1

    #     with open(vivado_log[0], "r") as fp:
    #         try:
    #             path_delay.append(re.findall(r'\d+.\d+', lines_that_contain("Path Delay", fp)[0])[0])
    #         except:
    #             path_delay.append(np.nan)
    #         try:
    #             fp.seek(0)
    #             logic_delay.append(re.findall(r'\d+.\d+', lines_that_contain("Logic Delay", fp)[0])[0])
    #         except:
    #             logic_delay.append(np.nan)
    #         try:
    #             fp.seek(0)
    #             net_delay.append(re.findall(r'\d+.\d+', lines_that_contain("Net Delay", fp)[0])[0])
    #         except:
    #             net_delay.append(np.nan)      
    #         try:      
    #             fp.seek(0)
    #             perecentage= re.findall(r'\d+%', lines_that_contain("Logic Delay", fp)[0])[0]
    #             Logic_Delay_Percentage.append(perecentage[:-1])
    #         except:
    #             Logic_Delay_Percentage.append(np.nan)
    #         try:
    #             fp.seek(0)
    #             Slice_LUTs.append(re.findall(r'\d+.\d+', lines_that_contain("Slice LUTs", fp)[0])[0])
    #         except:
    #             Slice_LUTs.append(np.nan)
    #         try:
    #             fp.seek(0)
    #             lut_Logic.append(re.findall(r'\d+.\d+', lines_that_contain("LUT as Logic", fp)[0])[0])
    #         except:
    #             lut_Logic.append(np.nan)
    #         try:
    #             fp.seek(0)
    #             lut_mem.append(re.findall(r'\d+.\d+', lines_that_contain("LUT as Memory", fp)[0])[0])
    #         except:
    #             lut_mem.append(np.nan)
    #         try:
    #             fp.seek(0)
    #             reg_ff.append(re.findall(r'\d+.\d+', lines_that_contain("Register as Flip Flop", fp)[0])[0])
    #         except:
    #             reg_ff.append(np.nan)
    #         try:
    #             fp.seek(0)
    #             reg_latch.append(re.findall(r'\d+.\d+', lines_that_contain("Register as Latch", fp)[0])[0])
    #         except:
    #             reg_latch.append(np.nan)
            
    #     with open(yosys_log[0], "r") as fp:
    #         last_if_line =  lines_that_contain("Del =", fp)[-1]
    #         try:
    #             abc_delay.append(re.findall(r'\d+.\d+',last_if_line)[0])
    #         except:
    #             abc_delay.append(np.nan)
    #         try:
    #             abc_area.append(re.findall(r'\d+.\d+', last_if_line)[1])
    #         except:
    #             abc_area.append(np.nan)
    # data["Benchmark"] = bmarks ; data["Sequence"] = seqs; data["Index"] = indices
    # data["Path_Delay"] = path_delay; data["Logic_Delay"] = logic_delay; data["Net_Delay"] = net_delay; data["Logic_Delay_Percentage"] = Logic_Delay_Percentage
    # data["Slice_LUTs"] = Slice_LUTs; data["LUT_as_Logic"] = lut_Logic; data["LUT_as_Memory"] = lut_mem
    # data["RegsFF"] = reg_ff; data["RegsLatch"] = reg_latch
    # data["ABC_Delay"] = abc_delay; data["ABC_Area"] = abc_area
    return data


def main():
    parser = argparse.ArgumentParser(
            description='Create ABC script for each permuted sequence of synthesis transformations')
    parser.add_argument('--i', type=str, help='Results Directory')
    args = parser.parse_args()
    dir = os.path.abspath(args.i)

    bmark = os.path.basename(dir)
    ip = bmark[bmark.find('run') + 4 :]
    data = load_data_from_dir(dir)

    
    # df["Benchmark"] = ip
    print("outputting to: " + ip+".pkl")
    # df = pd.DataFrame.from_dict(df_dict)
    # df.to_csv(ip+".out.csv", sep="\t",index=False)

    with open(ip+".pkl", "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(ip + ".json", "w") as f:
        json.dump(data, f, indent=4)
    # Generate scatterplots for random runs
    # dfs = load_data_from_dir("results/random*.csv")


if __name__ == "__main__":
    main()
