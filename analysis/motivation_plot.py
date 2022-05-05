import numpy as np
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob, os.path, os, pickle
import argparse

#global params
fig_dims = (12,8)
axis_label = 32
legend_label = 30
axis_scale = 2
default_script="dc2"

def load_data_pkl(pathname):
    filename=os.path.basename(pathname)
    tag=filename[filename.find('_') + 1 :filename.find('.pkl')]
    data = pd.read_pickle(pathname)
    data["Benchmark"] = tag
    return data, tag

# Load the data as Pandas DataFrame
def load_data(pathname):
    filename=os.path.basename(pathname)
    tag=filename[filename.find('_') + 1 :filename.find('.out.csv')]
    data = pd.read_csv(pathname, delimiter="\t")
    data["Benchmark"] = tag
    return data, tag

def load_data_from_dir(search):
    dfs = []
    search_path = os.path.normpath(os.path.join(os.getcwd(), search))
    csvs = glob.glob(search_path)
    for f in csvs:
        data = pd.DataFrame(pd.read_pickle(f)).transpose()
        dfs.append(data)
    return dfs

"""
    Plot a single (scatter/line/etc) plot for a benchmark
    @params: 
    
"""
def plot_single (df, title, y_vars, plot_type="scatter"):
    fig = plt.gcf()
    fig.set_size_inches(fig_dims)
    colors = [(0.277941, 0.056324, 0.381191), (0.257322, 0.25613, 0.526563), (0.136408, 0.541173, 0.554483), (0.506271, 0.828786, 0.300362)]
    with sns.plotting_context(font_scale=axis_scale):
        sns.set(font_scale=axis_scale)
        sns.set_style("ticks",{'axes.grid' : True})

        ax = sns.scatterplot(x=y_vars[0], y = y_vars[1], s=70, data=df)
        ax.set_xlabel("Area (# of LUTs)", fontsize=axis_label, weight='bold')
        ax.set_ylabel("Critical Path Delay (ns)", fontsize=axis_label, weight='bold')
        plt.title(title, fontsize=axis_label, weight='bold')
#        plt.legend(title="# Tech.-Ind.\n Passes", fontsize=legend_label,loc=1, prop={'weight': 'bold'}, title_fontproperties={'weight':'bold'})
#        plt.legend(title="Recipe\nLength", fontsize=legend_label,loc=1, prop={'weight': 'bold'}, title_fontproperties={'weight':'bold'})
    plt.savefig(title +'.png',  format='png', dpi=600, bbox_inches='tight')
    plt.close() 


def plot_circuit(bmark, largerDf):
    df = largerDf[largerDf['Benchmark'] == bmark]
 #   df['recipe_len'] = df.apply(lambda row: "<4" if len(row.Sequence.split(';')) <4 else "{}".format(len(row.Sequence.split(';'))), axis=1)
    plot_single(df, bmark, ['Slice_LUTs', 'Path_Delay'], plot_type="scatter")
        
    
def main():
    parser = argparse.ArgumentParser(
            description='Plot QoR distribution of Circuits')
    # Single run parameters
    parser.add_argument('-i','--input_circuits' , type=str, help='Circuits to plot') 
    args = parser.parse_args()
    epfl_arith =  pd.DataFrame(pd.read_pickle(("../dataset/run_epfl_arith.pkl")))
    epfl_control =  pd.DataFrame(pd.read_pickle(("../dataset/run_epfl_control.pkl")))

    plot_circuit('div', epfl_arith)
    plot_circuit('sqrt', epfl_arith)
    plot_circuit('mem_ctrl', epfl_control)
    # df_div = epfl_arith[epfl_arith['Benchmark'] == "div"]
    # df_div['Recipe\nLength'] = df_div.apply(lambda row: "<4" if len(row.Sequence.split(';')) <4 else "{}".format(len(row.Sequence.split(';'))), axis=1)
    # plot_single(df_div, "div", ['Slice_LUTs', 'Path_Delay'], plot_type="scatter")

    # sqrt = epfl_arith[epfl_arith['Benchmark'] == "squrt"]
    # sqrt['Recipe\nLength'] = sqrt.apply(lambda row: "<4" if len(row.Sequence.split(';')) <4 else "{}".format(len(row.Sequence.split(';'))), axis=1)
    # plot_single(df_div, "sqrt", ['Slice_LUTs', 'Path_Delay'], plot_type="scatter")
                                                                         


    # df_mctrl = epfl_control[epfl_control.Benchmark == "mem_ctrl"]
    # df_mctrl['Recipe\nLength'] = df_mctrl.apply(lambda row: "<4" if len(row.Sequence.split(';')) <4 else "{}".format(len(row.Sequence.split(';'))), axis=1)
    # plot_single(df_mctrl, "mem_ctrl", ['Slice_LUTs', 'Path_Delay'], plot_type="scatter")



if __name__ == "__main__":
    main()
