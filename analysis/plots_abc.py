import numpy as np
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob, os.path, os, pickle

#global params
fig_dims = (12,8)
axis_label = 12
legend_label = 12
axis_scale = 2.0
default_script="dc2;dch -f"

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
    sns.set_style("darkgrid")
    fig = plt.gcf()
    fig.set_size_inches(fig_dims)
    
    yosys_baseline = df.loc[df['Sequence'] == default_script]
    print(yosys_baseline)
    with sns.plotting_context(font_scale=axis_scale):
        plt.axvline(yosys_baseline.iloc[0][y_vars[0]], ls='--', color='black', lw=0.7)
        plt.axhline(yosys_baseline.iloc[0][y_vars[1]], ls='--', color='black', lw=0.7)

        if (plot_type == "scatter"):
            ax = sns.scatterplot(x=y_vars[0], y = y_vars[1],  data=df);
        else :
            ax = sns.lineplot(x=y_vars[0], y = y_vars[1],  data=df);
        ax.set_xlabel(y_vars[0], fontsize=axis_label, weight='bold')
        ax.set_ylabel(y_vars[1], fontsize=axis_label, weight='bold')
        #plt.legend(fontsize=legend_label,loc=1, prop={'weight': 'bold'})
        plt.title(title+" (N={})".format(df[y_vars[1]].count()), fontsize=axis_label, weight='bold')
    #plt.savefig(title+'_'+y_vars[0]+'_'+y_vars[1]+'.png',  format='png', dpi=300)
    plt.savefig(title + '.png',  format='png', dpi=300)
    plt.close() 

"""
    From a comined DataFrames, plot inidividual plots for each Benchmark
"""
def plot_singles_tiled(df, title, y_vars, plot_type):
    tiling = (2,2)
    sns.set_style("darkgrid")
    fig, axes = plt.subplots(tiling[0], tiling[1])
    fig.set_size_inches((tiling[1]*fig_dims[0], tiling[0]*fig_dims[1]))

    bmarks = df["Benchmark"].unique()
    for i in range(tiling[0]):
        for j in range(tiling[1]):
            bmark = bmarks[tiling[0]*i +j]
            this_df = df[df.Benchmark == bmark]
            sns.scatterplot(ax=axes[i,j], x=y_vars[0], y = y_vars[1],  data=this_df);
            axes[i,j].set_title(bmark, fontsize=axis_label, weight='bold')
    fig.suptitle(title+" each (N={})".format(this_df[y_vars[1]].count()), fontsize=axis_label, weight='bold')
    plt.savefig(title + '.png',  format='png', dpi=300)
    plt.close() 

"""
    From a comined DataFrames, plot inidividual plots for each Benchmark
"""
def plot_singles(df, title, y_vars, plot_type):
    for bmark in df["Benchmark"].unique():
        print(df[df.Benchmark == bmark])
        plot_single(df[df.Benchmark == bmark],bmark, y_vars, plot_type=plot_type)
"""
    From a list of DataFrames, plot all data in a single plot (with legend)
    Goal: compare the results of different benchmarks (of y_vars[1]) over some predictor (y_vars[0])
"""
def plot_stacked(dfs, y_vars, plot_type="scatter"):
    total_df = pd.DataFrame()
    min_size = np.amin([len(df.index) for df in dfs])
    for df in dfs:
        relative_df = df.copy()
        relative_df[y_vars[1]] = df[y_vars[1]] / df[y_vars[1]].median()
        total_df = total_df.append(relative_df)
    total_df = total_df.pivot(index=y_vars[0], columns='Benchmark', values=y_vars[1])

    fig = plt.gcf()
    fig.set_size_inches(fig_dims)
    sns.set_style("darkgrid")

    with sns.plotting_context(font_scale=axis_scale):
        if (plot_type == "scatter"):
            ax = sns.scatterplot(data=total_df.iloc[100:200])
        else :
            ax = sns.lineplot(data=total_df.iloc[100:400])
        ax.set_xlabel(y_vars[0]+' (Random Synthesis Flow)', fontsize=axis_label, weight='bold')
        ax.set_ylabel('Relative '+y_vars[1]+' (Normalized to Median)', fontsize=axis_label, weight='bold')
        plt.legend(fontsize=legend_label,loc=1, prop={'weight': 'bold'})

    plt.savefig(y_vars[0]+'_'+y_vars[1]+'.png',  format='png', dpi=300)
    plt.close() 



def main():
    # Generate scatterplots for random runs
    # dfs = load_data_from_dir("results/random*.csv")
    # plot_singles(dfs, "random", ['Slice_LUTs', 'Path_Delay'], plot_type="scatter-ratios")

    # Stacked plot to compare QoRs of same scripts on different benchmarks
    # plot_stacked(dfs, ['Index', 'Path_Delay'], plot_type="scatter")
    # plot_stacked(dfs, ['Index', 'Slice_LUTs'], plot_type="scatter")

    # for df in dfs:
    #     if df.iloc[0]['Benchmark'] == "or1200":
    #         plot_single(df, "Vivado_vs_ABC", ['ABC_Delay', 'Path_Delay'], plot_type="scatter")
    #         plot_single(df, "Vivado_vs_ABC", ['ABC_Area', 'Slice_LUTs'], plot_type="scatter")
    exh = load_data_from_dir("../dataset/*.pkl")

    plot_singles(exh[0], "test", ['Slice_LUTs', 'Path_Delay'], plot_type="scatter")
    #plot_singles_tiled(exh[0], "test", ['Slice_LUTs', 'Path_Delay'], plot_type="scatter")

    # plot_single(exh[0], "VTR_bgm", ['ABC_Delay', 'Path_Delay'], plot_type="scatter")
    # plot_single(exh[0], "VTR_bgm", ['ABC_Area', 'Slice_LUTs'], plot_type="scatter")
    # plot_single(exh[0], "Exhaustive_bgm", ['Slice_LUTs','Path_Delay'], plot_type="scatter-ratios")


if __name__ == "__main__":
    main()
