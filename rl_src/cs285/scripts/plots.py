from pandas.io.sql import DatabaseError
import numpy as np
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob, os.path, os
import tensorflow as tf


#Problem-specific setup variables
TAGNAME = "Experiment"
DIR = os.path.normpath(os.path.join(os.getcwd(), "plots"))
#global params
fig_dims = (12,8)
axis_label = 16
legend_label = 14
axis_scale = 3.0

def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    # for e in tf.train.summary_iterator(file):
    for e in tf.compat.v1.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_AverageReturn':
                X.append(v.simple_value)
            # elif v.tag == 'Eval_AverageReturn':
            elif v.tag == "Eval_AverageReturn":
                Y.append(v.simple_value)
    return X, Y

def load_data_from_dir(search):
    dfs = []
    search_path = os.path.normpath(os.path.join(os.getcwd(), search))
    csvs = glob.glob(search_path)
    for f in csvs:
        X = []
        Y = []
        columns = [] 
        filename=os.path.dirname(f)
        tag=filename[filename.find('/hw4_') + 1:filename.find('-v')]
        X, Y = get_section_results(f)
        columns = ["Train_AverageReturn", "Eval_AverageReturn" ]
        data = pd.DataFrame([ [x,y] for x,y in zip(X, Y)], columns = columns)
        data[TAGNAME] = tag
        data["Iteration"] = range(0, len(data[TAGNAME]))
        dfs.append(data)
    return dfs

"""
    From a list of DataFrames, plot all data in a single plot (with legend)
    Goal: compare learning curves with some score metric (y_vars[1]) over some predictor (y_vars[0])
"""
def plot_stacked_learning_curves(dfs, vars, title, plot_type="scatter", subtitle=""):
    total_df = pd.DataFrame()
    # min_size = np.amin([len(df.index) for df in dfs])
    for df in dfs:
        total_df = total_df.append(df)
    total_df = total_df.pivot(index=vars[0], columns=TAGNAME, values=vars[1])
    fig = plt.gcf()
    fig.set_size_inches(fig_dims)
    sns.set_style("darkgrid")

    with sns.plotting_context(font_scale=axis_scale):
        if (plot_type == "scatter"):
            ax = sns.scatterplot(data=total_df)
        else :
            ax = sns.lineplot(data=total_df)
        ax.set_xlabel(vars[0], fontsize=axis_label, weight='bold')
        ax.set_ylabel(vars[1], fontsize=axis_label, weight='bold')
        plt.xlim([-1,1])
        plt.ylim(dfs[0].iloc[0][vars[1]]*1.1, dfs[0].iloc[0][vars[1]]*0.9)
        ax.ticklabel_format(axis="x", style="plain")
        ax.ticklabel_format(axis="y", style="plain")
        plt.legend(fontsize=legend_label,loc="best", prop={'weight': 'bold'})
        plt.title(title+"\n"+subtitle, fontsize=axis_label, weight='bold')

    plt.savefig(DIR+"/"+title+'.png',  format='png', dpi=300)
    plt.close() 

"""
main function
"""
def main():
    if not os.path.exists(DIR):
        os.mkdir(DIR)
    dfs_q2 = load_data_from_dir("data/hw4_q2_*/event*")
    plot_stacked_learning_curves(dfs_q2, ['Iteration', 'Train_AverageReturn'], "Q2_Obstacles_TrainAvg", plot_type="scatter")
    plot_stacked_learning_curves(dfs_q2, ['Iteration', 'Eval_AverageReturn'], "Q2_Obstacles_EvalAvg", plot_type="scatter")

    # dfs_q3 = load_data_from_dir("data/hw4_q3_*/event*")
    # plot_stacked_learning_curves(dfs_q3, ['Iteration', 'Eval_AverageReturn'], "Q3_MBRL_Random_Shooting", plot_type="line")

    # dfs_q4_ensembles = load_data_from_dir("data/hw4_q4_*_ensemble*/event*")
    # plot_stacked_learning_curves(dfs_q4_ensembles, ['Iteration', 'Eval_AverageReturn'], "Q4_MBRL_Ensemble", plot_type="line", subtitle="Effect of Ensemble Size")
    # dfs_q4_numseq = load_data_from_dir("data/hw4_q4_*_numseq*/event*")
    # plot_stacked_learning_curves(dfs_q4_numseq, ['Iteration', 'Eval_AverageReturn'], "Q4_MBRL_Numseq", plot_type="line", subtitle="Effect of Number of Candidate Action Sequences")
    # dfs_q4_horizon = load_data_from_dir("data/hw4_q4_*_horizon*/event*")
    # plot_stacked_learning_curves(dfs_q4_horizon, ['Iteration', 'Eval_AverageReturn'], "Q4_MBRL_Horizon", plot_type="line", subtitle="Effect of Planning Horizon Length")
   
    # dfs_q5 = load_data_from_dir("data/hw4_q5_*/event*")
    # plot_stacked_learning_curves(dfs_q5, ['Iteration', 'Eval_AverageReturn'], "Q5_Random_vs_CEM", plot_type="line")
    



if __name__ == "__main__":
    main()
