from os import listdir
from os.path import isfile, join
from patsy.highlevel import dmatrices
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

import data


def get_result_data(result_mode):
    #
    # read result data (sparsity, elapsed_time)
    #
    result_path = './results'
    sparsity = []
    elapsed_time = []
    for result_file in [f for f in listdir(result_path) if isfile(join(result_path, f))]:
        result = data.Data(result_file)
        sparsity.extend(result.data[result_mode].size * [result.params['sparsity']])
        elapsed_time.extend(result.data[result_mode])

    return pd.DataFrame({'sparsity': sparsity, 'elapsed_time': elapsed_time})


def fit_model(model, q):
    res = model.fit(q=q)
    return [q, res.params["Intercept"], res.params["sparsity"]] + res.conf_int().loc["sparsity"].tolist()


def make_quantreg(result_df):
    reg_exp = "elapsed_time ~ sparsity"

    #
    # create quantile regression model
    #

    y_train, X_train = dmatrices(reg_exp, result_df, return_type='dataframe')

    #
    # fit quantile regression for quantiles (0.05, ..., 0.96)
    #
    coeff = []
    for q in np.arange(0.05, 0.96, 0.1):
        quantreg_model = smf.quantreg(formula=reg_exp, data=result_df)
        quantreg_model_results = quantreg_model.fit(q=q)

        coeff.append(quantreg_model_results.params['Intercept'])

    return coeff


def plot_result_data(result_data):
    plt.scatter(x=result_data['cpu_base']['sparsity'], y=result_data['cpu_base']['elapsed_time'])
    plt.scatter(x=result_data['cpu_parallel']['sparsity'], y=result_data['cpu_parallel']['elapsed_time'])
    plt.scatter(x=result_data['gpu_base']['sparsity'], y=result_data['gpu_base']['elapsed_time'])
    plt.xlabel('Sparsity')
    plt.ylabel('Time in [ns]')
    plt.title('Experiment Result')
    plt.legend(['CPU Baseline', 'CPU Parallel', 'GPU Baseline'])

    plt.show()


def plot_box_plot(result_data):
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(25, 15))

    plt.autoscale()
    axs[0].set_title('CPU Baseline')
    sns.boxplot(x=result_data['cpu_base']['sparsity'], y=result_data['cpu_base']['elapsed_time'], whis=0.5, ax=axs[0])

    axs[1].set_title('CPU Parallel')
    sns.boxplot(x=result_data['cpu_parallel']['sparsity'], y=result_data['cpu_parallel']['elapsed_time'], whis=0.5, ax=axs[1])

    axs[2].set_title('GPU Baseline')
    sns.boxplot(x=result_data['gpu_base']['sparsity'], y=result_data['gpu_base']['elapsed_time'], whis=0.5, ax=axs[2])

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    plt.show()


def plot_violin_plot(result_data):
    # fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(25, 15), sharex='all', sharey='all')
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(25, 15))

    axs[0].set_title('CPU Baseline')
    sns.violinplot(x=result_data['cpu_base']['sparsity'], y=result_data['cpu_base']['elapsed_time'], ax=axs[0])

    axs[1].set_title('CPU Parallel')
    sns.violinplot(x=result_data['cpu_parallel']['sparsity'], y=result_data['cpu_parallel']['elapsed_time'], ax=axs[1])

    axs[2].set_title('GPU Baseline')
    sns.violinplot(x=result_data['gpu_base']['sparsity'], y=result_data['gpu_base']['elapsed_time'], ax=axs[2])

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    plt.show()


def plot_quantreg(quantreg_result):
    # fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(25, 15), sharex='all', sharey='all')
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(25, 15))

    axs[0].set_title('CPU Baseline', fontsize=20)
    axs[0].set_xlabel("Quantities", fontsize=16)
    axs[0].set_ylabel("Intercept", fontsize=16)
    axs[0].plot(np.arange(0.05, 0.96, 0.1), quantreg_result['cpu_base'], linestyle="dotted", color="black")

    axs[1].set_title('CPU Parallel', fontsize=20)
    axs[1].set_xlabel("Quantities", fontsize=16)
    axs[1].set_ylabel("Intercept", fontsize=16)
    axs[1].plot(np.arange(0.05, 0.96, 0.1), quantreg_result['cpu_parallel'], linestyle="dotted", color="red")

    axs[2].set_title('GPU Baseline', fontsize=20)
    axs[2].set_xlabel("Quantities", fontsize=16)
    axs[2].set_ylabel("Intercept", fontsize=16)
    axs[2].plot(np.arange(0.05, 0.96, 0.1), quantreg_result['gpu_base'], linestyle="dotted", color="blue")

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    plt.show()


def plot():
    # read result data
    result_data = {
        'cpu_base': get_result_data('CPU Baseline'),
        'cpu_parallel': get_result_data('CPU Parallel'),
        'gpu_base': get_result_data('GPU Baseline')
    }

    # plot result data
    plot_result_data(result_data)

    # plot boxplot
    plot_box_plot(result_data)

    # plot violin plot
    plot_violin_plot(result_data)

    # plot percentile
    quantreg_result = {
        'cpu_base': make_quantreg(result_data['cpu_base']),
        'cpu_parallel': make_quantreg(result_data['cpu_parallel']),
        'gpu_base': make_quantreg(result_data['gpu_base'])
    }
    plot_quantreg(quantreg_result)



if __name__ == '__main__':
    plot()
