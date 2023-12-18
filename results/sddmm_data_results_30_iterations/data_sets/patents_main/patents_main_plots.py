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
    result_path = 'results/'
    k = []
    elapsed_time = []
    for result_file in [f for f in listdir(result_path) if isfile(join(result_path, f))]:
        result = data.Data(result_file)
        k.extend(result.data[result_mode].size * [result.params['K']])
        elapsed_time.extend(result.data[result_mode])

    return pd.DataFrame({'k': k, 'elapsed_time': elapsed_time})


def fit_model(model, q):
    res = model.fit(q=q)
    return [q, res.params["Intercept"], res.params["K"]] + res.conf_int().loc["K"].tolist()


def make_quantreg(result_df):
    reg_exp = "elapsed_time ~ k"

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
    plt.scatter(x=result_data['base']['k'], y=result_data['base']['elapsed_time'])
    plt.scatter(x=result_data['cuSparse']['k'], y=result_data['cuSparse']['elapsed_time'])
    plt.scatter(x=result_data['sm_l2']['k'], y=result_data['sm_l2']['elapsed_time'])
    plt.xlabel('K')
    plt.ylabel('Time in [ns]')
    plt.title('Experiment Result')
    plt.legend(['Baseline', 'cuSparse', 'sm_l2'])

    plt.show()


def plot_box_plot(result_data):
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(25, 15))

    plt.autoscale()
    axs[0].set_title('Baseline')
    sns.boxplot(x=result_data['base']['k'], y=result_data['base']['elapsed_time'], whis=0.5, ax=axs[0])

    axs[1].set_title('cuSPARSE')
    sns.boxplot(x=result_data['cuSparse']['k'], y=result_data['cuSparse']['elapsed_time'], whis=0.5, ax=axs[1])

    axs[2].set_title('SM_L2')
    sns.boxplot(x=result_data['sm_l2']['k'], y=result_data['sm_l2']['elapsed_time'], whis=0.5, ax=axs[2])

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    plt.show()


def plot_violin_plot(result_data):
    # fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(25, 15), sharex='all', sharey='all')
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(25, 15))

    axs[0].set_title('Baseline')
    sns.violinplot(x=result_data['base']['k'], y=result_data['base']['elapsed_time'], ax=axs[0])

    axs[1].set_title('cuSPARSE')
    sns.violinplot(x=result_data['cuSparse']['k'], y=result_data['cuSparse']['elapsed_time'], ax=axs[1])

    axs[2].set_title('SM_L2')
    sns.violinplot(x=result_data['sm_l2']['k'], y=result_data['sm_l2']['elapsed_time'], ax=axs[2])

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    plt.show()


def plot_quantreg(quantreg_result):
    # fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(25, 15), sharex='all', sharey='all')
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(25, 15))

    axs[0].set_title('Baseline', fontsize=20)
    axs[0].set_xlabel("Quantities", fontsize=16)
    axs[0].set_ylabel("Intercept", fontsize=16)
    axs[0].plot(np.arange(0.05, 0.96, 0.1), quantreg_result['base'], linestyle="dotted", color="black")

    axs[1].set_title('cuSPARSE', fontsize=20)
    axs[1].set_xlabel("Quantities", fontsize=16)
    axs[1].set_ylabel("Intercept", fontsize=16)
    axs[1].plot(np.arange(0.05, 0.96, 0.1), quantreg_result['cuSparse'], linestyle="dotted", color="red")

    axs[2].set_title('SM_L2', fontsize=20)
    axs[2].set_xlabel("Quantities", fontsize=16)
    axs[2].set_ylabel("Intercept", fontsize=16)
    axs[2].plot(np.arange(0.05, 0.96, 0.1), quantreg_result['sm_l2'], linestyle="dotted", color="blue")

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    plt.show()


def plot():
    # read result data
    result_data = {
        'base': get_result_data('Baseline'),
        'cuSparse': get_result_data('cuSPARSE'),
        'sm_l2': get_result_data('sm_l2')
    }

    # plot result data
    plot_result_data(result_data)

    # plot boxplot
    plot_box_plot(result_data)

    # plot violin plot
    plot_violin_plot(result_data)

    # plot percentile
    quantreg_result = {
        'base': make_quantreg(result_data['base']),
        'cuSparse': make_quantreg(result_data['cuSparse']),
        'sm_l2': make_quantreg(result_data['sm_l2'])
    }
    plot_quantreg(quantreg_result)



if __name__ == '__main__':
    plot()

