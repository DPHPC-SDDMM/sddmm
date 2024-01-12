from os import listdir
from os.path import isfile, join
from patsy.highlevel import dmatrices
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import matplotlib.text as plt_txt
import matplotlib.pyplot as plt
from decimal import Decimal

import data


def get_result_data(base_path, result_mode):
    #
    # read result data (sparsity, elapsed_time)
    #
    result_path = base_path
    variable = []
    elapsed_time = []
    name = []
    for result_file in [f for f in listdir(result_path) if isfile(join(result_path, f))]:
        if result_file == "readme.txt":
            continue
        result = data.Data(result_path[1:], result_file)
        variable.extend(result.data[result_mode].size * [result.params[result.params["variable"]]])
        elapsed_time.extend(result.data[result_mode])

    return pd.DataFrame({result.params["variable"]: variable, 'elapsed_time': elapsed_time, 'variable': result.params["variable"], 'description': result.params["description"]})


def fit_model(model, q):
    res = model.fit(q=q)
    return [q, res.params["Intercept"], res.params["K"]] + res.conf_int().loc["K"].tolist()


def make_quantreg(result_df):
    reg_exp = "elapsed_time ~ " + result_df["variable"][0]

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

def format_to_exp(value):
    temp = '%E' % (value)
    temp_l = temp.split('.')
    back_l = temp_l[1].split('E')
    expon = int(back_l[1])
    back_l_stripped = back_l[0].rstrip('0')
    if back_l_stripped == '':
        val_int = int(temp_l[0])
        if val_int == 1:
            val = ''
            fstr = "$10^{" + str(expon) + "}$"
        else:
            val = str(val_int)
            fstr = "$" + str(val) + " \cdot 10^{" + str(expon) + "}$"
            # fstr = "${} \cdot 10^{\{\}}$"
    else:
        val = round(float('0.' + str(back_l_stripped)) + int(temp_l[0]),2)
        # fstr = "${} \cdot 10^{\{\}}$"
        fstr = "$" + str(val) + " \cdot 10^{" + str(expon) + "}$"

    value_str = fstr
    # if back_l_stripped != '':
    #     commas = str(round(int(back_l_stripped)/10))
    #     value_str = fstr.format(val, expon) #r'$' + ints + r"." + commas + r"\\cdot 10e{" + expon + r'}$'
    # else:
    #     commas = ''
    #     value_str = r'$' + ints + r"\\cdot 10e{" + expon + r'}$'
    return value_str

def get_data(result_data):
    variable = result_data["base"]["variable"][0]
    if variable == "sparsity":
        x_label = "density"
        x_data_vals = pd.unique(result_data['base'][variable])
        x_data_ticks = [format_to_exp(val) for val in 1 - pd.unique(result_data['base'][variable])]

        x_data_base = result_data['base'][variable]
        x_data_cusparse = result_data['cuSparse'][variable]
        x_data_sml2 = result_data['sm_l2'][variable]
    elif variable == "K":
        x_label = "K"
        x_data_vals = pd.unique(result_data['base'][variable])
        x_data_ticks = pd.unique(result_data['base'][variable])

        x_data_base = result_data['base'][variable]
        x_data_cusparse = result_data['cuSparse'][variable]
        x_data_sml2 = result_data['sm_l2'][variable]

    return {
        "x_label" : x_label,
        "x_data_vals" : x_data_vals,
        "x_data_ticks" : x_data_ticks,
        "x_data_base" : x_data_base,
        "x_data_cusparse" : x_data_cusparse,
        "x_data_sml2" : x_data_sml2
    }

def plot_result_data(iterations, name, result_data, save_name=""):
    data = get_data(result_data)

    plt.figure(figsize=(25, 15))
    plt.scatter(x=data["x_data_base"], y=result_data['base']['elapsed_time'])
    plt.scatter(x=data["x_data_cusparse"], y=result_data['cuSparse']['elapsed_time'])
    plt.scatter(x=data["x_data_sml2"], y=result_data['sm_l2']['elapsed_time'])
    plt.yticks(fontsize=16)
    plt.ylabel(data["x_label"], fontsize=20)
    plt.xticks(data["x_data_vals"], data["x_data_ticks"], fontsize=16)
    plt.xlabel(data["x_label"], fontsize=20)
    plt.ylabel('Time in [ns]')
    # plt.title('Experiment Result: ' + name + " (" + str(iterations) + " iterations)")
    title = result_data["base"]["description"][0].replace("sparsity", "density")
    plt.title(title + " (" + str(iterations) + " iterations)", fontsize=24)
    plt.legend(['Baseline', 'cuSparse', 'sm_l2'])

    if save_name == "":
        plt.show()
    else:
        save_name += "-scatter"
        save_name += ".png"
        plt.savefig(save_name, bbox_inches="tight")


def plot_box_plot(iterations, name, result_data, save_name=""):
    data = get_data(result_data)

    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(25, 15))
    # fig.suptitle('Experiment Result: ' + name + " (" + str(iterations) + " iterations)")
    title = result_data["base"]["description"][0].replace("sparsity", "density")
    fig.suptitle(title + " (" + str(iterations) + " iterations)", fontsize=24)

    plt.autoscale()
    axs[0].set_title('Baseline', fontsize=20)
    s = sns.boxplot(x=data["x_data_base"], y=result_data['base']['elapsed_time'], whis=0.5, ax=axs[0])
    s.set_yticklabels(s.get_yticklabels(), fontsize=16)
    s.set_xticklabels([plt_txt.Text(ind,x,y) for ind,x,y in zip(range(len(data["x_data_vals"])), data["x_data_vals"], data["x_data_ticks"])], fontsize=16)
    s.set_ylabel("Runtime [ns]", fontsize=20)
    s.set_xlabel(data["x_label"], fontsize=20)

    axs[1].set_title('cuSPARSE', fontsize=20)
    s = sns.boxplot(x=data["x_data_cusparse"], y=result_data['cuSparse']['elapsed_time'], whis=0.5, ax=axs[1])
    s.set_yticklabels(s.get_yticklabels(), fontsize=16)
    s.set_xticklabels([plt_txt.Text(ind,x,y) for ind,x,y in zip(range(len(data["x_data_vals"])), data["x_data_vals"], data["x_data_ticks"])], fontsize=16)
    s.set_ylabel("Runtime [ns]", fontsize=20)
    s.set_xlabel(data["x_label"], fontsize=20)

    axs[2].set_title('SM_L2', fontsize=20)
    s = sns.boxplot(x=data["x_data_sml2"], y=result_data['sm_l2']['elapsed_time'], whis=0.5, ax=axs[2])
    s.set_yticklabels(s.get_yticklabels(), fontsize=16)
    s.set_xticklabels([plt_txt.Text(ind,x,y) for ind,x,y in zip(range(len(data["x_data_vals"])), data["x_data_vals"], data["x_data_ticks"])], fontsize=16)
    s.set_ylabel("Runtime [ns]", fontsize=20)
    s.set_xlabel(data["x_label"], fontsize=20)

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    if save_name == "":
        plt.show()
    else:
        save_name += "-box"
        save_name += ".png"
        plt.savefig(save_name, bbox_inches="tight")


def plot_violin_plot(iterations, name, result_data, save_name=""):
    data = get_data(result_data)

    # fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(25, 15), sharex='all', sharey='all')
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(25, 15))
    # fig.suptitle('Experiment Result: ' + name + " (" + str(iterations) + " iterations)")
    title = result_data["base"]["description"][0].replace("sparsity", "density")
    fig.suptitle(title + " (" + str(iterations) + " iterations)", fontsize=24)

    axs[0].set_title('Baseline', fontsize=20)
    s = sns.violinplot(x=data["x_data_base"], y=result_data['base']['elapsed_time'], ax=axs[0])
    s.set_yticklabels(s.get_yticklabels(), fontsize=16)
    s.set_xticklabels([plt_txt.Text(ind,x,y) for ind,x,y in zip(range(len(data["x_data_vals"])), data["x_data_vals"], data["x_data_ticks"])], fontsize=16)
    s.set_ylabel("Runtime [ns]", fontsize=20)
    s.set_xlabel(data["x_label"], fontsize=20)

    axs[1].set_title('cuSPARSE', fontsize=20)
    s = sns.violinplot(x=data["x_data_cusparse"], y=result_data['cuSparse']['elapsed_time'], ax=axs[1])
    s.set_yticklabels(s.get_yticklabels(), fontsize=16)
    s.set_xticklabels([plt_txt.Text(ind,x,y) for ind,x,y in zip(range(len(data["x_data_vals"])), data["x_data_vals"], data["x_data_ticks"])], fontsize=16)
    s.set_ylabel("Runtime [ns]", fontsize=20)
    s.set_xlabel(data["x_label"], fontsize=20)

    axs[2].set_title('SM_L2', fontsize=20)
    s = sns.violinplot(x=data["x_data_sml2"], y=result_data['sm_l2']['elapsed_time'], ax=axs[2])
    s.set_yticklabels(s.get_yticklabels(), fontsize=16)
    s.set_xticklabels([plt_txt.Text(ind,x,y) for ind,x,y in zip(range(len(data["x_data_vals"])), data["x_data_vals"], data["x_data_ticks"])], fontsize=16)
    s.set_ylabel("Runtime [ns]", fontsize=20)
    s.set_xlabel(data["x_label"], fontsize=20)

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    if save_name == "":
        plt.show()
    else:
        save_name += "-violin"
        save_name += ".png"
        plt.savefig(save_name, bbox_inches="tight")


def plot_quantreg(iterations, name, quantreg_result):
    # fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(25, 15), sharex='all', sharey='all')
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(25, 15))
    fig.suptitle('Experiment Result: ' + name + " (" + str(iterations) + " iterations)")

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


def plot(iterations, base_path, save_name=""):
    # read result data
    result_data = {
        'base': get_result_data(base_path, 'Baseline'),
        'cuSparse': get_result_data(base_path, 'cuSPARSE'),
        'sm_l2': get_result_data(base_path, 'sm_l2')
    }

    # plot result data
    plot_result_data(iterations, base_path.split("/")[-2], result_data, save_name)

    # plot boxplot
    plot_box_plot(iterations, base_path.split("/")[-2], result_data, save_name)

    # plot violin plot
    plot_violin_plot(iterations, base_path.split("/")[-2], result_data, save_name)

    # plot percentile
    # quantreg_result = {
    #     'base': make_quantreg(result_data['base']),
    #     'cuSparse': make_quantreg(result_data['cuSparse']),
    #     'sm_l2': make_quantreg(result_data['sm_l2'])
    # }
    # plot_quantreg(iterations, base_path.split("/")[-2], quantreg_result)



if __name__ == '__main__':
    plot(100, "./sddmm_data_results_100_iterations/data_sets/IMDB/", "imdb")
    plot(100, "./sddmm_data_results_100_iterations/data_sets/IMDB_companion/", "imdb-companion")
    plot(100, "./sddmm_data_results_100_iterations/data_sets/patents/", "patents")
    plot(100, "./sddmm_data_results_100_iterations/data_sets/patents_companion/", "patents-companion")
    plot(100, "./sddmm_data_results_100_iterations/data_sets/patents_main/", "patents-main")
    plot(100, "./sddmm_data_results_100_iterations/data_sets/patents_main_companion/", "patents-main-companion")
    plot(100, "./sddmm_data_results_100_iterations/sparsity_large_2/K32/", "sparsity-large-2-k32")
    plot(100, "./sddmm_data_results_100_iterations/sparsity_large_2/K128/", "sparsity-large-2-k128")
    plot(100, "./sddmm_data_results_100_iterations/sparsity_large_2/K512/", "sparsity-large-2-k512")
    plot(100, "./sddmm_data_results_100_iterations/sparsity_small/K32/", "sparsity-small-k32")
    plot(100, "./sddmm_data_results_100_iterations/sparsity_small/K128/", "sparsity-small-k128")
    plot(100, "./sddmm_data_results_100_iterations/sparsity_small/K512/", "sparsity-small-k512")

    # plot(30, "./sddmm_data_results_30_iterations/data_sets/IMDB/")
    # plot(30, "./sddmm_data_results_30_iterations/data_sets/IMDB_companion/")
    # plot(30, "./sddmm_data_results_30_iterations/data_sets/patents/")
    # plot(30, "./sddmm_data_results_30_iterations/data_sets/patents_companion/")
    # plot(30, "./sddmm_data_results_30_iterations/data_sets/patents_main/")
    # plot(30, "./sddmm_data_results_30_iterations/data_sets/patents_main_companion/")
    # plot(30, "./sddmm_data_results_30_iterations/sparsity_large_2/K32/")
    # plot(30, "./sddmm_data_results_30_iterations/sparsity_large_2/K128/")
    # plot(30, "./sddmm_data_results_30_iterations/sparsity_large_2/K512/")
    # plot(30, "./sddmm_data_results_30_iterations/sparsity_small/K32/")
    # plot(30, "./sddmm_data_results_30_iterations/sparsity_small/K128/")
    # plot(30, "./sddmm_data_results_30_iterations/sparsity_small/K512/")