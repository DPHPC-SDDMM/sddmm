from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

import data


def fit_model(mod, q):
    res = mod.fit(q=q)
    return [q, res.params["Intercept"], res.params["sparsity"]] + res.conf_int().loc["sparsity"].tolist()


def plot():
    #
    # get files of results directory
    #
    result_path = './results'
    result_files = [f for f in listdir(result_path) if isfile(join(result_path, f))]

    #
    # reformat result data to (sparsity, elapsed_time)
    #
    raw_data = {
        'cpu_base': {'sparsity': [], 'elapsed_time': []},
        'cpu_parallel': {'sparsity': [], 'elapsed_time': []},
        'gpu_base': {'sparsity': [], 'elapsed_time': []}
    }

    for result_file in result_files:
        result = data.Data(result_file)

        sparsity = result.params['sparsity']
        raw_data['cpu_base']['sparsity'].extend(result.data['CPU Baseline'].size * [sparsity])
        raw_data['cpu_base']['elapsed_time'].extend(result.data['CPU Baseline'])
        raw_data['cpu_parallel']['sparsity'].extend(result.data['CPU Parallel'].size * [sparsity])
        raw_data['cpu_parallel']['elapsed_time'].extend(result.data['CPU Parallel'])
        raw_data['gpu_base']['sparsity'].extend(result.data['GPU Baseline'].size * [sparsity])
        raw_data['gpu_base']['elapsed_time'].extend(result.data['GPU Baseline'])

    #
    # make dataframe (parallel, naive)
    #
    df_data = {
        'cpu_base': pd.DataFrame({'elapsed_time': raw_data['cpu_base']['elapsed_time'], 'sparsity': raw_data['cpu_base']['sparsity']}),
        'cpu_parallel': pd.DataFrame({'elapsed_time': raw_data['cpu_parallel']['elapsed_time'], 'sparsity': raw_data['cpu_parallel']['sparsity']}),
        'gpu_base': pd.DataFrame({'elapsed_time': raw_data['gpu_base']['elapsed_time'], 'sparsity': raw_data['gpu_base']['sparsity']})
    }

    #
    # quantiles (0.05, ..., 0.96)
    #
    quantiles = np.arange(0.05, 0.96, 0.1)

    #
    # quantile regression model
    #
    model = {
        'cpu_base': smf.quantreg("elapsed_time ~ sparsity", df_data['cpu_base']),
        'cpu_parallel': smf.quantreg("elapsed_time ~ sparsity", df_data['cpu_parallel']),
        'gpu_base': smf.quantreg("elapsed_time ~ sparsity", df_data['gpu_base'])
    }
    # print(model['cpu_base'].fit(q=0.5).summary())
    # print(model['cpu_parallel'].fit(q=0.5).summary())
    # print(model['gpu_base'].fit(q=0.5).summary())

    models = {
        'cpu_base': pd.DataFrame([fit_model(model['cpu_base'], x) for x in quantiles], columns=["q", "a", "b", "lb", "ub"]),
        'cpu_parallel': pd.DataFrame([fit_model(model['cpu_parallel'], x) for x in quantiles], columns=["q", "a", "b", "lb", "ub"]),
        'gpu_base': pd.DataFrame([fit_model(model['gpu_base'], x) for x in quantiles], columns=["q", "a", "b", "lb", "ub"])
    }
    # print(models['cpu_base'])
    # print(models['cpu_parallel'])
    # print(models['gpu_base'])

    #
    # Ordinary Least Squares (ols)
    #
    ols = {
        'cpu_base': smf.ols("elapsed_time ~ sparsity", df_data['cpu_base']).fit(),
        'cpu_parallel': smf.ols("elapsed_time ~ sparsity", df_data['cpu_parallel']).fit(),
        'gpu_base': smf.ols("elapsed_time ~ sparsity", df_data['gpu_base']).fit()
    }
    ols_ci = {
        'cpu_base': ols['cpu_base'].conf_int().loc["sparsity"].tolist(),
        'cpu_parallel': ols['cpu_parallel'].conf_int().loc["sparsity"].tolist(),
        'gpu_base': ols['gpu_base'].conf_int().loc["sparsity"].tolist()
    }
    ols_dict = {
        'cpu_base': dict(a=ols['cpu_base'].params["Intercept"], b=ols['cpu_base'].params["sparsity"],
                         lb=ols_ci['cpu_base'][0], ub=ols_ci['cpu_base'][1]),
        'cpu_parallel': dict(a=ols['cpu_parallel'].params["Intercept"], b=ols['cpu_parallel'].params["sparsity"],
                         lb=ols_ci['cpu_parallel'][0], ub=ols_ci['cpu_parallel'][1]),
        'gpu_base': dict(a=ols['gpu_base'].params["Intercept"], b=ols['gpu_base'].params["sparsity"],
                         lb=ols_ci['gpu_base'][0], ub=ols_ci['gpu_base'][1]),

    }
    print(ols_dict['cpu_base'])
    print(ols_dict['cpu_parallel'])
    print(ols_dict['gpu_base'])

    #
    # plot
    #
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(25, 15))

    x = np.arange(df_data['cpu_base'].sparsity.min(), df_data['cpu_base'].sparsity.max(), 30)
    get_y = lambda a, b: a + b * x

    for i in range(models['cpu_base'].shape[0]):
        y = get_y(models['cpu_base'].a[i], models['cpu_base'].b[i])
        axs[0, 0].plot(x, y, linestyle="dotted", color="grey")
    y = get_y(ols_dict['cpu_base']['a'], ols_dict['cpu_base']['b'])
    axs[0, 0].plot(x, y, color="blue", label="CPU Baseline")
    axs[0, 0].scatter(df_data['cpu_base'].sparsity, df_data['cpu_base'].elapsed_time, alpha=0.2)
    legend = axs[0, 0].legend()
    axs[0, 0].set_title("CPU Baseline", fontsize=16)
    axs[0, 0].set_xlabel("sparsity", fontsize=16)
    axs[0, 0].set_ylabel("elapsed time in [ns]", fontsize=16)

    for i in range(models['cpu_parallel'].shape[0]):
        y = get_y(models['cpu_parallel'].a[i], models['cpu_parallel'].b[i])
        axs[0, 1].plot(x, y, linestyle="dotted", color="grey")
    y = get_y(ols_dict['cpu_parallel']['a'], ols_dict['cpu_parallel']['b'])
    axs[0, 1].plot(x, y, color="blue", label="CPU Parallel")
    axs[0, 1].scatter(df_data['cpu_parallel'].sparsity, df_data['cpu_parallel'].elapsed_time, alpha=0.2)
    legend = axs[0, 1].legend()
    axs[0, 1].set_title("CPU Parallel", fontsize=16)
    axs[0, 1].set_xlabel("sparsity", fontsize=16)
    axs[0, 1].set_ylabel("elapsed time in [ns]", fontsize=16)

    for i in range(models['gpu_base'].shape[0]):
        y = get_y(models['gpu_base'].a[i], models['gpu_base'].b[i])
        axs[0, 2].plot(x, y, linestyle="dotted", color="grey")
    y = get_y(ols_dict['gpu_base']['a'], ols_dict['gpu_base']['b'])
    axs[0, 2].plot(x, y, color="blue", label="GPU Baseline")
    axs[0, 2].scatter(df_data['gpu_base'].sparsity, df_data['gpu_base'].elapsed_time, alpha=0.2)
    legend = axs[0, 2].legend()
    axs[0, 2].set_title("GPU Baseline", fontsize=16)
    axs[0, 2].set_xlabel("sparsity", fontsize=16)
    axs[0, 2].set_ylabel("elapsed time in [ns]", fontsize=16)

    n = models['cpu_base'].shape[0]
    axs[1, 0].plot(models['cpu_base'].q, models['cpu_base'].b, color="black", label="Quantile Reg.")
    axs[1, 0].plot(models['cpu_base'].q, models['cpu_base'].ub, linestyle="dotted", color="black")
    axs[1, 0].plot(models['cpu_base'].q, models['cpu_base'].lb, linestyle="dotted", color="black")
    axs[1, 0].plot(models['cpu_base'].q, [ols_dict['cpu_base']['b']] * n, color="red", label="OLS")
    axs[1, 0].plot(models['cpu_base'].q, [ols_dict['cpu_base']['lb']] * n, linestyle="dotted", color="red")
    axs[1, 0].plot(models['cpu_base'].q, [ols_dict['cpu_base']['ub']] * n, linestyle="dotted", color="red")
    axs[1, 0].set_ylabel(r"$\beta_{sparsity}$", fontsize=16)
    axs[1, 0].set_xlabel("Quantiles of elapsed time", fontsize=16)
    axs[1, 0].legend()

    n = models['cpu_parallel'].shape[0]
    axs[1, 1].plot(models['cpu_parallel'].q, models['cpu_parallel'].b, color="black", label="Quantile Reg.")
    axs[1, 1].plot(models['cpu_parallel'].q, models['cpu_parallel'].ub, linestyle="dotted", color="black")
    axs[1, 1].plot(models['cpu_parallel'].q, models['cpu_parallel'].lb, linestyle="dotted", color="black")
    axs[1, 1].plot(models['cpu_parallel'].q, [ols_dict['cpu_parallel']['b']] * n, color="red", label="OLS")
    axs[1, 1].plot(models['cpu_parallel'].q, [ols_dict['cpu_parallel']['lb']] * n, linestyle="dotted", color="red")
    axs[1, 1].plot(models['cpu_parallel'].q, [ols_dict['cpu_parallel']['ub']] * n, linestyle="dotted", color="red")
    axs[1, 1].set_ylabel(r"$\beta_{sparsity}$", fontsize=16)
    axs[1, 1].set_xlabel("Quantiles of elapsed time", fontsize=16)
    axs[1, 1].legend()

    n = models['gpu_base'].shape[0]
    axs[1, 2].plot(models['gpu_base'].q, models['gpu_base'].b, color="black", label="Quantile Reg.")
    axs[1, 2].plot(models['gpu_base'].q, models['gpu_base'].ub, linestyle="dotted", color="black")
    axs[1, 2].plot(models['gpu_base'].q, models['gpu_base'].lb, linestyle="dotted", color="black")
    axs[1, 2].plot(models['gpu_base'].q, [ols_dict['gpu_base']['b']] * n, color="red", label="OLS")
    axs[1, 2].plot(models['gpu_base'].q, [ols_dict['gpu_base']['lb']] * n, linestyle="dotted", color="red")
    axs[1, 2].plot(models['gpu_base'].q, [ols_dict['gpu_base']['ub']] * n, linestyle="dotted", color="red")
    axs[1, 2].set_ylabel(r"$\beta_{sparsity}$", fontsize=16)
    axs[1, 2].set_xlabel("Quantiles of elapsed time", fontsize=16)
    axs[1, 2].legend()

    plt.show()


if __name__ == '__main__':
    plot()
