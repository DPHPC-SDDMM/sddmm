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
    return [q, res.params["Intercept"], res.params["n_thread"]] + res.conf_int().loc["n_thread"].tolist()


def plot():
    #
    # get files of results directory
    #
    result_path = './results'
    result_files = [f for f in listdir(result_path) if isfile(join(result_path, f))]

    #
    # reformat result data to (thread, duration)
    #
    n_thread = []
    p_time = []
    n_time = []
    for result_file in result_files:
        result = data.Data(result_file)

        n_thread.append(result.params['n_cpu_threads'])
        p_time.append(np.average(result.data['parallel (CPU)']))
        n_time.append(np.average(result.data['naive (COO,CPU)']))

    #
    # make dataframe (parallel, naive)
    #
    df_parallel = pd.DataFrame(columns=['elapsed_time', 'n_thread'])
    df_parallel['elapsed_time'] = p_time
    df_parallel['n_thread'] = n_thread
    print(df_parallel.head())

    df_naive = pd.DataFrame(columns=['elapsed_time', 'n_thread'])
    df_naive['elapsed_time'] = n_time
    df_naive['n_thread'] = n_thread
    print(df_naive.head())

    #
    # quantiles (0.05, ..., 0.96)
    #
    quantiles = np.arange(0.05, 0.96, 0.1)

    #
    # quantile regression model
    #
    p_model = smf.quantreg("elapsed_time ~ n_thread", df_parallel)
    # print(p_model.fit(q=0.5).summary())

    n_model = smf.quantreg("elapsed_time ~ n_thread", df_naive)
    # print(n_model.fit(q=0.5).summary())

    p_models = [fit_model(p_model, x) for x in quantiles]
    p_models = pd.DataFrame(p_models, columns=["q", "a", "b", "lb", "ub"])
    print(p_models)

    n_models = [fit_model(n_model, x) for x in quantiles]
    n_models = pd.DataFrame(n_models, columns=["q", "a", "b", "lb", "ub"])
    print(n_models)

    #
    # Ordinary Least Squares (ols)
    #
    p_ols = smf.ols("elapsed_time ~ n_thread", df_parallel).fit()
    p_ols_ci = p_ols.conf_int().loc["n_thread"].tolist()
    p_ols = dict(a=p_ols.params["Intercept"], b=p_ols.params["n_thread"], lb=p_ols_ci[0], ub=p_ols_ci[1])
    print(p_ols)

    n_ols = smf.ols("elapsed_time ~ n_thread", df_naive).fit()
    n_ols_ci = n_ols.conf_int().loc["n_thread"].tolist()
    n_ols = dict(a=n_ols.params["Intercept"], b=n_ols.params["n_thread"], lb=n_ols_ci[0], ub=n_ols_ci[1])
    print(n_ols)

    #
    # plot
    #
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 11), sharex="row", sharey="row")

    get_y = lambda a, b: a + b * x

    x = np.arange(df_parallel.n_thread.min(), df_parallel.n_thread.max(), 30)
    for i in range(p_models.shape[0]):
        y = get_y(p_models.a[i], p_models.b[i])
        axs[0, 0].plot(x, y, linestyle="dotted", color="grey")
    y = get_y(p_ols["a"], p_ols["b"])
    axs[0, 0].plot(x, y, color="red", label="OLS")
    axs[0, 0].scatter(df_parallel.n_thread, df_parallel.elapsed_time, alpha=0.2)
    axs[0, 0].set_xlim((0, 30))
    axs[0, 0].set_ylim((0, 1000000))
    legend = axs[0, 0].legend()
    axs[0, 0].set_xlabel("n_thread", fontsize=16)
    axs[0, 0].set_ylabel("parallel elapsed time", fontsize=16)

    x = np.arange(df_naive.n_thread.min(), df_naive.n_thread.max(), 30)
    for i in range(n_models.shape[0]):
        y = get_y(n_models.a[i], n_models.b[i])
        axs[0, 1].plot(x, y, linestyle="dotted", color="grey")
    y = get_y(n_ols["a"], n_ols["b"])
    axs[0, 1].plot(x, y, color="red", label="OLS")
    axs[0, 1].scatter(df_naive.n_thread, df_naive.elapsed_time, alpha=0.2)
    axs[0, 1].set_xlim((0, 30))
    axs[0, 1].set_ylim((0, 1000000))
    legend = axs[0, 1].legend()
    axs[0, 1].set_xlabel("n_thread", fontsize=16)
    axs[0, 1].set_ylabel("naive elapsed time", fontsize=16)

    n = p_models.shape[0]
    axs[1, 0].plot(p_models.q, p_models.b, color="black", label="Quantile Reg.")
    axs[1, 0].plot(p_models.q, p_models.ub, linestyle="dotted", color="black")
    axs[1, 0].plot(p_models.q, p_models.lb, linestyle="dotted", color="black")
    axs[1, 0].plot(p_models.q, [p_ols["b"]] * n, color="red", label="OLS")
    axs[1, 0].plot(p_models.q, [p_ols["lb"]] * n, linestyle="dotted", color="red")
    axs[1, 0].plot(p_models.q, [p_ols["ub"]] * n, linestyle="dotted", color="red")
    axs[1, 0].set_ylabel("parallel elapsed time", fontsize=16)
    axs[1, 0].set_xlabel("Quantiles of threads", fontsize=16)
    axs[1, 0].legend()

    n = n_models.shape[0]
    axs[1, 1].plot(n_models.q, n_models.b, color="black", label="Quantile Reg.")
    axs[1, 1].plot(n_models.q, n_models.ub, linestyle="dotted", color="black")
    axs[1, 1].plot(n_models.q, n_models.lb, linestyle="dotted", color="black")
    axs[1, 1].plot(n_models.q, [n_ols["b"]] * n, color="red", label="OLS")
    axs[1, 1].plot(n_models.q, [n_ols["lb"]] * n, linestyle="dotted", color="red")
    axs[1, 1].plot(n_models.q, [n_ols["ub"]] * n, linestyle="dotted", color="red")
    axs[1, 1].set_ylabel("naive elapsed time", fontsize=16)
    axs[1, 1].set_xlabel("Quantiles of threads", fontsize=16)
    axs[1, 1].legend()

    plt.show()


if __name__ == '__main__':
    plot()
