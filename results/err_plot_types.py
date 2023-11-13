import matplotlib.pyplot as plt
import data
import numpy as np
import re

def break_lines(string, line_length):
    parts = string.split(' ')
    res = ""
    last = 0
    for p in parts:
        if len(res) + len(p) - last > line_length:
            res+='\n'
            last = len(res)
        elif len(res)>0:
            res+=' '
        res+=p
    return res


if __name__ == '__main__':
    data_file = 'Vec vs PP vs C benchmarks__rTS0_cTS0_rNum8192_cNum2048_nIt250_[Mon Nov 13 16-30-46 2023].txt'
    d_debug = data.Data(data_file)

    x_debug = [break_lines(s, 10) for s in  d_debug.data.keys()]
    y_debug = np.array([np.median(dd) for dd in d_debug.data.values()]) / 1.0
    # y_std_debug = np.array([abs(np.max(dd) - np.min(dd)) for dd in d_debug.data.values()]) / 1000.0
    y_std_debug = np.array([np.std(dd) for dd in d_debug.data.values()]) / 1.0
    p_y_std_debug = np.round(100.0 / y_debug * y_std_debug, 3)

    y_debug = np.round(y_debug,3)
    y_std_debug = np.round(y_std_debug,3)

    plt.errorbar(x_debug, y_debug, yerr=y_std_debug, color='blue', ecolor='green')
    for i in range(len(x_debug)):
        plt.text(i, y_debug[i], str(y_debug[i]) + "[us] +-" + str(p_y_std_debug[i]) + "%")

    data_file = 'Vec vs PP vs C benchmarks [Release]__rTS0_cTS0_rNum8192_cNum2048_nIt250_[Mon Nov 13 16-31-45 2023].txt'
    d_release = data.Data(data_file)

    x_release = [break_lines(s, 10) for s in  d_release.data.keys()]
    y_release = np.array([np.average(dd) for dd in d_release.data.values()]) / 1.0
    y_std_release = np.array([np.std(dd) for dd in d_release.data.values()]) / 1.0
    p_y_std_release = np.round(100.0 / y_release * y_std_release, 3)

    y_release = np.round(y_release,3)
    y_std_release = np.round(y_std_release,3)

    plt.errorbar(x_release, y_release, yerr=y_std_release, color='red', ecolor='orange')
    for i in range(len(x_release)):
        plt.text(i, y_release[i], str(y_release[i]) + "[us] +-" + str(p_y_std_release[i]) + "%")

    plt.title("{0} n_experiments = {1}".format(d_debug.params["experiment_name"], d_debug.params['n_experiment_iterations']))
    plt.xlabel("Experiment")
    plt.ylabel("Runtime [us]")

    plt.show()
