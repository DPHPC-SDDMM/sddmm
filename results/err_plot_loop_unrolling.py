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
    # data_file = 'Serial unrolling benchmark_tsR-50_tsI-30_tsC-60_dsR-1000_dsI-1500_dsC-2000_nIt-10_[Sat_Nov_11_15:58:42_2023].txt'
    # data_file = 'Serial unrolling benchmark_tsR-50_tsI-30_tsC-60_dsR-100_dsI-150_dsC-200_nIt-10_[Sat_Nov_11_16:00:15_2023].txt'
    # data_file = 'Serial unrolling benchmark_tsR-50_tsI-30_tsC-60_dsR-100_dsI-300_dsC-250_nIt-10_[Sat_Nov_11_16:02:36_2023].txt'
    # data_file = 'Serial unrolling benchmark_tsR-50_tsI-30_tsC-60_dsR-200_dsI-600_dsC-500_nIt-10_[Sat_Nov_11_16:07:43_2023].txt'
    # data_file = 'Serial unrolling benchmark_tsR-50_tsI-30_tsC-60_dsR-200_dsI-600_dsC-500_nIt-50_[Sat_Nov_11_16:09:45_2023].txt'
    # data_file = 'Serial unrolling benchmark_tsR-50_tsI-30_tsC-60_dsR-200_dsI-600_dsC-500_nIt-50_[Sat_Nov_11_16:40:02_2023].txt'
    # data_file = 'Serial unrolling benchmark_tsR-50_tsI-30_tsC-60_dsR-200_dsI-600_dsC-500_nIt-50_[Sat_Nov_11_17:04:13_2023].txt'
    # data_file = 'Serial unrolling benchmark_tsR-50_tsI-30_tsC-60_dsR-200_dsI-600_dsC-500_nIt-50_[Sat_Nov_11_17:12:08_2023].txt'
    # data_file = 'Serial unrolling benchmark_tsR-50_tsI-30_tsC-60_dsR-256_dsI-512_dsC-384_nIt-50_[Sat_Nov_11_18:01:07_2023].txt'
    # data_file = 'Serial unrolling benchmark_tsR-50_tsI-30_tsC-60_dsR-256_dsI-500_dsC-384_nIt-50_[Sat_Nov_11_18:34:32_2023].txt'
    data_file = 'Serial unrolling benchmark_tsR-50_tsI-30_tsC-60_dsR-512_dsI-1024_dsC-384_nIt-100_[Sat_Nov_11_18:59:27_2023].txt'
    # data_file = 'Serial unrolling benchmark_tsR-50_tsI-30_tsC-60_dsR-300_dsI-900_dsC-750_nIt-10_[Sat_Nov_11_16:04:24_2023].txt'
    d = data.Data(data_file)

    x = [break_lines(s, 10) for s in  d.data.keys()]
    y = np.array([np.average(dd) for dd in d.data.values()]) / 1000.0
    y_std = np.array([np.std(dd) for dd in d.data.values()]) / 1000.0
    p_y_std = np.round(100.0 / y * y_std, 3)

    y = np.round(y,3)
    y_std = np.round(y_std,3)

    plt.errorbar(x, y, yerr=y_std)
    
    for i in range(len(x)):
        plt.text(i, y[i], str(y[i]) + "[us] +-" + str(p_y_std[i]) + "%")

    plt.title("{0} n_experiments = {1}".format(d.params["experiment_name"], d.params['n_experiment_iterations']))
    plt.xlabel("Experiment")
    plt.ylabel("Runtime [us]")

    plt.show()
