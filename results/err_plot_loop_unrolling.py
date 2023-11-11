import matplotlib.pyplot as plt
import data
import numpy as np
 
if __name__ == '__main__':
    data_file = 'Serial unrolling benchmark_tsR-50_tsI-30_tsC-60_dsR-1000_dsI-1500_dsC-2000_nIt-10_[Sat_Nov_11_15:38:04_2023].txt'
    d = data.Data(data_file)

    x = d.data.keys()
    y = np.array([np.average(dd) for dd in d.data.values()]) / 1000000.0
    y_std = np.array([np.std(dd) for dd in d.data.values()]) / 1000000.0
    p_y_std = np.round(100.0 / y * y_std, 3)

    y = np.round(y,3)
    y_std = np.round(y_std,3)

    plt.errorbar(x, y, yerr=y_std)
    
    for i in range(len(x)):
        plt.text(i, y[i], str(y[i]) + "[ms] +-" + str(p_y_std[i]) + "%")

    plt.title("{0} n_experiments = {1}".format(d.params["experiment_name"], d.params['n_experiment_iterations']))
    plt.xlabel("Experiment")
    plt.ylabel("Runtime [ns]")

    plt.show()
