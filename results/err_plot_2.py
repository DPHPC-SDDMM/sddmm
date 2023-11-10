import matplotlib.pyplot as plt
import data
import numpy as np
 
if __name__ == '__main__':
    data_file = 'test_benchmark<NxK,KxM>Had<NxM>N500_M500_K800_sparsity-0.1_iters-50_cpu-t-24_[Fri_Nov_10_15:29:07_2023].txt'
    d = data.Data(data_file)

    x = d.data.keys()
    y = np.array([np.average(dd) for dd in d.data.values()]) / 1000000.0
    y_std = np.array([np.std(dd) for dd in d.data.values()]) / 1000000.0
    p_y_std = np.round(100.0 / y * y_std, 3)

    y = np.round(y,3)
    y_std = np.round(y_std,3)

    plt.errorbar(x, y, yerr=y_std)
    
    for i in range(len(x)):
        plt.text(i, y[i]*1.01, str(y[i]) + "[ms] +-" + str(p_y_std[i]) + "%")

    plt.title("SDDMM n_experiments = {0}".format(d.params['n_experiment_iterations']))
    plt.xlabel("Experiment")
    plt.ylabel("Runtime [ns]")

    plt.show()
