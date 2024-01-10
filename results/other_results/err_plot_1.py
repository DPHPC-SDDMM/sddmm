import matplotlib.pyplot as plt
import data
import numpy as np
 
if __name__ == '__main__':
    # https://www.geeksforgeeks.org/adding-value-labels-on-a-matplotlib-bar-chart/

    data_file = 'test_benchmark__[NxK,KxM]Had[NxM]N500_M500_K800_sparsity-0.1_iters-50_cpu-t-24_[Fri Nov 10 15-29-07 2023].txt'
    d = data.Data(data_file)

    x = d.data.keys()
    # y = np.array([np.average(dd) for dd in d.data.values()])
    # y_std = np.array([round(np.std(dd), 3) for dd in d.data.values()])
    # plt.subplot(2,1,1)
    # plt.bar(x, y)
    plt.hist(d.data[list(d.data.keys())[0]])
    
    # for i in range(len(x)):
    #     plt.text(i-0.4, y[i]*1.01, str(y[i]) + "[ns]")

    # plt.title("SDDMM n_experiments = {0}".format(d.params['n_experiment_iterations']))
    # plt.xlabel("Experiment")
    # plt.ylabel("Runtime [ns]")

    # plt.subplot(2,1,2)
    # plt.bar(x, y_std)
    
    # p_y_std = np.round(100.0 / y * y_std, 3)
    # for i in range(len(x)):
    #     plt.text(i-0.4, y_std[i]*1.01, str(y_std[i]) + " +- " + str(p_y_std[i]) + "[ns]")

    # plt.xlabel("Experiment")
    # plt.ylabel("Standard deviation in [ns]")

    plt.show()
