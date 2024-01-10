import matplotlib.pyplot as plt
import data
import numpy as np
 
if __name__ == '__main__':
    # https://www.geeksforgeeks.org/adding-value-labels-on-a-matplotlib-bar-chart/

    data_files = [
        'parallel_cpu_sddmm [Release]__vs4-es4__[NxK,KxM]Had[NxM]N500_M500_K400_sparsity-0.1_iters-300_cpu-t-13_[Tue Nov 21 18-39-57 2023].txt'
    ]
    p_ind = 1
    for data_file in data_files:
        d = data.Data(data_file)

        x = [data.Data.break_lines(s, 20) for s in  d.data.keys()]

        y = np.array([np.average(dd) for dd in d.data.values()])
        y_std = np.array([round(np.std(dd), 3) for dd in d.data.values()])
        plt.subplot(len(data_files),1,p_ind)
        plt.bar(x, y)
        
        for i in range(len(x)):
            plt.text(i-0.4, y[i]*1.01, str(y[i]) + "[ns]")
        p_ind+=1

    plt.title("SDDMM n_experiments = {0}".format(d.params['n_experiment_iterations']))
    plt.xlabel("Experiment")
    plt.ylabel("Runtime [ns]")

    # plt.subplot(2,1,2)
    # plt.bar(x, y_std)
    
    # p_y_std = np.round(100.0 / y * y_std, 3)
    # for i in range(len(x)):
    #     plt.text(i-0.4, y_std[i]*1.01, str(y_std[i]) + " +- " + str(p_y_std[i]) + "[ns]")

    # plt.xlabel("Experiment")
    # plt.ylabel("Standard deviation in [ns]")

    plt.show()
