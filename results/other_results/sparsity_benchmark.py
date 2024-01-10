import matplotlib.pyplot as plt
import data
import numpy as np

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
    # https://www.geeksforgeeks.org/adding-value-labels-on-a-matplotlib-bar-chart/

    data_file = 'Comparison Benchmark of Algorithms based on Sparsity__vs4-es4__[NxK,KxM]Had[NxM]N1000_M1000_K1000_sparsity-0.9_iters-10_cpu-t-17_[Fri Nov 24 10-31-43 2023].txt'
    d = data.Data(data_file)

    x = [break_lines(s, 8) for s in  d.data.keys()]
    y = np.array([np.average(dd) for dd in d.data.values()])
    y_std = np.array([round(np.std(dd), 3) for dd in d.data.values()])
    plt.subplot(2,1,1)
    plt.bar(x, y)
    
    for i in range(len(x)):
        plt.text(i-0.4, y[i]*1.01, str(y[i]) + "[ns]")

    plt.title("SDDMM n_experiments = {0}".format(d.params['n_experiment_iterations']))
    plt.xlabel("Experiment")
    plt.ylabel("Runtime [ns]")

    plt.subplot(2,1,2)
    plt.bar(x, y_std)
    
    p_y_std = np.round(100.0 / y * y_std, 3)
    for i in range(len(x)):
        plt.text(i-0.4, y_std[i]*1.01, str(y_std[i]) + " +- " + str(p_y_std[i]) + "[ns]")

    plt.xlabel("Experiment")
    plt.ylabel("Standard deviation in [ns]")

    plt.show()
