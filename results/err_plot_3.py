import matplotlib.pyplot as plt
import data
import numpy as np
 
if __name__ == '__main__':
    # https://www.geeksforgeeks.org/adding-value-labels-on-a-matplotlib-bar-chart/

    data_files = [
        # 'parallel_sddmm__[NxK,KxM]Had[NxM]N500_M500_K400_sparsity-0.1_iters-25_cpu-t-2_[Mon Nov 20 11-04-50 2023].txt',
        # 'parallel_sddmm__[NxK,KxM]Had[NxM]N500_M500_K400_sparsity-0.1_iters-25_cpu-t-2_[Mon Nov 20 11-08-33 2023].txt',
        # 'parallel_sddmm__[NxK,KxM]Had[NxM]N500_M500_K400_sparsity-0.1_iters-25_cpu-t-2_[Mon Nov 20 11-59-44 2023].txt',
        # 'parallel_sddmm__[NxK,KxM]Had[NxM]N500_M500_K400_sparsity-0.1_iters-25_cpu-t-2_[Mon Nov 20 12-02-16 2023].txt',
        # 'parallel_sddmm__[NxK,KxM]Had[NxM]N500_M500_K400_sparsity-0.1_iters-25_cpu-t-8_[Mon Nov 20 12-04-38 2023].txt',
        # 'parallel_sddmm__[NxK,KxM]Had[NxM]N500_M500_K400_sparsity-0.1_iters-25_cpu-t-8_[Mon Nov 20 12-06-44 2023].txt',
        # 'parallel_sddmm__[NxK,KxM]Had[NxM]N500_M500_K400_sparsity-0.1_iters-25_cpu-t-8_[Mon Nov 20 12-15-25 2023].txt',
        # 'parallel_sddmm__[NxK,KxM]Had[NxM]N500_M500_K400_sparsity-0.1_iters-25_cpu-t-8_[Mon Nov 20 12-18-04 2023].txt',
        # 'parallel_sddmm [Release]__[NxK,KxM]Had[NxM]N500_M500_K400_sparsity-0.1_iters-100_cpu-t-13_[Tue Nov 21 11-31-43 2023].txt',
        # 'parallel_sddmm [Release]__[NxK,KxM]Had[NxM]N500_M500_K400_sparsity-0.1_iters-100_cpu-t-13_[Tue Nov 21 11-33-46 2023].txt',
        # 'parallel_sddmm [Release]__[NxK,KxM]Had[NxM]N500_M500_K400_sparsity-0.1_iters-100_cpu-t-13_[Tue Nov 21 11-43-44 2023].txt',
        # 'parallel_sddmm [Release]__[NxK,KxM]Had[NxM]N500_M500_K400_sparsity-0.1_iters-100_cpu-t-13_[Tue Nov 21 11-51-49 2023].txt',
        # 'parallel_sddmm [Release]__[NxK,KxM]Had[NxM]N500_M500_K400_sparsity-0.1_iters-100_cpu-t-13_[Tue Nov 21 11-53-14 2023].txt',
        # 'parallel_sddmm [Release]__[NxK,KxM]Had[NxM]N500_M500_K400_sparsity-0.1_iters-300_cpu-t-13_[Tue Nov 21 11-54-38 2023].txt'
        # 'parallel_sddmm [Release]__[NxK,KxM]Had[NxM]N500_M500_K400_sparsity-0.1_iters-300_cpu-t-13_[Tue Nov 21 12-19-50 2023].txt'
        # 'parallel_sddmm [Release]__[NxK,KxM]Had[NxM]N500_M500_K400_sparsity-0.1_iters-300_cpu-t-13_[Tue Nov 21 12-26-12 2023].txt'
        # 'parallel_sddmm [Release]__[NxK,KxM]Had[NxM]N500_M500_K400_sparsity-0.1_iters-300_cpu-t-13_[Tue Nov 21 12-37-56 2023].txt',
        # 'parallel_sddmm [Release]__[NxK,KxM]Had[NxM]N500_M500_K400_sparsity-0.1_iters-300_cpu-t-13_[Tue Nov 21 12-40-04 2023].txt',
        # 'parallel_sddmm [Release]__[NxK,KxM]Had[NxM]N500_M500_K400_sparsity-0.1_iters-300_cpu-t-13_[Tue Nov 21 12-43-17 2023].txt',
        # 'parallel_sddmm [Release]__[NxK,KxM]Had[NxM]N500_M500_K400_sparsity-0.1_iters-300_cpu-t-13_[Tue Nov 21 12-47-29 2023].txt'
        # 'parallel_sddmm [Release]__[NxK,KxM]Had[NxM]N500_M500_K400_sparsity-0.1_iters-300_cpu-t-13_[Tue Nov 21 12-50-45 2023].txt',
        # 'parallel_sddmm [Release]__[NxK,KxM]Had[NxM]N500_M500_K400_sparsity-0.1_iters-300_cpu-t-13_[Tue Nov 21 12-52-37 2023].txt',
        # 'parallel_sddmm [Release]__[NxK,KxM]Had[NxM]N500_M500_K400_sparsity-0.1_iters-300_cpu-t-13_[Tue Nov 21 12-54-15 2023].txt',
        # 'parallel_sddmm [Release]__[NxK,KxM]Had[NxM]N500_M500_K400_sparsity-0.1_iters-300_cpu-t-13_[Tue Nov 21 12-56-18 2023].txt',
        # 'parallel_sddmm [Release]__[NxK,KxM]Had[NxM]N500_M500_K400_sparsity-0.1_iters-300_cpu-t-13_[Tue Nov 21 12-57-39 2023].txt',
        # 'parallel_sddmm [Release]__[NxK,KxM]Had[NxM]N500_M500_K400_sparsity-0.1_iters-300_cpu-t-13_[Tue Nov 21 12-59-35 2023].txt',
        # 'parallel_sddmm [Release]__[NxK,KxM]Had[NxM]N500_M500_K400_sparsity-0.1_iters-300_cpu-t-13_[Tue Nov 21 13-01-19 2023].txt'
        # 'parallel_sddmm [Release]__[NxK,KxM]Had[NxM]N500_M500_K400_sparsity-0.1_iters-300_cpu-t-13_[Tue Nov 21 13-03-32 2023].txt'
        # 'parallel_sddmm [Release]__[NxK,KxM]Had[NxM]N500_M500_K400_sparsity-0.1_iters-300_cpu-t-13_[Tue Nov 21 13-59-25 2023].txt'
        # 'parallel_sddmm [Release]__[NxK,KxM]Had[NxM]N500_M500_K400_sparsity-0.1_iters-300_cpu-t-13_[Tue Nov 21 14-04-44 2023].txt'
        'parallel_sddmm [Release]__[NxK,KxM]Had[NxM]N500_M500_K400_sparsity-0.1_iters-300_cpu-t-13_[Tue Nov 21 14-08-30 2023].txt',
        'parallel_sddmm [Release]__[NxK,KxM]Had[NxM]N500_M500_K400_sparsity-0.1_iters-300_cpu-t-13_[Tue Nov 21 14-12-14 2023].txt',
        'parallel_sddmm [Release]__[NxK,KxM]Had[NxM]N500_M500_K400_sparsity-0.1_iters-300_cpu-t-13_[Tue Nov 21 14-16-49 2023].txt',
        'parallel_sddmm [Release]__[NxK,KxM]Had[NxM]N500_M500_K400_sparsity-0.1_iters-300_cpu-t-13_[Tue Nov 21 14-20-22 2023].txt',
        'parallel_sddmm [Release]__[NxK,KxM]Had[NxM]N500_M500_K400_sparsity-0.1_iters-300_cpu-t-13_[Tue Nov 21 14-23-02 2023].txt'
    ]
    p_ind = 1
    for data_file in data_files:
        d = data.Data(data_file)

        if data_file == data_files[-1]:
            x = [data.Data.break_lines(s, 20) for s in  d.data.keys()]
        else:
            x = [ind for ind,s in  enumerate(d.data.keys())]
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
