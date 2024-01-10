# SDDMM Benchmarking of SM-L2 algorithm vs a naive version

### Setup
The benchmarking framework is set up to work on Windows 11, Visual Studio Community Edition 2022, CMake and C++20 standard. It probably also works on Linux since GCC compiler tends to be more forgiving in terms of obeying to C++ specifications. Note, that if you run this on Linux, you have to double-check that the paths always use the correct kind of path separators.

### Steps to reproduce
1. Install Nvidia CUDA framework
2. Open the repository in Visual Studio and run CMake config
3. If CMake config ends with an error, add the .cuda_arch file with the correct CUDA architecture as indicated in the CMake config output. Make sure, that there is NO newline after the architecture number and run the CMake config again
4. Build all exes in Visual Studio (make sure it's set to release mode)
5. Open ./sddmm_data/init.py and run it. This will copy the folder structure to create the benchmark data onto C drive (Note: if you use Linux, you have to change everything to a Linux folder structure :-D ) and copy all relevant exe files into C:/sddmm_data as well.
6. Open the folder C:/sddmm_data/data_sets where you will see IMDB, patents and patents_main subfolders. Inside each of those folders is a readme.txt describing how to download and unpack the data for the existing large sparse matrices. At the end of this step, each one of the mm_market subfolders should directly contain the martix market data files.
7. Run create_data_sets.bat, create_sparsity_large_2.bat and create_sparsity_small.bat indivisually inside cmd or Powershell to create all the test data. At the end of this step, each one of the subfolders inside data_sets and inside sparsity_large_2 and sparsity_small there should be a number of .binmat files that each contain two dense and one sparse matrix.
8. (This step is optional) Run generate_all_images.bat. This will create distribution images of all the generated, large sparse matrices. The images are on a scale from blue to red, where blue indicates the lowest and red the highest probability to encounter a nonzero value. The images have the same name as their respective .binmat file and are located in the same folder as well. Below are some samples of how the distributions look like.

   Low sparsity uniform matrix                               |  High sparsity uniform matrix
   :--------------------------------------------------------:|:----------------------------------------------------------:|
   ![](sample_images/low-sparsity-uniform-distribution.jpg)  |  ![](sample_images/high-sparsity-uniform-distribution.jpg)
   
   IMDB                         |  patents                         |  patents_main
   :---------------------------:|:---------------------------------|:------------------------------------:|
   ![](sample_images/imdb.jpg)  |  ![](sample_images/patents.jpg)  |  ![](sample_images/patents-main.jpg)

10.  Run run_all_benchmarks.bat. This will run all the benchmark tests and generate in each one of the folders that contain .binmat files corresponding .txt files with the results. Each txt file belongs to a sequence of experiments and by itself will generate one data point in the output plots. One results data file looks as follows:
```
    [INFO]
    experiment_name imdb
    variable K
    N 428440
    M 896308
    K 32
    sparsity 0.99999
    description Compare matrices with K=[32,128,256] for IMDB data set
    runtime 22
    n_warmup_iterations 5
    sequence_number 1
    [/INFO]
    [DATA]
    [L] Baseline
    [D] 2313 2327 2345 2330 2296 2286 2314 2926 2294 2961 14949 14946 16505 14781 15012 15061 14919 16717 14861 14692 15015 14732 14975 14836 14908 14958 14748 17525 14655 16485
    [L] cuSPARSE
    [D] 1677 1868 1939 1882 1888 1902 1554 2015 1543 2058 1614 1439 1583 1585 1568 1611 1582 1541 1595 1485 1573 1532 1478 1805 1460 1526 1910 1404 1554 1581
    [L] sm_l2
    [D] 1166 1021 1159 1138 1159 1158 1239 1169 1028 1180 1039 1168 1158 1273 1161 1180 1171 1138 1033 1025 1026 1162 1170 1161 1153 1178 1027 1159 1162 1153
    [/DATA]
```
 * The **[INFO]** section contains all information on how the test was generated.
   * **experiment_name** is the overall name of the benchmarking experiment
   * **variable** field contains the name of the variable that was varied in the benchmark. In this case this was the inner dimension K. 
   * **N, M, K** which are the dimensions of the matrices where dense A and B matrices are in NxK and KxM respetively and sparse matrix S is in NxM. 
   * **sparsity** indicates in this case that 99.999% of all entries in the sparse matrix are zero. 
   * **description** outlines the benchmark.
   * **runtime** indicates the runtime of this particular data point, meaning how long in *seconds* it took to run all algorithms in the benchmark with this particular data for x amount of times (wher x == length(one of the arrays with the numbers below))
   * **n_warmup_iterations** indicates how many times the data point was run without recording the data
   * **sequence_numner** indicates the place this data file has inside the entire benchmark run with multiple data points. In this case, this is the first results file.
 * The **[DATA]** section contains all the time measurements
   * each measurement consists of an L and a D part where the L contains the name of the algorithm and D the the time measurements for all the test runs
   * In this case, there were three algorithms 'Baseline', 'cuSPARSE' and 'sm_l2' and all of them were run 30 times with the same input data

11. Copy all the results files (all desired .txt files from all data subfolders) into ./results/analysis/[some_expressive_folder_name]. ./results/analysis contains two Python scripts data.py and plots.py. Open plots.py and at the bottom call the plot(..) function with the correct path to your result files. The following evaluation plots will be generated.

    Plot 1                                      |  Plot 2                                       |  Plot 3                              | Plot 4             
   :-------------------------------------------:|:----------------------------------------------|:------------------------------------:|:-------------------------------------------:|
   ![](sample_images/imdb-100-iters-plot1.png)  |  ![](sample_images/imdb-100-iters-plot2.png)  |  ![](sample_images/imdb-100-iters-plot3.png)  |   ![](sample_images/imdb-100-iters-plot4.png)


### cuSPARSE examples
https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuSPARSE

# How to run and compile the code with VSCode
### Information for vscode and cmake
https://code.visualstudio.com/docs/cpp/cmake-linux

### Howto set build folders debug/release for builds
* got vscode extension installation page
* select CMake Tools
* find **settings** cog-wheel next to installation buttions
* at option **Build Directory** set value to ${workspaceFolder}/build/${buildType}

### VSCode + CMake
* pres ctrl+p
* type "CMake:" in the search bar that shows up
* Relevant are
    * CMake: build, CMake: clean, CMake: Delete Cache and Reconfigure
* Deleting the "build" folder will cause CMake to have to rebuild everything
* In the first build run, matplotplusplus and all listed targets have to be built
* In subsequent build runs, only the missing stuff will be built
* Note, that the switch for Debug/Release mode is at the **bottom** of VSCode in a tiny box named "CMake: [Debug/Release]: Ready" XD

### Expected results
There should be a build folder containing either Debug, Release or both containing all executables that are listed as targets inside the toplevel CMakeLists.txt

