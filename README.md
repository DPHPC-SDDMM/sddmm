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
8. (This step is optional) Run generate_all_images.bat. This will create distribution images of all the generated, large sparse matrices. The images are on a scale from blue to red, where blue indicates the lowest and red the highest probability to encounter a nonzero value. The images have the same name as their respective .binmat file and are located in the same folder as well.

   Low sparsity uniform matrix  |  High sparsity uniform matrix
   :---------------------------:|:------------------------------:|
   ![](low-sparsity-uniform-distribution.jpg)  |  ![](high-sparsity-uniform-distribution.jpg)
   
   IMDB           |  patents           |  patents_main
   :-------------:|:-------------------|:----------------------:|
   ![](imdb.jpg)  |  ![](patents.jpg)  |  ![](patents-main.jpg)

10.  Run run_all_benchmarks.bat. This will run all the benchmark tests and generate in each one of the folders that contain .binmat files corresponding .txt files with the results. Each txt file is standalone and can be plotted individually.
11. Copy all the results files (all desired .txt files from all data subfolders) into ./results/analysis/[some_expressive_folder_name]. ./results/analysis contains two Python scripts data.py and plots.py. Open plots.py and at the bottom call the plot(..) function with the correct path to your result files

### cuSPARSE examples
https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuSPARSE

# Howto run and compile the code with VSCode
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

