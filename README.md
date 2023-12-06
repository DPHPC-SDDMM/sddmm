# sddmm

### Information for vscode and cmake
https://code.visualstudio.com/docs/cpp/cmake-linux

### Howto set build folders debug/release for builds
* got vscode extension installation page
* select CMake Tools
* find **settings** cog-wheel next to installation buttions
* at option **Build Directory** set value to ${workspaceFolder}/build/${buildType}

### Install gnuplot
http://www.gnuplot.info/download.html

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

### cuSPARSE examples
https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuSPARSE
