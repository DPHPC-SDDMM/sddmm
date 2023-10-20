#include <iostream>
#include "algos/algo.h"

// ################################################################################
// ################################################################################
// ██████  ███████ ██████  ██████  ███████  ██████  █████  ████████ ███████ ██████  
// ██   ██ ██      ██   ██ ██   ██ ██      ██      ██   ██    ██    ██      ██   ██ 
// ██   ██ █████   ██████  ██████  █████   ██      ███████    ██    █████   ██   ██ 
// ██   ██ ██      ██      ██   ██ ██      ██      ██   ██    ██    ██      ██   ██ 
// ██████  ███████ ██      ██   ██ ███████  ██████ ██   ██    ██    ███████ ██████ 
// ################################################################################
// # Create a new "whatever.cpp" with a main method for each experiment inside
// # the toplevel CMakeLists.txt and instantiate the experiment in there. 
// # See "plot_sample.cpp" as example
// ################################################################################ 
                                                                                 
                                                                                 

/**
 * All algos as cpp files
*/
#include "algos/algo_sample.cpp"

/**
 * This is the algo that will run
*/
#define SAMPLE_ALGO

int main(int argc, char** argv){
    /**
     * Perform some global initialization code here
     * Add results into InitParams and pass to subsequent algos
    */
    const SDDMM::InitParams initParams {
        .sampleParam=5
    };

    /**
     * TODO: Probably we may want to add if/else if/else based on some string here
     *  => could be more convenient
     * Alternative would be to set "#define SAMPLE_ALGO" inside CMakeLists.txt
    */
#ifdef SAMPLE_ALGO
    SDDMM::AlgoSample().main(argc, argv, initParams);
#endif

}
