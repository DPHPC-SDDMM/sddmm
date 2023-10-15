#pragma once

#include "algo.h" // *.h files need "..." and the relative path
#include <iostream> // all other files need <...>

/**
 * Use SDDMM as namespace, just to be on the safe side
*/
namespace SDDMM {
    /**
     * Inherit from Algo with virtual method main(...) to enforce some
     * minimal format consistency
    */
    class AlgoSample: public Algo {
    public:
        /**
         * Must add ..."override" at the end here
        */
        int main(int argc, char** argv, const InitParams& initParams) override{

            // note: static stuff needs :: not .
            // LibSample::sampleDataGenerator() not LibSample.sampleDataGenerator()
            int x = LibSample::sampleDataGenerator();

            std::cout << "Running sample algo " << x << std::endl;
            // do whatever...
            return 0;
        }
    };
}
