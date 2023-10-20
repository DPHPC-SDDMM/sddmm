#pragma once

/**
 * Add whatever generator classes here
*/
#include "lib_sample.h"

namespace SDDMM {
    struct InitParams {
        int sampleParam;
    };

    class Algo {
    public:
        virtual int main(int argc, char** argv, const InitParams& initParams) = 0;
    };
}