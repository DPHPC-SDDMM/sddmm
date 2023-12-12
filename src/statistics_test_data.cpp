#include <iostream>
#include "defines.h"
#include "experiments/test_data_sddmm.cpp"

/**
 * This is the algo that will run
*/

using namespace SDDMM;

int main(int argc, char** argv) {

    Experiments::ExperimentTestData::benchmark_static("sddmm_test", 100);

    return 0;
}
