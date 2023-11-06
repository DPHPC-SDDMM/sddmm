#include "test_helpers.hpp"
#include "../src/algos/cuda_sddmm.cpp"

UTEST_MAIN();

UTEST(Cuda_tiled_SDDMM, Init_test) {
    SDDMM::Algo::CudaTiledSDDMM();
}