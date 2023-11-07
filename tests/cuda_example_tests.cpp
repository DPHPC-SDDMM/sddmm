#include "test_helpers.hpp"
#include "../src/cuda_examples/cuda_plus_x.cpp"
#include <iostream>

UTEST_MAIN();

UTEST(Cuda_Example_Tests, Plus_X) {
    for(SDDMM::Types::expmt_t x=0; x<100; x+=0.7f){
        std::vector<SDDMM::Types::expmt_t> input;
        std::vector<SDDMM::Types::expmt_t> expected;

        for(int i=0; i<1000; ++i){
            input.push_back(i);
            expected.push_back(i+x);
        }

        std::vector<SDDMM::Types::expmt_t> output = SDDMM::CUDA_EXAMPLES::CuPlusX(input, x);

        ASSERT_TRUE(output.size() == expected.size());
        ASSERT_TRUE(TestHelpers::compare_vectors(output, expected));
    }
}