#include "test_helpers.hpp"
#include "../src/data_structures/coo/coo.h"
#include "../src/cuda_examples/cuda_plus_x.cpp"
#include "../src/cuda_examples/cuda_plus_x_struct.cpp"
#include <iostream>

UTEST_MAIN();

UTEST(Cuda_Example_Tests, Plus_X_Struct) {
    for(SDDMM::Types::expmt_t x=0; x<100; x+=0.7f){
        std::vector<SDDMM::CUDA_EXAMPLES::triplet> input;
        std::vector<SDDMM::CUDA_EXAMPLES::triplet> expected;

        for(int i=0; i<1000; ++i){
            input.push_back(SDDMM::CUDA_EXAMPLES::triplet{
                .row = static_cast<SDDMM::Types::vec_size_t>(i-1),
                .col = static_cast<SDDMM::Types::vec_size_t>(i+1),
                .value = static_cast<SDDMM::Types::expmt_t>(i)
            });

            expected.push_back(SDDMM::CUDA_EXAMPLES::triplet{
                .row = static_cast<SDDMM::Types::vec_size_t>(i),
                .col = static_cast<SDDMM::Types::vec_size_t>(i),
                .value = static_cast<SDDMM::Types::expmt_t>(i+x)
            });
        }

        std::vector<SDDMM::CUDA_EXAMPLES::triplet> output = SDDMM::CUDA_EXAMPLES::CuPlusXStruct(input, x);

        const TestHelpers::FEquals<SDDMM::CUDA_EXAMPLES::triplet> compare = [](
            SDDMM::CUDA_EXAMPLES::triplet x, 
            SDDMM::CUDA_EXAMPLES::triplet y
        ) {
            return x.value == y.value && x.row == y.row && x.col == y.col;
        };

        ASSERT_TRUE(output.size() == expected.size());
        ASSERT_TRUE(TestHelpers::compare_vectors(output, expected, compare));
    }
}

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