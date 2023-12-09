#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>
#include "../defines.h"
#include "../data_structures/coo/coo.h"

namespace SDDMM {
    namespace CUDA_EXAMPLES {
        struct triplet {
            Types::vec_size_t row;
            Types::vec_size_t col;
            Types::expmt_t value;
        };
    }
}

extern "C" void run_k_struct(
    SDDMM::CUDA_EXAMPLES::triplet *in, 
    SDDMM::CUDA_EXAMPLES::triplet *out, 
    SDDMM::Types::vec_size_t len, 
    SDDMM::Types::expmt_t x
);
