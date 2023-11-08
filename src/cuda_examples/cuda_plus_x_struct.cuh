#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>
#include "../defines.h"
#include "../data_structures/coo/coo.h"

extern "C" void run_k_struct(
    SDDMM::Types::COO::triplet *in, 
    SDDMM::Types::COO::triplet *out, 
    SDDMM::Types::vec_size_t len, 
    SDDMM::Types::expmt_t x
);
