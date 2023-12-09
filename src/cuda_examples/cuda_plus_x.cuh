#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>
#include "../defines.h"

extern "C" void run_k(SDDMM::Types::expmt_t *in, SDDMM::Types::expmt_t *out, SDDMM::Types::vec_size_t len, SDDMM::Types::expmt_t x);
