#include "cuda_runtime.h"
#include "device_launch_parameters.h"

extern "C" void run(float* d_out, float* d_a, float* d_b, int N);
