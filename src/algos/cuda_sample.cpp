/**
 * Some example from a cuda tutorial
*/

#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>

#include "../defines.h"

#include "cuda_sample.cuh"
extern "C" void run(float* d_out, float* d_a, float* d_b, int N);

#define N 10000000
#define MAX_ERR 1e-6

namespace SDDMM {
    namespace Algo {
        class SampleCudaAlgo {
        public:
        SampleCudaAlgo(){
            float *a, *b, *out;
            float *d_a, *d_b, *d_out; 

            // Allocate host memory
            a   = (float*)malloc(sizeof(float) * N);
            b   = (float*)malloc(sizeof(float) * N);
            out = (float*)malloc(sizeof(float) * N);

            // Initialize host arrays
            for(int i = 0; i < N; i++){
                a[i] = 1.0f;
                b[i] = 2.0f;
            }

            // Allocate device memory 
            cudaMalloc((void**)&d_a, sizeof(float) * N);
            cudaMalloc((void**)&d_b, sizeof(float) * N);
            cudaMalloc((void**)&d_out, sizeof(float) * N);

            // Transfer data from host to device memory
            cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
            cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

            // Executing kernel 
            // vector_add<<<1,256>>>(d_out, d_a, d_b, N);
            run(d_out, d_a, d_b, N);
            
            // Transfer data back to host memory
            cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

            // Verification
            for(int i = 0; i < N; i++){
                assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
            }

            printf("PASSED\n");

            // Deallocate device memory
            cudaFree(d_a);
            cudaFree(d_b);
            cudaFree(d_out);

            // Deallocate host memory
            free(a); 
            free(b); 
            free(out);
        }
        };
    }
}