#include <vector>
#include "../defines.h"
#include "../data_structures/coo/coo.h"
#include "cuda_plus_x_struct.cuh"

/**
 * Create input and output std vector
 * Copy contents of std vector to cuda
 * Change values inside cuda
 * Copy vector back and back into an output vector
*/

namespace SDDMM {
    namespace CUDA_EXAMPLES {
        std::vector<triplet> CuPlusXStruct(std::vector<triplet>& in, SDDMM::Types::expmt_t x){

            // define type pointers: initialize with 'nullptr' => good practice because
            // it causes a nullptr exception rather than random behaviour if something accesses them
            // before intended
            triplet* in_d = nullptr;
            SDDMM::Types::vec_size_t len = in.size();
            // all memcpy operations use bytes as unit => make SURE to use the correct type
            // for these calculations. If the sizes are incorrect it will likely randomly cause
            // cryptic 'memory freed twice or not at all or just half-ass' errors at some point
            SDDMM::Types::vec_size_t len_d = sizeof(triplet) * len;

            // allocate and copy input
            cudaMalloc(reinterpret_cast<void**>(&in_d), len_d);
            cudaMemcpy(in_d, in.data(), len_d, cudaMemcpyHostToDevice);

            // allocate space for the output
            triplet* out_d = nullptr;
            cudaMalloc(reinterpret_cast<void**>(&out_d), len_d);

            // run the cuda kernel
            // pass 'len' as in the actual count of the elements for loops because cuda knows
            // basic data types like float or int
            run_k_struct(in_d, out_d, len, x);

            // declare output type
            std::vector<triplet> out;
            // resize (!!!) data part of the output type to fit
            out.resize(in.size());
            // memcpy the result into the data space of the output type
            cudaMemcpy(out.data(), out_d, len_d, cudaMemcpyDeviceToHost);

            // free on-device memory
            cudaFree(in_d);
            cudaFree(out_d);

            // set freed pointers to 'nullptr' => good practice: get a nullptr exception if anything uses
            // them later rather than random behaviour.
            in_d = nullptr;
            out_d  = nullptr;

            // return the result (unless explicitly instructed, the compiler will substitude this with
            // a "move" instruction which does not copy the vector and is efficient)
            return out;
        }
    }
}

