#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>

#include "../defines.h"

#include "cuda_sddmm.cuh"
extern "C" void cuda_tiled_sddmm();

namespace SDDMM {
    namespace Algo {
        void CudaTiledSDDMM() {
            cuda_tiled_sddmm();
        }
    }
}