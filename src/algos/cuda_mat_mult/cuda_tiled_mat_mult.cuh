#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../../defines.h"

namespace SDDMM {
    namespace Algo {
        namespace CUDA_TILED_MAT_MULT {
            extern "C" void CudaTiledMatMult(
                SDDMM::Types::expmt_t* X_d,
                SDDMM::Types::vec_size_t X_n,
                SDDMM::Types::vec_size_t X_m,
                SDDMM::Types::expmt_t* Y_d,
                SDDMM::Types::vec_size_t Y_m,
                SDDMM::Types::vec_size_t ts,
                SDDMM::Types::expmt_t* XY_out_d
            );
        }
    }
}
