#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>
#include "../data_structures/matrix/matrix.h"
#include "../data_structures/coo/coo.h"

#include "../defines.h"

#include "cuda_sddmm.cuh"

namespace SDDMM {
    namespace Algo {
        void CudaTiledSDDMM(const Types::COO& A_sparse, const Types::Matrix& X_dense, const Types::Matrix& Y_dense) {
            assert(X_dense.m == Y_dense.n && "Size of cols(X_dense) and rows(Y) must match!");
            assert(A_sparse.n>0 && A_sparse.m>0 && X_dense.n>0 && X_dense.m>0 && Y_dense.n>0 && Y_dense.m && "All involved matrices must be non-empty!");
            assert(A_sparse.n==X_dense.n && A_sparse.m==Y_dense.m && "Matrix dimensions must match!");

            
            Types::COO out_sparse;
            out_sparse.n = A_sparse.n;
            out_sparse.m = A_sparse.m;
            // make some space
            out_sparse.data.resize(A_sparse.data.size());

            size_t sp_base_size = sizeof(SDDMM::Types::COO::triplet);
            size_t dense_base_size = sizeof(SDDMM::Types::expmt_t);

            Types::vec_size_t sp_size = A_sparse.data.size()*sp_base_size;
            Types::vec_size_t dense_size_x = X_dense.data.size()*dense_base_size;
            Types::vec_size_t dense_size_y = Y_dense.data.size()*dense_base_size;

            cuda_tiled_sddmm(
                A_sparse.data.data(), 
                sp_size,
                X_dense.data.data(),
                dense_size_x,
                Y_dense.data.data(),
                dense_size_y, 
                out_sparse.data.data()
            );
        }
    }
}