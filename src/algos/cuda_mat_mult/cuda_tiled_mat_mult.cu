#include "cuda_tiled_mat_mult.cuh"

__global__ void k_mat_mult(
    SDDMM::Types::expmt_t* X_d,
    SDDMM::Types::vec_size_t X_n,
    SDDMM::Types::vec_size_t X_m,
    SDDMM::Types::expmt_t* Y_d,
    SDDMM::Types::vec_size_t Y_m,
    SDDMM::Types::vec_size_t ts,
    SDDMM::Types::expmt_t* XY_out_d
) {
    // only use one CUDA core and run exactly the same as on a CPU
    for(SDDMM::Types::vec_size_t i=0; i<X_n; i+=ts){
        for(SDDMM::Types::vec_size_t j=0; j<Y_m; j+=ts){
            for(SDDMM::Types::vec_size_t k=0; k<X_m; k+=ts){

                SDDMM::Types::vec_size_t i_p_ts = i+ts;
                for(SDDMM::Types::vec_size_t r=i; r<i_p_ts; ++r){

                    SDDMM::Types::vec_size_t j_p_ts = j+ts;
                    for(SDDMM::Types::vec_size_t c=j; c<j_p_ts; ++c){

                        SDDMM::Types::expmt_t inner_p = 0;
                        SDDMM::Types::vec_size_t k_p_ts = k+ts;
                        // Types::vec_size_t ind = r*r_num + c;
                        for(SDDMM::Types::vec_size_t kk=k; kk<k_p_ts; ++kk){

                            inner_p += X_d[r*X_m + kk] * Y_d[kk*Y_m + c];
                            // inner_p += A.data[r*c_num+kk]*B.data[kk*c_num+c];
                        }
                        XY_out_d[r*Y_m + c] += inner_p;
                    }
                }
            }
        }
    }
}

void CudaTiledMatMult(
    SDDMM::Types::expmt_t* X_d,
    SDDMM::Types::vec_size_t X_n,
    SDDMM::Types::vec_size_t X_m,
    SDDMM::Types::expmt_t* Y_d,
    SDDMM::Types::vec_size_t Y_m,
    SDDMM::Types::vec_size_t ts,
    SDDMM::Types::expmt_t* XY_out_d
) {
    k_mat_mult<<<1,1>>>(
        X_d, X_n, X_m,
        Y_d, Y_m,
        ts, XY_out_d
    );
}
