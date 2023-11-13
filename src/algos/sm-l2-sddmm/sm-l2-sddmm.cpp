#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "sm-l2-gpu.cuh"
#include "../../data_structures/coo/coo.h"

// "proper cuda error checking"
// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

int compute_tile_size_using_model(int L2_size, double c, double p) {
    // cast? precision?
    return sqrt((L2_size) / (c * p));
}

int compute_k_slice_using_auto_tuning() {
    return 64;  // TODO
}

SDDMM::Types::COO partition(const SDDMM::Types::COO& S, SDDMM::Types::vec_size_t Tj, SDDMM::Types::vec_size_t tile_id) {
    SDDMM::Types::COO S_tile;

    // start and end col indices for the tile
    SDDMM::Types::vec_size_t start_ind = tile_id * Tj;
    SDDMM::Types::vec_size_t end_ind = start_ind + Tj;

    // limit the max col
    if (end_ind > S.m) {
        end_ind = S.m;
    }

    // extract rows
    for (auto t : S.data) {
        if (t.col >= start_ind && t.col < end_ind) {
            S_tile.data.push_back(t);
        }
    }

    S_tile.n = S.n;
    S_tile.m = end_ind - start_ind;

    return S_tile;
}

std::vector<SDDMM::Types::vec_size_t> process_active_rows(const SDDMM::Types::COO& S_tile, std::vector<SDDMM::Types::vec_size_t>& active_rows, std::vector<SDDMM::Types::vec_size_t>& S_tile_starts, std::vector<SDDMM::Types::vec_size_t>& S_tile_rows_local, SDDMM::Types::vec_size_t Ti) {
    // iterate over elements of S_tile and push the encountered rows
    // assumption: S_tile is sorted, otherwise we'd need a set()
    if (!S_tile.data.empty()) {
        active_rows.push_back(S_tile.data[0].row);
        S_tile_starts.push_back(0);

        int c = 1;

        int local_row = 0;
        S_tile_rows_local.push_back(local_row);

        for (int i = 1; i < S_tile.data.size(); i++) {
            if (S_tile.data[i].row != active_rows.back()) {
                active_rows.push_back(S_tile.data[i].row);

                c++;
                if (c > Ti) {
                    // add new tile start
                    S_tile_starts.push_back(i);
                    c = 1;
                }

                local_row++;
            }

            S_tile_rows_local.push_back(local_row);
        }

        S_tile_starts.push_back(S_tile.data.size());
    }

    return active_rows;
}

//std::vector<SDDMM::Types::vec_size_t> get_rows(const SDDMM::Types::COO& M) {
//    std::vector<SDDMM::Types::vec_size_t> rows;
//    for (auto & i : M.data) {
//        rows.push_back(i.row);
//    }
//
//    return rows;
//}

std::vector<SDDMM::Types::vec_size_t> get_cols(const SDDMM::Types::COO& M) {
    std::vector<SDDMM::Types::vec_size_t> cols;
    for (auto & i : M.data) {
        cols.push_back(i.col);
    }

    return cols;
}

std::vector<float> get_values(const SDDMM::Types::COO& M) {
    std::vector<float> values;
    for (auto & i : M.data) {
        values.push_back(i.value);
    }

    return values;
}

//void merge_and_sort(SDDMM::Types::COO& target_M,
//                    std::vector<SDDMM::Types::vec_size_t>& source_rows,
//                    std::vector<SDDMM::Types::vec_size_t>& source_cols,
//                    std::vector<float>& source_values) {
//
//    target_M.data.reserve(target_M.data.size() + source_rows.size());
//    for (int i = 0; i < source_rows.size(); i++) {
//        target_M.data.push_back({
//            source_rows[i],
//            source_cols[i],
//            source_values[i]
//        });
//    }
//
//    target_M.sort();
//}

SDDMM::Types::COO get_correct_result_tile(const SDDMM::Types::COO& S_tile, SDDMM::Types::vec_size_t k,
                                          const SDDMM::Types::Matrix& A, const SDDMM::Types::Matrix& B) {
    // check
    SDDMM::Types::COO R;
    R.n = S_tile.n;
    R.m = S_tile.m;

    for (const auto & t : S_tile.data) {
        auto row = t.row;
        auto col = t.col;
        auto v = t.value;

        float sum = 0.;
        for (auto i = 0; i < k; i++) {
            sum += A.at(row, i) * B.at(col, i);
        }

        R.data.push_back({
             row,
             col,
             v * sum
     });
    }

    return R;
}

namespace SDDMM {
    namespace Algo {
        class SML2SDDMM {
        public:
            SML2SDDMM() = default;

            // assumptions: sparse matrix not empty
            static int run() {
                // dimensions
//                Types::vec_size_t n = 1024;
//                Types::vec_size_t m = 4096;
//                Types::vec_size_t k = 128;

                Types::vec_size_t n = 1024;
                Types::vec_size_t m = 256;
                // has to be a multiple of Tk (for now)
                Types::vec_size_t k = 256;

                float sparsity = 0.97;

                std::cout << "Matrix sizes:" << std::endl;
                std::cout << "N: " << n << ";  M: " << m << ";  K: " << k << ";  Sparsity: " << sparsity << std::endl;
                std::cout << std::endl;

                std::cout << "Creating matrices..." << std::endl;

                // matrix S
                auto S_dense = SDDMM::Types::Matrix::generate(n, m, sparsity);
                SDDMM::Types::COO S = S_dense.to_coo();

                // matrices A and B
                auto A = SDDMM::Types::Matrix::generate(n, k, sparsity=0.);
                auto B = SDDMM::Types::Matrix::generate(m, k, sparsity=0.);

                // result matrix
                SDDMM::Types::COO P;
                P.n = S.n;
                P.m = S.m;

                std::cout << "S nxm: " << S.n * S.m << ";  S non-zero: " << S.data.size() << std::endl;
                std::cout << std::endl;

                int l2_cache_capacity = 2097152;  // 2MB for testing
                int shared_mem_size = 49152;  // 48KB for testing
                int c = 3; // 3 for COO

                // not used for now
//                std::cout << "Parameters:" << std::endl;
//                std::cout << "L2: " << l2_cache_capacity << "B;  SM " << shared_mem_size << "B;  c: " << c << std::endl;
//                std::cout << std::endl;

//                int Tj = compute_tile_size_using_model(l2_cache_capacity, c, sparsity);
                auto Tj = S.m / 4;
                auto num_J_tiles = (m  + Tj - 1) / Tj;
//                std::cout << "Dimension Tj (from model):" << std::endl;
                std::cout << "Dimension Tj:" << std::endl;
                std::cout << "size: " << Tj << ";  count: " << num_J_tiles << std::endl;
                std::cout << std::endl;

//                std::cout << "Starting autotuning..." << std::endl;
//                int Tk = compute_k_slice_using_auto_tuning();
//                auto Tk = k / 2;
                auto Tk = 32;
                Types::vec_size_t num_Tks = (k + Tk - 1) / Tk;
                Types::vec_size_t Ti = std::min(static_cast<Types::vec_size_t>(shared_mem_size / sizeof(float) / Tk), S.n);
//                std::cout << "Autotuning completed!" << std::endl;
//                std::cout << "Dimension Tk (from autotuning):" << std::endl;
                std::cout << "Dimension Tk:" << std::endl;
                std::cout << "size: " << Tk << ";  count: " << num_Tks << std::endl;
                std::cout << "Dimension Ti:" << std::endl;
                std::cout << "size: " << Ti << std::endl;
                std::cout << std::endl;

                std::cout << "Starting processing..." << std::endl << std::endl;

                for (int tile_j_id = 0; tile_j_id < num_J_tiles; tile_j_id++) {
                    std::cout << "Tile J id: " << tile_j_id << std::endl << std::endl;

                    std::cout << "Partitioning matrix S..." << std::endl;
                    SDDMM::Types::COO S_tile = partition(S, Tj, tile_j_id);
                    std::cout << "size: " << S_tile.data.size() << ";  S_tile.n: " << S_tile.n << ";  S_tile.m: " << S_tile.m << std::endl;
                    std::cout << std::endl;

                    std::cout << "Collecting active rows in S_tile..." << std::endl;
                    std::vector<SDDMM::Types::vec_size_t> active_rows;
                    std::vector<SDDMM::Types::vec_size_t> S_tile_starts;
                    // localised slice row indices
                    std::vector<SDDMM::Types::vec_size_t> S_tile_rows_local;
                    process_active_rows(S_tile, active_rows, S_tile_starts, S_tile_rows_local, Ti);
                    std::cout << "size: " << active_rows.size() << std::endl;
//                    std::cout << " rows: [";
//                    for (auto i : active_rows) {
//                        std::cout << i << ", ";
//                    }
//                    std::cout << "]" << std::endl;
                    std::cout << std::endl;

                    std::cout << "Calculating the number of threadblocks..." << std::endl;
                    int num_threadblocks = (active_rows.size() + Ti - 1) / Ti;
                    std::cout << "size: " << num_threadblocks << std::endl;
                    std::cout << std::endl;

                    auto P_tile_check = get_correct_result_tile(S_tile, k, A, B);

                    // TODO allocate memory and/or load once

                    // S_tile
                    SDDMM::Types::vec_size_t* S_tile_rows_d;
                    auto size = S_tile.data.size() * sizeof(SDDMM::Types::vec_size_t);
                    gpuErrchk(cudaMalloc((void**)& S_tile_rows_d, size));
                    gpuErrchk(cudaMemcpy(S_tile_rows_d, S_tile_rows_local.data(), size, cudaMemcpyHostToDevice));

                    SDDMM::Types::vec_size_t* S_tile_cols_d;
                    auto S_tile_cols = get_cols(S_tile);
                    gpuErrchk(cudaMalloc((void**)& S_tile_cols_d, size));
                    gpuErrchk(cudaMemcpy(S_tile_cols_d, S_tile_cols.data(), size, cudaMemcpyHostToDevice));

                    float* S_tile_values_d;
                    auto size_values = S_tile.data.size() * sizeof(float);
                    gpuErrchk(cudaMalloc((void**)&S_tile_values_d, size_values));
                    gpuErrchk(cudaMemcpy(S_tile_values_d, get_values(S_tile).data(), size_values, cudaMemcpyHostToDevice));

                    // result tile
                    float* P_tile_values_d;
                    gpuErrchk(cudaMalloc((void**)&P_tile_values_d, size_values));
                    gpuErrchk(cudaMemset(P_tile_values_d, 0, size_values));

                    // P_tile (the result)
                    std::vector<float> P_tile_values = std::vector<float>(S_tile.data.size());

                    // tile starts
                    SDDMM::Types::vec_size_t* S_tile_starts_d;
                    auto S_tile_starts_size = S_tile_starts.size() * sizeof(SDDMM::Types::vec_size_t);
                    gpuErrchk(cudaMalloc((void**)&S_tile_starts_d, size));
                    gpuErrchk(cudaMemcpy(S_tile_starts_d, S_tile_starts.data(), S_tile_starts_size, cudaMemcpyHostToDevice));

                    // active_rows
                    SDDMM::Types::vec_size_t* active_rows_d;
                    auto active_rows_size = active_rows.size() * sizeof(SDDMM::Types::vec_size_t);
                    gpuErrchk(cudaMalloc((void**)&active_rows_d, size));
                    gpuErrchk(cudaMemcpy(active_rows_d, active_rows.data(), active_rows_size, cudaMemcpyHostToDevice));

                    // A
                    float* A_d;
                    auto A_size = n * k * sizeof(float);
                    gpuErrchk(cudaMalloc((void**)&A_d, A_size));
                    gpuErrchk(cudaMemcpy(A_d, A.data.data(), A_size, cudaMemcpyHostToDevice));

                    // B
                    float* B_d;
                    auto B_size = m * k * sizeof(float);
                    gpuErrchk(cudaMalloc((void**)&B_d, B_size));
                    gpuErrchk(cudaMemcpy(B_d, B.data.data(), B_size, cudaMemcpyHostToDevice));

                    // iterate over Tk tiles and launch a kernel for each Tk tile
                    for (int tile_k_id = 0; tile_k_id < num_Tks; tile_k_id++) {
                        // the innermost loop, streaming is done along dimension i (assuming that i is the smaller dimension, i.e. n < m)
                        std::cout << "Tile K id: " << tile_k_id << std::endl;

                        // load matrices into device memory

                        // launch num_threadblocks with 512 threads in each
                        run_kernel(
                            num_threadblocks,
                            S_tile_rows_d, S_tile_cols_d, S_tile_values_d, S_tile.data.size(),
                            S_tile_starts_d, S_tile_starts.size(),
                            P_tile_values_d,
                            active_rows_d, active_rows.size(),
                            A_d, B_d,
                            Tj, Tk, Ti,
                            tile_j_id,
                            tile_k_id,
                            num_Tks,
                            n, m, k
                        );

                        gpuErrchk(cudaPeekAtLastError());
                        gpuErrchk(cudaDeviceSynchronize());
                    }

                    // read from device (P_tile)
                    float diff = 0.;
                    gpuErrchk(cudaMemcpy(P_tile_values.data(), P_tile_values_d, size_values, cudaMemcpyDeviceToHost));
                    for (int i = 0; i < P_tile_values.size(); i++) {
                        diff += abs(P_tile_check.data.at(i).value - P_tile_values.at(i));
//                        std::cout << P_tile_values.at(i) << " [" << P_tile_check.data.at(i).value << "]" << std::endl;
                    }
                    std::cout << std::endl << "DIFF: " << diff << std::endl << std::endl;

                    // deallocate cuda memory
                    gpuErrchk(cudaFree(S_tile_rows_d));
                    gpuErrchk(cudaFree(S_tile_cols_d));
                    gpuErrchk(cudaFree(S_tile_values_d));

                    gpuErrchk(cudaFree(S_tile_starts_d));
                    gpuErrchk(cudaFree(active_rows_d));

                    gpuErrchk(cudaFree(A_d));
                    gpuErrchk(cudaFree(B_d));
                }

                cudaDeviceReset();

                return 0;
            }
        };
    }
}


