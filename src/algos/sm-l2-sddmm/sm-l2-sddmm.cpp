#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "sm-l2-gpu.cuh"
#include "../../data_structures/coo/coo.h"
#include "../../results.h"

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

SDDMM::Types::vec_size_t compute_tile_size_using_model(unsigned int L2_size, double c, double p) {
    // cast? precision?
    return sqrt(L2_size / (c * p));
}

int compute_k_slice_using_auto_tuning() {
    return 32;  // TODO
}

//std::vector<float> get_correct_result(const SDDMM::Types::COO& S, SDDMM::Types::vec_size_t k, const SDDMM::Types::Matrix& A, const SDDMM::Types::Matrix& B, SDDMM::Types::vec_size_t num_J_tiles, SDDMM::Types::vec_size_t Tj) {
//    // calculate the resulting matrix
//    SDDMM::Types::COO R;
//    R.n = S.n;
//    R.m = S.m;
//
//    for (const auto & t : S.data) {
//        auto row = t.row;
//        auto col = t.col;
//        auto v = t.value;
//
//        float sum = 0.;
//        for (auto i = 0; i < k; i++) {
//            sum += A.at(row, i) * B.at(col, i);
//        }
//
//        R.data.push_back({
//             row,
//             col,
//             v * sum
//        });
//    }
//
//    std::vector<float> R_values;
//
//    // put it in the slice-by-slice form to be able to compare with P
//    for (int j = 0; j < num_J_tiles; j++) {
//        // start and end col indices for the tile
//        SDDMM::Types::vec_size_t start_ind = j * Tj;
//        SDDMM::Types::vec_size_t end_ind = start_ind + Tj;
//
//        // limit the max col
//        if (end_ind > R.m) {
//            end_ind = R.m;
//        }
//
//        // extract elements
//        for (auto t: R.data) {
//            if (t.col >= start_ind && t.col < end_ind) {
//                R_values.push_back(t.value);
//            }
//        }
//    }
//
//    return R_values;
//}

namespace SDDMM {
    namespace Algo {
        class SML2SDDMM {
        public:
            SML2SDDMM() = default;

            static int run() {
                // generate matrices
                auto matrix_params = prepare_matrices();

                // calculate tile sizes
                auto tiling_params = determine_tiling_params(
                        matrix_params.N,
                        matrix_params.M,
                        matrix_params.K,
                        matrix_params.sparsity
                );

                // assumptions: sparse matrix not empty, no empty slices (for now), K multiple of 32
                auto sparse_params = prepare_sparse(matrix_params.S, tiling_params.Tj, tiling_params.Ti, tiling_params.num_J_tiles);

                // allocate GPU memory and run the algorithm
                auto res = run_algo(
                        matrix_params,
                        tiling_params,
                        sparse_params
                );

                // TODO postprocess (if needed), like cleaning up zeroes from res.values

                // check correctness
                check_result(matrix_params.S, matrix_params.A, matrix_params.B, res, matrix_params.K, tiling_params.Tj, tiling_params.num_J_tiles);

                return 0;
            }

            struct MatrixParams {
                const Types::COO S;
                const Types::Matrix A;
                const Types::Matrix B;

                const Types::vec_size_t N;
                const Types::vec_size_t M;
                const Types::vec_size_t K;

                const double sparsity;
            };

            struct TilingParams {
                const Types::vec_size_t Ti;
                const Types::vec_size_t Tj;
                const Types::vec_size_t Tk;

                const Types::vec_size_t num_J_tiles;
                const Types::vec_size_t num_K_tiles;
            };

            struct SparseParams {
                const std::vector<Types::vec_size_t> rows;
                const std::vector<Types::vec_size_t> rows_local;
                const std::vector<Types::vec_size_t> cols;
                const std::vector<float> values;

                std::vector<SDDMM::Types::vec_size_t> slice_sizes;

                std::vector<SDDMM::Types::vec_size_t> active_rows;
                std::vector<SDDMM::Types::vec_size_t> active_rows_sizes;

                std::vector<SDDMM::Types::vec_size_t> S_tile_starts;
            };

            struct Result {
                const std::vector<float> values;
                long long duration_ms;
            };

            static MatrixParams prepare_matrices() {
                // dimensions
                // NOTE: K has to be a multiple of 32
//                Types::vec_size_t N = 1024;
//                Types::vec_size_t M = 256;
//                Types::vec_size_t K = 256;
                Types::vec_size_t N = 234234;
                Types::vec_size_t M = 2344;
                Types::vec_size_t K = 256;

                float sparsity = 0.97;

                std::cout << "Matrix sizes:" << std::endl;
                std::cout << "N: " << N << ";  M: " << M << ";  K: " << K << ";  Sparsity: " << sparsity << std::endl;
                std::cout << std::endl;

                std::cout << "Creating matrices..." << std::endl;

                // matrix S
                auto S_dense = SDDMM::Types::Matrix::generate(N, M, sparsity);
                SDDMM::Types::COO S = S_dense.to_coo();

                // matrices A and B
                auto A = SDDMM::Types::Matrix::generate(N, K, sparsity=0.);
                auto B = SDDMM::Types::Matrix::generate(M, K, sparsity=0.);

                // result matrix
                SDDMM::Types::COO P;
                P.n = S.n;
                P.m = S.m;

                std::cout << "S nxm: " << S.n * S.m << ";  S non-zero: " << S.data.size() << std::endl;
                std::cout << "A nxm: " << A.n * A.m << std::endl;
                std::cout << "B nxm: " << B.n * B.m << std::endl;
                std::cout << std::endl;

                return {
                        S,
                        A,
                        B,
                        N,
                        M,
                        K,
                        sparsity
                };
            }

            static TilingParams determine_tiling_params(Types::vec_size_t N, Types::vec_size_t M, Types::vec_size_t K, double sparsity) {
                // GPU and format params
//                Types::vec_size_t l2_cache_capacity = 6291456;  // 6MB 3080Ti
//                Types::vec_size_t shared_mem_size = 101376;  // 99KB 3080Ti
                unsigned int l2_cache_capacity = 2097152;  // 2MB for testing
                unsigned int shared_mem_size = 49152;  // 48KB for testing
                double c = 3.; // 3 for COO

                std::cout << "Parameters:" << std::endl;
                std::cout << "L2: " << l2_cache_capacity << "B;  SM " << shared_mem_size << "B;  c: " << c << std::endl;
                std::cout << std::endl;

                auto Tj = std::min(compute_tile_size_using_model(l2_cache_capacity, c, 1 - sparsity), M);
//                auto Tj = M / 4;
                auto num_J_tiles = (M + Tj - 1) / Tj;
//                std::cout << "Dimension Tj (from model):" << std::endl;
                std::cout << "Dimension Tj:" << std::endl;
                std::cout << "size: " << Tj << ";  count: " << num_J_tiles << std::endl;
                std::cout << std::endl;

//                std::cout << "Starting autotuning..." << std::endl;
//                Types::vec_size_t Tk = compute_k_slice_using_auto_tuning();
                Types::vec_size_t Tk = 32;
                Types::vec_size_t num_K_tiles = (K + Tk - 1) / Tk;
                Types::vec_size_t Ti = std::min(static_cast<Types::vec_size_t>(shared_mem_size / sizeof(float) / Tk), N);
//                std::cout << "Autotuning completed!" << std::endl;
                std::cout << "Dimension Tk:" << std::endl;
                std::cout << "size: " << Tk << ";  count: " << num_K_tiles << std::endl;
                std::cout << "Dimension Ti:" << std::endl;
                std::cout << "size: " << Ti << std::endl;
                std::cout << std::endl;

                return {
                    Ti,
                    Tj,
                    Tk,
                    num_J_tiles,
                    num_K_tiles
                };
            }

            static SparseParams prepare_sparse(const Types::COO& S, Types::vec_size_t Tj, Types::vec_size_t Ti, Types::vec_size_t num_J_tiles) {
                std::vector<SDDMM::Types::vec_size_t> rows;
                std::vector<SDDMM::Types::vec_size_t> rows_local;
                std::vector<SDDMM::Types::vec_size_t> cols;
                std::vector<float> values;

                std::vector<SDDMM::Types::vec_size_t> slice_sizes;

                std::vector<SDDMM::Types::vec_size_t> active_rows;
                std::vector<SDDMM::Types::vec_size_t> active_rows_sizes;

                std::vector<SDDMM::Types::vec_size_t> S_tile_starts;

                for (int j = 0; j < num_J_tiles; j++) {
                    SDDMM::Types::COO S_tile;

                    // start and end col indices for the tile
                    SDDMM::Types::vec_size_t start_ind = j * Tj;
                    SDDMM::Types::vec_size_t end_ind = start_ind + Tj;

                    // limit the max col
                    if (end_ind > S.m) {
                        end_ind = S.m;
                    }

                    // extract elements
                    for (auto t: S.data) {
                        if (t.col >= start_ind && t.col < end_ind) {
                            S_tile.data.push_back(t);
                        }
                    }

                    S_tile.n = S.n;
                    S_tile.m = end_ind - start_ind;

                    // process the slice
                    if (!S_tile.data.empty()) {
                        int a = 1;
                        active_rows.push_back(S_tile.data[0].row);

                        // counter for the elements in a single Ti x Tj tile
                        int c = 1;
                        S_tile_starts.push_back(0);

                        int local_row = 0;
                        rows_local.push_back(local_row);

                        rows.push_back(S_tile.data[0].row);
                        cols.push_back(S_tile.data[0].col);
                        values.push_back(S_tile.data[0].value);

                        // number of elements in the slice
                        int n = 1;

                        for (int i = 1; i < S_tile.data.size(); i++) {
                            if (S_tile.data[i].row != active_rows.back()) {
                                a++;
                                active_rows.push_back(S_tile.data[i].row);

                                c++;
                                if (c > Ti) {
                                    // add new tile start
                                    S_tile_starts.push_back(i);
                                    c = 1;
                                }

                                local_row++;
                            }

                            // populate row, col and vals lists
                            rows_local.push_back(local_row);
                            rows.push_back(S_tile.data[i].row);
                            cols.push_back(S_tile.data[i].col);
                            values.push_back(S_tile.data[i].value);

                            n++;
                        }

                        // push the number of elements in the slice
                        slice_sizes.push_back(n);

                        // push the number of active rows
                        active_rows_sizes.push_back(a);

                        S_tile_starts.push_back(S_tile.data.size());
                    } else {
                        // TODO properly process empty slices
                        slice_sizes.push_back(0);
                    }
                }

                return {
                        rows,
                        rows_local,
                        cols,
                        values,

                        slice_sizes,

                        active_rows,
                        active_rows_sizes,

                        S_tile_starts
                };
            }

            static void check_result(const Types::COO& S, const Types::Matrix& A, const Types::Matrix& B, const Result& res, Types::vec_size_t K, Types::vec_size_t Tj, Types::vec_size_t num_J_tiles) {
                std::cout << "Calculating the correct result..." << std::endl << std::endl;

                auto start_time = std::chrono::high_resolution_clock::now();

//                // compare with the correct (rearranged) result
//                auto R_values = get_correct_result(S, K, A, B, num_J_tiles, Tj);

                // calculate the resulting matrix
                SDDMM::Types::COO R;
                R.n = S.n;
                R.m = S.m;

                for (const auto & t : S.data) {
                    auto row = t.row;
                    auto col = t.col;
                    auto v = t.value;

                    float sum = 0.;
                    for (auto i = 0; i < K; i++) {
                        sum += A.at(row, i) * B.at(col, i);
                    }

                    R.data.push_back({
                             row,
                             col,
                             v * sum
                     });
                }

                std::vector<float> R_values;

                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration_ms = (end_time - start_time) / std::chrono::milliseconds(1);
                std::cout << "Duration (CPU, single-threaded): " << duration_ms << "ms" << std::endl << std::endl;

                std::cout << "Speedup: " << duration_ms / res.duration_ms << "x" << std::endl << std::endl;

                // put it in the slice-by-slice form to be able to compare with P
                for (int j = 0; j < num_J_tiles; j++) {
                    // start and end col indices for the tile
                    SDDMM::Types::vec_size_t start_ind = j * Tj;
                    SDDMM::Types::vec_size_t end_ind = start_ind + Tj;

                    // limit the max col
                    if (end_ind > R.m) {
                        end_ind = R.m;
                    }

                    // extract elements
                    for (auto t: R.data) {
                        if (t.col >= start_ind && t.col < end_ind) {
                            R_values.push_back(t.value);
                        }
                    }
                }

                std::cout << "Difference check:" << std::endl;
                float diff = 0.;
                for (int i = 0; i < res.values.size(); i++) {
                    diff += abs(R_values.at(i) - res.values.at(i));
                    // std::cout << P_values.at(i) << " [" << R_values.at(i) << "]" << std::endl;
                }
                std::cout << diff << std::endl << std::endl;
            }

            static Result run_algo(
                        MatrixParams& matrix_params,
                        TilingParams& tiling_params,
                        SparseParams& sparse_params
                    ) {
                auto S_size = matrix_params.S.data.size();

                // transfer data to GPU
                std::cout << "Allocating memory & transferring data..." << std::endl;

                // sparse matrix S
                SDDMM::Types::vec_size_t* rows_d;
                SDDMM::Types::vec_size_t* rows_local_d;
                SDDMM::Types::vec_size_t* cols_d;
                float* values_d;

                auto S_ind_size = S_size * sizeof(SDDMM::Types::vec_size_t);
                auto S_values_size = S_size * sizeof(float);

                gpuErrchk(cudaMalloc((void**)& rows_d, S_ind_size));
                gpuErrchk(cudaMalloc((void**)& rows_local_d, S_ind_size));
                gpuErrchk(cudaMalloc((void**)& cols_d, S_ind_size));
                gpuErrchk(cudaMalloc((void**)& values_d, S_values_size));

                gpuErrchk(cudaMemcpy(rows_d, sparse_params.rows.data(), S_ind_size, cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(rows_local_d, sparse_params.rows_local.data(), S_ind_size, cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(cols_d, sparse_params.cols.data(), S_ind_size, cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(values_d, sparse_params.values.data(), S_values_size, cudaMemcpyHostToDevice));

                // P result
                float* result_values_d;
                gpuErrchk(cudaMalloc((void**)&result_values_d, S_values_size));
                gpuErrchk(cudaMemset(result_values_d, 0, S_values_size));

                // tile starts
                SDDMM::Types::vec_size_t* starts_d;
                auto starts_size = sparse_params.S_tile_starts.size() * sizeof(SDDMM::Types::vec_size_t);
                gpuErrchk(cudaMalloc((void**)&starts_d, starts_size));
                gpuErrchk(cudaMemcpy(starts_d, sparse_params.S_tile_starts.data(), starts_size, cudaMemcpyHostToDevice));

                // active_rows
                SDDMM::Types::vec_size_t* active_rows_d;
                auto active_rows_size = sparse_params.active_rows.size() * sizeof(SDDMM::Types::vec_size_t);
                gpuErrchk(cudaMalloc((void**)&active_rows_d, active_rows_size));
                gpuErrchk(cudaMemcpy(active_rows_d, sparse_params.active_rows.data(), active_rows_size, cudaMemcpyHostToDevice));

                // A & B
                float* A_d;
                float* B_d;

                auto A_size = matrix_params.N * matrix_params.K * sizeof(float);
                auto B_size = matrix_params.M * matrix_params.K * sizeof(float);

                gpuErrchk(cudaMalloc((void**)&A_d, A_size));
                gpuErrchk(cudaMalloc((void**)&B_d, B_size));

                gpuErrchk(cudaMemcpy(A_d, matrix_params.A.data.data(), A_size, cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(B_d, matrix_params.B.data.data(), B_size, cudaMemcpyHostToDevice));

                std::cout << std::endl << "Starting processing..." << std::endl << std::endl;

                Types::vec_size_t slice_start_ind = 0;
                Types::vec_size_t tile_starts_start_ind = 0;
                Types::vec_size_t active_rows_start_ind = 0;

                auto start_time = std::chrono::high_resolution_clock::now();

                for (int tile_j_id = 0; tile_j_id < tiling_params.num_J_tiles; tile_j_id++) {
                    std::cout << "Tile J id: " << tile_j_id << std::endl << std::endl;

                    std::cout << "Calculating the number of threadblocks..." << std::endl;
                    int num_threadblocks = (sparse_params.active_rows_sizes.at(tile_j_id) + tiling_params.Ti - 1) / tiling_params.Ti;
                    std::cout << "size: " << num_threadblocks << std::endl;
                    std::cout << std::endl;

                    // iterate over Tk tiles and launch a kernel for each Tk tile
                    for (int tile_k_id = 0; tile_k_id < tiling_params.num_K_tiles; tile_k_id++) {
                        // the innermost loop, streaming is done along dimension i (assuming that i is the smaller dimension, i.e. N < M)
                        std::cout << "Tile K id: " << tile_k_id << std::endl;

                        // launch num_threadblocks with 512 threads in each
                        run_kernel(
                                num_threadblocks,
                                // S
                                &rows_local_d[slice_start_ind],
                                &cols_d[slice_start_ind],
                                &values_d[slice_start_ind],
                                // P
                                &result_values_d[slice_start_ind],
                                // starts
                                &starts_d[tile_starts_start_ind],
                                // active rows
                               &active_rows_d[active_rows_start_ind], sparse_params.active_rows_sizes.at(tile_j_id),
                                // A & B
                                A_d, B_d,
                                // tiles
                                tiling_params.Tk, tiling_params.Ti,
                                tile_k_id,
                                tiling_params.num_K_tiles,
                                matrix_params.K
                        );
                    }

                    // prepare the next slice
                    slice_start_ind += sparse_params.slice_sizes.at(tile_j_id);
                    tile_starts_start_ind += num_threadblocks + 1;
                    active_rows_start_ind += sparse_params.active_rows_sizes.at(tile_j_id);
                }

                gpuErrchk(cudaPeekAtLastError());
                gpuErrchk(cudaDeviceSynchronize());

                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration_ms = (end_time - start_time) / std::chrono::milliseconds(1);

                std::cout << std::endl << "Done processing!" << std::endl << std::endl;
                std::cout << "Duration (GPU, without preprocessing): " << duration_ms << "ms" << std::endl << std::endl;

                // read the result from device
                std::vector<float> P_values = std::vector<float>(S_size);
                gpuErrchk(cudaMemcpy(P_values.data(), result_values_d, S_values_size, cudaMemcpyDeviceToHost));

                // deallocate cuda memory
                gpuErrchk(cudaFree(rows_d));
                gpuErrchk(cudaFree(rows_local_d));
                gpuErrchk(cudaFree(cols_d));
                gpuErrchk(cudaFree(values_d));

                gpuErrchk(cudaFree(result_values_d));

                gpuErrchk(cudaFree(starts_d));
                gpuErrchk(cudaFree(active_rows_d));

                gpuErrchk(cudaFree(A_d));
                gpuErrchk(cudaFree(B_d));

                cudaDeviceReset();

                return {
                    P_values,
                    duration_ms
                };
            }
        };
    }
}