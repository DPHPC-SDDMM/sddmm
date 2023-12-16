#pragma once

#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../../defines.h"
#include "sm-l2-gpu.cuh"
#include "../../data_structures/coo/coo.h"
#include "../../results.h"

// use this one to get some output that disrupts the output of the unit tests
// #define LOCAL_PRINT

inline void local_print(const std::string& message){
#ifdef LOCAL_PRINT
    std::cout << message << std::endl;
#endif
}

SDDMM::Types::vec_size_t compute_tile_size_using_model(unsigned int L2_size, double c, float p) {
    // cast? precision?
    return sqrt(L2_size / (c * p));
}

namespace SDDMM {
    namespace Algo {
        class SML2SDDMM {
        public:
            SML2SDDMM() = default;

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

            struct Params {
                TilingParams tiling_params;
                SparseParams sparse_params;
            };

            static Types::vec_size_t compute_k_slice_using_auto_tuning(
                    unsigned int shared_mem_size,
                    const SDDMM::Types::COO& S, float sparsity, SDDMM::Types::Matrix& A, SDDMM::Types::Matrix& B,
                    Types::vec_size_t N, Types::vec_size_t M, Types::vec_size_t K,
                    Types::vec_size_t Tj, Types::vec_size_t num_J_tiles
                    ) {

                if (K == 32) {
                    return K;
                }

                local_print("Starting autotuning...");

                auto* measurements = new Results::ExperimentData;
                uint32_t repetitions = 3;

                Types::vec_size_t best_Tk = 32;
                Types::time_duration_unit best_measurement = std::numeric_limits<Types::time_duration_unit>::max();
                for (Types::vec_size_t Tk = 32; Tk <= K; Tk += 32) {
                    if (K % Tk == 0) {
                        // compute the tiling params that depend on Tk
                        Types::vec_size_t num_K_tiles = (K + Tk - 1) / Tk;
                        Types::vec_size_t Ti = std::min(
                                static_cast<Types::vec_size_t>(shared_mem_size / sizeof(float) / Tk), N);

                        local_print("Trying Tk=" + std::to_string(Tk));
                        local_print("count: " + std::to_string(num_K_tiles));
                        local_print("Dimension Ti:");
                        local_print("size: " + std::to_string(Ti));
                        local_print("");

                        TilingParams tiling_params{
                                Ti,
                                Tj,
                                Tk,
                                num_J_tiles,
                                num_K_tiles
                        };

                        // assumptions: sparse matrix not empty, no empty slices (for now), K multiple of 32
                        auto sparse_params = prepare_sparse(
                                S,
                                tiling_params.Tj,
                                tiling_params.Ti,
                                tiling_params.num_J_tiles
                        );

                        Params params{
                                .tiling_params=tiling_params,
                                .sparse_params=sparse_params
                        };

                        Types::time_duration_unit total_runtime = 0;
                        for (int i = 0; i < repetitions; i++) {
                            auto result = SDDMM::Algo::SML2SDDMM::run_sm_l2(
                                    S, sparsity,
                                    A, B,
                                    // N, M, K
                                    A.n, B.m, B.n,
                                    params,
                                    measurements
                            );

                            auto last_measurement = measurements->durations.back();
                            total_runtime += last_measurement;
                        }

                        // check runtime
                        auto avg_runtime = total_runtime / repetitions;
                        local_print(std::to_string(avg_runtime));

                        if (best_measurement > avg_runtime) {
                            best_measurement = avg_runtime;
                            best_Tk = Tk;
                        }
                    }
                }

                //local_print("Autotuning completed. Best Tk=" + std::to_string(best_Tk));

                //// return the best value of Tk
                //return best_Tk;
            }

            static Params preparations(
                SDDMM::Types::COO& S, 
                float sparsity,
                Types::vec_size_t N, Types::vec_size_t M, Types::vec_size_t K, 
                SDDMM::Types::Matrix& A, 
                SDDMM::Types::Matrix& B
            ) {
                // calculate tile sizes
                // A.n = N,
                // B.m = M,
                // B.n == A.m = K,
                auto tiling_params = determine_tiling_params(
                        N,
                        M,
                        K,
                        sparsity,
                        A,
                        B,
                        S
                );

                // assumptions: sparse matrix not empty, no empty slices (for now), K multiple of 32
                auto sparse_params = prepare_sparse(
                    S,
                    tiling_params.Tj,
                    tiling_params.Ti,
                    tiling_params.num_J_tiles
                );

                return Params {
                    // .matrix_params=matrix_params,
                    .tiling_params=tiling_params,
                    .sparse_params=sparse_params
                };
            }

            static TilingParams determine_tiling_params(Types::vec_size_t N, Types::vec_size_t M, Types::vec_size_t K, float sparsity, SDDMM::Types::Matrix& A, SDDMM::Types::Matrix& B, const Types::COO& S) {
                // GPU and format params
//                Types::vec_size_t l2_cache_capacity = 6291456;  // 6MB 3080Ti
//                Types::vec_size_t shared_mem_size = 101376;  // 99KB 3080Ti
                unsigned int l2_cache_capacity = 2097152;  // 2MB for testing
                unsigned int shared_mem_size = 49152;  // 48KB for testing
                double c = 3.; // 3 for COO

                assert(l2_cache_capacity % 32 == 0 && shared_mem_size % 32 == 0 && "L2 cache capacity and shared memory size must be multiples of 32!");

                local_print("Parameters:");
                local_print("L2: " + std::to_string(l2_cache_capacity) + "B;  SM " + std::to_string(shared_mem_size) + "B;  c: " + std::to_string(c));
                local_print("");

                auto Tj = std::min(compute_tile_size_using_model(l2_cache_capacity, c, 1 - sparsity), M);
                auto num_J_tiles = (M + Tj - 1) / Tj;
                local_print("Dimension Tj:");
                local_print("size: " + std::to_string(Tj) + ";  count: " + std::to_string(num_J_tiles));
                local_print("");

//                Types::vec_size_t Tk = 32;
                Types::vec_size_t Tk = compute_k_slice_using_auto_tuning(
                    shared_mem_size,
                    S,
                    sparsity,
                    A, B,
                    N, M, K,
                    Tj, num_J_tiles
                    );
                Types::vec_size_t num_K_tiles = (K + Tk - 1) / Tk;
                Types::vec_size_t Ti = std::min(static_cast<Types::vec_size_t>(shared_mem_size / sizeof(float) / Tk), N);
                local_print("Dimension Tk:");
                local_print("size: " + std::to_string(Tk) + ";  count: " + std::to_string(num_K_tiles));
                local_print("Dimension Ti:");
                local_print("size: " + std::to_string(Ti));
                local_print("");

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
                    // SDDMM::Types::COO S_tile;
                    std::vector<Types::expmt_t> S_tile_values;
                    std::vector<Types::vec_size_t> S_tile_cols;
                    std::vector<Types::vec_size_t> S_tile_rows;

                    // start and end col indices for the tile
                    SDDMM::Types::vec_size_t start_ind = j * Tj;
                    SDDMM::Types::vec_size_t end_ind = start_ind + Tj;

                    // limit the max col
                    if (end_ind > S.m) {
                        end_ind = S.m;
                    }

                    for (int i = 0; i < S.values.size(); i++) {
                        auto row = S.rows[i];
                        auto col = S.cols[i];
                        auto value = S.values[i];

                        if (col >= start_ind && col < end_ind) {
                            S_tile_rows.push_back(row);
                            S_tile_cols.push_back(col);
                            S_tile_values.push_back(value);
                        }
                    }

                    // process the slice
                    if(!S_tile_values.empty()) {
                        int a = 1;
                        active_rows.push_back(S_tile_rows[0]);

                        // counter for the elements in a single Ti x Tj tile
                        int c = 1;
                        S_tile_starts.push_back(0);

                        int local_row = 0;
                        rows_local.push_back(local_row);

                        rows.push_back(S_tile_rows[0]);
                        cols.push_back(S_tile_cols[0]);
                        values.push_back(S_tile_values[0]);

                        // number of elements in the slice
                        int n = 1;

                        for (int i = 1; i < S_tile_values.size(); i++) {
                            if (S_tile_rows[i] != active_rows.back()) {
                                a++;
                                // active_rows.push_back(S_tile.data[i].row);
                                active_rows.push_back(S_tile_rows[i]);

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
                            rows.push_back(S_tile_rows[i]);
                            cols.push_back(S_tile_cols[i]);
                            values.push_back(S_tile_values[i]);

                            n++;
                        }

                        // push the number of elements in the slice
                        slice_sizes.push_back(n);

                        // push the number of active rows
                        active_rows_sizes.push_back(a);

                        S_tile_starts.push_back(S_tile_values.size());
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

            static bool check_result(
                const Types::vec_size_t K,
                const Types::COO& expected_res,
                const Types::COO& res, 
                const Params& params
            ) {
                Types::vec_size_t Tj = params.tiling_params.Tj; 
                Types::vec_size_t num_J_tiles = params.tiling_params.num_J_tiles;

                auto start_time = std::chrono::high_resolution_clock::now();

                std::vector<float> R_values;

                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration_ms = (end_time - start_time) / std::chrono::milliseconds(1);

                // put it in the slice-by-slice form to be able to compare with P
                for (int j = 0; j < num_J_tiles; j++) {
                    // start and end col indices for the tile
                    SDDMM::Types::vec_size_t start_ind = j * Tj;
                    SDDMM::Types::vec_size_t end_ind = start_ind + Tj;

                    // limit the max col
                    if (end_ind > expected_res.m) {
                        end_ind = expected_res.m;
                    }

                    for (int i=0; i<expected_res.values.size(); i++) {
                        if (expected_res.cols[i] >= start_ind && expected_res.cols[i] < end_ind) {
                            R_values.push_back(expected_res.values[i]);
                        }
                    }
                }

                local_print("Difference check:");
                float diff = 0.;
                for (int i = 0; i < res.values.size(); i++) {
                    if(abs(R_values.at(i) - res.values.at(i)) > SDDMM::Defines::epsilon){
                        return false;
                    }
                }
                return true;
            }

            static Types::COO run_sm_l2(
                // MatrixParams& matrix_params,
                const SDDMM::Types::COO& S, float sparsity, SDDMM::Types::Matrix& A, SDDMM::Types::Matrix& B,
                Types::vec_size_t N, Types::vec_size_t M, Types::vec_size_t K,
                Params& params,
                Results::ExperimentData* measurements = nullptr
            ) {
                local_print("Matrix sizes:");
                local_print("N: " + std::to_string(N) + ";  M: " + std::to_string(M) + ";  K: " + std::to_string(K) + ";  Sparsity: " + std::to_string(sparsity));
                local_print("\n");

                TilingParams& tiling_params = params.tiling_params;
                SparseParams& sparse_params = params.sparse_params;

                auto S_size = S.values.size();

                // transfer data to GPU
                local_print("Allocating memory & transferring data...");

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

                auto A_size = N * K * sizeof(float);
                auto B_size = M * K * sizeof(float);

                gpuErrchk(cudaMalloc((void**)&A_d, A_size));
                gpuErrchk(cudaMalloc((void**)&B_d, B_size));

                gpuErrchk(cudaMemcpy(A_d, A.data.data(), A_size, cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(B_d, B.data.data(), B_size, cudaMemcpyHostToDevice));

                local_print("\nStarting processing...\n");

                Types::vec_size_t slice_start_ind = 0;
                Types::vec_size_t tile_starts_start_ind = 0;
                Types::vec_size_t active_rows_start_ind = 0;

                // create streams for parallel execution
                auto stream_n = tiling_params.num_J_tiles * tiling_params.num_K_tiles;
                std::vector<cudaStream_t> streams(stream_n);
                for (int i = 0; i < stream_n; i++) {
                    gpuErrchk(cudaStreamCreate(&streams[i]));
                }

                auto start = std::chrono::high_resolution_clock::now();

                // measure execution time
//                cudaEvent_t start_c, stop_c;
//                gpuErrchk(cudaEventCreate(&start_c));
//                gpuErrchk(cudaEventCreate(&stop_c));
//
//                gpuErrchk(cudaEventRecord(start_c));

                //for (int tile_j_id = 0; tile_j_id < tiling_params.num_J_tiles; tile_j_id++) {
                // some rounding error? => don't have time for that
                auto num_J_tiles = sparse_params.active_rows_sizes.size();
                for (int tile_j_id = 0; tile_j_id < num_J_tiles; tile_j_id++) {
                    local_print("Tile J id: " + std::to_string(tile_j_id) + "\n");

                    local_print("Calculating the number of threadblocks...");
                    int num_threadblocks = (sparse_params.active_rows_sizes.at(tile_j_id) + tiling_params.Ti - 1) / tiling_params.Ti;
                    local_print("size: " + std::to_string(num_threadblocks));
                    local_print("\n");

                    // iterate over Tk tiles and launch a kernel for each Tk tile
                    for (int tile_k_id = 0; tile_k_id < tiling_params.num_K_tiles; tile_k_id++) {
                        // the innermost loop, streaming is done along dimension i (assuming that i is the smaller dimension, i.e. N < M)
                        local_print("Tile K id: " + std::to_string(tile_k_id));

                        // launch num_threadblocks with 512 threads in each
                        SML2SDDMM_Kernel::run_kernel(
                                num_threadblocks,
                                // execute each kernel in own stream
//                                streams.at(tile_j_id * tiling_params.num_K_tiles + tile_k_id),
                                // execute everything in a single stream
                                streams.at(0),
                                // S
                                &rows_local_d[slice_start_ind],
                                &cols_d[slice_start_ind],
                                &values_d[slice_start_ind],
                                // P
                                &result_values_d[slice_start_ind],
                                // starts
                                &starts_d[tile_starts_start_ind],
                                // active rows
                                &active_rows_d[active_rows_start_ind], 
                                sparse_params.active_rows_sizes.at(tile_j_id),
                                // A & B
                                A_d, 
                                B_d,
                                // tiles
                                tiling_params.Tk, 
                                tiling_params.Ti,
                                tile_k_id,
                                tiling_params.num_K_tiles,
                                K
                        );
                    }

                    // prepare the next slice
                    slice_start_ind += sparse_params.slice_sizes.at(tile_j_id);
                    tile_starts_start_ind += num_threadblocks + 1;
                    active_rows_start_ind += sparse_params.active_rows_sizes.at(tile_j_id);
                }


//                gpuErrchk(cudaEventRecord(stop_c));
//                gpuErrchk(cudaEventSynchronize(stop_c));
//
//                float milliseconds = 0;
//                cudaEventElapsedTime(&milliseconds, start_c, stop_c);

                gpuErrchk(cudaPeekAtLastError());
                gpuErrchk(cudaDeviceSynchronize());

                auto end = std::chrono::high_resolution_clock::now();
                if(measurements != nullptr){
                    Types::time_duration_unit duration = std::chrono::duration_cast<Types::time_measure_unit>(end - start).count();
                    measurements->durations.push_back(duration);
                    local_print("Duration (CPU chrono): " + std::to_string(duration));
                }

//                local_print("Duration (GPU events): " + std::to_string(milliseconds) + " ms");

                // clean up the streams
                for (int i = 0; i < tiling_params.num_J_tiles * tiling_params.num_K_tiles; ++i) {
                    cudaStreamDestroy(streams[i]);
                }

                // clean up the events
//                gpuErrchk(cudaEventDestroy(start_c));
//                gpuErrchk(cudaEventDestroy(stop_c));

                local_print("Done processing!");

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

                Types::COO result;
                result.n = S.n;
                result.m = S.m;
                result.cols.resize(P_values.size());
                result.rows.resize(P_values.size());
                result.values.resize(P_values.size());

                std::copy(P_values.begin(), P_values.end(), result.values.begin());
                std::copy(S.cols.begin(), S.cols.end(), result.cols.begin());
                std::copy(S.rows.begin(), S.rows.end(), result.rows.begin());
                return result;
            }
        };
    }
}