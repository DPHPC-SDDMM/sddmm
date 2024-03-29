#pragma once

#include <iostream>
#include <fstream>
// Libraries for randomness
#include <random>
// Libraries for exception handling
#include <stdexcept>
// Data structures
#include <tuple>
#include <map>
#include <vector>
#include <execution>
#include <algorithm> // std::sort
#include <cassert>
#include <chrono>
#include <unordered_set>
#include <set>
#include <cuda_runtime.h>
#include <curand.h>

#include "../../defines.h"
#include "../matrix/matrix.h"
#include "../csr/csr.h"
#include "../../matrix_file_reading/mmio.h"

namespace SDDMM{
    namespace Types{
        // forward declarations
        class CSR;
        class Matrix;

        class COO {
            /**
             * 'triplet' is a triplet of (Types::vec_size_t, Types::vec_size_t, Types::expmt_t), whose elements represent:
             * 0: row index
             * 1: column index
             * 2: value of cell
             * respectively.
            */
            struct triplet {
                Types::vec_size_t row;
                Types::vec_size_t col;
                Types::expmt_t value;

                bool operator<(const triplet& other) const {
                    if (row == other.row) {
                        return col < other.col;
                    }

                    return row < other.row;
                }
            };

            inline static const char* _ptr_cast(const Types::vec_size_t* src){
                const void* ptot = static_cast<const void*>(src);
                return static_cast<const char*>(ptot);
            }

            inline static const char* _ptr_cast(const Types::expmt_t* src){
                const void* ptot = static_cast<const void*>(src);
                return static_cast<const char*>(ptot);
            }

            inline static const char* _uint_ptr_cast(const uint32_t* src){
                const void* ptot = static_cast<const void*>(src);
                return static_cast<const char*>(ptot);
            }

            inline static const char* _float_ptr_cast(const float* src){
                const void* ptot = static_cast<const void*>(src);
                return static_cast<const char*>(ptot);
            }

            inline static Types::vec_size_t scale_rand(const float r, const Types::vec_size_t m){
                Types::vec_size_t res = static_cast<Types::vec_size_t>(std::ceil(r*m))-1;
                if(res > m-1) res = m-1;
                return res;
            }

        public:
            // structs could be a bit easier to access than tuples but that's up for a discussion

            // std::vector<triplet> data;
            std::vector<Types::vec_size_t> rows;
            std::vector<Types::vec_size_t> cols;
            std::vector<Types::expmt_t> values;

            // std::vector<triplet> init_data;

            Types::vec_size_t n, m;

            COO(): n(), m() {};

            void sort() {
                // std::sort(data.begin(), data.end());
            }

            static bool no_filter_condition(float sparsity, Types::vec_size_t N, Types::vec_size_t M) {
                return sparsity < 0.989f && N > 90000 && M > 90000;
            }

            // pretty prints COO
            friend std::ostream &operator<<(std::ostream &os, const COO &mat) {
                os << std::endl << "COO (" << mat.n << ", " << mat.m << "):" << std::endl;

                os << "Triplets: " << std::endl;
                auto s = mat.values.size();
                for (auto i=0; i<s; ++i) {
                    os << "(" << mat.rows[i] << ", " << mat.cols[i] << ", " << mat.values[i] << ")" << std::endl;
                }

                return os;
            }

            /**
             * @param other: A reference to a COO sparse matrix.
             * @returns Whether all elements of both matrices are equal within an error margin of `Defines::epsilon`.
            */
            bool operator==(const COO& other){
                return equals(other);
            }

            bool equals(const COO& other) {
                if (values.size() != other.values.size())
                    return false;

                Types::vec_size_t s = values.size();
                const Types::expmt_t epsilon = Defines::epsilon;
                for (Types::vec_size_t i = 0; i < s; ++i) {
                    auto a = std::fabs(cols[i] - other.cols[i]);
                    if (a > epsilon) return false;
                    auto b = std::fabs(rows[i] - other.rows[i]);
                    if (b > epsilon) return false;
                    auto c = std::abs(values[i] - other.values[i]);
                    if (c > epsilon) return false;
                }

                return true;
            }

            private:
            /**
            * @brief Generate a randomized, uniformly distributed, sparse matrix S^{NxM} using cuRAND (https://docs.nvidia.com/cuda/curand/index.html). 
            * This can be used to generate large random matrices for benchmarking.
            * 
            * The method uses cuRAND to generate a sufficient number of random (n_i, m_i) indices by genreating pairs of numbers in (0,1] then scaling
            * them to the required size for an NxM matrix, then filtering indices that are generated twice (may happen, especially for very full and small matrices)
            * using the binary tree version of the C++ std lib set. The downside is, that this takes a lot of time, the upside is that the resulting indices are
            * already sorted (NOTE: sorting the indices is mandatory to be able to convert the COO matrix to CSR, which is the required input format for cuSPARSE SDDMM).
            *
            * @param n: dimension 1 of desired matrix
            * @param m: dimensino 2 of desired matrix
            * @param t: current try of the generator (used for some verbose output only)
            * @param tries: number of retries in case to few random indices are generated
            * @param sparsity: required amount of zeros (0 == no zeros, 1.0 all zeros)
            * @param verbose: produce some progress output
            * @param report_sparsity: after how many indices is the progress report updated (#gen_entries % report_sparsity ==) => print something
            * @return generated random matrix in COO format
            * 
            * @warning This is only suitable to generate larger matrices of 90% or more zeros (sparsity >= 0.9). For less sparse matrices, luck is a real ingredient ;-)
            *
            * @sa
            * - [COO matrix format](https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO))
            * - [Hadamard product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices))
            */
            static bool _generate_row_major_curand(
                Types::vec_size_t n, 
                Types::vec_size_t m, 
                int t,
                int tries,
                COO& result,
                float sparsity,
                bool verbose,
                uint64_t report_sparsity
            ){
                result.cols.clear();
                result.rows.clear();
                result.values.clear();

                uint64_t total = static_cast<uint64_t>(std::ceil(n*m*(1.0f - sparsity)));
                uint64_t gen_total = static_cast<uint64_t>(1.2f*total);

                curandGenerator_t gen;
                std::vector<float> n_rows(gen_total);
                std::vector<float> n_cols(gen_total);
                float* n_rows_d;
                float* n_cols_d;

                gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&n_rows_d), gen_total*sizeof(float)));
                gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&n_cols_d), gen_total*sizeof(float)));
                rand_gpuErrchk(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
                auto seed = static_cast<uint64_t>(std::chrono::system_clock::now().time_since_epoch().count());
                rand_gpuErrchk(curandSetPseudoRandomGeneratorSeed(gen, seed));
                rand_gpuErrchk(curandGenerateUniform(gen, n_rows_d, gen_total));
                rand_gpuErrchk(curandGenerateUniform(gen, n_cols_d, gen_total));

                gpuErrchk(cudaMemcpy(n_rows.data(), n_rows_d, gen_total*sizeof(float), cudaMemcpyDeviceToHost));
                gpuErrchk(cudaMemcpy(n_cols.data(), n_cols_d, gen_total*sizeof(float), cudaMemcpyDeviceToHost));

                rand_gpuErrchk(curandDestroyGenerator(gen));
                gpuErrchk(cudaFree(n_cols_d));
                gpuErrchk(cudaFree(n_rows_d));

                result.n = n;
                result.m = m;
                result.values.resize(total);
                // doesn't matter which values we use here as long as we have enough of them
                // => no filtering etc... necessary, just copy
                memcpy(result.values.data(), n_rows.data(), total*sizeof(Types::expmt_t));

                // try to multithreaded sort the stuff (or maybe not ^^)
                if (!no_filter_condition(sparsity, n, m)) {
                    result.cols.reserve(total);
                    result.rows.reserve(total);
                    if (verbose) TEXT::Gadgets::print_colored_text_line(std::string("...Filter coords..."), TEXT::BRIGHT_BLUE);
                    Types::sorted_coo_collector collector;
                    uint64_t i = 0;
                    while (collector.size() < total && i <= gen_total) {
                        if (i == gen_total) {
                            return false;
                        }
                        collector.insert({ scale_rand(n_rows[i], n), scale_rand(n_cols[i], m) });
                        i++;
                        if (collector.size() % report_sparsity == 0) {
                            if (verbose) TEXT::Gadgets::print_progress_percent(collector.size(), static_cast<double>(total), report_sparsity);
                        }
                    }

                    if (verbose) TEXT::Gadgets::print_colored_text_line(std::string("...Split coords..."), TEXT::BRIGHT_BLUE);

                    for (auto& p : collector) {
                        result.rows.push_back(p.first);
                        result.cols.push_back(p.second);
                        if (result.rows.size() % report_sparsity == 0) {
                            if (verbose) TEXT::Gadgets::print_progress_percent(result.rows.size(), static_cast<double>(total), report_sparsity);
                        }
                    }
                }

                result.values.shrink_to_fit();
                result.cols.shrink_to_fit();
                result.rows.shrink_to_fit();

                if(verbose){
                    TEXT::Gadgets::print_colored_text_line(
                        std::string("Summary: generated ") + 
                        std::to_string(gen_total) + 
                        std::string(" random pairs, out of which at least ") +
                        std::to_string(total) +
                        std::string(" were distinct using ") +
                        std::to_string(t) + std::string("/") + std::to_string(tries) +
                        std::string(" tries"),
                        TEXT::BRIGHT_GREEN
                    );
                }

                return true;
            }

            public:

            /**
            * @brief Generate a randomized, uniformly distributed, sparse matrix S^{NxM} using cuRAND (https://docs.nvidia.com/cuda/curand/index.html).
            * This can be used to generate large random matrices for benchmarking.
            *
            * The method uses cuRAND to generate a sufficient number of random (n_i, m_i) indices by genreating pairs of numbers in (0,1] then scaling
            * them to the required size for an NxM matrix, then filtering indices that are generated twice (may happen, especially for very full and small matrices)
            * using the binary tree version of the C++ std lib set. The downside is, that this takes a lot of time, the upside is that the resulting indices are
            * already sorted (NOTE: sorting the indices is mandatory to be able to convert the COO matrix to CSR, which is the required input format for cuSPARSE SDDMM).
            *
            * @param n: dimension 1 of desired matrix
            * @param m: dimensino 2 of desired matrix
            * @param sparsity: required amount of zeros (0 == no zeros, 1.0 all zeros)
            * @param verbose: produce some progress output
            * @param report_sparsity: after how many indices is the progress report updated (#gen_entries % report_sparsity ==) => print something
            * @return generated random matrix in COO format
            *
            * @warning This is only suitable to generate larger matrices of 90% or more zeros (sparsity >= 0.9). For less sparse matrices, luck is a real ingredient ;-)
            *
            * @sa
            * - [COO matrix format](https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO))
            * - [Hadamard product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices))
            */
            static COO generate_row_major_curand(
                Types::vec_size_t n, 
                Types::vec_size_t m, 
                float sparsity = 1.0,
                bool verbose = true,
                uint64_t report_sparsity = 10000
            ){
                if(verbose) TEXT::Gadgets::print_colored_line(100, '#', TEXT::BRIGHT_YELLOW);
                if(verbose) TEXT::Gadgets::print_colored_text_line(std::string("Generate sparse col maj [") + std::to_string(n) + "x" + std::to_string(m) + "], sparsity: " + std::to_string(sparsity), TEXT::BRIGHT_RED);

                COO result;
                int tries = 100;
                auto start = std::chrono::high_resolution_clock::now();
                for(int t=1; t<=tries; ++t){
                    if(_generate_row_major_curand(n, m, t, tries, result, sparsity, verbose, report_sparsity)){
                        auto stop = std::chrono::high_resolution_clock::now();
                        if(verbose) TEXT::Gadgets::print_colored_text_line(std::string("..Finished in [") + std::to_string(std::chrono::duration_cast<SDDMM::Types::time_measure_unit>(stop - start).count()/1000.0) + std::string("ms]"), TEXT::BRIGHT_RED);
                        return result;
                    }

                    throw std::runtime_error("Not enough distinct coordinate pairs generated");
                }
                return COO();
            }

            /**
            * @brief Generate a randomized, uniformly distributed, sorted sparse matrix S^{NxM} using the standard C++ random generator.
            *
            * @param n: dimension 1 of desired matrix
            * @param m: dimensino 2 of desired matrix
            * @param sparsity: required amount of zeros (0 == no zeros, 1.0 all zeros)
            * @param min: smallest number in the matrix
            * @param max: largest number in the matrix
            * @param verbose: produce some progress output
            * @param report_sparsity: after how many indices is the progress report updated (#gen_entries % report_sparsity ==) => print something
            * @return generated random matrix in COO format
            *
            * @sa
            * - [COO matrix format](https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO))
            * - [Hadamard product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices))
            */
            static COO generate_row_major_sorted(
                Types::vec_size_t n, 
                Types::vec_size_t m, 
                float sparsity = 1.0, 
                SDDMM::Types::expmt_t min = -1.0, 
                SDDMM::Types::expmt_t max = 1.0,
                bool verbose = false,
                uint64_t report_sparsity = 10000
            ){
                return generate_row_major<Types::sorted_coo_collector>(n, m, sparsity, min, max, verbose, report_sparsity);
            }

            /**
            * @brief Generate a randomized, uniformly distributed, **unsorted** sparse matrix S^{NxM} using the standard C++ random generator.
            *
            * @param n: dimension 1 of desired matrix
            * @param m: dimensino 2 of desired matrix
            * @param sparsity: required amount of zeros (0 == no zeros, 1.0 all zeros)
            * @param min: smallest number in the matrix
            * @param max: largest number in the matrix
            * @param verbose: produce some progress output
            * @param report_sparsity: after how many indices is the progress report updated (#gen_entries % report_sparsity ==) => print something
            * @return generated random matrix in COO format
            *
            * @warning This generation method is not suitable if the COO matrix has to be converted to CSR at some point since converting to CSR
            * requires sorted indices.
            * 
            * @sa
            * - [COO matrix format](https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO))
            * - [Hadamard product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices))
            */
            static COO generate_row_major_unsorted(
                Types::vec_size_t n, 
                Types::vec_size_t m, 
                float sparsity = 1.0, 
                SDDMM::Types::expmt_t min = -1.0, 
                SDDMM::Types::expmt_t max = 1.0,
                bool verbose = false,
                uint64_t report_sparsity = 10000
            ){
                return generate_row_major<Types::unsorted_coo_collector>(n, m, sparsity, min, max, verbose, report_sparsity);
            }

            // generates an NxM Matrix with elements in range [min,max] and desired sparsity (sparsity 0.7 means that
            // the matrix will be 70% empty)
            template<typename set_type>
            static COO generate_row_major(
                Types::vec_size_t n, 
                Types::vec_size_t m, 
                float sparsity = 1.0, 
                SDDMM::Types::expmt_t min = -1.0, 
                SDDMM::Types::expmt_t max = 1.0,
                bool verbose = false,
                uint64_t report_sparsity = 10000
            ) {
                // https://stackoverflow.com/questions/18501081/generating-random-number-within-cuda-kernel-in-a-varying-range
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<> value_dist(min, max);
                // std::uniform_real_distribution<> sparsity_dist(0, 1.0);
                std::uniform_int_distribution<> r_dist(0, n-1);
                std::uniform_int_distribution<> c_dist(0, m-1);

                set_type nnz_locs;

                uint64_t total = static_cast<uint64_t>(std::ceil(n*m*(1.0f - sparsity)));
                uint64_t counter = 0;
                COO output;
                output.n = n;
                output.m = m;
                if(verbose) TEXT::Gadgets::print_colored_line(100, '#', TEXT::BRIGHT_YELLOW);
                if(verbose) TEXT::Gadgets::print_colored_text_line(std::string("Generate sparse row maj [") + std::to_string(n) + "x" + std::to_string(m) + "], sparsity: " + std::to_string(sparsity), TEXT::BRIGHT_RED);
                if(verbose) TEXT::Gadgets::print_colored_text_line(std::string("...Generate coords..."), TEXT::BRIGHT_BLUE);
                while(nnz_locs.size() < total){
                    Types::vec_size_t r = r_dist(gen);
                    Types::vec_size_t c = c_dist(gen);
                    nnz_locs.insert({r, c});

                    if(nnz_locs.size()%report_sparsity == 0){
                        if(verbose) TEXT::Gadgets::print_progress_percent(nnz_locs.size(), static_cast<double>(total), report_sparsity);
                    }
                }

                // if(verbose) TEXT::Gadgets::print_colored_text_line(std::string("... => Gave up count: ") + std::to_string(gave_up), TEXT::BRIGHT_CYAN);
                total = nnz_locs.size();

                counter = 0;
                for(auto& p : nnz_locs){
                    output.rows.push_back(p.first);
                    output.cols.push_back(p.second);
                    output.values.push_back(value_dist(gen));
                    counter++;
                    if(counter%report_sparsity == 0){
                        if(verbose) TEXT::Gadgets::print_progress_percent(counter, static_cast<double>(total), report_sparsity);
                    }
                }

                output.values.shrink_to_fit();
                output.cols.shrink_to_fit();
                output.rows.shrink_to_fit();

                return output;
            }

            /**
             * @param other: A reference to a dense matrix.
             * @returns The per-element product of the input dense matrix with this one.
            */
            COO hadamard(const Matrix& other){
                assert(n>0 && m>0 && other.n>0 && other.m>0 && "All involved matrices must be non-empty!");
                assert(n==other.n && m==other.m && "Matrix dimensions must match!");
                
                COO res;
                res.n = n;
                res.m = m;
                // reserve space
                res.values.reserve(values.size());
                res.rows.reserve(rows.size());
                res.cols.reserve(cols.size());

                // Types::vec_size_t o_col = 0;
                // Types::vec_size_t o_row = 0;
                Types::vec_size_t s = values.size();
                for(Types::vec_size_t t=0; t<s; ++t){
                    Types::vec_size_t col = cols[t];
                    Types::vec_size_t row = rows[t];
                    Types::expmt_t new_val = values[t] * other.at(row, col);
#ifdef SDDMM_PARALLEL_CPU_ZERO_FILTER
                    if(new_val != 0){
                        res.cols.push_back(col);
                        res.rows.push_back(row);
                        res.values.push_back(new_val);
                    }
#else
                    res.cols.push_back(col);
                    res.rows.push_back(row);
                    res.values.push_back(new_val);
#endif
                }

                return res;
            }

            /**
            * @brief Store a generated set of matrices for a Hadamard product (X @ Y).Hadamard(sparse) into a binary file, because regenerate matrices every time
            * takes time and ruins reproducibility
            *
            * @param path: storage location for the binary file (must end with a path splitter like / or \\ depending on the operating system)
            * @param sparse: COO matrix
            * @param sparse_sparsity: the proportion of zeros in the sparse matrix (0 == no zeros, 1.0 == all zeros)
            * @param X: left-hand dense matrix for the Hadamard product
            * @param X_sparsity: the proportion of zeros in the matrix X (0 == no zeros, 1.0 == all zeros)
            * @param Y: right-hand dense matrix for the Hadamard product
            * @param Y_sparsity: the proportion of zeros in the matrix Y (0 == no zeros, 1.0 == all zeros)
            * @out out_size_written: uint64 reference will contain the written number of bytes
            * @return the full path of the generated binary file
            *
            * @warning Make sure to use the correct path splitter
            */
            static std::string hadamard_to_bin_file(
                std::string path, 
                const SDDMM::Types::COO& sparse, 
                const float sparse_sparsity,
                const SDDMM::Types::Matrix& X, 
                const float X_sparsity,
                const SDDMM::Types::Matrix& Y,
                const float Y_sparsity,
                uint64_t& out_size_written,
                bool verbose = false
            )
            {
                auto last = path[path.size()-1];
                assert((last == '/' || last == '\\') && "path must end with a path separator / or \\");

                auto created_at = std::chrono::system_clock::now();
                auto created_at_t = std::chrono::system_clock::to_time_t(created_at);
                std::string time = std::string(std::ctime(&created_at_t));
                std::replace(time.begin(), time.end(), ' ', '_');
                std::replace(time.begin(), time.end(), ':', '-');
                time = time.substr(0, time.size()-1);

                std::stringstream name;
                name << "hadamard_S-" << sparse.n << "x" << sparse.m << "-" << sparse_sparsity 
                     << "_X-" << X.n << "x" << X.m << "-" << X_sparsity 
                     << "_Y-" << Y.n << "x" << Y.m << "-" << Y_sparsity 
                     << "___" << time
                     << "___" << created_at.time_since_epoch().count()
                     << ".bindat";

                std::ofstream output_file(path + name.str(), std::ios::out | std::ios::binary);

                // header
                uint32_t s_t_vec_size = sizeof(Types::vec_size_t);
                uint32_t s_t_expmt_size = sizeof(Types::expmt_t);
                output_file.write(_uint_ptr_cast(&s_t_vec_size), sizeof(uint32_t));
                output_file.write(_uint_ptr_cast(&s_t_expmt_size), sizeof(uint32_t));

                // sparse matrix
                Types::vec_size_t s = sparse.values.size();
                output_file.write(_float_ptr_cast(&sparse_sparsity), sizeof(float));
                output_file.write(_ptr_cast(&s), sizeof(Types::vec_size_t));
                output_file.write(_ptr_cast(&sparse.n), sizeof(Types::vec_size_t));
                output_file.write(_ptr_cast(&sparse.m), sizeof(Types::vec_size_t));
                output_file.write(_ptr_cast(sparse.values.data()), sparse.values.size() * sizeof(Types::expmt_t));
                output_file.write(_ptr_cast(sparse.rows.data()), sparse.rows.size() * sizeof(Types::vec_size_t));
                output_file.write(_ptr_cast(sparse.cols.data()), sparse.cols.size() * sizeof(Types::vec_size_t));

                uint64_t size0 = 3 * sizeof(Types::vec_size_t) + sizeof(float);
                uint64_t size1 = sparse.values.size() * sizeof(Types::expmt_t);
                uint64_t size2 = sparse.rows.size() * sizeof(Types::vec_size_t);
                uint64_t size3 = sparse.cols.size() * sizeof(Types::vec_size_t);

                if (verbose) {
                    std::cout << "write: " << size0 << std::endl;
                    std::cout << "write: " << size1 << std::endl;
                    std::cout << "write: " << size2 << std::endl;
                    std::cout << "write: " << size3 << std::endl;
                }

                // matrix X
                s = X.data.size();
                output_file.write(_float_ptr_cast(&X_sparsity), sizeof(float));
                output_file.write(_ptr_cast(&s), sizeof(Types::vec_size_t));
                output_file.write(_ptr_cast(&X.n), sizeof(Types::vec_size_t));
                output_file.write(_ptr_cast(&X.m), sizeof(Types::vec_size_t));
                output_file.write(_uint_ptr_cast(X.is_row_major() ? &Constants::row_storage : &Constants::col_storage), sizeof(uint32_t));
                output_file.write(_ptr_cast(X.data.data()), X.data.size() * sizeof(Types::expmt_t));

                uint64_t size4 = 3 * sizeof(Types::vec_size_t) + sizeof(float) + sizeof(uint32_t);
                uint64_t size5 = X.data.size() * sizeof(Types::expmt_t);
                
                if (verbose) {
                    std::cout << "write: " << size4 << std::endl;
                    std::cout << "write: " << size5 << std::endl;
                }

                // matrix Y
                s = Y.data.size();
                output_file.write(_float_ptr_cast(&Y_sparsity), sizeof(float));
                output_file.write(_ptr_cast(&s), sizeof(Types::vec_size_t));
                output_file.write(_ptr_cast(&Y.n), sizeof(Types::vec_size_t));
                output_file.write(_ptr_cast(&Y.m), sizeof(Types::vec_size_t));
                output_file.write(_uint_ptr_cast(Y.is_row_major() ? &Constants::row_storage : &Constants::col_storage), sizeof(uint32_t));
                output_file.write(_ptr_cast(Y.data.data()), Y.data.size() * sizeof(Types::expmt_t));
                
                uint64_t size6 = 3 * sizeof(Types::vec_size_t) + sizeof(float) + sizeof(uint32_t);
                uint64_t size7 = Y.data.size() * sizeof(Types::expmt_t);
                
                if (verbose) {
                    std::cout << "write: " << size6 << std::endl;
                    std::cout << "write: " << size7 << std::endl;
                    std::cout << "total: " << (size0 + size1 + size2 + size3 + size4 + size5 + size6 + size7) << std::endl;
                }

                output_file.close();

                out_size_written = size0 + size1 + size2 + size3 + size4 + size5 + size6 + size7;

                return path + name.str();
            }

            /**
            * @brief Load a stored set of Hadamard matrices stored with hadamard_to_bin_file from a binary file.
            *
            * @param path: full path of the binary file or path relative to the executable (must end with a path splitter like / or \\ depending on the operating system)
            * @out out_coo: loaded sparse matrix in COO format
            * @out out_csr: loaded sparse matrix in CSR format
            * @out out_sparse_sparsity: the proportion of zeros in the loaded sparse matrix (0 == no zeros, 1.0 == all zeros)
            * @out out_X: loaded left-hand dense matrix X
            * @out out_x_sparsity: the proportion of zeros in the matrix X (0 == no zeros, 1.0 == all zeros)
            * @out out_Y: loaded right-hand dense matrix Y
            * @out out_y_sparsity: the proportion of zeros in the matrix Y (0 == no zeros, 1.0 == all zeros)
            * @out out_size_read: uint64 containing the number of bytes read from file
            * @param verbose: set to true for some verbose output
            *
            * @warning Make sure to use the correct path splitter
            */
            static void hadamard_from_bin_file(
                std::string path, 
                SDDMM::Types::COO& out_coo,
                SDDMM::Types::CSR& out_csr,
                float& out_sparse_sparsity,
                SDDMM::Types::Matrix& out_X,
                float& out_x_sparsity, 
                SDDMM::Types::Matrix& out_Y,
                float& out_y_sparsity,
                uint64_t& out_size_read,
                bool verbose = false
                )
            {
                std::ifstream input_file(path, std::ios::in | std::ios::binary);
                input_file.seekg (0, input_file.end);
                uint64_t length = input_file.tellg();
                input_file.seekg (0, input_file.beg);

                char* buffer = new char[length];
                input_file.read(buffer, length);

                uint32_t s_t_vec_size;
                memcpy(&s_t_vec_size, &buffer[0], sizeof(uint32_t));
                uint32_t s_t_expmt_size;
                memcpy(&s_t_expmt_size, &buffer[4], sizeof(uint32_t));
                // uint32_t next_size = static_cast<uint32_t>(buffer[8]);
                // memcpy(&next_size, &buffer[8], sizeof(uint32_t));

                if(sizeof(Types::vec_size_t) != s_t_vec_size){
                    throw std::runtime_error(
                        std::string("Impossible to import file ") + 
                        path + 
                        std::string("\n...required sizeof vec_size_t is ") + 
                        std::to_string(s_t_vec_size) + 
                        std::string(" but program sizeof vec_size_t is ") + 
                        std::to_string(sizeof(Types::vec_size_t)) +
                        std::string("\nRecompile the program with the appropriate type size")
                    );
                }

                if(sizeof(Types::expmt_t) != s_t_expmt_size){
                    throw std::runtime_error(
                        std::string("Impossible to import file ") + 
                        path +
                        std::string("\n...required sizeof expmt_t is ") + 
                        std::to_string(s_t_vec_size) + 
                        std::string(" but program sizeof expmt_t is ") + 
                        std::to_string(sizeof(Types::expmt_t)) +
                        std::string("\nRecompile the program with the appropriate type size")
                    );
                }

                uint64_t f_index = 8;
                                      memcpy(&out_sparse_sparsity, &buffer[f_index], sizeof(float)); f_index += sizeof(float);
                Types::vec_size_t s1; memcpy(&s1, &buffer[f_index], sizeof(Types::vec_size_t)); f_index += sizeof(Types::vec_size_t);
                Types::vec_size_t n1; memcpy(&n1, &buffer[f_index], sizeof(Types::vec_size_t)); f_index += sizeof(Types::vec_size_t);
                Types::vec_size_t m1; memcpy(&m1, &buffer[f_index], sizeof(Types::vec_size_t)); f_index += sizeof(Types::vec_size_t);
                out_coo.n = n1;
                out_coo.m = m1;
                out_coo.values.resize(s1); memcpy(out_coo.values.data(), &buffer[f_index], s1*sizeof(Types::expmt_t));    f_index += s1*sizeof(Types::expmt_t);
                out_coo.rows.resize(s1);   memcpy(out_coo.rows.data(),   &buffer[f_index], s1*sizeof(Types::vec_size_t)); f_index += s1*sizeof(Types::vec_size_t);
                out_coo.cols.resize(s1);   memcpy(out_coo.cols.data(),   &buffer[f_index], s1*sizeof(Types::vec_size_t)); f_index += s1*sizeof(Types::vec_size_t);

                uint64_t size0 = sizeof(out_sparse_sparsity) + sizeof(s1) + sizeof(n1) + sizeof(m1);
                uint64_t size1 = out_coo.values.size() * sizeof(Types::expmt_t);
                uint64_t size2 = out_coo.rows.size() * sizeof(Types::vec_size_t);
                uint64_t size3 = out_coo.cols.size() * sizeof(Types::vec_size_t);

                if (verbose) {
                    std::cout << "read: " << size0 << std::endl;
                    std::cout << "read: " << size1 << std::endl;
                    std::cout << "read: " << size2 << std::endl;
                    std::cout << "read: " << size3 << std::endl;
                }

                                      memcpy(&out_x_sparsity, &buffer[f_index], sizeof(float)); f_index += sizeof(float);
                Types::vec_size_t s2; memcpy(&s2, &buffer[f_index], sizeof(Types::vec_size_t)); f_index += sizeof(Types::vec_size_t);
                Types::vec_size_t n2; memcpy(&n2, &buffer[f_index], sizeof(Types::vec_size_t)); f_index += sizeof(Types::vec_size_t);
                Types::vec_size_t m2; memcpy(&m2, &buffer[f_index], sizeof(Types::vec_size_t)); f_index += sizeof(Types::vec_size_t);
                uint32_t          t2; memcpy(&t2, &buffer[f_index], sizeof(uint32_t));          f_index += sizeof(uint32_t);
                out_X.n = n2;
                out_X.m = m2;
                out_X.set_matrix_format(t2 == Constants::col_storage ? Types::MatrixFormat::ColMajor : Types::MatrixFormat::RowMajor);
                out_X.data.resize(s2);
                memcpy(out_X.data.data(), &buffer[f_index], s2*sizeof(Types::expmt_t)); f_index += s2*sizeof(Types::expmt_t);
                
                uint64_t size4 = sizeof(out_x_sparsity) + sizeof(s2) + sizeof(n2) + sizeof(m2) + sizeof(t2);
                uint64_t size5 = out_X.data.size() * sizeof(Types::expmt_t);
                
                if (verbose) {
                    std::cout << "read: " << size4 << std::endl;
                    std::cout << "read: " << size5 << std::endl;
                }

                                      memcpy(&out_y_sparsity, &buffer[f_index], sizeof(float)); f_index += sizeof(float);
                Types::vec_size_t s3; memcpy(&s3, &buffer[f_index], sizeof(Types::vec_size_t)); f_index += sizeof(Types::vec_size_t);
                Types::vec_size_t n3; memcpy(&n3, &buffer[f_index], sizeof(Types::vec_size_t)); f_index += sizeof(Types::vec_size_t);
                Types::vec_size_t m3; memcpy(&m3, &buffer[f_index], sizeof(Types::vec_size_t)); f_index += sizeof(Types::vec_size_t);
                uint32_t          t3; memcpy(&t3, &buffer[f_index], sizeof(uint32_t));          f_index += sizeof(uint32_t);
                out_Y.n = n3;
                out_Y.m = m3;
                out_Y.set_matrix_format(t3 == Constants::col_storage ? Types::MatrixFormat::ColMajor : Types::MatrixFormat::RowMajor);
                out_Y.data.resize(s3);
                memcpy(out_Y.data.data(), &buffer[f_index], s3*sizeof(Types::expmt_t)); f_index += s3*sizeof(Types::expmt_t);
                
                uint64_t size6 = sizeof(out_y_sparsity) + sizeof(s3) + sizeof(n3) + sizeof(m3) + sizeof(t3);
                uint64_t size7 = out_Y.data.size() * sizeof(Types::expmt_t);
                
                if (verbose) {
                    std::cout << "read: " << size6 << std::endl;
                    std::cout << "read: " << size7 << std::endl;
                    std::cout << "total: " << (size0 + size1 + size2 + size3 + size4 + size5 + size6 + size7) << std::endl;
                }

                assert(f_index == length && "f_index must be the same as length at the end!!");

                out_csr = out_coo.to_csr();
                delete[] buffer;

                out_size_read = size0 + size1 + size2 + size3 + size4 + size5 + size6 + size7;
            }

            /**
            * @brief Load a matrix stored in matrix market format (https://math.nist.gov/MatrixMarket/formats.html)
            *
            * @param filepath: full path to the matrix file (NOTE: make sure to use the correct path splitter, even though, it should work...)
            * @out out_size_read: uint64 containing the number of bytes read from file
            * @return loaded matrix in COO format
            *
            * @warning Make sure to use the correct path splitter
            */
            static COO read_matrix_market_file(const char* filepath, uint64_t& out_size_read, bool verbose=false)
            {
                // convert to whatever os we are on...
                std::string temp = filepath;
                if (Defines::path_separator == '/') {
                    std::replace(temp.begin(), temp.end(), '\\', '/');
                }
                else {
                    std::replace(temp.begin(), temp.end(), '/', '\\');
                }
                filepath = temp.c_str();

                // Output variable
                Types::COO output;

                Types::vec_size_t nr_nonzeroes;

                Types::vec_size_t r_in, c_in; // Dummy variables for loading file content.
                Types::expmt_t v; // Dummy variable for loading file content.
                char test_string[MM_MAX_LINE_LENGTH]; // Dummy variable for skipping file content.


                /*
                C part of the code START
                This part is to extract the `matcode`.
                */
                MM_typecode matcode;
                FILE *f;

                // Make sure that the correct file is read.
                if ((f = fopen(filepath, "r")) == NULL) 
                    exit(1);

                if (mm_read_banner(f, &matcode) != 0)
                {
                    std::cout << "Could not process Matrix Market banner" << std::endl;
                    exit(1);
                }
                /*
                C part of the code FINISH
                This part is to extract the `matcode`.
                */
                

                std::ifstream in;
                in.open(filepath);
                if (!in.is_open()){
                    std::cout << "File was not opened correctly" << std::endl;
                    exit(1);
                }

                // Skip unneeded parts of the file (file acknowledgements).
                while(in.peek() == '%'){ in.getline(test_string, MM_MAX_LINE_LENGTH); }

                // Read number of rows, columns and number of non-zero elements of the sparse matrix.
                in >> output.n >> output.m >> nr_nonzeroes;
                std::vector<triplet> triplets;
                triplets.reserve(nr_nonzeroes);

                Types::vec_size_t index = 0;
                // The second check is made in case
                // an additional final line is added in the matrix file.
                while(!in.eof() && index < nr_nonzeroes){ 
                    if (mm_is_pattern(matcode)){
                        in >> r_in >> c_in;
                        v = 1;
                    }
                    else { in >> r_in >> c_in >> v; }

                    // adjust from 1-based to 0-based
                    triplets.push_back({
                        r_in-1,
                        c_in-1, 
                        v 
                    });
                    index++;
                    // output.data[index++] = {r_in-1, c_in-1, v};
                }

                in.close(); // Don't forget to close the file!

                std::sort(triplets.begin(), triplets.end(), [](auto& e1, auto& e2) {
                    //return e1.x < e2.x || (e1.x == e2.x && e1.y < e2.y);
                    return e1 < e2;
                });

                // Allocate memory for data.
                // output.data.resize(nr_nonzeroes);
                output.values.reserve(nr_nonzeroes);
                output.cols.reserve(nr_nonzeroes);
                output.rows.reserve(nr_nonzeroes);

                for (auto& t : triplets) {
                    output.rows.push_back(t.row);
                    output.cols.push_back(t.col);
                    output.values.push_back(t.value);
                }

                return output;
            }

            [[nodiscard]] CSR to_csr() const;

            [[nodiscard]] Matrix to_matrix() const;
        };
    }
}