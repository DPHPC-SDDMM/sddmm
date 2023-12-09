#pragma once

#include <random>
#include <tuple>
#include <vector>
#include <map>
#include <iostream>
#include <iomanip>

#include "../../defines.h"
#include "../matrix/matrix.h"


namespace SDDMM{
    namespace Types {
        // forward declarations
        // class Matrix;
        class COO;

        /* NOTE:
        The data primitive (arbitrarily) decided to store the Matrix content
        are `double`.
        This was chosen, because they offer large accuracy,
        without encumbering the system with too much memory per element.
        Although I do not have citations to support this claim,
        I have heard that the use of double in numerical computations
        is also very common.
        */
        class CSR {
        public:
            std::vector<SDDMM::Types::expmt_t> values;
            std::vector<Types::vec_size_t> col_idx;
            std::vector<Types::vec_size_t> row_ptr;
            Types::vec_size_t n, m;

            CSR() : row_ptr(), col_idx(), values(), n(), m() { }

            // pretty print CSR
            friend std::ostream &operator<<(std::ostream &os, const CSR &mat) {
                os << std::endl << "CSR (" << mat.n << ", " << mat.m << "):" << std::endl;

                os << "Values: ";
                for (SDDMM::Types::expmt_t v : mat.values) os << v << " ";
                os << "\nCol idx: ";
                for (Types::vec_size_t col : mat.col_idx) os << col << " ";
                os << "\nRow ptr: ";
                for (Types::vec_size_t row : mat.row_ptr) os << row << " ";
                os << std::endl;

                return os;
            }

            // bool operator==(const CSR& other){
            //     return std::equal(values.begin(), values.end(), other.values.begin())
            //         && std::equal(col_idx.begin(), col_idx.end(), other.col_idx.begin())
            //         && std::equal(row_ptr.begin(), row_ptr.end(), other.row_ptr.begin())
            //         && n == other.n && m == other.m;
            // }
            /**
             * @param other: A reference to a CSR sparse matrix.
             * @returns Whether all elements of both matrices are equal within an error margin of `Defines::epsilon`.
            */
            bool operator==(const CSR& other){
                return equals(other);
            }

            bool equals(const CSR& other) {
                if (values.size() != other.values.size())
                    return false;
                if (row_ptr.size() != other.row_ptr.size())
                    return false;
                if (values.size() != other.col_idx.size())
                    return false;

                const Types::expmt_t epsilon = Defines::epsilon;

                Types::vec_size_t s = values.size();
                for (Types::vec_size_t i = 0; i < s; ++i) {
                    auto c = std::abs(values[i] - other.values[i]);
                    if (c > epsilon) return false;
                }

                s = row_ptr.size();
                for (Types::vec_size_t i = 0; i < s; ++i) {
                    Types::vec_size_t c = std::fabs(row_ptr[i] - other.row_ptr[i]);
                    if (c > epsilon) return false;
                }

                s = col_idx.size();
                for (Types::vec_size_t i = 0; i < s; ++i) {
                    Types::vec_size_t c = std::fabs(col_idx[i] - other.col_idx[i]);
                    if (c > epsilon) return false;
                }

                return true;
            }

            CSR hadamard(const Matrix& other){
                assert(n>0 && m>0 && other.n>0 && other.m>0 && "All involved matrices must be non-empty!");
                assert(n==other.n && m==other.m && "Matrix dimensions must match!");
                
                CSR res;
                res.n = n;
                res.m = m;
                // reserve space
                res.values.reserve(values.size());
                res.col_idx.reserve(col_idx.size());
                res.row_ptr.reserve(row_ptr.size());

                Types::vec_size_t s = row_ptr.size()-1;
                Types::vec_size_t v_ind = 0;
                
                // init val
                Types::vec_size_t new_ci = 0;
                res.row_ptr.push_back(new_ci);

                // row_ptr
                for(Types::vec_size_t r=0; r<s; ++r){
                    
                    Types::vec_size_t from = row_ptr[r];
                    Types::vec_size_t to = row_ptr[r+1];

                    // col_idx
                    Types::vec_size_t ci;
                    for(ci=from; ci<to; ++ci){
                        
                        Types::vec_size_t c = col_idx[ci];
                        Types::expmt_t new_value = values[v_ind] * other.at(r, c);
                        
                        if(new_value != 0){
                            res.values.push_back(new_value);
                            res.col_idx.push_back(c);
                            new_ci++;
                        }
                        v_ind++;
                    }

                    res.row_ptr.push_back(new_ci);
                }

                return res;
            }

            [[nodiscard]] Matrix to_matrix() const;
            [[nodiscard]] COO to_coo() const;


    //            /**
    //             * Generate a random Matrix represented in the CSR format.
    //             * For details about the COO format, you may read
    //             * https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)
    //             *
    //             * @param n: Number of rows of the generated Matrix.
    //             * @param m: Number of columns of the generated Matrix.
    //             * @param sparsity: A percentage expressing the ratio of non-zero elements to all the elements (n * m).
    //             * @param distribution: Distribution according to which Matrix elements are generated.
    //            */
    //            std::tuple<std::vector<int>, std::vector<int>, std::vector<double>> static Generate(
    //                int n, int m, float sparsity,
    //                bool sort=true, std::string distribution = "uniform"
    //                )
    //            {
    //
    //                // TODO: Implement numerous random distribution schemes.
    //                // Currently, only the default (uniform) is implemented.
    //
    //                // Dynamic sanity check.
    //                if (sparsity < 0 | sparsity > 1.0)
    //                    throw std::out_of_range("Expected a sparsity value from zero (0) to one (1).");
    //
    //
    //                // Compute number of non-zero elements of the Matrix.
    //                int nr_elements = (int) n * m * sparsity;
    //
    //                // Define the three output 1D arrays.
    //                //? Does `reserve` and afterwards `push_back` create less overhead
    //                //? than `row_index(n+1)` and then `row_index[i] = row`?
    //                std::vector<int> row_ptrs(n+1), column_index; std::vector<double> values;
    //                //row_ptrs.reserve(n+1);
    //                column_index.reserve(nr_elements); values.reserve(nr_elements);
    //
    //                // Define random generator
    //                // and distribution functions for the random generation.
    //                // NOTE: Source for random number generator:
    //                // https://stackoverflow.com/questions/15461140/stddefault-random-engine-generate-values-between-0-0-and-1-0
    //                std::random_device rd;
    //                std::default_random_engine generator(rd());
    //                std::uniform_int_distribution<int> row_distribution(0, nr_elements);
    //                std::uniform_int_distribution<int> column_distribution(0, m-1);
    //                // [-1, 1] values were selected because neural networks often deal
    //                // with smalle values.
    //                std::uniform_real_distribution<double> value_distribution(-1.0, 1.0);
    //
    //                // Construct the `row_ptrs` data structure.
    //                // row_ptrs.push_back(0); // The first element of the CSR row pointer array is 0.
    //                for (int i = 1 ; i <= n ; ++i)
    //                {
    //                    // Ensure that all rows have the same number elements.
    //                    // This is how uniformity is ensured globally.
    //                    row_ptrs[i] = row_ptrs[i-1] + nr_elements / n;
    //                    //! Possible bug, because of rounding during the definition of `nr_elements`?
    //                }
    //
    //                // Define the data structure (hash map) which will ensure
    //                // that no (row, column) duplicates are inserted.
    //                // NOTE: In practice, that probability will be fairly low.
    //                std::map<int, int> row_to_column;
    //
    //                // Construct the `column_index` and `values` data structures.
    //                for (int row = 1 ; row <= n ; ++row)
    //                {
    //                    size_t elements_remaining = row_ptrs[row] - row_ptrs[row-1];
    //
    //                    while (elements_remaining)
    //                    {
    //                        int column = column_distribution(generator);
    //                        bool successful_insertion = std::get<1>(row_to_column.emplace(row, column));
    //
    //                        if (successful_insertion)
    //                        {
    //                            double value = value_distribution(generator); // Generate cell value.
    //
    //                            column_index.push_back(column);
    //                            values.push_back(value);
    //                            --elements_remaining;
    //                        }
    //                    }
    //
    //
    //                }
    //
    //                return std::tuple<std::vector<int>, std::vector<int>, std::vector<double>>(row_ptrs, column_index, values);
    //
    //            }

        };
    }
}
