#pragma once

#include <vector>
#include <random>
#include <map>
#include <iostream>
#include <iomanip>
#include <assert.h>

#include "../../defines.h"


namespace SDDMM {
    namespace Types {
        // forward declarations
        class CSR;
        class COO;

        enum class MatrixFormat {
            RowMajor,
            ColMajor
        };

        class Matrix {
        private:
            MatrixFormat _format = MatrixFormat::RowMajor;

            void flip_matrix_format(){
                std::vector<SDDMM::Types::expmt_t> d;
                d.reserve(data.size());

                for (Types::vec_size_t j = 0; j < m; j++) {
                    for (Types::vec_size_t i = 0; i < n; i++) {
                        d.push_back(this->at(i,j));
                    }
                }

                data = d;
            }

        public:
            std::vector<SDDMM::Types::expmt_t> data;
            // n == number of rows
            // m == number of columns
            Types::vec_size_t n, m;

            Matrix(Types::vec_size_t n, Types::vec_size_t m) : data(n * m, 0.0), n(n), m(m) {}

            // modifiable access mat(i,j), so cannot be used in const matrices
            SDDMM::Types::expmt_t &operator()(const Types::vec_size_t& i, const Types::vec_size_t& j) {
                if(_format == MatrixFormat::RowMajor)
                    return data[i * m + j];
                return data[j*n + i];
            }

            SDDMM::Types::expmt_t at(const Types::vec_size_t i, const Types::vec_size_t j) const {
                if(_format == MatrixFormat::RowMajor)
                    return data[i * m + j];
                return data[j*n + i];
            }

            static Matrix deterministic_gen(Types::vec_size_t n, Types::vec_size_t m, const std::vector<SDDMM::Types::expmt_t>& vals){
                assert(n*m == vals.size() && "Size of the values must correspond to given size!");
                Matrix newM(n, m);
                std::copy(vals.begin(), vals.end(), newM.data.begin());
                return newM;
            }

            // generates an NxM Matrix with elements in range [min,max] and desired sparsity (sparsity 0.7 means that
            // the matrix will be 70% empty)
            static Matrix generate(Types::vec_size_t n, Types::vec_size_t m, float sparsity = 1.0, SDDMM::Types::expmt_t min = -1.0, SDDMM::Types::expmt_t max = 1.0) {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<> value_dist(min, max);
                std::uniform_real_distribution<> sparsity_dist(0.0, 1.0);

                Matrix output(n, m);
                for (Types::vec_size_t i = 0; i < n; i++) {
                    for (Types::vec_size_t j = 0; j < m; j++) {
                        if (sparsity_dist(gen) < sparsity) {
                            output(i, j) = 0.0;
                        } else {
                            output(i, j) = value_dist(gen);
                        }
                    }
                }

                return output;
            }

            // compares two matrices
            bool operator==(const Matrix& other) const {
                // check dimensions

                // return std::equal(data.begin(), data.end(), other.data.begin()) && n==other.n && m==other.m;

                if (n != other.n || m != other.m) {
                    return false;
                }

                // check elements
                for (Types::vec_size_t i = 0; i < n; i++) {
                    for (Types::vec_size_t j = 0; j < m; j++) {
                        // allow a margin for FP comparisons
                        Types::expmt_t a = this->at(i,j);
                        Types::expmt_t b = other.at(i,j);
                        if (std::abs(a - b) > Defines::epsilon) {
                            // std::cout << i << " " << j << " " << std::setprecision(16) << a << " " << std::setprecision(16) << b << std::endl;
                            return false;
                        }
                    }
                }

                return true;
            }

            // pretty prints Matrix
            friend std::ostream &operator<<(std::ostream &os, const Matrix &mat) {
                os << std::endl << "Matrix (" << mat.n << ", " << mat.m << "):" << std::endl;

                for (Types::vec_size_t i = 0; i < mat.n; i++) {
                    for (Types::vec_size_t j = 0; j < mat.m; j++) {
                        os << std::setw(12) << std::left << mat.at(i, j) << ' ';
                    }
                    os << std::endl;
                }

                return os;
            }

            // override multiplication
            Matrix operator*(Matrix& other){
                assert(m == other.n && "Dimensions of matrices must match");
                Matrix newM(n, other.m);
                for(int l_row=0; l_row < n; ++l_row){
                    for(int r_col=0; r_col<other.m; ++r_col){
                        SDDMM::Types::expmt_t result = 0;
                        for(int i=0; i<m; ++i){
                            result += (*this)(l_row, i)*(other(i, r_col));
                        }
                        newM(l_row, r_col) = result;
                    }
                }
                return newM;
            }

             MatrixFormat format() const {
                return _format;
            }

            [[nodiscard]] CSR to_csr() const;

            [[nodiscard]] COO to_coo() const;

            void to_dense_row_major() {
                if(_format == MatrixFormat::RowMajor) return;
                flip_matrix_format();
                _format = MatrixFormat::RowMajor;
            }

            void to_dense_col_major() {
                if(_format == MatrixFormat::ColMajor) return;
                flip_matrix_format();
                _format = MatrixFormat::ColMajor;
            }
        };
    }
}