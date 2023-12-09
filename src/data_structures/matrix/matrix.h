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

            /**
             * Just flip the format identifier without changing the data
            */
            void set_matrix_format(MatrixFormat format){
                _format = format;
            }

            bool is_col_major() const {
                return _format == MatrixFormat::ColMajor;
            }

            bool is_row_major() const {
                return _format == MatrixFormat::RowMajor;
            }

            std::vector<SDDMM::Types::expmt_t> data;
            // n == number of rows
            // m == number of columns
            Types::vec_size_t n, m;

            Matrix(Types::vec_size_t n, Types::vec_size_t m) : data(n * m, 0.0), n(n), m(m) {}

            // modifiable access mat(i,j), so cannot be used in const matrices
            inline SDDMM::Types::expmt_t &operator()(const Types::vec_size_t& i, const Types::vec_size_t& j) {
                if(_format == MatrixFormat::RowMajor)
                    return data[i * m + j];
                return data[j*n + i];
            }

            inline SDDMM::Types::expmt_t at(const Types::vec_size_t i, const Types::vec_size_t j) const {
                if(_format == MatrixFormat::RowMajor)
                    return data[i * m + j];
                return data[j*n + i];
            }

            static Matrix deterministic_gen_row_major(Types::vec_size_t n, Types::vec_size_t m, const std::vector<SDDMM::Types::expmt_t>& vals){
                assert(n*m == vals.size() && "Size of the values must correspond to given size!");
                Matrix newM(n, m);
                std::copy(vals.begin(), vals.end(), newM.data.begin());

                newM.set_matrix_format(MatrixFormat::RowMajor);
                return newM;
            }

            static Matrix deterministic_gen_col_major(Types::vec_size_t n, Types::vec_size_t m, const std::vector<SDDMM::Types::expmt_t>& vals){
                assert(n*m == vals.size() && "Size of the values must correspond to given size!");
                Matrix newM(n, m);
                newM.set_matrix_format(MatrixFormat::ColMajor);

                Types::vec_size_t index = 0;
                for (Types::vec_size_t i = 0; i < n; i++) {
                    for (Types::vec_size_t j = 0; j < m; j++) {
                        newM(i, j) = vals[index];
                        index++;
                    }
                }

                return newM;
            }

            // generates an NxM Matrix with elements in range [min,max] and desired sparsity (sparsity 0.7 means that
            // the matrix will be 70% empty)
            static Matrix generate_row_major(
                Types::vec_size_t n, 
                Types::vec_size_t m, 
                float sparsity = 1.0, 
                SDDMM::Types::expmt_t min = -1.0, 
                SDDMM::Types::expmt_t max = 1.0,
                bool verbose = false,
                uint64_t report_sparsity = 10000
            ) {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<> value_dist(min, max);
                std::uniform_real_distribution<> sparsity_dist(0.0, 1.0);

                double total = n*m;
                uint64_t counter = 0;
                Matrix output(n, m);
                if(verbose) TEXT::Gadgets::print_colored_line(100, '#', TEXT::BRIGHT_YELLOW);
                if(verbose) TEXT::Gadgets::print_colored_text_line(std::string("Generate row maj [") + std::to_string(n) + "x" + std::to_string(m) + "], sparsity: " + std::to_string(sparsity), TEXT::BRIGHT_RED);
                for (Types::vec_size_t i = 0; i < n; i++) {
                    for (Types::vec_size_t j = 0; j < m; j++) {
                        if (sparsity_dist(gen) < sparsity) {
                            output(i, j) = 0.0;
                        } else {
                            output(i, j) = value_dist(gen);
                        }
                        counter++;
                        if(counter%report_sparsity == 0){
                            if(verbose) TEXT::Gadgets::print_progress_percent(counter, total, report_sparsity);
                        }
                    }
                }

                output.set_matrix_format(MatrixFormat::RowMajor);

                output.data.shrink_to_fit();

                return output;
            }

            // generates an NxM Matrix with elements in range [min,max] and desired sparsity (sparsity 0.7 means that
            // the matrix will be 70% empty)
            static Matrix generate_col_major(
                Types::vec_size_t n, 
                Types::vec_size_t m, 
                float sparsity = 1.0, 
                SDDMM::Types::expmt_t min = -1.0, 
                SDDMM::Types::expmt_t max = 1.0,
                bool verbose = false,
                uint64_t report_sparsity = 10000
            ) {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<> value_dist(min, max);
                std::uniform_real_distribution<> sparsity_dist(0.0, 1.0);

                double total = n*m;
                uint64_t counter = 0;
                Matrix output(n, m);
                output.set_matrix_format(MatrixFormat::ColMajor);
                if(verbose) TEXT::Gadgets::print_colored_line(100, '#', TEXT::BRIGHT_YELLOW);
                if(verbose) TEXT::Gadgets::print_colored_text_line(std::string("Generate col maj [") + std::to_string(n) + "x" + std::to_string(m) + "], sparsity: " + std::to_string(sparsity), TEXT::BRIGHT_RED);
                for (Types::vec_size_t i = 0; i < n; i++) {
                    for (Types::vec_size_t j = 0; j < m; j++) {
                        if (sparsity_dist(gen) < sparsity) {
                            output(i, j) = 0.0;
                        } else {
                            output(i, j) = value_dist(gen);
                        }
                        counter++;
                        if(counter%report_sparsity == 0){
                            if(verbose) TEXT::Gadgets::print_progress_percent(counter, total, report_sparsity);
                        }
                    }
                }

                output.data.shrink_to_fit();

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
                            std::cout << i << " " << j << " " << std::setprecision(16) << a << " " << std::setprecision(16) << b << std::endl;
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

            Types::Matrix tmult(
                Types::vec_size_t ts, 
                // Types::Matrix& A, 
                Types::Matrix& B
            ){
                assert(m == B.n);
                Types::vec_size_t r_num = n;
                Types::vec_size_t k_num = m;
                Types::vec_size_t c_num = B.m;

                Types::Matrix res(n,B.m);

                for(Types::vec_size_t i=0; i<r_num; i+=ts){
                    for(Types::vec_size_t j=0; j<c_num; j+=ts){
                        for(Types::vec_size_t k=0; k<k_num; k+=ts){
                            Types::vec_size_t i_p_ts = i+ts;
                            for(Types::vec_size_t r=i; r<i_p_ts; ++r){
                                Types::vec_size_t j_p_ts = j+ts;
                                for(Types::vec_size_t c=j; c<j_p_ts; ++c){
                                    Types::expmt_t inner_p = 0;
                                    Types::vec_size_t k_p_ts = k+ts;
                                    // Types::vec_size_t ind = r*r_num + c;
                                    for(Types::vec_size_t kk=k; kk<k_p_ts; ++kk){
                                        // inner_p += (*this)(r, kk)*B(kk,c);
                                        inner_p += data[r*k_num + kk]*B.data[kk*c_num + c];
                                    }
                                    // res(r, c) += inner_p;
                                    res.data[r*c_num + c] += inner_p;
                                }
                            }
                        }
                    }
                }
                return res;
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

            /**
             * Create new matrix in row major order from a col major order matrix
            */
            Matrix get_dense_row_major() {
                assert(is_col_major() && "Attempted to convert row major matrix to row major");

                return get_major_version(*this, MatrixFormat::RowMajor);
            }

            /**
             * Create new matrix in col major order from a row major order matrix
            */
            Matrix get_dense_col_major() {
                assert(is_row_major() && "Attempted to convert col major matrix to col major");

                return get_major_version(*this, MatrixFormat::ColMajor);
            }

            static Matrix get_major_version(Matrix& matrix, MatrixFormat to_format) {
                Matrix res(matrix.n, matrix.m);
                res.data.resize(matrix.data.size());
                res.set_matrix_format(to_format);

                auto m = matrix.m;
                auto n = matrix.n;
                for (Types::vec_size_t i = 0; i < n; i++) {
                    for (Types::vec_size_t j = 0; j < m; j++) {
                        res(i,j) = matrix.at(i,j);
                    }
                }

                return res;
            }

            Matrix get_transposed() {
                Matrix res(this->m, this->n);
                res.data.resize(this->data.size());
                res.set_matrix_format(this->format());

                auto m = this->m;
                auto n = this->n;
                for (Types::vec_size_t i = 0; i < n; i++) {
                    for (Types::vec_size_t j = 0; j < m; j++) {
                        res(j,i) = this->at(i,j);
                    }
                }

                return res;
            }
        };
    }
}