#pragma once

#include <vector>
#include <random>
#include <map>
#include <iostream>
#include <iomanip>

#include "../../defines.h"


namespace SDDMM {
    // forward declarations
    class CSR;
    class COO;

    class Matrix {
    private:
        static constexpr double EPSILON = 1e-9;

    public:
        std::vector<double> data;
        vec_size_t n, m;

        Matrix(vec_size_t n, vec_size_t m) : data(n * m, 0.0), n(n), m(m) {}

        // modifiable access mat(i,j), so cannot be used in const matrices
        double &operator()(vec_size_t i, vec_size_t j) {
            return data[i * m + j];
        }

        // non-modifiable access mat(i,j), so can be called with const matrices
        const double &operator()(vec_size_t i, vec_size_t j) const {
            return data[i * m + j];
        }

        // generates an NxM Matrix with elements in range [min,max] and desired sparsity (sparsity 0.7 means that
        // the matrix will be 70% empty)
        Matrix static generate(vec_size_t n, vec_size_t m, double sparsity = 1.0, double min = -1.0, double max = 1.0) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> value_dist(min, max);
            std::uniform_real_distribution<> sparsity_dist(0.0, 1.0);

            Matrix output(n, m);
            for (vec_size_t i = 0; i < n; i++) {
                for (vec_size_t j = 0; j < m; j++) {
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
            if (n != other.n || m != other.m) {
                return false;
            }

            // check elements
            for (vec_size_t i = 0; i < n; i++) {
                for (vec_size_t j = 0; j < m; j++) {
                    // allow a margin for FP comparisons
                    if (std::abs(data[i * m + j] - other.data[i * m + j]) > EPSILON) {
                        return false;
                    }
                }
            }

            return true;
        }

        // pretty prints Matrix
        friend std::ostream &operator<<(std::ostream &os, const Matrix &mat) {
            os << std::endl << "Matrix (" << mat.n << ", " << mat.m << "):" << std::endl;

            for (vec_size_t i = 0; i < mat.n; i++) {
                for (vec_size_t j = 0; j < mat.m; j++) {
                    os << std::setw(12) << std::left << mat(i, j) << ' ';
                }
                os << std::endl;
            }

            return os;
        }

        [[nodiscard]] CSR to_csr() const;

        [[nodiscard]] COO to_coo() const;
    };
}