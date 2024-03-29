#include "matrix.h"
#include "../csr/csr.h"
#include "../coo/coo.h"

namespace SDDMM {
    namespace Types {
        [[nodiscard]] CSR Matrix::to_csr() const {
            // create an empty CSR matrix
            CSR mat;

            mat.n = n;
            mat.m = m;

            // reserve space
            mat.values.reserve(n * m);
            mat.col_idx.reserve(n * m);
            mat.row_ptr.reserve(n + 1);

            // first row_ptr is always 0
            mat.row_ptr.push_back(0);

            for (Types::vec_size_t i = 0; i < n; i++) {
                for (Types::vec_size_t j = 0; j < m; j++) {
                    SDDMM::Types::expmt_t v = this->at(i, j);
                    if (v != 0.0) {
                        // add values and column indices
                        mat.values.push_back(v);
                        mat.col_idx.push_back(j);
                    }
                }

                // add the next row's starting pointer
                mat.row_ptr.push_back(mat.values.size());
            }

            return mat;
        }

        [[nodiscard]] COO Matrix::to_coo() const {
            // create an empty COO matrix
            COO mat;
            mat.n = n;
            mat.m = m;

            // iterate over all rows and columns
            for (Types::vec_size_t i = 0; i < n; i++) {
                for (Types::vec_size_t j = 0; j < m; j++) {
                    SDDMM::Types::expmt_t value = data[i * m + j];

                    // push a triplet
                    if (value != 0.0) {
                        // mat.init_data.push_back({i, j, value});
                        mat.rows.push_back(i);
                        mat.cols.push_back(j);
                        mat.values.push_back(value);
                    }
                }
            }

            return mat;
        }
    }
}