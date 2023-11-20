#include "../csr/csr.h"
#include "../coo/coo.h"
#include "../matrix/matrix.h"


namespace SDDMM {
    namespace Types{
        [[nodiscard]] Matrix CSR::to_matrix() const {
            Matrix mat(n, m);

            // iterate over rows
            for (Types::vec_size_t i = 0; i < n; i++) {
                // iterate over columns
                for (Types::vec_size_t j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
                    mat(i, col_idx[j]) = values[j];
                }
            }

            return mat;
        }

        [[nodiscard]] COO CSR::to_coo() const {
            COO mat;

            mat.values.reserve(values.size());
            mat.rows.reserve(values.size());
            mat.cols.reserve(values.size());
            mat.n = n;
            mat.m = m;

            // iterate over rows
            for (Types::vec_size_t row = 0; row < n; row++) {
                Types::vec_size_t start_idx = row_ptr[row];
                Types::vec_size_t end_idx = row_ptr[row + 1];

                // iterate over columns
                for (Types::vec_size_t j = start_idx; j < end_idx; j++) {
                    Types::vec_size_t col = col_idx[j];
                    SDDMM::Types::expmt_t value = values[j];
                    mat.values.emplace_back(value);
                    mat.rows.emplace_back(row);
                    mat.cols.emplace_back(col);
                }
            }

            return mat;
        }
    }
}
