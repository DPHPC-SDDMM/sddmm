#include "coo.h"
#include "../csr/csr.h"
#include "../matrix/matrix.h"

namespace SDDMM {
    namespace Types {
        [[nodiscard]] CSR COO::to_csr() const {
            // create an empty CSR matrix
            CSR mat;
            mat.n = n;
            mat.m = m;

            // reserve space and initialise row pointers
            mat.values.reserve(values.size());
            mat.col_idx.reserve(values.size());
            mat.row_ptr.resize(n + 1); // make sure the empty case contains one zero per row + one additional zero

            int last_row = 0;
            int row_counter = 0;
            // iterate over each triplet and insert components into the corresponding array
            auto s = values.size();
            Types::vec_size_t r_ind = 0;
            mat.row_ptr[r_ind] = row_counter;
            r_ind++;
            for (auto i=0; i<s; ++i) {
                // column indices and values
                mat.col_idx.push_back(cols[i]);
                mat.values.push_back(values[i]);

                // row pointer
                if (last_row != rows[i]) {
                    mat.row_ptr[r_ind] = row_counter;
                    r_ind++;
                }
                last_row = rows[i];
                row_counter++;
            }

            mat.row_ptr[r_ind] = mat.values.size();

            return mat;
        }

        [[nodiscard]] Matrix COO::to_matrix() const {
            // create and initialise an empty matrix
            Matrix mat(n, m);
            mat.set_matrix_format(Types::MatrixFormat::RowMajor);
            // iterate over each triplet
            auto s = values.size();
            for (auto i=0; i<s; ++i) {
                Types::vec_size_t row = rows[i];
                Types::vec_size_t col = cols[i];
                Types::expmt_t value = values[i];

                mat(row, col) = value;
            }

            return mat;
        }    
    }
}