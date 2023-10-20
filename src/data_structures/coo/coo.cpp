#include "../type_defs.h"
#include "coo.h"
#include "../csr/csr.h"
#include "../matrix/matrix.h"

namespace SDDMM {
    [[nodiscard]] CSR COO::to_csr() const {
        // create an empty CSR matrix
        CSR mat;
        mat.n = n;
        mat.m = m;

        // reserve space and initialise row pointers
        mat.values.reserve(data.size());
        mat.col_idx.reserve(data.size());
        mat.row_ptr.resize(n + 1, 0);

        int last_row = -1;
        // iterate over each triplet and insert components into the corresponding array
        for (const auto& t : data) {
            // column indices and values
            mat.col_idx.push_back(t.col);
            mat.values.push_back(t.value);

            // row pointer
            while (last_row < t.row) {
                mat.row_ptr[++last_row] = mat.values.size();
            }
        }

        // leftover rows
        while (last_row < n) {
            mat.row_ptr[++last_row] = mat.values.size();
        }

        return mat;
    }

    [[nodiscard]] Matrix COO::to_matrix() const {
        // create and initialise an empty matrix
        Matrix mat(n, m);

        // iterate over each triplet
        for (const auto& triplet : data) {
            vec_size_t row = triplet.row;
            vec_size_t col = triplet.col;
            double value = triplet.value;

            mat(row, col) = value;
        }

        return mat;
    }
}