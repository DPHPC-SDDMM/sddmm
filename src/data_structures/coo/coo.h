#include <iostream>
// Libraries for randomness
#include <random>
// Libraries for exception handling
#include <stdexcept>
// Data structures
#include <tuple>
#include <map>
#include <algorithm> // std::sort
#include <cassert>

#include "../../defines.h"

namespace SDDMM{
    namespace Types{
        // forward declarations
        class CSR;
        class Matrix;

        class COO {
        public:
            /**
             * 'triplet' is a triplet of (vec_size_t, vec_size_t, double), whose elements represent:
             * 0: row index
             * 1: column index
             * 2: value of cell
             * respectively.
            */

            // structs could be a bit easier to access than tuples but that's up for a discussion
            struct triplet {
                Types::vec_size_t row;
                Types::vec_size_t col;
                double value;

                bool operator<(const triplet& other) const {
                    if (row == other.row) {
                        return col < other.col;
                    }

                    return row < other.row;
                }
            };

            std::vector<triplet> data;
            Types::vec_size_t n, m;

            COO(): n(), m() {};

            void sort() {
                std::sort(data.begin(), data.end());
            }

            /**
             * Generate a random Matrix represented in the COO format.
             * For details about the COO format, you may read
             * https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO)
             *
             * @param n: Number of rows of the generated Matrix.
             * @param m: Number of columns of the generated Matrix.
             * @param sparsity: A percentage expressing the ratio of non-zero elements to all the elements (n * m).
             * @param sort: Determines whether the output is returned in a sorted manner or not.
             *              The sorting takes place first by ascending row and then by ascending column.
             * @param distribution: Distribution according to which Matrix elements are generated.
            */
            COO static generate(
                    Types::vec_size_t n, Types::vec_size_t m, float sparsity,
                    bool sort=true, const std::string& distribution = "uniform"
                )
            {

                // TODO: Implement numerous random distribution schemes.
                // Currently, only the default (uniform) is implemented.

                assert(sparsity >= 0. && sparsity <= 1. && "Expected a sparsity value from 0.0 to 1.0");

                // Define the `output` data structure and allocate sufficient memory for it in advance.
                COO output;
                auto total_elements = n * m;
                auto nr_elements = static_cast<Types::vec_size_t>(total_elements * sparsity);

                // Define random generator
                // and distribution functions for the random generation.
                // NOTE: Source for random number generator:
                // https://stackoverflow.com/questions/15461140/stddefault-random-engine-generate-values-between-0-0-and-1-0
                std::random_device rd;
                std::default_random_engine generator(rd());
                std::uniform_int_distribution<Types::vec_size_t> row_distribution(0, n-1);
                std::uniform_int_distribution<Types::vec_size_t> column_distribution(0, m-1);
                // [-1, 1] values were selected because neural networks often deal with smaller values.
                std::uniform_real_distribution<double> value_distribution(-1.0, 1.0);

                // Define the data structure (hash map) which will ensure
                // that no (row, column) duplicates are inserted.
                // NOTE: In practice, that probability will be fairly low.
                std::map<Types::vec_size_t, Types::vec_size_t> row_to_column;

                auto elements_remaining = nr_elements;

                while (elements_remaining)
                {
                    Types::vec_size_t row = row_distribution(generator);
                    Types::vec_size_t column = column_distribution(generator);
                    bool successful_insertion = std::get<1>(row_to_column.emplace(row, column));

                    if (successful_insertion)
                    {
                        double value = value_distribution(generator); // Generate cell value.
                        output.data.push_back({row, column, value}); // Add element to output.
                        --elements_remaining; // Decrease counter of further elements required to add.
                    }
                }

                if (sort) { output.sort(); }

                return output;
            }

            // pretty prints COO
            friend std::ostream &operator<<(std::ostream &os, const COO &mat) {
                os << std::endl << "COO (" << mat.n << ", " << mat.m << "):" << std::endl;

                os << "Triplets: " << std::endl;
                for (const auto& t : mat.data) {
                    os << "(" << t.row << ", " << t.col << ", " << t.value << ")" << std::endl;
                }

                return os;
            }

            [[nodiscard]] CSR to_csr() const;

            [[nodiscard]] Matrix to_matrix() const;
        };
    }
}