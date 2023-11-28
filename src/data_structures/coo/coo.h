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
#include <algorithm> // std::sort
#include <cassert>
#include <chrono>

#include "../../defines.h"
#include "../matrix/matrix.h"
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
            // COO static generate(
            //         Types::vec_size_t n, Types::vec_size_t m, float sparsity,
            //         bool sort=true, const std::string& distribution = "uniform"
            //     )
            // {

            //     // TODO: Implement numerous random distribution schemes.
            //     // Currently, only the default (uniform) is implemented.

            //     assert(sparsity >= 0. && sparsity <= 1. && "Expected a sparsity value from 0.0 to 1.0");

            //     // Define the `output` data structure and allocate sufficient memory for it in advance.
            //     COO output;
            //     auto total_elements = n * m;
            //     auto nr_elements = static_cast<Types::vec_size_t>(total_elements * sparsity);

            //     // Define random generator
            //     // and distribution functions for the random generation.
            //     // NOTE: Source for random number generator:
            //     // https://stackoverflow.com/questions/15461140/stddefault-random-engine-generate-values-between-0-0-and-1-0
            //     std::random_device rd;
            //     std::default_random_engine generator(rd());
            //     std::uniform_int_distribution<Types::vec_size_t> row_distribution(0, n-1);
            //     std::uniform_int_distribution<Types::vec_size_t> column_distribution(0, m-1);
            //     // [-1, 1] values were selected because neural networks often deal with smaller values.
            //     std::uniform_real_distribution<Types::expmt_t> value_distribution(-1.0, 1.0);

            //     // Define the data structure (hash map) which will ensure
            //     // that no (row, column) duplicates are inserted.
            //     // NOTE: In practice, that probability will be fairly low.
            //     std::map<Types::vec_size_t, Types::vec_size_t> row_to_column;

            //     auto elements_remaining = nr_elements;

            //     while (elements_remaining)
            //     {
            //         Types::vec_size_t row = row_distribution(generator);
            //         Types::vec_size_t column = column_distribution(generator);
            //         bool successful_insertion = std::get<1>(row_to_column.emplace(row, column));

            //         if (successful_insertion)
            //         {
            //             Types::expmt_t value = value_distribution(generator); // Generate cell value.
            //             output.data.push_back({row, column, value}); // Add element to output.
            //             --elements_remaining; // Decrease counter of further elements required to add.
            //         }
            //     }

            //     if (sort) { output.sort(); }

            //     return output;
            // }

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
             * @param other: A reference to a sparse matrix.
             * @returns Whether all elements of both matrices are equal within an error margin of `Defines::epsilon`.
            */
            bool operator==(const COO& other){
                if(values.size() != other.values.size())
                    return false;

                Types::vec_size_t s = values.size();
                const Types::expmt_t epsilon = Defines::epsilon;
                for(Types::vec_size_t i=0; i<s; ++i){
                    auto a = std::fabs(cols[i] - other.cols[i]);
                    if(a > epsilon) return false;
                    auto b = std::fabs(rows[i] - other.rows[i]);
                    if(b  > epsilon) return false;
                    auto c = std::abs(values[i] - other.values[i]);
                    if(c  > epsilon) return false;
                }

                return true;
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
                    if(new_val != 0){
                        res.cols.push_back(col);
                        res.rows.push_back(row);
                        res.values.push_back(new_val);
                    }
                }

                return res;
            }

            static std::string hadamard_to_file(
                std::string path, 
                const SDDMM::Types::COO& sparse, 
                const float sparse_sparsity,
                const SDDMM::Types::Matrix& X, 
                const float X_sparsity,
                const SDDMM::Types::Matrix& Y,
                const float Y_sparsity)
            {
                auto last = path[path.size()-1];
                assert(last == SDDMM::Defines::path_separator && "path must end with a path separator / or \\");

                auto created_at = std::chrono::system_clock::now();
                auto created_at_t = std::chrono::system_clock::to_time_t(created_at);
                std::string time = std::string(std::ctime(&created_at_t));
                std::replace(time.begin(), time.end(), ' ', '_');
                time = time.substr(0, time.size()-1);

                std::stringstream name;
                name << "hadamard_S-" << sparse.n << "x" << sparse.m << "-" << sparse_sparsity 
                     << "_X-" << X.n << "x" << X.m << "-" << X_sparsity 
                     << "_Y-" << Y.n << "x" << Y.m << "-" << Y_sparsity 
                     << "___" << time
                     << "___" << created_at.time_since_epoch().count()
                     << ".txt";

                std::ofstream output_file;
                output_file.open(path + name.str());
                output_file << sparse.n << " " << sparse.m << "\n";
                // for(const triplet& val : sparse.data){
                auto s = sparse.values.size();
                for(auto i=0; i<s; ++i){
                    output_file << sparse.rows[i] << " " << sparse.cols[i] << " " << std::setprecision(12) << sparse.values[i] << "|";
                }
                output_file << "\n" << X.n << " " << X.m << "\n";
                output_file << (X.is_row_major() ? Constants::row_storage : Constants::col_storage) << "\n";
                for(const SDDMM::Types::expmt_t& val : X.data){
                    output_file << std::setprecision(12) << val << " ";
                }
                output_file << "\n" << Y.n << " " << Y.m << "\n";
                output_file << (Y.is_row_major() ? Constants::row_storage : Constants::col_storage) << "\n";
                for(const SDDMM::Types::expmt_t& val : Y.data){
                    output_file << std::setprecision(12) << val << " ";
                }
                output_file << "\n";
                output_file.close();
                return name.str();
            }

            static void hadamard_from_file(
                std::string file_name, 
                SDDMM::Types::COO& out_sparse, 
                SDDMM::Types::Matrix& out_X, 
                SDDMM::Types::Matrix& out_Y)
            {
                std::ifstream input_file;
                input_file.open(file_name);

                int state = 0;
                // Variable which stores the numerical content of a single line of the file.
                std::vector<SDDMM::Types::expmt_t> nums;
                while(!input_file.eof()){
                    std::string input;
                    std::getline(input_file, input, '\n');
                    /*
                    NOTE: 
                    Although in most cases a switch-case block is harder to maintain than multiple if-elses,
                    switch statements are generally faster than nested if-else statements.
                    This is because during compilation the compiler generates a jump table
                    that is used to select the path of execution.
                    Since our software is performance-critical,
                    as minimally useful as it may be in practice, it should be considered.
                    */
                    switch (state){
                        case 0: // size of sparse matrix
                            nums = string_to_num_vec(input);
                            out_sparse.n = static_cast<SDDMM::Types::vec_size_t>(nums[0]);
                            out_sparse.m = static_cast<SDDMM::Types::vec_size_t>(nums[1]);
                            break;
                        
                        case 1: // value triplets of sparse matrix
                            // out_sparse.data
                            {
                                auto trip = string_to_triplets(input);
                                for(auto t : trip){
                                    out_sparse.rows.push_back(t.row);
                                    out_sparse.cols.push_back(t.col);
                                    out_sparse.values.push_back(t.value);
                                }
                            }
                            break;
                        case 2: // size of X
                            nums = string_to_num_vec(input);
                            out_X.n = static_cast<SDDMM::Types::vec_size_t>(nums[0]);
                            out_X.m = static_cast<SDDMM::Types::vec_size_t>(nums[1]);
                            break;
                        case 3:
                            // row/col of X
                            if(input.compare(std::to_string(Constants::row_storage)) == 0){
                                out_X.set_matrix_format(Types::MatrixFormat::RowMajor);
                            } 
                            else{
                                out_X.set_matrix_format(Types::MatrixFormat::ColMajor);
                            }
                            break;
                        case 4: // values of X
                            out_X.data = string_to_num_vec(input);
                            break;

                        case 5: // size of Y
                            nums = string_to_num_vec(input);
                            out_Y.n = static_cast<SDDMM::Types::vec_size_t>(nums[0]);
                            out_Y.m = static_cast<SDDMM::Types::vec_size_t>(nums[1]);
                            break;
                        case 6:
                            // row/col of Y
                            if(input.compare(std::to_string(Constants::row_storage)) == 0){
                                out_Y.set_matrix_format(Types::MatrixFormat::RowMajor);
                            } 
                            else{
                                out_Y.set_matrix_format(Types::MatrixFormat::ColMajor);
                            }
                            break;
                        case 7: // values of Y
                            out_Y.data = string_to_num_vec(input);
                            break;
                    }
                    state++; // Transition to the next state of reading
                }

                
            }

            static std::vector<triplet> string_to_triplets(std::string input){
                std::vector<triplet> values;
                std::stringstream temp;
                for(char s : input){
                    if(s == '|'){
                        std::string str = temp.str();
                        triplet t = string_to_triplet(str);
                        values.push_back(t);
                        temp.str(std::string());
                    }
                    else{
                        temp << s;
                    }
                }
                return values;
            }

            static triplet string_to_triplet(std::string input){
                std::vector<SDDMM::Types::expmt_t> vals = string_to_num_vec(input);
                triplet t = {
                    static_cast<SDDMM::Types::vec_size_t>(vals[0]),
                    static_cast<SDDMM::Types::vec_size_t>(vals[1]),
                    static_cast<SDDMM::Types::expmt_t>(vals[2])
                };

                return t;
            }

            static std::vector<SDDMM::Types::expmt_t> string_to_num_vec(std::string input){
                std::vector<SDDMM::Types::expmt_t> values;
                std::stringstream temp;
                for(char s : input){
                    if(s == ' '){ // Element (number) of file is finished 
                        std::string t = temp.str(); // Copy content of buffer
                        if(!t.empty()){
                            SDDMM::Types::expmt_t val = std::stod(t);
                            val = std::round(1e12*val)/1e12;
                            values.push_back(static_cast<SDDMM::Types::expmt_t>(val));
                        }
                        temp.str(std::string()); // Empty content of buffer
                    }
                    else{
                        temp << s; // Read character from file 
                    }
                }
                //? Why is this not captured by the above loop?
                // last one
                std::string t = temp.str();
                if(!t.empty()){
                    SDDMM::Types::expmt_t val = std::stod(t);
                    val = std::round(1e12*val)/1e12;
                    values.push_back(static_cast<SDDMM::Types::expmt_t>(val));
                }
                return values;
            }

            static COO read_matrix_market_file(const char* filepath)
            {
                // Output variable
                Types::COO output;;

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

                // Allocate memory for data.
                // output.data.resize(nr_nonzeroes);
                output.values.resize(nr_nonzeroes);
                output.cols.resize(nr_nonzeroes);
                output.rows.resize(nr_nonzeroes);

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
                    output.values[index++] = v;
                    output.cols[index++] = c_in-1;
                    output.rows[index++] = r_in-1;
                    // output.data[index++] = {r_in-1, c_in-1, v};
                }

                in.close(); // Don't forget to close the file!

                return output;
            }

            [[nodiscard]] CSR to_csr() const;

            [[nodiscard]] Matrix to_matrix() const;
        };
    }
}