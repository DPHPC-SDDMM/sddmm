
#include "../defines.h"
#include <vector>
#include "../data_structures/matrix/matrix.h"
#include "../data_structures/coo/coo.h"
#include <omp.h>
#include <iostream>
#include <math.h>

namespace SDDMM {
    namespace Algo {
        Types::COO ParallelSDDMM(const Types::COO& A_sparse, const Types::Matrix& X_dense, const Types::Matrix& Y_dense, Types::vec_size_t num_threads) {
            assert(X_dense.m == Y_dense.n && "Size of cols(X_dense) and rows(Y) must match!");
            assert(A_sparse.n>0 && A_sparse.m>0 && X_dense.n>0 && X_dense.m>0 && Y_dense.n>0 && Y_dense.m && "All involved matrices must be non-empty!");
            assert(A_sparse.n==X_dense.n && A_sparse.m==Y_dense.m && "Matrix dimensions must match!");

            
            Types::COO res;
            res.n = A_sparse.n;
            res.m = A_sparse.m;

            std::vector<Types::COO::triplet> block(num_threads, {0,0,0});
            auto s = A_sparse.data.size();
            for(SDDMM::Types::vec_size_t i=0; i<s; i+=num_threads){
                // auto m = std::min(s - i, num_threads);
                #pragma omp parallel
                // for(int tn=0; tn<num_threads; ++tn)
                {
                    auto tn = omp_get_thread_num();
                    if(i+tn < s) {
                        auto idx = i+tn;
                        Types::COO::triplet p = A_sparse.data.at(idx);
                        Types::expmt_t inner_product = 0;
                        
                        // the ind index has to be tiled later
                        for(SDDMM::Types::vec_size_t ind=0; ind < X_dense.m; ++ind){
                            inner_product += X_dense.at(p.row, ind)*Y_dense.at(ind, p.col);
                        }

                        // here we have the entire inner prouct inside inner_product
                        block[tn] = {p.row, p.col, p.value * inner_product};
                    }
                }
                res.data.insert(res.data.end(), block.begin(), block.begin() + std::min(s - i, num_threads));
            }

            // // reserve space
            // res.values.reserve(A_sparse.values.size());
            // res.col_idx.reserve(A_sparse.col_idx.size());
            // res.row_ptr.reserve(A_sparse.row_ptr.size());
            // std::copy(A_sparse.values.begin(), A_sparse.values.end(), std::back_inserter(res.values));
            // std::copy(A_sparse.col_idx.begin(), A_sparse.col_idx.end(), std::back_inserter(res.col_idx));
            // std::copy(A_sparse.row_ptr.begin(), A_sparse.row_ptr.end(), std::back_inserter(res.row_ptr));

            // SDDMM::Types::vec_size_t o_col = 0;
            // SDDMM::Types::vec_size_t o_row = 0;
            // SDDMM::Types::vec_size_t s = A_sparse.row_ptr.size()-1;
            // SDDMM::Types::vec_size_t v_s_ind = 0;

            // SDDMM::Types::vec_size_t v_t_ind = 0;
            // SDDMM::Types::vec_size_t c_t_ind = 0;
            // SDDMM::Types::vec_size_t r_t_ind = 0;

            // omp_set_num_threads(num_threads);
            // std::vector<SDDMM::Defines::RC> entries(num_threads,{0,0,0});
            // for(SDDMM::Types::vec_size_t r=0; r<s; r+=num_threads){
            //     // this is essentially a to_COO...
            //     int t_ind = 0;
            //     while(t_ind<num_threads){
            //         SDDMM::Types::vec_size_t from = A_sparse.row_ptr[r+t_ind];
            //         SDDMM::Types::vec_size_t to = A_sparse.row_ptr[r+t_ind+1];
            //         if(num_threads < (t_ind + to-from)) break;
            //         for(SDDMM::Types::vec_size_t ci=from; ci<to; ++ci){
            //             entries[t_ind] = {r, A_sparse.col_idx[ci], from};
            //         }
            //         t_ind+=(to-from);
            //     }

            //     #pragma omp parallel
            //     {
            //         auto tn = omp_get_thread_num();
            //         if(tn < t_ind) {
            //             auto p = entries.at(tn);
            //             Types::expmt_t inner_product = 0;
                        
            //             // the ind index has to be tiled later
            //             for(SDDMM::Types::vec_size_t ind=0; ind < X_dense.m; ++ind){
            //                 inner_product += X_dense.at(p.row, ind)*Y_dense.at(ind, p.col);
            //             }

            //             // here we have the entire inner prouct inside inner_product
            //             res.values[v_s_ind + p.val_offset + tn] *= inner_product;
            //             // inner_prod_id++; // don't do in threaded version
            //         }
            //     }
            //     v_s_ind += t_ind;


                
                // // std::vector<Types::expmt_t> inner_products(to - from, 0.0);
                // {
                //     // this would be the threadid
                //     // openMP parallel loop
                //     // for(int inner_prod_id = 0; inner_prod_id < inner_products.size(); ++inner_prod_id){
                //         // int inner_prod_id = 0; // => this is the thread id
                //         // #pragma omp parallel for num_threads(12)
                //         // {
                //             for(SDDMM::Types::vec_size_t ci=from; ci<to; ++ci){
                //             SDDMM::Types::vec_size_t c = A_sparse.col_idx[ci];
                            
                //             Types::expmt_t inner_product = 0;
                //             // the ind index has to be tiled later
                //             for(SDDMM::Types::vec_size_t ind=0; ind < X_dense.m; ++ind){
                //                 inner_product += X_dense.at(r, ind)*Y_dense.at(ind, c);
                //             }

                //             // here we have the entire inner prouct inside inner_product
                //             res.values[v_s_ind + ci - from] *= inner_product;
                //             // inner_prod_id++; // don't do in threaded version
                //             }
                //         // }
                //     // }
                // }
                // v_s_ind += (to - from);
            // }

            // So, now, we start cheating because we are too lazy to 
            // think of a correct, efficient solution that works without cheating
            // (Hehehehehehe... ^^ Muahahahahahaaaaaaaa XD)


            return res;
        }
    }
}