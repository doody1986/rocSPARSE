/* ************************************************************************
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once
#ifndef MCSCM_DEVICE_H
#define MCSCM_DEVICE_H

#include <hip/hip_runtime.h>

template <typename T, rocsparse_int BLOCKSIZE, rocsparse_int WF_SIZE>
static __device__ void mcscmnn_general_device(rocsparse_int M,
                                              rocsparse_int N,
                                              rocsparse_int K,
                                              T alpha,
                                              const T* __restrict__ A,
                                              rocsparse_int lda,
                                              rocsparse_int nnz,
                                              const rocsparse_int* __restrict__ csc_col_ptr,
                                              const rocsparse_int* __restrict__ csc_row_ind,
                                              const T* __restrict__ csc_val,
                                              T beta,
                                              T* __restrict__ C,
                                              rocsparse_int ldc,
                                              rocsparse_index_base idx_base)
{
    rocsparse_int tid = hipThreadIdx_x;
    rocsparse_int gid = hipBlockIdx_x * hipBlockDim_x + tid;
    rocsparse_int lid = gid & (WF_SIZE - 1);
    rocsparse_int wid = tid / WF_SIZE;
    rocsparse_int nwf = hipGridDim_x * hipBlockDim_x / WF_SIZE;
    rocsparse_int row = lid + hipBlockIdx_y * WF_SIZE;

    rocsparse_int rowA = row;
    rocsparse_int rowC = row;

    __shared__ rocsparse_int shared_row[BLOCKSIZE / WF_SIZE][WF_SIZE];
    __shared__ T shared_val[BLOCKSIZE / WF_SIZE][WF_SIZE];

    for(rocsparse_int col = gid / WF_SIZE; col < N; col += nwf)
    {
        rocsparse_int col_start = csc_col_ptr[col] - idx_base;
        rocsparse_int col_end   = csc_col_ptr[col + 1] - idx_base;

        T sum = static_cast<T>(0);

        for(rocsparse_int j = col_start; j < col_end; j += WF_SIZE)
        {
            rocsparse_int k = j + lid;

            shared_row[wid][lid] = (k < col_end) ? csc_row_ind[k] - idx_base : 0;
            shared_val[wid][lid] = (k < col_end) ? alpha * csc_val[k] : static_cast<T>(0);

            __syncthreads();
            for(rocsparse_int i = 0; i < WF_SIZE && rowA < M; ++i)
            {
                sum = fma(shared_val[wid][i], A[rowA + shared_row[wid][i]*lda], sum);
            }
        }

        if(rowC < M)
        {
            if(beta == 0.0)
            {
                C[rowC + col * ldc] = sum;
            }
            else
            {
                C[rowC + col * ldc] = fma(beta, C[rowC + col * ldc], sum);
            }
        }
    }
}

template <typename T, rocsparse_int BLOCKSIZE, rocsparse_int WF_SIZE>
static __device__ void mcscmtn_general_device(rocsparse_int offset,
                                              rocsparse_int ncol,
                                              rocsparse_int M,
                                              rocsparse_int N,
                                              rocsparse_int K,
                                              T alpha,
                                              const T* __restrict__ A,
                                              rocsparse_int lda,
                                              rocsparse_int nnz,
                                              const rocsparse_int* __restrict__ csc_col_ptr,
                                              const rocsparse_int* __restrict__ csc_row_ind,
                                              const T* __restrict__ csc_val,
                                              T beta,
                                              T* __restrict__ C,
                                              rocsparse_int ldc,
                                              rocsparse_index_base idx_base)
{
    // rocsparse_int tid = hipThreadIdx_x;
    // rocsparse_int gid = hipBlockIdx_x * hipBlockDim_x + tid;
    // rocsparse_int row = gid / WF_SIZE;
    // rocsparse_int lid = tid & (WF_SIZE - 1);
    // rocsparse_int wid = tid / WF_SIZE;
    //
    // if(row >= M)
    // {
    //     return;
    // }
    //
    // __shared__ rocsparse_int shared_col[BLOCKSIZE / WF_SIZE][WF_SIZE];
    // __shared__ T shared_val[BLOCKSIZE / WF_SIZE][WF_SIZE];
    //
    // rocsparse_int row_start = csr_row_ptr[row] - idx_base;
    // rocsparse_int row_end   = csr_row_ptr[row + 1] - idx_base;
    //
    // for(rocsparse_int l = offset; l < ncol; l += WF_SIZE)
    // {
    //     rocsparse_int col = l + lid;
    //     T sum             = static_cast<T>(0);
    //
    //     for(rocsparse_int j = row_start; j < row_end; j += WF_SIZE)
    //     {
    //         rocsparse_int k = j + lid;
    //
    //         __syncthreads();
    //
    //         shared_col[wid][lid] = (k < row_end) ? N * (csr_col_ind[k] - idx_base) : 0;
    //         shared_val[wid][lid] = (k < row_end) ? alpha * csr_val[k] : static_cast<T>(0);
    //
    //         __syncthreads();
    //
    //         for(rocsparse_int i = 0; i < WF_SIZE; ++i)
    //         {
    //             T val_B = (col < ncol) ? __ldg(B + col + shared_col[wid][i]) : static_cast<T>(0);
    //             sum     = fma(shared_val[wid][i], val_B, sum);
    //         }
    //     }
    //
    //     if(col < ncol)
    //     {
    //         if(beta == static_cast<T>(0))
    //         {
    //             C[row + col * ldc] = sum;
    //         }
    //         else
    //         {
    //             C[row + col * ldc] = fma(beta, C[row + col * ldc], sum);
    //         }
    //     }
    // }
}

#endif // MCSCM_DEVICE_H
