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
#ifndef ROCSPARSE_MCSCM_HPP
#define ROCSPARSE_MCSCM_HPP

#include "rocsparse.h"
#include "handle.h"
#include "utility.h"
#include "mcscm_device.h"

#include <hip/hip_runtime.h>

template <typename T, rocsparse_int BLOCKSIZE, rocsparse_int WF_SIZE>
__launch_bounds__(256) __global__
    void mcscmnn_kernel_host_pointer(rocsparse_int m,
                                     rocsparse_int n,
                                     rocsparse_int k,
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
    mcscmnn_general_device<T, BLOCKSIZE, WF_SIZE>(
        m, n, k, alpha, A, lda, nnz, csc_col_ptr, csc_row_ind, csc_val, beta, C, ldc, idx_base);
}

template <typename T, rocsparse_int BLOCKSIZE, rocsparse_int WF_SIZE>
__launch_bounds__(256) __global__
    void mcscmnn_kernel_device_pointer(rocsparse_int m,
                                       rocsparse_int n,
                                       rocsparse_int k,
                                       const T* alpha,
                                       const T* __restrict__ A,
                                       rocsparse_int lda,
                                       rocsparse_int nnz,
                                       const rocsparse_int* __restrict__ csc_col_ptr,
                                       const rocsparse_int* __restrict__ csc_row_ind,
                                       const T* __restrict__ csc_val,
                                       const T* beta,
                                       T* __restrict__ C,
                                       rocsparse_int ldc,
                                       rocsparse_index_base idx_base)
{
    if(*alpha == 0.0 && *beta == 1.0)
    {
        return;
    }

    mcscmnn_general_device<T, BLOCKSIZE, WF_SIZE>(
        m, n, k, *alpha, A, lda, nnz, csc_col_ptr, csc_row_ind, csc_val, *beta, C, ldc, idx_base);
}

template <typename T, rocsparse_int BLOCKSIZE, rocsparse_int WF_SIZE>
__launch_bounds__(256) __global__
    void mcscmtn_kernel_host_pointer(rocsparse_int offset,
                                     rocsparse_int ncol,
                                     rocsparse_int m,
                                     rocsparse_int n,
                                     rocsparse_int k,
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
    mcscmtn_general_device<T, BLOCKSIZE, WF_SIZE>(offset,
                                                  ncol,
                                                  m,
                                                  n,
                                                  k,
                                                  alpha,
                                                  A,
                                                  lda,
                                                  nnz,
                                                  csc_col_ptr,
                                                  csc_row_ind,
                                                  csc_val,
                                                  beta,
                                                  C,
                                                  ldc,
                                                  idx_base);
}

template <typename T, rocsparse_int BLOCKSIZE, rocsparse_int WF_SIZE>
__launch_bounds__(256) __global__
    void mcscmtn_kernel_device_pointer(rocsparse_int offset,
                                       rocsparse_int ncol,
                                       rocsparse_int m,
                                       rocsparse_int n,
                                       rocsparse_int k,
                                       const T* alpha,
                                       const T* __restrict__ A,
                                       rocsparse_int lda,
                                       rocsparse_int nnz,
                                       const rocsparse_int* __restrict__ csc_col_ptr,
                                       const rocsparse_int* __restrict__ csc_row_ind,
                                       const T* __restrict__ csc_val,
                                       const T* beta,
                                       T* __restrict__ C,
                                       rocsparse_int ldc,
                                       rocsparse_index_base idx_base)
{
    if(*alpha == 0.0 && *beta == 1.0)
    {
        return;
    }

    mcscmtn_general_device<T, BLOCKSIZE, WF_SIZE>(offset,
                                                  ncol,
                                                  m,
                                                  n,
                                                  k,
                                                  *alpha,
                                                  A,
                                                  lda,
                                                  nnz,
                                                  csc_col_ptr,
                                                  csc_row_ind,
                                                  csc_val,
                                                  *beta,
                                                  C,
                                                  ldc,
                                                  idx_base);
}

template <typename T>
rocsparse_status rocsparse_mcscm_template(rocsparse_handle handle,
                                          rocsparse_operation trans_A,
                                          rocsparse_operation trans_B,
                                          rocsparse_int m,
                                          rocsparse_int n,
                                          rocsparse_int k,
                                          const T* alpha,
                                          const T* A,
                                          rocsparse_int lda,
                                          rocsparse_int nnz,
                                          const rocsparse_mat_descr descr,
                                          const T* csc_val,
                                          const rocsparse_int* csc_col_ptr,
                                          const rocsparse_int* csc_row_ind,
                                          const T* beta,
                                          T* C,
                                          rocsparse_int ldc)
{
    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Logging TODO bench logging
    if(handle->pointer_mode == rocsparse_pointer_mode_host)
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xmcscm"),
                  trans_A,
                  trans_B,
                  m,
                  n,
                  k,
                  *alpha,
                  (const void*&)A,
                  lda,
                  nnz,
                  (const void*&)descr,
                  (const void*&)csc_val,
                  (const void*&)csc_col_ptr,
                  (const void*&)csc_row_ind,
                  *beta,
                  (const void*&)C,
                  ldc);
    }
    else
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xmcscm"),
                  trans_A,
                  trans_B,
                  m,
                  n,
                  k,
                  (const void*&)alpha,
                  (const void*&)A,
                  lda,
                  nnz,
                  (const void*&)descr,
                  (const void*&)csc_val,
                  (const void*&)csc_col_ptr,
                  (const void*&)csc_row_ind,
                  (const void*&)beta,
                  (const void*&)C,
                  ldc);
    }

    // Check index base
    if(descr->base != rocsparse_index_base_zero && descr->base != rocsparse_index_base_one)
    {
        return rocsparse_status_invalid_value;
    }
    if(descr->type != rocsparse_matrix_type_general)
    {
        // TODO
        return rocsparse_status_not_implemented;
    }

    // Check sizes
    if(m < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if(n < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if(k < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if(nnz < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check pointer arguments
    if(csc_val == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(csc_col_ptr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(csc_row_ind == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(A == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(C == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(alpha == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(beta == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || k == 0 || nnz == 0)
    {
        return rocsparse_status_success;
    }

    // Check leading dimension of B
    rocsparse_int one = 1;
    if(trans_A == rocsparse_operation_none)
    {
        if(lda < std::max(one, m))
        {
            return rocsparse_status_invalid_size;
        }
    }
    else
    {
        if(lda < std::max(one, k))
        {
            return rocsparse_status_invalid_size;
        }
    }

    // Check leading dimension of C
    if(trans_A == rocsparse_operation_none)
    {
        if(ldc < std::max(one, m))
        {
            return rocsparse_status_invalid_size;
        }
    }
    else
    {
        if(ldc < std::max(one, k))
        {
            return rocsparse_status_invalid_size;
        }
    }

    // Stream
    hipStream_t stream = handle->stream;

    // Run different mcscm kernels
    if(trans_A == rocsparse_operation_none)
    {
        if(trans_B == rocsparse_operation_none)
        {
#define MCSCMNN_DIM 256
#define SUB_WF_SIZE 8
            dim3 mcscmnn_blocks((SUB_WF_SIZE * n - 1) / MCSCMNN_DIM + 1, (m - 1) / SUB_WF_SIZE + 1);
            dim3 mcscmnn_threads(MCSCMNN_DIM);

            if(handle->pointer_mode == rocsparse_pointer_mode_device)
            {
                hipLaunchKernelGGL((mcscmnn_kernel_device_pointer<T, MCSCMNN_DIM, SUB_WF_SIZE>),
                                   mcscmnn_blocks,
                                   mcscmnn_threads,
                                   0,
                                   stream,
                                   m,
                                   n,
                                   k,
                                   alpha,
                                   A,
                                   lda,
                                   nnz,
                                   csc_col_ptr,
                                   csc_row_ind,
                                   csc_val,
                                   beta,
                                   C,
                                   ldc,
                                   descr->base);
            }
            else
            {
                if(*alpha == 0.0 && *beta == 1.0)
                {
                    return rocsparse_status_success;
                }

                hipLaunchKernelGGL((mcscmnn_kernel_host_pointer<T, MCSCMNN_DIM, SUB_WF_SIZE>),
                                   mcscmnn_blocks,
                                   mcscmnn_threads,
                                   0,
                                   stream,
                                   m,
                                   n,
                                   k,
                                   *alpha,
                                   A,
                                   lda,
                                   nnz,
                                   csc_col_ptr,
                                   csc_row_ind,
                                   csc_val,
                                   *beta,
                                   C,
                                   ldc,
                                   descr->base);
            }
#undef SUB_WF_SIZE
#undef MCSCMNN_DIM
        }
        else if(trans_B == rocsparse_operation_transpose)
        {
            return rocsparse_status_not_implemented;
        }
    }
    else
    {
        return rocsparse_status_not_implemented;
    }
    return rocsparse_status_success;
}

#endif // ROCSPARSE_MCSCM_HPP
