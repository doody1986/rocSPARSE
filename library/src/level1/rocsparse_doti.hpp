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
#ifndef ROCSPARSE_DOTI_HPP
#define ROCSPARSE_DOTI_HPP

#include "rocsparse.h"
#include "definitions.h"
#include "handle.h"
#include "utility.h"
#include "doti_device.h"

#include <hip/hip_runtime.h>

template <typename T>
rocsparse_status rocsparse_doti_template(rocsparse_handle handle,
                                         rocsparse_int nnz,
                                         const T* x_val,
                                         const rocsparse_int* x_ind,
                                         const T* y,
                                         T* result,
                                         rocsparse_index_base idx_base)
{
    // Check for valid handle
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Logging
    if(handle->pointer_mode == rocsparse_pointer_mode_host)
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xdoti"),
                  nnz,
                  (const void*&)x_val,
                  (const void*&)x_ind,
                  (const void*&)y,
                  *result,
                  idx_base);

        log_bench(handle, "./rocsparse-bench -f doti -r", replaceX<T>("X"), "--mtx <vector.mtx> ");
    }
    else
    {
        log_trace(handle,
                  replaceX<T>("rocsparse_Xdoti"),
                  nnz,
                  (const void*&)x_val,
                  (const void*&)x_ind,
                  (const void*&)y,
                  (const void*&)result,
                  idx_base);
    }

    // Check index base
    if(idx_base != rocsparse_index_base_zero && idx_base != rocsparse_index_base_one)
    {
        return rocsparse_status_invalid_value;
    }

    // Check size
    if(nnz < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Check pointer arguments
    if(x_val == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(x_ind == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(y == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if(result == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Quick return if possible
    if(nnz == 0)
    {
        return rocsparse_status_success;
    }

    // Stream
    hipStream_t stream = handle->stream;

#define DOTI_DIM 1024
    rocsparse_int nblocks = DOTI_DIM;

    // Get workspace from handle device buffer
    T* workspace = reinterpret_cast<T*>(handle->buffer);

    dim3 doti_blocks(nblocks);
    dim3 doti_threads(DOTI_DIM);

    hipLaunchKernelGGL((doti_kernel_part1<T, DOTI_DIM>),
                       doti_blocks,
                       doti_threads,
                       0,
                       stream,
                       nnz,
                       x_val,
                       x_ind,
                       y,
                       workspace,
                       idx_base);

    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        hipLaunchKernelGGL((doti_kernel_part2<T, DOTI_DIM, 1>),
                           dim3(1),
                           doti_threads,
                           0,
                           stream,
                           nblocks,
                           workspace,
                           result);
    }
    else
    {
        hipLaunchKernelGGL((doti_kernel_part2<T, DOTI_DIM, 0>),
                           dim3(1),
                           doti_threads,
                           0,
                           stream,
                           nblocks,
                           workspace,
                           result);

        RETURN_IF_HIP_ERROR(hipMemcpy(result, workspace, sizeof(T), hipMemcpyDeviceToHost));
    }
#undef DOTI_DIM

    return rocsparse_status_success;
}

#endif // ROCSPARSE_DOTI_HPP
