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
#ifndef ROCSPARSE_IM2COL_HPP
#define ROCSPARSE_IM2COL_HPP

#include "rocsparse.h"
#include "handle.h"
#include "utility.h"
#include "im2col_device.h"

#include <hip/hip_runtime.h>

template <typename T>
rocsparse_status rocsparse_im2col_template(rocsparse_handle handle,
                                           const T* im,
                                           rocsparse_int n,
                                           rocsparse_int c,
                                           rocsparse_int h,
                                           rocsparse_int w,
                                           rocsparse_int wei_h,
                                           rocsparse_int wei_w,
                                           rocsparse_int out_h,
                                           rocsparse_int out_w,
                                           rocsparse_int pad_h,
                                           rocsparse_int pad_w,
                                           rocsparse_int stride_h,
                                           rocsparse_int stride_w,
                                           rocsparse_im2col_type type,
                                           T* col)
{
    // Check for valid handle and matrix descriptor
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }
    else if (im == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else if (col == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    log_trace(handle,
              replaceX<T>("rocsparse_Xim2col"),
              (const void*&)im,
              n,
              c,
              h,
              w,
              wei_h,
              wei_w,
              out_h,
              out_w,
              pad_h,
              pad_w,
              stride_h,
              stride_w,
              type,
              (const void*&)col);


    // Check sizes
    if(n < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if(c < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if(h < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if(w < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if(wei_h < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if(wei_w < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if(out_h < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if(out_w < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if(pad_h < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if(pad_w < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if(stride_h < 0)
    {
        return rocsparse_status_invalid_size;
    }
    else if(stride_w < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(n == 0 || c == 0 || h == 0 || w == 0 || wei_h == 0 || wei_w == 0)
    {
        return rocsparse_status_success;
    }

    // Stream
    hipStream_t stream = handle->stream;

    // Run different im2col kernels
    if(type == rocsparse_im2col_nchw)
    {
#define TILE_SZ_X 32
#define TILE_SZ_Y 8
          int lds_size = (TILE_SZ_X * stride_w + wei_w) * (TILE_SZ_Y * stride_h + wei_h);
          int num_blks_x = std::ceil(static_cast<float>(out_w) / TILE_SZ_X);
          int num_blks   = num_blks_x * int(std::ceil(static_cast<float>(out_h) / TILE_SZ_Y));
          dim3 dimBlock(256, 1, 1);//Number of threads in each block
          dim3 dimGrid(num_blks, c*n, 1);//Number of blocks

          hipLaunchKernelGGL((im2col_nchw_kernel<T, TILE_SZ_X, TILE_SZ_Y>),
                             dim3(dimGrid), dim3(dimBlock),
                             lds_size, stream,
                             im,
                             c, h, w,
                             wei_h, wei_w,
                             out_h, out_w,
                             pad_h, pad_w,
                             stride_h, stride_w,
                             col);
#undef TILE_SZ_X
#undef TILE_SZ_Y
    }
    else if (type == rocsparse_im2col_nhwc)
    {
#define TILE_SZ_X 32
#define TILE_SZ_Y 8
          int lds_size = (TILE_SZ_X * stride_w + wei_w) * (TILE_SZ_Y * stride_h + wei_h);
          int num_blks_x = std::ceil(static_cast<float>(out_w) / TILE_SZ_X);
          int num_blks   = num_blks_x * int(std::ceil(static_cast<float>(out_h) / TILE_SZ_Y));
          dim3 dimBlock(256, 1, 1);//Number of threads in each block
          dim3 dimGrid(num_blks, c*n, 1);//Number of blocks

          hipLaunchKernelGGL((im2col_nhwc_kernel<T, TILE_SZ_X, TILE_SZ_Y>),
                             dim3(dimGrid), dim3(dimBlock),
                             lds_size, stream,
                             im,
                             c, h, w,
                             wei_h, wei_w,
                             out_h, out_w,
                             pad_h, pad_w,
                             stride_h, stride_w,
                             col);
#undef TILE_SZ_X
#undef TILE_SZ_Y
    }
    else
    {
        return rocsparse_status_not_implemented;
    }
    return rocsparse_status_success;
}

#endif // ROCSPARSE_MCSCM_HPP
