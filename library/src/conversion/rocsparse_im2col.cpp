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

#include "rocsparse.h"
#include "rocsparse_im2col.hpp"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_sim2col( rocsparse_handle handle,
                                               const float* im,
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
                                               float* col )
{
    return rocsparse_im2col_template<float>(handle,
                                           im,
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
                                           col);
}

extern "C" rocsparse_status rocsparse_dim2col( rocsparse_handle handle,
                                               const double* im,
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
                                               double* col )
{
  return rocsparse_im2col_template<double>(handle,
                                           im,
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
                                           col);
}
