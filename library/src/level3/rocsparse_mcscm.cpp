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
#include "rocsparse_mcscm.hpp"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_smcscm(rocsparse_handle handle,
                                             rocsparse_operation trans_A,
                                             rocsparse_operation trans_B,
                                             rocsparse_int m,
                                             rocsparse_int n,
                                             rocsparse_int k,
                                             const float* alpha,
                                             const float* A,
                                             rocsparse_int lda,
                                             rocsparse_int nnz,
                                             const rocsparse_mat_descr descr,
                                             const float* csc_val,
                                             const rocsparse_int* csc_col_ptr,
                                             const rocsparse_int* csc_row_ind,
                                             const float* beta,
                                             float* C,
                                             rocsparse_int ldc)
{
    return rocsparse_mcscm_template<float>(handle,
                                           trans_A,
                                           trans_B,
                                           m,
                                           n,
                                           k,
                                           alpha,
                                           A,
                                           lda,
                                           nnz,
                                           descr,
                                           csc_val,
                                           csc_col_ptr,
                                           csc_row_ind,
                                           beta,
                                           C,
                                           ldc);
}

extern "C" rocsparse_status rocsparse_dmcscm(rocsparse_handle handle,
                                             rocsparse_operation trans_A,
                                             rocsparse_operation trans_B,
                                             rocsparse_int m,
                                             rocsparse_int n,
                                             rocsparse_int k,
                                             const double* alpha,
                                             const double* A,
                                             rocsparse_int lda,
                                             rocsparse_int nnz,
                                             const rocsparse_mat_descr descr,
                                             const double* csc_val,
                                             const rocsparse_int* csc_col_ptr,
                                             const rocsparse_int* csc_row_ind,
                                             const double* beta,
                                             double* C,
                                             rocsparse_int ldc)
{
    return rocsparse_mcscm_template<double>(handle,
                                            trans_A,
                                            trans_B,
                                            m,
                                            n,
                                            k,
                                            alpha,
                                            A,
                                            lda,
                                            nnz,
                                            descr,
                                            csc_val,
                                            csc_col_ptr,
                                            csc_row_ind,
                                            beta,
                                            C,
                                            ldc);
}
