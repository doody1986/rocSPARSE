#pragma once
#ifndef IM2COL_DEVICE_H
#define IM2COL_DEVICE_H

#include <hip/hip_runtime.h>

template <typename T, int TILE_SZ_X, int TILE_SZ_Y>
__global__ void im2col_nchw_kernel(const T* __restrict__ im,
                                          const int c,
                                          const int h,
                                          const int w,
                                          const int wei_h,
                                          const int wei_w,
                                          const int out_h,
                                          const int out_w,
                                          const int pad_h,
                                          const int pad_w,
                                          const int stride_h,
                                          const int stride_w,
                                          T* __restrict__ col)
{
    int t_id = hipThreadIdx_x;
    //int img_tile_size = hipBlockDim_x;
    int img_tile_id = hipBlockIdx_x;
    //int n_img_tiles = hipGridDim_x;
    int c_id = hipBlockIdx_y % c;
    int n_id = hipBlockIdx_y / c;

    // Load image into LDS
    HIP_DYNAMIC_SHARED(T, local_im);
    int num_img_blks_x = (out_w + TILE_SZ_X - 1) / TILE_SZ_X;
    int im_x = (img_tile_id % num_img_blks_x) * TILE_SZ_X;
    int im_y = (img_tile_id / num_img_blks_x) * TILE_SZ_Y;

    int out_cols_wg = im_x + TILE_SZ_X <= out_w ? TILE_SZ_X : out_w - im_x;
    int out_rows_wg = im_y + TILE_SZ_Y <= out_h ? TILE_SZ_Y : out_h - im_y;

    int im_cols_wg = TILE_SZ_X * stride_w + wei_w;
    int im_rows_wg = TILE_SZ_Y * stride_h + wei_h;
    int inner_tid  = t_id;

    while(inner_tid < im_cols_wg * im_rows_wg)
    {
        int row_to_use = inner_tid / im_cols_wg;
        int col_to_use = inner_tid % im_cols_wg;
        int lm_offset  = row_to_use * im_cols_wg + col_to_use;
        if(im_y * stride_h + row_to_use >= pad_h && im_y * stride_h + row_to_use < h + pad_h &&
           im_x * stride_w + col_to_use >= pad_w && im_x * stride_w + col_to_use < w + pad_w &&
           row_to_use < im_rows_wg)
        {
            int im_off_h        = im_y * stride_h + row_to_use - pad_h;
            int im_off_w        = im_x * stride_w + col_to_use - pad_w;
            local_im[lm_offset] = im[n_id * c * h * w + c_id * h * w + im_off_h * w + im_off_w];
        }
        else
            local_im[lm_offset] = 0;

        inner_tid += 256;
    }
    __syncthreads();

    inner_tid = t_id;
    while(inner_tid < out_cols_wg * out_rows_wg)
    {
        int out_x = inner_tid % out_cols_wg;
        int out_y = inner_tid / out_cols_wg;

        int col_x = (im_y + out_y) * out_w + im_x + out_x;
        int col_sz = wei_h * wei_w * c;

        for(int y = 0; y < wei_h; y++)
        {
            for(int x = 0; x < wei_w; x++)
            {
                int im_off_h = out_y * stride_h + y;
                int im_off_w = out_x * stride_w + x;
                col[n_id*out_h*out_w*col_sz + col_x*col_sz + c_id*wei_h*wei_w + y*wei_w + x] =
                    local_im[im_off_h*im_cols_wg + im_off_w];
            }
        }
        inner_tid += 256;
    }
}

template <typename T, int TILE_SZ_X, int TILE_SZ_Y>
__global__ void im2col_nhwc_kernel(const T* __restrict__ im,
                                          const int c,
                                          const int h,
                                          const int w,
                                          const int wei_h,
                                          const int wei_w,
                                          const int out_h,
                                          const int out_w,
                                          const int pad_h,
                                          const int pad_w,
                                          const int stride_h,
                                          const int stride_w,
                                          T* __restrict__ col)
{
    int t_id = hipThreadIdx_x;
    //int img_tile_size = hipBlockDim_x;
    int img_tile_id = hipBlockIdx_x;
    //int n_img_tiles = hipGridDim_x;
    int c_id = hipBlockIdx_y % c;
    int n_id = hipBlockIdx_y / c;

    // Load image into LDS
    HIP_DYNAMIC_SHARED(T, local_im);
    int num_img_blks_x = (out_w + TILE_SZ_X - 1) / TILE_SZ_X;
    int im_x = (img_tile_id % num_img_blks_x) * TILE_SZ_X;
    int im_y = (img_tile_id / num_img_blks_x) * TILE_SZ_Y;

    int out_cols_wg = im_x + TILE_SZ_X <= out_w ? TILE_SZ_X : out_w - im_x;
    int out_rows_wg = im_y + TILE_SZ_Y <= out_h ? TILE_SZ_Y : out_h - im_y;

    int im_cols_wg = TILE_SZ_X * stride_w + wei_w;
    int im_rows_wg = TILE_SZ_Y * stride_h + wei_h;
    int inner_tid  = t_id;

    while(inner_tid < im_cols_wg * im_rows_wg)
    {
        int row_to_use = inner_tid / im_cols_wg;
        int col_to_use = inner_tid % im_cols_wg;
        int lm_offset  = row_to_use * im_cols_wg + col_to_use;
        if(im_y * stride_h + row_to_use >= pad_h && im_y * stride_h + row_to_use < h + pad_h &&
           im_x * stride_w + col_to_use >= pad_w && im_x * stride_w + col_to_use < w + pad_w &&
           row_to_use < im_rows_wg)
        {
            int im_off_h        = im_y * stride_h + row_to_use - pad_h;
            int im_off_w        = im_x * stride_w + col_to_use - pad_w;
            local_im[lm_offset] = im[n_id * c * h * w + im_off_h * w * c + im_off_w * c + c_id];
        }
        else
            local_im[lm_offset] = 0;

        inner_tid += 256;
    }
    __syncthreads();

    inner_tid = t_id;
    while(inner_tid < out_cols_wg * out_rows_wg)
    {
        int out_x = inner_tid % out_cols_wg;
        int out_y = inner_tid / out_cols_wg;

        int col_x = (im_y + out_y) * out_w + im_x + out_x;
        int col_sz = wei_h * wei_w * c;

        for(int y = 0; y < wei_h; y++)
        {
            for(int x = 0; x < wei_w; x++)
            {
                int im_off_h = out_y * stride_h + y;
                int im_off_w = out_x * stride_w + x;
                col[n_id*out_h*out_w*col_sz + col_x*col_sz + y*wei_w*c + x*c + c_id] =
                    local_im[im_off_h*im_cols_wg + im_off_w];
            }
        }
        inner_tid += 256;
    }
}

#endif // IM2COL_DEVICE_H
