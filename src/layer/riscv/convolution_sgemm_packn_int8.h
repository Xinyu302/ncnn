// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

static void im2col_sgemm_packn_int8_rvv(const Mat& bottom_im2col, Mat& top_blob, const Mat& kernel, const Option& opt)
{
    const int packn = csrr_vlenb() / 2;
    const size_t vl = vsetvl_e8m1(packn);

    // Mat bottom_im2col(size, maxk, inch, 1u * packn, packn, opt.workspace_allocator);

    const int size = bottom_im2col.w;
    const int maxk = bottom_im2col.h;
    const int inch = bottom_im2col.c;

    const int outch = top_blob.c;

    // permute
    Mat tmp;
    if (size >= 8)
        tmp.create(8 * maxk, inch, size / 8 + (size % 8) / 4 + (size % 4) / 2 + size % 2, 1u * packn, packn, opt.workspace_allocator);
    else if (size >= 4)
        tmp.create(4 * maxk, inch, size / 4 + (size % 4) / 2 + size % 2, 1u * packn, packn, opt.workspace_allocator);
    else if (size >= 2)
        tmp.create(2 * maxk, inch, size / 2 + size % 2, 1u * packn, packn, opt.workspace_allocator);
    else
        tmp.create(maxk, inch, size, 1u * packn, packn, opt.workspace_allocator);
    {
        int remain_size_start = 0;
        int nn_size = size >> 3;

#pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 8;

            int8_t* tmpptr = tmp.channel(i / 8);

            for (int q = 0; q < inch; q++)
            {
                const int8_t* img0 = (const int8_t*)bottom_im2col.channel(q) + i * packn;

                for (int k = 0; k < maxk; k++)
                {
                    vint8m1_t _val0 = vle8_v_i8m1(img0, vl);
                    vint8m1_t _val1 = vle8_v_i8m1(img0 + packn, vl);
                    vint8m1_t _val2 = vle8_v_i8m1(img0 + packn * 2, vl);
                    vint8m1_t _val3 = vle8_v_i8m1(img0 + packn * 3, vl);
                    vint8m1_t _val4 = vle8_v_i8m1(img0 + packn * 4, vl);
                    vint8m1_t _val5 = vle8_v_i8m1(img0 + packn * 5, vl);
                    vint8m1_t _val6 = vle8_v_i8m1(img0 + packn * 6, vl);
                    vint8m1_t _val7 = vle8_v_i8m1(img0 + packn * 7, vl);
                    vsseg8e8_v_i8m1(tmpptr, _val0, _val1, _val2, _val3, _val4, _val5, _val6, _val7, vl);

                    img0 += size * packn;
                    tmpptr += packn * 8;
                }
            }
        }

        remain_size_start += nn_size << 3;
        nn_size = (size - remain_size_start) >> 2;

#pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 4;

            int8_t* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);

            for (int q = 0; q < inch; q++)
            {
                const int8_t* img0 = (const int8_t*)bottom_im2col.channel(q) + i * packn;

                for (int k = 0; k < maxk; k++)
                {
                    vint8m1_t _val0 = vle8_v_i8m1(img0, vl);
                    vint8m1_t _val1 = vle8_v_i8m1(img0 + packn, vl);
                    vint8m1_t _val2 = vle8_v_i8m1(img0 + packn * 2, vl);
                    vint8m1_t _val3 = vle8_v_i8m1(img0 + packn * 3, vl);
                    vsseg4e8_v_i8m1(tmpptr, _val0, _val1, _val2, _val3, vl);

                    img0 += size * packn;
                    tmpptr += packn * 4;
                }
            }
        }

        remain_size_start += nn_size << 2;

        nn_size = (size - remain_size_start) >> 1;

#pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 2;

            int8_t* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2);

            for (int q = 0; q < inch; q++)
            {
                const int8_t* img0 = (const int8_t*)bottom_im2col.channel(q) + i * packn;

                for (int k = 0; k < maxk; k++)
                {
                    vint8m1_t _val0 = vle8_v_i8m1(img0, vl);
                    vint8m1_t _val1 = vle8_v_i8m1(img0 + packn, vl);
                    vsseg2e8_v_i8m1(tmpptr, _val0, _val1, vl);

                    img0 += size * packn;
                    tmpptr += packn * 2;
                }
            }
        }

        remain_size_start += nn_size << 1;

#pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_size_start; i < size; i++)
        {
            int8_t* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2);

            for (int q = 0; q < inch; q++)
            {
                const int8_t* img0 = (const int8_t*)bottom_im2col.channel(q) + i * packn;

                for (int k = 0; k < maxk; k++)
                {
                    vint8m1_t _val = vle8_v_i8m1(img0, vl);
                    vse8_v_i8m1(tmpptr, _val, vl);

                    img0 += size * packn;
                    tmpptr += packn;
                }
            }
        }
    }
    int p = 0;
#pragma omp parallel for num_threads(opt.num_threads)
    for (; p + 2 < outch; p += 3)
    {
        int32_t* outptr0 = top_blob.channel(p + 0);
        int32_t* outptr1 = top_blob.channel(p + 1);
        int32_t* outptr2 = top_blob.channel(p + 2);

        int i = 0;
        for (; i + 7 < size; i += 8)
        {
            {
                const int8_t* tmpptr = tmp.channel(i / 8);
                const int8_t* kptr0 = kernel.channel(p + 0);
                const int8_t* kptr1 = kernel.channel(p + 1);
                const int8_t* kptr2 = kernel.channel(p + 2);

                int nn = inch * maxk * packn; // inch always > 0

                vint32m2_t _sum0_0 = vmv_v_x_i32m2(0, vl);
                vint32m2_t _sum0_1 = vmv_v_x_i32m2(0, vl);
                vint32m2_t _sum0_2 = vmv_v_x_i32m2(0, vl);
                vint32m2_t _sum0_3 = vmv_v_x_i32m2(0, vl);
                vint32m2_t _sum1_0 = vmv_v_x_i32m2(0, vl);
                vint32m2_t _sum1_1 = vmv_v_x_i32m2(0, vl);
                vint32m2_t _sum1_2 = vmv_v_x_i32m2(0, vl);
                vint32m2_t _sum1_3 = vmv_v_x_i32m2(0, vl);
                vint32m2_t _sum2_0 = vmv_v_x_i32m2(0, vl);
                vint32m2_t _sum2_1 = vmv_v_x_i32m2(0, vl);
                vint32m2_t _sum2_2 = vmv_v_x_i32m2(0, vl);
                vint32m2_t _sum2_3 = vmv_v_x_i32m2(0, vl);

                int j = 0;
                for (; j < nn; j += 2)
                {
                    int8_t val0_0 = *tmpptr++;
                    int8_t val1_0 = *tmpptr++;
                    int8_t val2_0 = *tmpptr++;
                    int8_t val3_0 = *tmpptr++;
                    tmpptr += 4;
                    int8_t val0_1 = *tmpptr++;
                    int8_t val1_1 = *tmpptr++;
                    int8_t val2_1 = *tmpptr++;
                    int8_t val3_1 = *tmpptr++;
                    tmpptr += 4;

                    vint8m1_t _w0_0 = vle8_v_i8m1(kptr0, vl * 2);
                    vint8m1_t _w1_0 = vle8_v_i8m1(kptr1, vl * 2);
                    vint8m1_t _w2_0 = vle8_v_i8m1(kptr2, vl * 2);
                    _sum0_0 = vwadd_wv_i32m2(_sum0_0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0_0, val0_0, vl), 0), vl);
                    _sum0_1 = vwadd_wv_i32m2(_sum0_1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0_0, val1_0, vl), 0), vl);
                    _sum0_2 = vwadd_wv_i32m2(_sum0_2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0_0, val2_0, vl), 0), vl);
                    _sum0_3 = vwadd_wv_i32m2(_sum0_3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0_0, val3_0, vl), 0), vl);
                    _sum1_0 = vwadd_wv_i32m2(_sum1_0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1_0, val0_0, vl), 0), vl);
                    _sum1_1 = vwadd_wv_i32m2(_sum1_1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1_0, val1_0, vl), 0), vl);
                    _sum1_2 = vwadd_wv_i32m2(_sum1_2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1_0, val2_0, vl), 0), vl);
                    _sum1_3 = vwadd_wv_i32m2(_sum1_3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1_0, val3_0, vl), 0), vl);
                    _sum2_0 = vwadd_wv_i32m2(_sum2_0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w2_0, val0_0, vl), 0), vl);
                    _sum2_1 = vwadd_wv_i32m2(_sum2_1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w2_0, val1_0, vl), 0), vl);
                    _sum2_2 = vwadd_wv_i32m2(_sum2_2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w2_0, val2_0, vl), 0), vl);
                    _sum2_3 = vwadd_wv_i32m2(_sum2_3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w2_0, val3_0, vl), 0), vl);

                    vint8m1_t _w0_1 = vslidedown_vx_i8m1(_w0_0, _w0_0, vl, vl);
                    vint8m1_t _w1_1 = vslidedown_vx_i8m1(_w1_0, _w1_0, vl, vl);
                    vint8m1_t _w2_1 = vslidedown_vx_i8m1(_w2_0, _w2_0, vl, vl);
                    _sum0_0 = vwadd_wv_i32m2(_sum0_0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0_1, val0_1, vl), 0), vl);
                    _sum0_1 = vwadd_wv_i32m2(_sum0_1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0_1, val1_1, vl), 0), vl);
                    _sum0_2 = vwadd_wv_i32m2(_sum0_2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0_1, val2_1, vl), 0), vl);
                    _sum0_3 = vwadd_wv_i32m2(_sum0_3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0_1, val3_1, vl), 0), vl);
                    _sum1_0 = vwadd_wv_i32m2(_sum1_0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1_1, val0_1, vl), 0), vl);
                    _sum1_1 = vwadd_wv_i32m2(_sum1_1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1_1, val1_1, vl), 0), vl);
                    _sum1_2 = vwadd_wv_i32m2(_sum1_2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1_1, val2_1, vl), 0), vl);
                    _sum1_3 = vwadd_wv_i32m2(_sum1_3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1_1, val3_1, vl), 0), vl);
                    _sum2_0 = vwadd_wv_i32m2(_sum2_0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w2_1, val0_1, vl), 0), vl);
                    _sum2_1 = vwadd_wv_i32m2(_sum2_1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w2_1, val1_1, vl), 0), vl);
                    _sum2_2 = vwadd_wv_i32m2(_sum2_2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w2_1, val2_1, vl), 0), vl);
                    _sum2_3 = vwadd_wv_i32m2(_sum2_3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w2_1, val3_1, vl), 0), vl);

                    kptr0 += packn * 2;
                    kptr1 += packn * 2;
                    kptr2 += packn * 2;
                }

                for (; j < nn; j++)
                {
                    int8_t val0 = *tmpptr++;
                    int8_t val1 = *tmpptr++;
                    int8_t val2 = *tmpptr++;
                    int8_t val3 = *tmpptr++;
                    tmpptr += 4;

                    vint8m1_t _w0 = vle8_v_i8m1(kptr0, vl);
                    vint8m1_t _w1 = vle8_v_i8m1(kptr1, vl);
                    vint8m1_t _w2 = vle8_v_i8m1(kptr2, vl);
                    _sum0_0 = vwadd_wv_i32m2(_sum0_0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0, val0, vl), 0), vl);
                    _sum0_1 = vwadd_wv_i32m2(_sum0_1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0, val1, vl), 0), vl);
                    _sum0_2 = vwadd_wv_i32m2(_sum0_2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0, val2, vl), 0), vl);
                    _sum0_3 = vwadd_wv_i32m2(_sum0_3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0, val3, vl), 0), vl);
                    _sum1_0 = vwadd_wv_i32m2(_sum1_0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1, val0, vl), 0), vl);
                    _sum1_1 = vwadd_wv_i32m2(_sum1_1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1, val1, vl), 0), vl);
                    _sum1_2 = vwadd_wv_i32m2(_sum1_2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1, val2, vl), 0), vl);
                    _sum1_3 = vwadd_wv_i32m2(_sum1_3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1, val3, vl), 0), vl);
                    _sum2_0 = vwadd_wv_i32m2(_sum2_0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w2, val0, vl), 0), vl);
                    _sum2_1 = vwadd_wv_i32m2(_sum2_1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w2, val1, vl), 0), vl);
                    _sum2_2 = vwadd_wv_i32m2(_sum2_2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w2, val2, vl), 0), vl);
                    _sum2_3 = vwadd_wv_i32m2(_sum2_3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w2, val3, vl), 0), vl);

                    kptr0 += packn;
                    kptr1 += packn;
                    kptr2 += packn;
                }

                vse32_v_i32m2(outptr0 + packn * 0, _sum0_0, vl);
                vse32_v_i32m2(outptr0 + packn * 1, _sum0_1, vl);
                vse32_v_i32m2(outptr0 + packn * 2, _sum0_2, vl);
                vse32_v_i32m2(outptr0 + packn * 3, _sum0_3, vl);
                vse32_v_i32m2(outptr1 + packn * 0, _sum1_0, vl);
                vse32_v_i32m2(outptr1 + packn * 1, _sum1_1, vl);
                vse32_v_i32m2(outptr1 + packn * 2, _sum1_2, vl);
                vse32_v_i32m2(outptr1 + packn * 3, _sum1_3, vl);
                vse32_v_i32m2(outptr2 + packn * 0, _sum2_0, vl);
                vse32_v_i32m2(outptr2 + packn * 1, _sum2_1, vl);
                vse32_v_i32m2(outptr2 + packn * 2, _sum2_2, vl);
                vse32_v_i32m2(outptr2 + packn * 3, _sum2_3, vl);
            }
            {
                const int8_t* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
                const int8_t* kptr0 = kernel.channel(p + 0);
                const int8_t* kptr1 = kernel.channel(p + 1);
                const int8_t* kptr2 = kernel.channel(p + 2);

                int nn = inch * maxk * packn; // inch always > 0

                vint32m2_t _sum0_4 = vmv_v_x_i32m2(0, vl);
                vint32m2_t _sum0_5 = vmv_v_x_i32m2(0, vl);
                vint32m2_t _sum0_6 = vmv_v_x_i32m2(0, vl);
                vint32m2_t _sum0_7 = vmv_v_x_i32m2(0, vl);
                vint32m2_t _sum1_4 = vmv_v_x_i32m2(0, vl);
                vint32m2_t _sum1_5 = vmv_v_x_i32m2(0, vl);
                vint32m2_t _sum1_6 = vmv_v_x_i32m2(0, vl);
                vint32m2_t _sum1_7 = vmv_v_x_i32m2(0, vl);
                vint32m2_t _sum2_4 = vmv_v_x_i32m2(0, vl);
                vint32m2_t _sum2_5 = vmv_v_x_i32m2(0, vl);
                vint32m2_t _sum2_6 = vmv_v_x_i32m2(0, vl);
                vint32m2_t _sum2_7 = vmv_v_x_i32m2(0, vl);

                int j = 0;
                for (; j < nn; j += 2)
                {
                    tmpptr += 4;
                    int8_t val4_0 = *tmpptr++;
                    int8_t val5_0 = *tmpptr++;
                    int8_t val6_0 = *tmpptr++;
                    int8_t val7_0 = *tmpptr++;
                    tmpptr += 4;
                    int8_t val4_1 = *tmpptr++;
                    int8_t val5_1 = *tmpptr++;
                    int8_t val6_1 = *tmpptr++;
                    int8_t val7_1 = *tmpptr++;

                    vint8m1_t _w0_0 = vle8_v_i8m1(kptr0, vl * 2);
                    vint8m1_t _w1_0 = vle8_v_i8m1(kptr1, vl * 2);
                    vint8m1_t _w2_0 = vle8_v_i8m1(kptr2, vl * 2);
                    _sum0_4 = vwadd_wv_i32m2(_sum0_4, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0_0, val4_0, vl), 0), vl);
                    _sum0_5 = vwadd_wv_i32m2(_sum0_5, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0_0, val5_0, vl), 0), vl);
                    _sum0_6 = vwadd_wv_i32m2(_sum0_6, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0_0, val6_0, vl), 0), vl);
                    _sum0_7 = vwadd_wv_i32m2(_sum0_7, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0_0, val7_0, vl), 0), vl);
                    _sum1_4 = vwadd_wv_i32m2(_sum1_4, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1_0, val4_0, vl), 0), vl);
                    _sum1_5 = vwadd_wv_i32m2(_sum1_5, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1_0, val5_0, vl), 0), vl);
                    _sum1_6 = vwadd_wv_i32m2(_sum1_6, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1_0, val6_0, vl), 0), vl);
                    _sum1_7 = vwadd_wv_i32m2(_sum1_7, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1_0, val7_0, vl), 0), vl);
                    _sum2_4 = vwadd_wv_i32m2(_sum2_4, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w2_0, val4_0, vl), 0), vl);
                    _sum2_5 = vwadd_wv_i32m2(_sum2_5, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w2_0, val5_0, vl), 0), vl);
                    _sum2_6 = vwadd_wv_i32m2(_sum2_6, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w2_0, val6_0, vl), 0), vl);
                    _sum2_7 = vwadd_wv_i32m2(_sum2_7, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w2_0, val7_0, vl), 0), vl);

                    vint8m1_t _w0_1 = vslidedown_vx_i8m1(_w0_0, _w0_0, vl, vl);
                    vint8m1_t _w1_1 = vslidedown_vx_i8m1(_w1_0, _w1_0, vl, vl);
                    vint8m1_t _w2_1 = vslidedown_vx_i8m1(_w2_0, _w2_0, vl, vl);
                    _sum0_4 = vwadd_wv_i32m2(_sum0_4, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0_1, val4_1, vl), 0), vl);
                    _sum0_5 = vwadd_wv_i32m2(_sum0_5, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0_1, val5_1, vl), 0), vl);
                    _sum0_6 = vwadd_wv_i32m2(_sum0_6, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0_1, val6_1, vl), 0), vl);
                    _sum0_7 = vwadd_wv_i32m2(_sum0_7, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0_1, val7_1, vl), 0), vl);
                    _sum1_4 = vwadd_wv_i32m2(_sum1_4, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1_1, val4_1, vl), 0), vl);
                    _sum1_5 = vwadd_wv_i32m2(_sum1_5, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1_1, val5_1, vl), 0), vl);
                    _sum1_6 = vwadd_wv_i32m2(_sum1_6, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1_1, val6_1, vl), 0), vl);
                    _sum1_7 = vwadd_wv_i32m2(_sum1_7, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1_1, val7_1, vl), 0), vl);
                    _sum2_4 = vwadd_wv_i32m2(_sum2_4, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w2_1, val4_1, vl), 0), vl);
                    _sum2_5 = vwadd_wv_i32m2(_sum2_5, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w2_1, val5_1, vl), 0), vl);
                    _sum2_6 = vwadd_wv_i32m2(_sum2_6, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w2_1, val6_1, vl), 0), vl);
                    _sum2_7 = vwadd_wv_i32m2(_sum2_7, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w2_1, val7_1, vl), 0), vl);

                    kptr0 += packn * 2;
                    kptr1 += packn * 2;
                    kptr2 += packn * 2;
                }

                for (; j < nn; j++)
                {
                    tmpptr += 4;
                    int8_t val4 = *tmpptr++;
                    int8_t val5 = *tmpptr++;
                    int8_t val6 = *tmpptr++;
                    int8_t val7 = *tmpptr++;

                    vint8m1_t _w0 = vle8_v_i8m1(kptr0, vl);
                    vint8m1_t _w1 = vle8_v_i8m1(kptr1, vl);
                    vint8m1_t _w2 = vle8_v_i8m1(kptr2, vl);
                    _sum0_4 = vwadd_wv_i32m2(_sum0_4, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0, val4, vl), 0), vl);
                    _sum0_5 = vwadd_wv_i32m2(_sum0_5, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0, val5, vl), 0), vl);
                    _sum0_6 = vwadd_wv_i32m2(_sum0_6, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0, val6, vl), 0), vl);
                    _sum0_7 = vwadd_wv_i32m2(_sum0_7, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0, val7, vl), 0), vl);
                    _sum1_4 = vwadd_wv_i32m2(_sum1_4, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1, val4, vl), 0), vl);
                    _sum1_5 = vwadd_wv_i32m2(_sum1_5, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1, val5, vl), 0), vl);
                    _sum1_6 = vwadd_wv_i32m2(_sum1_6, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1, val6, vl), 0), vl);
                    _sum1_7 = vwadd_wv_i32m2(_sum1_7, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1, val7, vl), 0), vl);
                    _sum2_4 = vwadd_wv_i32m2(_sum2_4, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w2, val4, vl), 0), vl);
                    _sum2_5 = vwadd_wv_i32m2(_sum2_5, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w2, val5, vl), 0), vl);
                    _sum2_6 = vwadd_wv_i32m2(_sum2_6, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w2, val6, vl), 0), vl);
                    _sum2_7 = vwadd_wv_i32m2(_sum2_7, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w2, val7, vl), 0), vl);

                    kptr0 += packn;
                    kptr1 += packn;
                    kptr2 += packn;
                }

                vse32_v_i32m2(outptr0 + packn * 4, _sum0_4, vl);
                vse32_v_i32m2(outptr0 + packn * 5, _sum0_5, vl);
                vse32_v_i32m2(outptr0 + packn * 6, _sum0_6, vl);
                vse32_v_i32m2(outptr0 + packn * 7, _sum0_7, vl);
                vse32_v_i32m2(outptr1 + packn * 4, _sum1_4, vl);
                vse32_v_i32m2(outptr1 + packn * 5, _sum1_5, vl);
                vse32_v_i32m2(outptr1 + packn * 6, _sum1_6, vl);
                vse32_v_i32m2(outptr1 + packn * 7, _sum1_7, vl);
                vse32_v_i32m2(outptr2 + packn * 4, _sum2_4, vl);
                vse32_v_i32m2(outptr2 + packn * 5, _sum2_5, vl);
                vse32_v_i32m2(outptr2 + packn * 6, _sum2_6, vl);
                vse32_v_i32m2(outptr2 + packn * 7, _sum2_7, vl);
            }
            outptr0 += packn * 8;
            outptr1 += packn * 8;
            outptr2 += packn * 8;
        }
        for (; i + 3 < size; i += 4)
        {
            const int8_t* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
            const int8_t* kptr0 = kernel.channel(p + 0);
            const int8_t* kptr1 = kernel.channel(p + 1);
            const int8_t* kptr2 = kernel.channel(p + 2);

            int nn = inch * maxk * packn; // inch always > 0

            vint32m2_t _sum0_0 = vmv_v_x_i32m2(0, vl);
            vint32m2_t _sum0_1 = vmv_v_x_i32m2(0, vl);
            vint32m2_t _sum0_2 = vmv_v_x_i32m2(0, vl);
            vint32m2_t _sum0_3 = vmv_v_x_i32m2(0, vl);
            vint32m2_t _sum1_0 = vmv_v_x_i32m2(0, vl);
            vint32m2_t _sum1_1 = vmv_v_x_i32m2(0, vl);
            vint32m2_t _sum1_2 = vmv_v_x_i32m2(0, vl);
            vint32m2_t _sum1_3 = vmv_v_x_i32m2(0, vl);
            vint32m2_t _sum2_0 = vmv_v_x_i32m2(0, vl);
            vint32m2_t _sum2_1 = vmv_v_x_i32m2(0, vl);
            vint32m2_t _sum2_2 = vmv_v_x_i32m2(0, vl);
            vint32m2_t _sum2_3 = vmv_v_x_i32m2(0, vl);

            int j = 0;
            for (; j < nn; j += 2)
            {
                int8_t val0_0 = *tmpptr++;
                int8_t val1_0 = *tmpptr++;
                int8_t val2_0 = *tmpptr++;
                int8_t val3_0 = *tmpptr++;
                int8_t val0_1 = *tmpptr++;
                int8_t val1_1 = *tmpptr++;
                int8_t val2_1 = *tmpptr++;
                int8_t val3_1 = *tmpptr++;

                vint8m1_t _w0_0 = vle8_v_i8m1(kptr0, vl * 2);
                vint8m1_t _w1_0 = vle8_v_i8m1(kptr1, vl * 2);
                vint8m1_t _w2_0 = vle8_v_i8m1(kptr2, vl * 2);
                _sum0_0 = vwadd_wv_i32m2(_sum0_0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0_0, val0_0, vl), 0), vl);
                _sum0_1 = vwadd_wv_i32m2(_sum0_1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0_0, val1_0, vl), 0), vl);
                _sum0_2 = vwadd_wv_i32m2(_sum0_2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0_0, val2_0, vl), 0), vl);
                _sum0_3 = vwadd_wv_i32m2(_sum0_3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0_0, val3_0, vl), 0), vl);
                _sum1_0 = vwadd_wv_i32m2(_sum1_0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1_0, val0_0, vl), 0), vl);
                _sum1_1 = vwadd_wv_i32m2(_sum1_1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1_0, val1_0, vl), 0), vl);
                _sum1_2 = vwadd_wv_i32m2(_sum1_2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1_0, val2_0, vl), 0), vl);
                _sum1_3 = vwadd_wv_i32m2(_sum1_3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1_0, val3_0, vl), 0), vl);
                _sum2_0 = vwadd_wv_i32m2(_sum2_0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w2_0, val0_0, vl), 0), vl);
                _sum2_1 = vwadd_wv_i32m2(_sum2_1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w2_0, val1_0, vl), 0), vl);
                _sum2_2 = vwadd_wv_i32m2(_sum2_2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w2_0, val2_0, vl), 0), vl);
                _sum2_3 = vwadd_wv_i32m2(_sum2_3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w2_0, val3_0, vl), 0), vl);

                vint8m1_t _w0_1 = vslidedown_vx_i8m1(_w0_0, _w0_0, vl, vl);
                vint8m1_t _w1_1 = vslidedown_vx_i8m1(_w1_0, _w1_0, vl, vl);
                vint8m1_t _w2_1 = vslidedown_vx_i8m1(_w2_0, _w2_0, vl, vl);
                _sum0_0 = vwadd_wv_i32m2(_sum0_0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0_1, val0_1, vl), 0), vl);
                _sum0_1 = vwadd_wv_i32m2(_sum0_1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0_1, val1_1, vl), 0), vl);
                _sum0_2 = vwadd_wv_i32m2(_sum0_2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0_1, val2_1, vl), 0), vl);
                _sum0_3 = vwadd_wv_i32m2(_sum0_3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0_1, val3_1, vl), 0), vl);
                _sum1_0 = vwadd_wv_i32m2(_sum1_0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1_1, val0_1, vl), 0), vl);
                _sum1_1 = vwadd_wv_i32m2(_sum1_1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1_1, val1_1, vl), 0), vl);
                _sum1_2 = vwadd_wv_i32m2(_sum1_2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1_1, val2_1, vl), 0), vl);
                _sum1_3 = vwadd_wv_i32m2(_sum1_3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1_1, val3_1, vl), 0), vl);
                _sum2_0 = vwadd_wv_i32m2(_sum2_0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w2_1, val0_1, vl), 0), vl);
                _sum2_1 = vwadd_wv_i32m2(_sum2_1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w2_1, val1_1, vl), 0), vl);
                _sum2_2 = vwadd_wv_i32m2(_sum2_2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w2_1, val2_1, vl), 0), vl);
                _sum2_3 = vwadd_wv_i32m2(_sum2_3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w2_1, val3_1, vl), 0), vl);

                kptr0 += packn * 2;
                kptr1 += packn * 2;
                kptr2 += packn * 2;
            }

            for (; j < nn; j++)
            {
                int8_t val0 = *tmpptr++;
                int8_t val1 = *tmpptr++;
                int8_t val2 = *tmpptr++;
                int8_t val3 = *tmpptr++;

                vint8m1_t _w0 = vle8_v_i8m1(kptr0, vl);
                vint8m1_t _w1 = vle8_v_i8m1(kptr1, vl);
                vint8m1_t _w2 = vle8_v_i8m1(kptr2, vl);
                _sum0_0 = vwadd_wv_i32m2(_sum0_0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0, val0, vl), 0), vl);
                _sum0_1 = vwadd_wv_i32m2(_sum0_1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0, val1, vl), 0), vl);
                _sum0_2 = vwadd_wv_i32m2(_sum0_2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0, val2, vl), 0), vl);
                _sum0_3 = vwadd_wv_i32m2(_sum0_3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0, val3, vl), 0), vl);
                _sum1_0 = vwadd_wv_i32m2(_sum1_0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1, val0, vl), 0), vl);
                _sum1_1 = vwadd_wv_i32m2(_sum1_1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1, val1, vl), 0), vl);
                _sum1_2 = vwadd_wv_i32m2(_sum1_2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1, val2, vl), 0), vl);
                _sum1_3 = vwadd_wv_i32m2(_sum1_3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1, val3, vl), 0), vl);
                _sum2_0 = vwadd_wv_i32m2(_sum2_0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w2, val0, vl), 0), vl);
                _sum2_1 = vwadd_wv_i32m2(_sum2_1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w2, val1, vl), 0), vl);
                _sum2_2 = vwadd_wv_i32m2(_sum2_2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w2, val2, vl), 0), vl);
                _sum2_3 = vwadd_wv_i32m2(_sum2_3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w2, val3, vl), 0), vl);

                kptr0 += packn;
                kptr1 += packn;
                kptr2 += packn;
            }

            vse32_v_i32m2(outptr0 + packn * 0, _sum0_0, vl);
            vse32_v_i32m2(outptr0 + packn * 1, _sum0_1, vl);
            vse32_v_i32m2(outptr0 + packn * 2, _sum0_2, vl);
            vse32_v_i32m2(outptr0 + packn * 3, _sum0_3, vl);
            vse32_v_i32m2(outptr1 + packn * 0, _sum1_0, vl);
            vse32_v_i32m2(outptr1 + packn * 1, _sum1_1, vl);
            vse32_v_i32m2(outptr1 + packn * 2, _sum1_2, vl);
            vse32_v_i32m2(outptr1 + packn * 3, _sum1_3, vl);
            vse32_v_i32m2(outptr2 + packn * 0, _sum2_0, vl);
            vse32_v_i32m2(outptr2 + packn * 1, _sum2_1, vl);
            vse32_v_i32m2(outptr2 + packn * 2, _sum2_2, vl);
            vse32_v_i32m2(outptr2 + packn * 3, _sum2_3, vl);

            outptr0 += packn * 4;
            outptr1 += packn * 4;
            outptr2 += packn * 4;
        }
        for (; i + 1 < size; i += 2)
        {
            const int8_t* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2);
            const int8_t* kptr0 = kernel.channel(p + 0);
            const int8_t* kptr1 = kernel.channel(p + 1);
            const int8_t* kptr2 = kernel.channel(p + 2);

            int nn = inch * maxk * packn; // inch always > 0

            vint32m2_t _sum0_0 = vmv_v_x_i32m2(0, vl);
            vint32m2_t _sum0_1 = vmv_v_x_i32m2(0, vl);
            vint32m2_t _sum1_0 = vmv_v_x_i32m2(0, vl);
            vint32m2_t _sum1_1 = vmv_v_x_i32m2(0, vl);
            vint32m2_t _sum2_0 = vmv_v_x_i32m2(0, vl);
            vint32m2_t _sum2_1 = vmv_v_x_i32m2(0, vl);

            int j = 0;
            for (; j < nn; j += 2)
            {
                int8_t val0_0 = *tmpptr++;
                int8_t val1_0 = *tmpptr++;
                int8_t val0_1 = *tmpptr++;
                int8_t val1_1 = *tmpptr++;

                vint8m1_t _w0_0 = vle8_v_i8m1(kptr0, vl * 2);
                vint8m1_t _w1_0 = vle8_v_i8m1(kptr1, vl * 2);
                vint8m1_t _w2_0 = vle8_v_i8m1(kptr2, vl * 2);
                _sum0_0 = vwadd_wv_i32m2(_sum0_0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0_0, val0_0, vl), 0), vl);
                _sum0_1 = vwadd_wv_i32m2(_sum0_1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0_0, val1_0, vl), 0), vl);
                _sum1_0 = vwadd_wv_i32m2(_sum1_0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1_0, val0_0, vl), 0), vl);
                _sum1_1 = vwadd_wv_i32m2(_sum1_1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1_0, val1_0, vl), 0), vl);
                _sum2_0 = vwadd_wv_i32m2(_sum2_0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w2_0, val0_0, vl), 0), vl);
                _sum2_1 = vwadd_wv_i32m2(_sum2_1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w2_0, val1_0, vl), 0), vl);

                vint8m1_t _w0_1 = vslidedown_vx_i8m1(_w0_0, _w0_0, vl, vl);
                vint8m1_t _w1_1 = vslidedown_vx_i8m1(_w1_0, _w1_0, vl, vl);
                vint8m1_t _w2_1 = vslidedown_vx_i8m1(_w2_0, _w2_0, vl, vl);
                _sum0_0 = vwadd_wv_i32m2(_sum0_0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0_1, val0_1, vl), 0), vl);
                _sum0_1 = vwadd_wv_i32m2(_sum0_1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0_1, val1_1, vl), 0), vl);
                _sum1_0 = vwadd_wv_i32m2(_sum1_0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1_1, val0_1, vl), 0), vl);
                _sum1_1 = vwadd_wv_i32m2(_sum1_1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1_1, val1_1, vl), 0), vl);
                _sum2_0 = vwadd_wv_i32m2(_sum2_0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w2_1, val0_1, vl), 0), vl);
                _sum2_1 = vwadd_wv_i32m2(_sum2_1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w2_1, val1_1, vl), 0), vl);

                kptr0 += packn * 2;
                kptr1 += packn * 2;
                kptr2 += packn * 2;
            }

            for (; j < nn; j++)
            {
                int8_t val0 = *tmpptr++;
                int8_t val1 = *tmpptr++;

                vint8m1_t _w0 = vle8_v_i8m1(kptr0, vl);
                vint8m1_t _w1 = vle8_v_i8m1(kptr1, vl);
                vint8m1_t _w2 = vle8_v_i8m1(kptr2, vl);
                _sum0_0 = vwadd_wv_i32m2(_sum0_0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0, val0, vl), 0), vl);
                _sum0_1 = vwadd_wv_i32m2(_sum0_1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0, val1, vl), 0), vl);
                _sum1_0 = vwadd_wv_i32m2(_sum1_0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1, val0, vl), 0), vl);
                _sum1_1 = vwadd_wv_i32m2(_sum1_1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1, val1, vl), 0), vl);
                _sum2_0 = vwadd_wv_i32m2(_sum2_0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w2, val0, vl), 0), vl);
                _sum2_1 = vwadd_wv_i32m2(_sum2_1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w2, val1, vl), 0), vl);

                kptr0 += packn;
                kptr1 += packn;
                kptr2 += packn;
            }

            vse32_v_i32m2(outptr0 + packn * 0, _sum0_0, vl);
            vse32_v_i32m2(outptr0 + packn * 1, _sum0_1, vl);
            vse32_v_i32m2(outptr1 + packn * 0, _sum1_0, vl);
            vse32_v_i32m2(outptr1 + packn * 1, _sum1_1, vl);
            vse32_v_i32m2(outptr2 + packn * 0, _sum2_0, vl);
            vse32_v_i32m2(outptr2 + packn * 1, _sum2_1, vl);

            outptr0 += packn * 2;
            outptr1 += packn * 2;
            outptr2 += packn * 2;
        }
        for (; i < size; i++)
        {
            const int8_t* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2);
            const int8_t* kptr0 = kernel.channel(p + 0);
            const int8_t* kptr1 = kernel.channel(p + 1);
            const int8_t* kptr2 = kernel.channel(p + 2);

            int nn = inch * maxk * packn; // inch always > 0

            vint32m2_t _sum0_0 = vmv_v_x_i32m2(0, vl);
            vint32m2_t _sum1_0 = vmv_v_x_i32m2(0, vl);
            vint32m2_t _sum2_0 = vmv_v_x_i32m2(0, vl);

            int j = 0;
            for (; j < nn; j += 2)
            {
                int8_t val0_0 = *tmpptr++;
                int8_t val0_1 = *tmpptr++;

                vint8m1_t _w0_0 = vle8_v_i8m1(kptr0, vl * 2);
                vint8m1_t _w1_0 = vle8_v_i8m1(kptr1, vl * 2);
                vint8m1_t _w2_0 = vle8_v_i8m1(kptr2, vl * 2);
                _sum0_0 = vwadd_wv_i32m2(_sum0_0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0_0, val0_0, vl), 0), vl);
                _sum1_0 = vwadd_wv_i32m2(_sum1_0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1_0, val0_0, vl), 0), vl);
                _sum2_0 = vwadd_wv_i32m2(_sum2_0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w2_0, val0_0, vl), 0), vl);

                vint8m1_t _w0_1 = vslidedown_vx_i8m1(_w0_0, _w0_0, vl, vl);
                vint8m1_t _w1_1 = vslidedown_vx_i8m1(_w1_0, _w1_0, vl, vl);
                vint8m1_t _w2_1 = vslidedown_vx_i8m1(_w2_0, _w2_0, vl, vl);
                _sum0_0 = vwadd_wv_i32m2(_sum0_0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0_1, val0_1, vl), 0), vl);
                _sum1_0 = vwadd_wv_i32m2(_sum1_0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1_1, val0_1, vl), 0), vl);
                _sum2_0 = vwadd_wv_i32m2(_sum2_0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w2_1, val0_1, vl), 0), vl);

                kptr0 += packn * 2;
                kptr1 += packn * 2;
                kptr2 += packn * 2;
            }

            for (; j < nn; j++)
            {
                int8_t val0 = *tmpptr++;

                vint8m1_t _w0 = vle8_v_i8m1(kptr0, vl);
                vint8m1_t _w1 = vle8_v_i8m1(kptr1, vl);
                vint8m1_t _w2 = vle8_v_i8m1(kptr2, vl);
                _sum0_0 = vwadd_wv_i32m2(_sum0_0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0, val0, vl), 0), vl);
                _sum1_0 = vwadd_wv_i32m2(_sum1_0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1, val0, vl), 0), vl);
                _sum2_0 = vwadd_wv_i32m2(_sum2_0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w2, val0, vl), 0), vl);

                kptr0 += packn;
                kptr1 += packn;
                kptr2 += packn;
            }

            vse32_v_i32m2(outptr0 + packn * 0, _sum0_0, vl);
            vse32_v_i32m2(outptr1 + packn * 0, _sum1_0, vl);
            vse32_v_i32m2(outptr2 + packn * 0, _sum2_0, vl);

            outptr0 += packn;
            outptr1 += packn;
            outptr2 += packn;
        }
    }
#pragma omp parallel for num_threads(opt.num_threads)
    for (; p < outch; p++)
    {
        int32_t* outptr0 = top_blob.channel(p);

        int i = 0;
        for (; i + 7 < size; i += 8)
        {
            const int8_t* tmpptr = tmp.channel(i / 8);
            const int8_t* kptr0 = kernel.channel(p);

            int nn = inch * maxk * packn; // inch always > 0

            vint32m2_t _sum0 = vmv_v_x_i32m2(0, vl);
            vint32m2_t _sum1 = vmv_v_x_i32m2(0, vl);
            vint32m2_t _sum2 = vmv_v_x_i32m2(0, vl);
            vint32m2_t _sum3 = vmv_v_x_i32m2(0, vl);
            vint32m2_t _sum4 = vmv_v_x_i32m2(0, vl);
            vint32m2_t _sum5 = vmv_v_x_i32m2(0, vl);
            vint32m2_t _sum6 = vmv_v_x_i32m2(0, vl);
            vint32m2_t _sum7 = vmv_v_x_i32m2(0, vl);
            int j = 0;
            for (; j + 1 < nn; j += 2)
            {
                int8_t val0_0 = *tmpptr++;
                int8_t val1_0 = *tmpptr++;
                int8_t val2_0 = *tmpptr++;
                int8_t val3_0 = *tmpptr++;
                int8_t val4_0 = *tmpptr++;
                int8_t val5_0 = *tmpptr++;
                int8_t val6_0 = *tmpptr++;
                int8_t val7_0 = *tmpptr++;
                int8_t val0_1 = *tmpptr++;
                int8_t val1_1 = *tmpptr++;
                int8_t val2_1 = *tmpptr++;
                int8_t val3_1 = *tmpptr++;
                int8_t val4_1 = *tmpptr++;
                int8_t val5_1 = *tmpptr++;
                int8_t val6_1 = *tmpptr++;
                int8_t val7_1 = *tmpptr++;

                vint8m1_t _w0 = vle8_v_i8m1(kptr0, vl * 2);
                _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0, val0_0, vl), 0), vl);
                _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0, val1_0, vl), 0), vl);
                _sum2 = vwadd_wv_i32m2(_sum2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0, val2_0, vl), 0), vl);
                _sum3 = vwadd_wv_i32m2(_sum3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0, val3_0, vl), 0), vl);
                _sum4 = vwadd_wv_i32m2(_sum4, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0, val4_0, vl), 0), vl);
                _sum5 = vwadd_wv_i32m2(_sum5, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0, val5_0, vl), 0), vl);
                _sum6 = vwadd_wv_i32m2(_sum6, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0, val6_0, vl), 0), vl);
                _sum7 = vwadd_wv_i32m2(_sum7, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0, val7_0, vl), 0), vl);

                vint8m1_t _w1 = vslidedown_vx_i8m1(_w0, _w0, 8, vl);
                _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1, val0_1, vl), 0), vl);
                _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1, val1_1, vl), 0), vl);
                _sum2 = vwadd_wv_i32m2(_sum2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1, val2_1, vl), 0), vl);
                _sum3 = vwadd_wv_i32m2(_sum3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1, val3_1, vl), 0), vl);
                _sum4 = vwadd_wv_i32m2(_sum4, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1, val4_1, vl), 0), vl);
                _sum5 = vwadd_wv_i32m2(_sum5, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1, val5_1, vl), 0), vl);
                _sum6 = vwadd_wv_i32m2(_sum6, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1, val6_1, vl), 0), vl);
                _sum7 = vwadd_wv_i32m2(_sum7, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1, val7_1, vl), 0), vl);

                kptr0 += 2 * packn;
            }

            for (; j < nn; j++)
            {
                int8_t val0 = *tmpptr++;
                int8_t val1 = *tmpptr++;
                int8_t val2 = *tmpptr++;
                int8_t val3 = *tmpptr++;
                int8_t val4 = *tmpptr++;
                int8_t val5 = *tmpptr++;
                int8_t val6 = *tmpptr++;
                int8_t val7 = *tmpptr++;
                vint8m1_t _w0 = vle8_v_i8m1(kptr0, vl);
                _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0, val0, vl), 0), vl);
                _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0, val1, vl), 0), vl);
                _sum2 = vwadd_wv_i32m2(_sum2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0, val2, vl), 0), vl);
                _sum3 = vwadd_wv_i32m2(_sum3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0, val3, vl), 0), vl);
                _sum4 = vwadd_wv_i32m2(_sum4, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0, val4, vl), 0), vl);
                _sum5 = vwadd_wv_i32m2(_sum5, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0, val5, vl), 0), vl);
                _sum6 = vwadd_wv_i32m2(_sum6, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0, val6, vl), 0), vl);
                _sum7 = vwadd_wv_i32m2(_sum7, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0, val7, vl), 0), vl);

                kptr0 += packn;
            }

            vse32_v_i32m2(outptr0, _sum0, vl);
            vse32_v_i32m2(outptr0 + packn, _sum1, vl);
            vse32_v_i32m2(outptr0 + packn * 2, _sum2, vl);
            vse32_v_i32m2(outptr0 + packn * 3, _sum3, vl);
            vse32_v_i32m2(outptr0 + packn * 4, _sum4, vl);
            vse32_v_i32m2(outptr0 + packn * 5, _sum5, vl);
            vse32_v_i32m2(outptr0 + packn * 6, _sum6, vl);
            vse32_v_i32m2(outptr0 + packn * 7, _sum7, vl);

            outptr0 += packn * 8;
        }
        for (; i + 3 < size; i += 4)
        {
            const int8_t* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
            const int8_t* kptr0 = kernel.channel(p);

            int nn = inch * maxk * packn; // inch always > 0

            vint32m2_t _sum0 = vmv_v_x_i32m2(0, vl);
            vint32m2_t _sum1 = vmv_v_x_i32m2(0, vl);
            vint32m2_t _sum2 = vmv_v_x_i32m2(0, vl);
            vint32m2_t _sum3 = vmv_v_x_i32m2(0, vl);

            int j = 0;
            for (; j < nn; j += 2)
            {
                int8_t val0_0 = *tmpptr++;
                int8_t val1_0 = *tmpptr++;
                int8_t val2_0 = *tmpptr++;
                int8_t val3_0 = *tmpptr++;
                int8_t val0_1 = *tmpptr++;
                int8_t val1_1 = *tmpptr++;
                int8_t val2_1 = *tmpptr++;
                int8_t val3_1 = *tmpptr++;
                vint8m1_t _w0 = vle8_v_i8m1(kptr0, vl * 2);
                _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0, val0_0, vl), 0), vl);
                _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0, val1_0, vl), 0), vl);
                _sum2 = vwadd_wv_i32m2(_sum2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0, val2_0, vl), 0), vl);
                _sum3 = vwadd_wv_i32m2(_sum3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0, val3_0, vl), 0), vl);

                vint8m1_t _w1 = vslidedown_vx_i8m1(_w0, _w0, 8, vl);

                _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1, val0_1, vl), 0), vl);
                _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1, val1_1, vl), 0), vl);
                _sum2 = vwadd_wv_i32m2(_sum2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1, val2_1, vl), 0), vl);
                _sum3 = vwadd_wv_i32m2(_sum3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1, val3_1, vl), 0), vl);

                kptr0 += packn * 2;
            }

            for (; j < nn; j++)
            {
                int8_t val0 = *tmpptr++;
                int8_t val1 = *tmpptr++;
                int8_t val2 = *tmpptr++;
                int8_t val3 = *tmpptr++;
                vint8m1_t _w0 = vle8_v_i8m1(kptr0, vl);
                _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0, val0, vl), 0), vl);
                _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0, val1, vl), 0), vl);
                _sum2 = vwadd_wv_i32m2(_sum2, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0, val2, vl), 0), vl);
                _sum3 = vwadd_wv_i32m2(_sum3, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0, val3, vl), 0), vl);

                kptr0 += packn;
            }

            vse32_v_i32m2(outptr0, _sum0, vl);
            vse32_v_i32m2(outptr0 + packn, _sum1, vl);
            vse32_v_i32m2(outptr0 + packn * 2, _sum2, vl);
            vse32_v_i32m2(outptr0 + packn * 3, _sum3, vl);

            outptr0 += packn * 4;
        }
        for (; i + 1 < size; i += 2)
        {
            const int8_t* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2);
            const int8_t* kptr0 = kernel.channel(p);

            int nn = inch * maxk * packn; // inch always > 0

            vint32m2_t _sum0 = vmv_v_x_i32m2(0, vl);
            vint32m2_t _sum1 = vmv_v_x_i32m2(0, vl);

            int j = 0;
            for (; j < nn; j += 2)
            {
                int8_t val0_0 = *tmpptr++;
                int8_t val1_0 = *tmpptr++;
                int8_t val0_1 = *tmpptr++;
                int8_t val1_1 = *tmpptr++;

                vint8m1_t _w0 = vle8_v_i8m1(kptr0, vl * 2);
                _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0, val0_0, vl), 0), vl);
                _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0, val1_0, vl), 0), vl);
                vint8m1_t _w1 = vslidedown_vx_i8m1(_w0, _w0, 8, vl);
                _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1, val0_1, vl), 0), vl);
                _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1, val1_1, vl), 0), vl);

                kptr0 += packn * 2;
            }

            for (; j < nn; j++)
            {
                int8_t val0 = *tmpptr++;
                int8_t val1 = *tmpptr++;
                vint8m1_t _w0 = vle8_v_i8m1(kptr0, vl);
                _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0, val0, vl), 0), vl);
                _sum1 = vwadd_wv_i32m2(_sum1, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0, val1, vl), 0), vl);

                kptr0 += packn;
            }

            vse32_v_i32m2(outptr0, _sum0, vl);
            vse32_v_i32m2(outptr0 + packn, _sum1, vl);

            outptr0 += packn * 2;
        }
        for (; i < size; i++)
        {
            const int8_t* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2);
            const int8_t* kptr0 = kernel.channel(p);

            int nn = inch * maxk * packn; // inch always > 0

            vint32m2_t _sum0 = vmv_v_x_i32m2(0, vl);

            int j = 0;
            for (; j < nn; j += 2)
            {
                int8_t val0_0 = *tmpptr++;
                int8_t val0_1 = *tmpptr++;
                vint8m1_t _w0 = vle8_v_i8m1(kptr0, vl * 2);
                _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0, val0_0, vl), 0), vl);
                vint8m1_t _w1 = vslidedown_vx_i8m1(_w0, _w0, 8, vl);
                _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w1, val0_1, vl), 0), vl);

                kptr0 += packn * 2;
            }

            for (; j < nn; j++)
            {
                int8_t val0 = *tmpptr++;
                vint8m1_t _w0 = vle8_v_i8m1(kptr0, vl);
                _sum0 = vwadd_wv_i32m2(_sum0, vget_v_i16m2_i16m1(vwmul_vx_i16m2(_w0, val0, vl), 0), vl);

                kptr0 += packn;
            }

            vse32_v_i32m2(outptr0, _sum0, vl);

            outptr0 += packn;
        }
    }
}

static void convolution_im2col_sgemm_packn_int8_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
{
    const int packn = csrr_vlenb() / 2;
    const size_t vl = vsetvl_e8m1(packn);

    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    const int size = outw * outh;

    const int maxk = kernel_w * kernel_h;

    // im2col
    Mat bottom_im2col(size, maxk, inch, 1u * packn, packn, opt.workspace_allocator);
    {
        const int gap = (w * stride_h - outw * stride_w) * packn;

#pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < inch; p++)
        {
            const Mat img = bottom_blob.channel(p);
            int8_t* ptr = bottom_im2col.channel(p);

            for (int u = 0; u < kernel_h; u++)
            {
                for (int v = 0; v < kernel_w; v++)
                {
                    const int8_t* sptr = img.row<const int8_t>(dilation_h * u) + dilation_w * v * packn;

                    for (int i = 0; i < outh; i++)
                    {
                        int j = 0;
                        for (; j < outw; j++)
                        {
                            vint8m1_t _val = vle8_v_i8m1(sptr, vl);
                            vse8_v_i8m1(ptr, _val, vl);

                            sptr += stride_w * packn;
                            ptr += packn;
                        }

                        sptr += gap;
                    }
                }
            }
        }
    }

    im2col_sgemm_packn_int8_rvv(bottom_im2col, top_blob, kernel, opt);
}
