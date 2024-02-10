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

#include "innerproduct_riscv.h"

#include "layer_type.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "riscv_activation.h"
#include "riscv_usability.h"

namespace ncnn {

InnerProduct_riscv::InnerProduct_riscv()
{
#if __riscv_vector
    support_packing = true;
#if __riscv_zfh
    support_fp16_storage = true;
#endif
#endif // __riscv_vector

    flatten = 0;
}

int InnerProduct_riscv::create_pipeline(const Option& opt)
{
    {
        flatten = ncnn::create_layer_cpu(ncnn::LayerType::Flatten);

        ncnn::ParamDict pd;

        flatten->load_param(pd);

        flatten->create_pipeline(opt);
    }

#if NCNN_INT8
    if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
    {
        // TODO implement int8
        return create_pipeline_int8(opt);
    }
#endif

#if __riscv_vector && __riscv_zfh
    if (opt.use_fp16_storage)
    {
        return create_pipeline_fp16s(opt);
    }
#endif

    int out_elempack = 1;

#if __riscv_vector
    const int packn = csrr_vlenb() / 4;

    const int num_input = weight_data_size / num_output;

    if (opt.use_packing_layout)
    {
        out_elempack = num_output % packn == 0 ? packn : 1;
    }

    if (out_elempack == packn)
    {
        // src = inch-outch
        // dst = packn-inch-outch/packn
        {
            Mat weight_data_r2 = weight_data.reshape(num_input, num_output);

            weight_data_tm.create(num_input, num_output / packn, (size_t)4u * packn, packn);

            for (int q = 0; q + (packn - 1) < num_output; q += packn)
            {
                float* g0 = weight_data_tm.row(q / packn);

                for (int p = 0; p < num_input; p++)
                {
                    for (int j = 0; j < packn; j++)
                    {
                        *g0++ = weight_data_r2.row(q + j)[p];
                    }
                }
            }
        }
    }
#endif // __riscv_vector

    if (out_elempack == 1)
    {
        weight_data_tm = weight_data;
    }

    weight_data.release();

    return 0;
}

int InnerProduct_riscv::destroy_pipeline(const Option& opt)
{
    if (flatten)
    {
        flatten->destroy_pipeline(opt);
        delete flatten;
        flatten = 0;
    }

    return 0;
}

int InnerProduct_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
#if NCNN_INT8
    if (opt.use_int8_inference && int8_scale_term)
    {
        // Mat bottom_blob_unpacked = bottom_blob;
        // if (bottom_blob.elempack != 1)
        // {
        //     Option opt_pack1 = opt;
        //     opt_pack1.blob_allocator = opt.workspace_allocator;

        //     convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_pack1);
        // }

        // Mat bottom_blob_unpacked_fp32 = bottom_blob_unpacked;
        // if (bottom_blob_unpacked.elembits() == 16)
        // {
        //     Option opt_pack1 = opt;
        //     opt_pack1.blob_allocator = opt.workspace_allocator;

        //     cast_float16_to_float32(bottom_blob_unpacked, bottom_blob_unpacked_fp32, opt_pack1);
        // }

        // Option opt_unpacked = opt;
        // opt_unpacked.use_packing_layout = false;
        return forward_int8(bottom_blob, top_blob, opt);
    }
#endif

    int elembits = bottom_blob.elembits();

#if __riscv_vector && __riscv_zfh
    if (opt.use_fp16_storage && elembits == 16)
    {
        if (opt.use_fp16_arithmetic)
            return forward_fp16sa(bottom_blob, top_blob, opt);
        else
            return forward_fp16s(bottom_blob, top_blob, opt);
    }
#endif

#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
#endif

    const int num_input = weight_data_size / num_output;

    if (bottom_blob.dims == 2 && bottom_blob.w == num_input)
    {
        // gemm
        int h = bottom_blob.h;
        size_t elemsize = bottom_blob.elemsize;
        int elempack = bottom_blob.elempack;

        top_blob.create(num_output, h, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        int num_output_elempack = 1;
#if __riscv_vector
        if (opt.use_packing_layout)
        {
            num_output_elempack = num_output % packn == 0 ? packn : 1;
        }
#endif

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int j = 0; j < h; j++)
        {
#if __riscv_vector
            if (elempack == packn && num_output_elempack == packn)
            {
                const size_t vl = vsetvl_e32m1(packn);

                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output / num_output_elempack; p++)
                {
                    for (int l = 0; l < packn; l++)
                    {
                        const float* kptr = weight_data_tm.row(p) + l;
                        const float* m = bottom_blob.row(j);

                        vfloat32m1_t _sum = vfmv_v_f_f32m1(0.f, vl);

                        if (bias_term)
                        {
                            _sum = vfmv_v_f_f32m1(bias_data[p * packn + l], vl);
                        }

                        int n = num_input;
                        while (n > 0)
                        {
                            vfloat32m1_t _val = vle32_v_f32m1(m, vl);
                            _sum = vfmacc_vf_f32m1(_sum, *kptr, _val, vl);

                            m += packn;
                            kptr += packn;
                            n -= 1;
                        }

                        _sum = activation_ps(_sum, activation_type, activation_params, vl);

                        vse32_v_f32m1(outptr, _sum, vl);
                        outptr += packn;
                    }
                }
            }

            if (elempack == 1 && num_output_elempack == packn)
            {
                const size_t vl = vsetvl_e32m1(packn);

                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output / num_output_elempack; p++)
                {
                    const float* kptr = weight_data_tm.row(p);
                    const float* m = bottom_blob.row(j);

                    vfloat32m1_t _sum = vfmv_v_f_f32m1(0.f, vl);

                    if (bias_term)
                    {
                        _sum = vle32_v_f32m1((const float*)bias_data + p * packn, vl);
                    }

                    int n = num_input;
                    while (n > 0)
                    {
                        vfloat32m1_t _w = vle32_v_f32m1(kptr, vl);
                        _sum = vfmacc_vf_f32m1(_sum, *m, _w, vl);

                        m += 1;
                        kptr += packn;
                        n -= 1;
                    }

                    _sum = activation_ps(_sum, activation_type, activation_params, vl);

                    vse32_v_f32m1(outptr, _sum, vl);
                    outptr += packn;
                }
            }

            if (elempack == packn && num_output_elempack == 1)
            {
                const size_t vl = vsetvl_e32m1(packn);

                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output; p++)
                {
                    const float* kptr = (const float*)weight_data_tm + num_input * p;
                    const float* m = bottom_blob.row(j);

                    vfloat32m1_t _sum = vfmv_v_f_f32m1(0.f, vl);

                    if (bias_term)
                    {
                        _sum = vfmv_v_f_f32m1(bias_data[p], vl);
                    }

                    int n = num_input;
                    while (n > 0)
                    {
                        vfloat32m1_t _val = vle32_v_f32m1(m, vl);
                        _sum = vfmacc_vf_f32m1(_sum, *kptr, _val, vl);

                        m += packn;
                        kptr += 1;
                        n -= 1;
                    }

                    _sum = activation_ps(_sum, activation_type, activation_params, vl);

                    vse32_v_f32m1(outptr, _sum, vl);
                    outptr += packn;
                }
            }
#endif // __riscv_vector

            if (elempack == 1 && num_output_elempack == 1)
            {
                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output; p++)
                {
                    const float* kptr = (const float*)weight_data_tm + num_input * p;
                    const float* m = bottom_blob.row(j);

                    float sum = 0.f;

                    if (bias_term)
                    {
                        sum = bias_data[p];
                    }

                    for (int i = 0; i < num_input; i++)
                    {
                        sum += m[i] * kptr[i];
                    }

                    sum = activation_ss(sum, activation_type, activation_params);

                    outptr[0] = sum;
                    outptr += 1;
                }
            }
        }

        return 0;
    }

    // flatten
    Mat bottom_blob_flattened = bottom_blob;
    if (bottom_blob.dims != 1)
    {
        Option opt_flatten = opt;
        opt_flatten.blob_allocator = opt.workspace_allocator;

        flatten->forward(bottom_blob, bottom_blob_flattened, opt_flatten);
    }

    size_t elemsize = bottom_blob_flattened.elemsize;
    int elempack = bottom_blob_flattened.elempack;

    int out_elempack = 1;
#if __riscv_vector
    if (opt.use_packing_layout)
    {
        out_elempack = num_output % packn == 0 ? packn : 1;
    }
#endif
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

#if __riscv_vector
    if (out_elempack == packn)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            const size_t vl = vsetvl_e32m1(packn);
            vfloat32m1_t _sum = vfmv_v_f_f32m1(0.f, vl);

            if (bias_term)
            {
                _sum = vle32_v_f32m1((const float*)bias_data + p * packn, vl);
            }

            const float* kptr = weight_data_tm.row(p);

            const float* sptr = bottom_blob_flattened;

            int n = num_input;
            while (n > 0)
            {
                vfloat32m1_t _w = vle32_v_f32m1(kptr, vl);
                _sum = vfmacc_vf_f32m1(_sum, *sptr, _w, vl);

                sptr += 1;
                kptr += packn;
                n -= 1;
            }

            _sum = activation_ps(_sum, activation_type, activation_params, vl);

            float* outptr = top_blob;
            vse32_v_f32m1(outptr + p * packn, _sum, vl);
        }
    }
#endif // __riscv_vector

    if (out_elempack == 1)
    {
#if __riscv_vector
        int nn_num_output = num_output / packn;
        int remain_num_output_start = nn_num_output * packn;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp = 0; pp < nn_num_output; pp++)
        {
            int p = pp * packn;

            const size_t vl = vsetvl_e32m1(packn);
            vfloat32m1_t _sum = vfmv_v_f_f32m1(0.f, vl);

            if (bias_term)
            {
                _sum = vle32_v_f32m1((const float*)bias_data + p, vl);
            }

            const float* w = (const float*)weight_data_tm + num_input * p;

            const float* m = bottom_blob_flattened;

            int n = num_input;
            while (n > 0)
            {
                vfloat32m1_t _w = vlse32_v_f32m1(w, num_input * sizeof(float), vl);

                _sum = vfmacc_vf_f32m1(_sum, *m, _w, vl);

                m += 1;
                w += 1;
                n -= 1;
            }

            _sum = activation_ps(_sum, activation_type, activation_params, vl);

            vse32_v_f32m1((float*)top_blob + p, _sum, vl);
        }
#else // __riscv_vector
        int nn_num_output = num_output / 4;
        int remain_num_output_start = nn_num_output * 4;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp = 0; pp < nn_num_output; pp++)
        {
            int p = pp * 4;

            float sum0 = 0.f;
            float sum1 = 0.f;
            float sum2 = 0.f;
            float sum3 = 0.f;

            if (bias_term)
            {
                sum0 = bias_data[p];
                sum1 = bias_data[p + 1];
                sum2 = bias_data[p + 2];
                sum3 = bias_data[p + 3];
            }

            const float* w0 = (const float*)weight_data_tm + num_input * p;
            const float* w1 = (const float*)weight_data_tm + num_input * (p + 1);
            const float* w2 = (const float*)weight_data_tm + num_input * (p + 2);
            const float* w3 = (const float*)weight_data_tm + num_input * (p + 3);

            const float* m = bottom_blob_flattened;

            for (int i = 0; i < num_input; i++)
            {
                sum0 += *m * *w0;
                sum1 += *m * *w1;
                sum2 += *m * *w2;
                sum3 += *m * *w3;

                m++;
                w0++;
                w1++;
                w2++;
                w3++;
            }

            sum0 = activation_ss(sum0, activation_type, activation_params);
            sum1 = activation_ss(sum1, activation_type, activation_params);
            sum2 = activation_ss(sum2, activation_type, activation_params);
            sum3 = activation_ss(sum3, activation_type, activation_params);

            top_blob[p] = sum0;
            top_blob[p + 1] = sum1;
            top_blob[p + 2] = sum2;
            top_blob[p + 3] = sum3;
        }
#endif // __riscv_vector

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = remain_num_output_start; p < num_output; p++)
        {
            float sum = 0.f;

            if (bias_term)
                sum = bias_data[p];

            const float* w = (const float*)weight_data_tm + num_input * p;

            const float* m = bottom_blob_flattened;

            for (int i = 0; i < num_input; i++)
            {
                sum += *m * *w;

                m++;
                w++;
            }

            sum = activation_ss(sum, activation_type, activation_params);

            top_blob[p] = sum;
        }
    }

    return 0;
}

#if __riscv_vector && __riscv_zfh
int InnerProduct_riscv::create_pipeline_fp16s(const Option& opt)
{
    const int packn = csrr_vlenb() / 2;

    const int num_input = weight_data_size / num_output;

    int out_elempack = 1;

    if (opt.use_packing_layout)
    {
        out_elempack = num_output % packn == 0 ? packn : 1;
    }

    // src = inch-outch
    // dst = pb-inch-outch/pb
    {
        Mat weight_data_r2 = weight_data.reshape(num_input, num_output);

        weight_data_tm.create(num_input, num_output / out_elempack, (size_t)2u * out_elempack, out_elempack);

        for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
        {
            __fp16* g0 = weight_data_tm.row<__fp16>(q / out_elempack);

            for (int p = 0; p < num_input; p++)
            {
                for (int j = 0; j < out_elempack; j++)
                {
                    *g0++ = (__fp16)(weight_data_r2.row(q + j)[p]);
                }
            }
        }
    }

    ncnn::cast_float32_to_float16(bias_data, bias_data_fp16, opt);

    weight_data.release();

    return 0;
}

int InnerProduct_riscv::forward_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int packn = csrr_vlenb() / 2;

    const int num_input = weight_data_size / num_output;

    if (bottom_blob.dims == 2 && bottom_blob.w == num_input)
    {
        // gemm
        int h = bottom_blob.h;
        size_t elemsize = bottom_blob.elemsize;
        int elempack = bottom_blob.elempack;

        top_blob.create(num_output, h, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        int num_output_elempack = opt.use_packing_layout && num_output % packn == 0 ? packn : 1;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int j = 0; j < h; j++)
        {
            if (elempack == packn && num_output_elempack == packn)
            {
                const size_t vl = vsetvl_e16m1(packn);

                __fp16* outptr = top_blob.row<__fp16>(j);

                for (int p = 0; p < num_output / num_output_elempack; p++)
                {
                    for (int l = 0; l < packn; l++)
                    {
                        const __fp16* kptr = (const __fp16*)weight_data_tm + num_input * p * packn + l;
                        const __fp16* m = bottom_blob.row<const __fp16>(j);

                        vfloat32m2_t _sum = vfmv_v_f_f32m2(0.f, vl);

                        if (bias_term)
                        {
                            _sum = vfmv_v_f_f32m2(bias_data[p * packn + l], vl);
                        }

                        int n = num_input;
                        while (n > 0)
                        {
                            vfloat32m2_t _val = vfwcvt_f_f_v_f32m2(vle16_v_f16m1(m, vl), vl);

                            _sum = vfmacc_vf_f32m2(_sum, *kptr, _val, vl);

                            m += packn;
                            kptr += packn;
                            n -= 1;
                        }

                        _sum = activation_ps(_sum, activation_type, activation_params, vl);

                        vse16_v_f16m1(outptr, vfncvt_f_f_w_f16m1(_sum, vl), vl);
                        outptr += packn;
                    }
                }
            }

            if (elempack == 1 && num_output_elempack == packn)
            {
                const size_t vl = vsetvl_e16m1(packn);

                __fp16* outptr = top_blob.row<__fp16>(j);

                for (int p = 0; p < num_output / num_output_elempack; p++)
                {
                    const __fp16* kptr = (const __fp16*)weight_data_tm + num_input * p * packn;
                    const __fp16* m = bottom_blob.row<const __fp16>(j);

                    vfloat32m2_t _sum = vfmv_v_f_f32m2(0.f, vl);

                    if (bias_term)
                    {
                        _sum = vle32_v_f32m2((const float*)bias_data + p * packn, vl);
                    }

                    int n = num_input;
                    while (n > 0)
                    {
                        vfloat32m2_t _w = vfwcvt_f_f_v_f32m2(vle16_v_f16m1(kptr, vl), vl);

                        _sum = vfmacc_vf_f32m2(_sum, *m, _w, vl);

                        m += 1;
                        kptr += packn;
                        n -= 1;
                    }

                    _sum = activation_ps(_sum, activation_type, activation_params, vl);

                    vse16_v_f16m1(outptr, vfncvt_f_f_w_f16m1(_sum, vl), vl);
                    outptr += packn;
                }
            }

            if (elempack == packn && num_output_elempack == 1)
            {
                const size_t vl = vsetvl_e16m1(packn);

                __fp16* outptr = top_blob.row<__fp16>(j);

                for (int p = 0; p < num_output; p++)
                {
                    const __fp16* kptr = (const __fp16*)weight_data_tm + num_input * p;
                    const __fp16* m = bottom_blob.row<const __fp16>(j);

                    vfloat32m2_t _sum = vfmv_v_f_f32m2(0.f, vl);

                    if (bias_term)
                    {
                        _sum = vfmv_v_f_f32m2(bias_data[p], vl);
                    }

                    int n = num_input;
                    while (n > 0)
                    {
                        vfloat32m2_t _val = vfwcvt_f_f_v_f32m2(vle16_v_f16m1(m, vl), vl);

                        _sum = vfmacc_vf_f32m2(_sum, *kptr, _val, vl);

                        m += packn;
                        kptr += 1;
                        n -= 1;
                    }

                    _sum = activation_ps(_sum, activation_type, activation_params, vl);

                    vse16_v_f16m1(outptr, vfncvt_f_f_w_f16m1(_sum, vl), vl);
                    outptr += packn;
                }
            }

            if (elempack == 1 && num_output_elempack == 1)
            {
                __fp16* outptr = top_blob.row<__fp16>(j);

                for (int p = 0; p < num_output; p++)
                {
                    const __fp16* kptr = (const __fp16*)weight_data_tm + num_input * p;
                    const __fp16* m = bottom_blob.row<const __fp16>(j);

                    float sum = 0.f;

                    if (bias_term)
                    {
                        sum = bias_data[p];
                    }

                    for (int i = 0; i < num_input; i++)
                    {
                        sum += (float)m[i] * (float)kptr[i];
                    }

                    sum = activation_ss(sum, activation_type, activation_params);

                    outptr[0] = (__fp16)sum;
                    outptr += 1;
                }
            }
        }

        return 0;
    }

    // flatten
    Mat bottom_blob_flattened = bottom_blob;
    if (bottom_blob.dims != 1)
    {
        Option opt_flatten = opt;
        opt_flatten.blob_allocator = opt.workspace_allocator;

        flatten->forward(bottom_blob, bottom_blob_flattened, opt_flatten);
    }

    size_t elemsize = bottom_blob_flattened.elemsize;
    int elempack = bottom_blob_flattened.elempack;

    int out_elempack = opt.use_packing_layout && num_output % packn == 0 ? packn : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (out_elempack == packn)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            const size_t vl = vsetvl_e16m1(packn);
            vfloat32m2_t _sum = vfmv_v_f_f32m2(0.f, vl);

            if (bias_term)
            {
                _sum = vle32_v_f32m2((const float*)bias_data + p * packn, vl);
            }

            const __fp16* kptr = weight_data_tm.row<const __fp16>(p);

            const __fp16* sptr = bottom_blob_flattened;

            int n = num_input;
            while (n > 0)
            {
                vfloat32m2_t _w = vfwcvt_f_f_v_f32m2(vle16_v_f16m1(kptr, vl), vl);

                _sum = vfmacc_vf_f32m2(_sum, (float)(*sptr), _w, vl);

                sptr += 1;
                kptr += packn;
                n -= 1;
            }

            _sum = activation_ps(_sum, activation_type, activation_params, vl);

            __fp16* outptr = (__fp16*)top_blob;
            vse16_v_f16m1(outptr + p * packn, vfncvt_f_f_w_f16m1(_sum, vl), vl);
        }
    }

    if (out_elempack == 1)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output; p++)
        {
            float sum = 0.f;

            if (bias_term)
                sum = bias_data[p];

            const __fp16* kptr = weight_data_tm.row<__fp16>(p);

            const __fp16* sptr = bottom_blob_flattened;

            int i = 0;
            for (; i < num_input; i++)
            {
                float v = (float)(*sptr);
                float k = (float)(*kptr);

                sum += v * k;

                sptr++;
                kptr++;
            }

            sum = activation_ss(sum, activation_type, activation_params);

            __fp16* outptr = (__fp16*)top_blob;
            outptr[p] = (__fp16)sum;
        }
    }

    return 0;
}

int InnerProduct_riscv::forward_fp16sa(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int packn = csrr_vlenb() / 2;

    const int num_input = weight_data_size / num_output;

    if (bottom_blob.dims == 2 && bottom_blob.w == num_input)
    {
        // gemm
        int h = bottom_blob.h;
        size_t elemsize = bottom_blob.elemsize;
        int elempack = bottom_blob.elempack;

        top_blob.create(num_output, h, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        int num_output_elempack = opt.use_packing_layout && num_output % packn == 0 ? packn : 1;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int j = 0; j < h; j++)
        {
            if (elempack == packn && num_output_elempack == packn)
            {
                const size_t vl = vsetvl_e16m1(packn);

                __fp16* outptr = top_blob.row<__fp16>(j);

                for (int p = 0; p < num_output / num_output_elempack; p++)
                {
                    for (int l = 0; l < packn; l++)
                    {
                        const __fp16* kptr = (const __fp16*)weight_data_tm + num_input * p * packn + l;
                        const __fp16* m = bottom_blob.row<const __fp16>(j);

                        vfloat16m1_t _sum = vfmv_v_f_f16m1((__fp16)0.f, vl);

                        if (bias_term)
                        {
                            _sum = vfmv_v_f_f16m1(((const __fp16*)bias_data_fp16)[p * packn + l], vl);
                        }

                        int n = num_input;
                        while (n > 0)
                        {
                            vfloat16m1_t _val = vle16_v_f16m1(m, vl);

                            _sum = vfmacc_vf_f16m1(_sum, *kptr, _val, vl);

                            m += packn;
                            kptr += packn;
                            n -= 1;
                        }

                        _sum = activation_ps(_sum, activation_type, activation_params, vl);

                        vse16_v_f16m1(outptr, _sum, vl);
                        outptr += packn;
                    }
                }
            }

            if (elempack == 1 && num_output_elempack == packn)
            {
                const size_t vl = vsetvl_e16m1(packn);

                __fp16* outptr = top_blob.row<__fp16>(j);

                for (int p = 0; p < num_output / num_output_elempack; p++)
                {
                    const __fp16* kptr = (const __fp16*)weight_data_tm + num_input * p * packn;
                    const __fp16* m = bottom_blob.row<const __fp16>(j);

                    vfloat16m1_t _sum = vfmv_v_f_f16m1(0.f, vl);

                    if (bias_term)
                    {
                        _sum = vle16_v_f16m1((const __fp16*)bias_data_fp16 + p * packn, vl);
                    }

                    int n = num_input;
                    while (n > 0)
                    {
                        vfloat16m1_t _w = vle16_v_f16m1(kptr, vl);

                        _sum = vfmacc_vf_f16m1(_sum, *m, _w, vl);

                        m += 1;
                        kptr += packn;
                        n -= 1;
                    }

                    _sum = activation_ps(_sum, activation_type, activation_params, vl);

                    vse16_v_f16m1(outptr, _sum, vl);
                    outptr += packn;
                }
            }

            if (elempack == packn && num_output_elempack == 1)
            {
                const size_t vl = vsetvl_e16m1(packn);

                __fp16* outptr = top_blob.row<__fp16>(j);

                for (int p = 0; p < num_output; p++)
                {
                    const __fp16* kptr = (const __fp16*)weight_data_tm + num_input * p;
                    const __fp16* m = bottom_blob.row<const __fp16>(j);

                    vfloat16m1_t _sum = vfmv_v_f_f16m1(0.f, vl);

                    if (bias_term)
                    {
                        _sum = vfmv_v_f_f16m1(((const __fp16*)bias_data_fp16)[p], vl);
                    }

                    int n = num_input;
                    while (n > 0)
                    {
                        vfloat16m1_t _val = vle16_v_f16m1(m, vl);

                        _sum = vfmacc_vf_f16m1(_sum, *kptr, _val, vl);

                        m += packn;
                        kptr += 1;
                        n -= 1;
                    }

                    _sum = activation_ps(_sum, activation_type, activation_params, vl);

                    vse16_v_f16m1(outptr, _sum, vl);
                    outptr += packn;
                }
            }

            if (elempack == 1 && num_output_elempack == 1)
            {
                __fp16* outptr = top_blob.row<__fp16>(j);

                for (int p = 0; p < num_output; p++)
                {
                    const __fp16* kptr = (const __fp16*)weight_data_tm + num_input * p;
                    const __fp16* m = bottom_blob.row<const __fp16>(j);

                    float sum = 0.f;

                    if (bias_term)
                    {
                        sum = bias_data[p];
                    }

                    for (int i = 0; i < num_input; i++)
                    {
                        sum += (float)(m[i] * kptr[i]);
                    }

                    sum = activation_ss(sum, activation_type, activation_params);

                    outptr[0] = (__fp16)sum;
                    outptr += 1;
                }
            }
        }

        return 0;
    }

    // flatten
    Mat bottom_blob_flattened = bottom_blob;
    if (bottom_blob.dims != 1)
    {
        Option opt_flatten = opt;
        opt_flatten.blob_allocator = opt.workspace_allocator;

        flatten->forward(bottom_blob, bottom_blob_flattened, opt_flatten);
    }

    size_t elemsize = bottom_blob_flattened.elemsize;
    int elempack = bottom_blob_flattened.elempack;

    int out_elempack = opt.use_packing_layout && num_output % packn == 0 ? packn : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (out_elempack == packn)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            const size_t vl = vsetvl_e16m1(packn);
            vfloat16m1_t _sum = vfmv_v_f_f16m1(0.f, vl);

            if (bias_term)
            {
                _sum = vle16_v_f16m1((const __fp16*)bias_data_fp16 + p * packn, vl);
            }

            const __fp16* kptr = weight_data_tm.row<const __fp16>(p);

            const __fp16* sptr = bottom_blob_flattened;

            int n = num_input;
            while (n > 0)
            {
                vfloat16m1_t _w = vle16_v_f16m1(kptr, vl);

                _sum = vfmacc_vf_f16m1(_sum, *sptr, _w, vl);

                sptr += 1;
                kptr += packn;
                n -= 1;
            }

            _sum = activation_ps(_sum, activation_type, activation_params, vl);

            __fp16* outptr = (__fp16*)top_blob;
            vse16_v_f16m1(outptr + p * packn, _sum, vl);
        }
    }

    if (out_elempack == 1)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output; p++)
        {
            float sum = 0.f;

            if (bias_term)
                sum = bias_data[p];

            const __fp16* kptr = weight_data_tm.row<__fp16>(p);

            const __fp16* sptr = bottom_blob_flattened;

            int i = 0;
            for (; i < num_input; i++)
            {
                __fp16 v = *sptr;
                __fp16 k = *kptr;

                sum += (float)(v * k);

                sptr++;
                kptr++;
            }

            sum = activation_ss(sum, activation_type, activation_params);

            __fp16* outptr = (__fp16*)top_blob;
            outptr[p] = (__fp16)sum;
        }
    }

    return 0;
}
#endif // __riscv_vector && __riscv_zfh

#if NCNN_INT8
int InnerProduct_riscv::create_pipeline_int8(const Option& opt)
{
    const int num_input = weight_data_size / num_output;

    int out_elempack = 1;
#if __riscv_vector
    if (opt.use_packing_layout)
    {
        out_elempack = num_output % 8 == 0 ? 8 : 1;
    }
#endif

    // src = inch-outch
    // dst = pb-inch-outch/pb
    {
        Mat weight_data_r2 = weight_data.reshape(num_input, num_output);

        weight_data_tm.create(num_input, num_output / out_elempack, (size_t)out_elempack, out_elempack);

        for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
        {
            signed char* g0 = weight_data_tm.row<signed char>(q / out_elempack);

            for (int p = 0; p < num_input; p++)
            {
                for (int j = 0; j < out_elempack; j++)
                {
                    *g0++ = weight_data_r2.row<signed char>(q + j)[p];
                }
            }
        }
    }

    scale_in_data.create(num_output);
    for (int p = 0; p < num_output; p++)
    {
        // dequantize
        float scale_in;
        if (weight_data_int8_scales[p] == 0)
            scale_in = 0;
        else
            scale_in = 1.f / (bottom_blob_int8_scales[0] * weight_data_int8_scales[p]);

        scale_in_data[p] = scale_in;
    }

    weight_data.release();

    return 0;
}

int InnerProduct_riscv::forward_int8(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    fprintf(stderr, "InnerProduct_riscv::forward_int8\n");
    int vl;
    const int num_input = weight_data_size / num_output;

    int elembits = bottom_blob.elembits();

    Mat bottom_blob_int8 = bottom_blob;
    fprintf(stderr, "elembits: %d\n", elembits);
    fprintf(stderr, "bottom_blob.dims: %d\n", bottom_blob.dims);
    // print w, h, d, c
    fprintf(stderr, "bottom_blob.w: %d\n", bottom_blob.w);
    fprintf(stderr, "bottom_blob.h: %d\n", bottom_blob.h);
    fprintf(stderr, "bottom_blob.c: %d\n", bottom_blob.c);
    fprintf(stderr, "bottom_blob.elemsize: %d\n", bottom_blob.elemsize);
    fprintf(stderr, "bottom_blob.elempack: %d\n", bottom_blob.elempack);


    if (elembits != 8)
    {
        Option opt_q = opt;
        opt_q.blob_allocator = opt.workspace_allocator;
        quantize_to_int8(bottom_blob, bottom_blob_int8, bottom_blob_int8_scales, opt_q);
    }

    fprintf(stderr, "bottom_blob_int8.dims: %d\n", bottom_blob_int8.dims);
    fprintf(stderr, "bottom_blob_int8.w: %d\n", bottom_blob_int8.w);
    fprintf(stderr, "bottom_blob_int8.h: %d\n", bottom_blob_int8.h);
    fprintf(stderr, "bottom_blob_int8.c: %d\n", bottom_blob_int8.c);
    fprintf(stderr, "bottom_blob_int8.elemsize: %d\n", bottom_blob_int8.elemsize);
    fprintf(stderr, "bottom_blob_int8.elempack: %d\n", bottom_blob_int8.elempack);


    if (bottom_blob_int8.dims == 2 && bottom_blob_int8.w == num_input)
    {
        fprintf(stderr, "in 1163\n");
        // gemm
        Mat bottom_blob_int8_unpacked;
        Option opt_unpack = opt;
        opt_unpack.blob_allocator = opt.workspace_allocator;
        convert_packing(bottom_blob_int8, bottom_blob_int8_unpacked, 1, opt_unpack);

        int h = bottom_blob_int8_unpacked.h;

        int out_elempack = 1;
#if __riscv_vector
        if (opt.use_packing_layout)
        {
            out_elempack = h % 4 == 0 ? 4 : 1;
        }
#endif

        int outh = h / out_elempack;

        top_blob.create(num_output, outh, (size_t)(4u * out_elempack), out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        int num_output_elempack = 1;
#if __riscv_vector
        if (opt.use_packing_layout)
        {
            num_output_elempack = num_output % 8 == 0 ? 8 : 1;
        }
#endif

#if __riscv_vector
        if (num_output_elempack == 8 && out_elempack == 4)
        {
            fprintf(stderr, "in 1195\n");
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int j = 0; j < outh; j++)
            {
                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output / num_output_elempack; p++)
                {
                    const signed char* kptr = weight_data_tm.row<const signed char>(p);
                    const signed char* m0 = bottom_blob_int8_unpacked.row<const signed char>(j * 4);
                    const signed char* m1 = bottom_blob_int8_unpacked.row<const signed char>(j * 4 + 1);
                    const signed char* m2 = bottom_blob_int8_unpacked.row<const signed char>(j * 4 + 2);
                    const signed char* m3 = bottom_blob_int8_unpacked.row<const signed char>(j * 4 + 3);

                    vl = 8;
                    vint32m2_t _sum0 = vmv_v_x_i32m2(0, vl);
                    // vint32m2_t _sum01 = vmv_v_x_i32m2(0, vl);
                    vint32m2_t _sum1 = vmv_v_x_i32m2(0, vl);
                    // vint32m2_t _sum11 = vmv_v_x_i32m2(0, vl);
                    vint32m2_t _sum2 = vmv_v_x_i32m2(0, vl);
                    // vint32m2_t _sum21 = vmv_v_x_i32m2(0, vl);
                    vint32m2_t _sum3 = vmv_v_x_i32m2(0, vl);
                    // vint32m2_t _sum31 = vmv_v_x_i32m2(0, vl);

                    // int32x4_t _sum00 = vdupq_n_s32(0);
                    // int32x4_t _sum01 = vdupq_n_s32(0);
                    // int32x4_t _sum10 = vdupq_n_s32(0);
                    // int32x4_t _sum11 = vdupq_n_s32(0);
                    // int32x4_t _sum20 = vdupq_n_s32(0);
                    // int32x4_t _sum21 = vdupq_n_s32(0);
                    // int32x4_t _sum30 = vdupq_n_s32(0);
                    // int32x4_t _sum31 = vdupq_n_s32(0);

                    int i = 0;
                    for (; i < num_input; i++)
                    {
                        vint8m1_t _val0 = vmv_v_x_i8m1(m0[0], vl);
                        vint8m1_t _val1 = vmv_v_x_i8m1(m1[0], vl);
                        vint8m1_t _val2 = vmv_v_x_i8m1(m2[0], vl);
                        vint8m1_t _val3 = vmv_v_x_i8m1(m3[0], vl);

                        vint8m1_t _w = vle8_v_i8m1(kptr, vl);

                        // vint16m2_t _s0 = vmv_v_x_i16m2(0, vl);

                        // int8x8_t _val0 = vld1_dup_s8(m0);
                        // int8x8_t _val1 = vld1_dup_s8(m1);
                        // int8x8_t _val2 = vld1_dup_s8(m2);
                        // int8x8_t _val3 = vld1_dup_s8(m3);

                        // int8x8_t _w = vld1_s8(kptr);
                        vint16m1_t _s0 = vget_v_i16m2_i16m1(vwmul_vv_i16m2(_val0, _w, vl), 0);
                        vint16m1_t _s1 = vget_v_i16m2_i16m1(vwmul_vv_i16m2(_val1, _w, vl), 0);
                        vint16m1_t _s2 = vget_v_i16m2_i16m1(vwmul_vv_i16m2(_val2, _w, vl), 0);
                        vint16m1_t _s3 = vget_v_i16m2_i16m1(vwmul_vv_i16m2(_val3, _w, vl), 0);

                        _sum0 = vwadd_wv_i32m2(_sum0, _s0, vl);
                        _sum1 = vwadd_wv_i32m2(_sum1, _s1, vl);
                        _sum2 = vwadd_wv_i32m2(_sum2, _s2, vl);
                        _sum3 = vwadd_wv_i32m2(_sum3, _s3, vl);

                        // int16x8_t _s0 = vmull_s8(_val0, _w);
                        // int16x8_t _s1 = vmull_s8(_val1, _w);
                        // int16x8_t _s2 = vmull_s8(_val2, _w);
                        // int16x8_t _s3 = vmull_s8(_val3, _w);
                        // _sum00 = vaddw_s16(_sum00, vget_low_s16(_s0));
                        // _sum01 = vaddw_s16(_sum01, vget_high_s16(_s0));
                        // _sum10 = vaddw_s16(_sum10, vget_low_s16(_s1));
                        // _sum11 = vaddw_s16(_sum11, vget_high_s16(_s1));
                        // _sum20 = vaddw_s16(_sum20, vget_low_s16(_s2));
                        // _sum21 = vaddw_s16(_sum21, vget_high_s16(_s2));
                        // _sum30 = vaddw_s16(_sum30, vget_low_s16(_s3));
                        // _sum31 = vaddw_s16(_sum31, vget_high_s16(_s3));

                        m0++;
                        m1++;
                        m2++;
                        m3++;
                        kptr += 8;
                    }

                    // dequantize and relu
                    vfloat32m2_t _scale_in = vle32_v_f32m2((const float*)scale_in_data + p * 8, vl);

                    vfloat32m2_t _sumfp32_0 = vfcvt_f_x_v_f32m2(_sum0, vl);
                    vfloat32m2_t _sumfp32_1 = vfcvt_f_x_v_f32m2(_sum1, vl);
                    vfloat32m2_t _sumfp32_2 = vfcvt_f_x_v_f32m2(_sum2, vl);
                    vfloat32m2_t _sumfp32_3 = vfcvt_f_x_v_f32m2(_sum3, vl);

                    // float32x4_t _scale_in0 = vld1q_f32((const float*)scale_in_data + p * 8);
                    // float32x4_t _scale_in1 = vld1q_f32((const float*)scale_in_data + p * 8 + 4);

                    // float32x4_t _sumfp32_00 = vcvtq_f32_s32(_sum00);
                    // float32x4_t _sumfp32_01 = vcvtq_f32_s32(_sum01);
                    // float32x4_t _sumfp32_10 = vcvtq_f32_s32(_sum10);
                    // float32x4_t _sumfp32_11 = vcvtq_f32_s32(_sum11);
                    // float32x4_t _sumfp32_20 = vcvtq_f32_s32(_sum20);
                    // float32x4_t _sumfp32_21 = vcvtq_f32_s32(_sum21);
                    // float32x4_t _sumfp32_30 = vcvtq_f32_s32(_sum30);
                    // float32x4_t _sumfp32_31 = vcvtq_f32_s32(_sum31);
                    if (bias_term)
                    {
                        vfloat32m2_t _bias = vle32_v_f32m2((const float*)bias_data + p * 8, vl);
                        _sumfp32_0 = vfmacc_vv_f32m2(_bias, _sumfp32_0, _scale_in, vl);
                        _sumfp32_1 = vfmacc_vv_f32m2(_bias, _sumfp32_1, _scale_in, vl);
                        _sumfp32_2 = vfmacc_vv_f32m2(_bias, _sumfp32_2, _scale_in, vl);
                        _sumfp32_3 = vfmacc_vv_f32m2(_bias, _sumfp32_3, _scale_in, vl);

                        // float32x4_t _bias0 = vld1q_f32((const float*)bias_data + p * 8);
                        // float32x4_t _bias1 = vld1q_f32((const float*)bias_data + p * 8 + 4);
                        // _sumfp32_00 = vmlaq_f32(_bias0, _sumfp32_00, _scale_in0);
                        // _sumfp32_01 = vmlaq_f32(_bias1, _sumfp32_01, _scale_in1);
                        // _sumfp32_10 = vmlaq_f32(_bias0, _sumfp32_10, _scale_in0);
                        // _sumfp32_11 = vmlaq_f32(_bias1, _sumfp32_11, _scale_in1);
                        // _sumfp32_20 = vmlaq_f32(_bias0, _sumfp32_20, _scale_in0);
                        // _sumfp32_21 = vmlaq_f32(_bias1, _sumfp32_21, _scale_in1);
                        // _sumfp32_30 = vmlaq_f32(_bias0, _sumfp32_30, _scale_in0);
                        // _sumfp32_31 = vmlaq_f32(_bias1, _sumfp32_31, _scale_in1);
                    }
                    else
                    {
                        _sumfp32_0 = vfmul_vv_f32m2(_sumfp32_0, _scale_in, vl);
                        _sumfp32_1 = vfmul_vv_f32m2(_sumfp32_1, _scale_in, vl);
                        _sumfp32_2 = vfmul_vv_f32m2(_sumfp32_2, _scale_in, vl);
                        _sumfp32_3 = vfmul_vv_f32m2(_sumfp32_3, _scale_in, vl);

                        // _sumfp32_00 = vmulq_f32(_sumfp32_00, _scale_in0);
                        // _sumfp32_01 = vmulq_f32(_sumfp32_01, _scale_in1);
                        // _sumfp32_10 = vmulq_f32(_sumfp32_10, _scale_in0);
                        // _sumfp32_11 = vmulq_f32(_sumfp32_11, _scale_in1);
                        // _sumfp32_20 = vmulq_f32(_sumfp32_20, _scale_in0);
                        // _sumfp32_21 = vmulq_f32(_sumfp32_21, _scale_in1);
                        // _sumfp32_30 = vmulq_f32(_sumfp32_30, _scale_in0);
                        // _sumfp32_31 = vmulq_f32(_sumfp32_31, _scale_in1);
                    }

                    _sumfp32_0 = activation_ps(_sumfp32_0, activation_type, activation_params, vl);
                    // _sumfp32_01 = activation_ps(_sumfp32_01, activation_type, activation_params);
                    _sumfp32_1 = activation_ps(_sumfp32_1, activation_type, activation_params, vl);
                    // _sumfp32_11 = activation_ps(_sumfp32_11, activation_type, activation_params);
                    _sumfp32_2 = activation_ps(_sumfp32_2, activation_type, activation_params, vl);
                    // _sumfp32_21 = activation_ps(_sumfp32_21, activation_type, activation_params);
                    _sumfp32_3 = activation_ps(_sumfp32_3, activation_type, activation_params, vl);
                    // _sumfp32_31 = activation_ps(_sumfp32_31, activation_type, activation_params);

                    vsseg4e32_v_f32m2(outptr, _sumfp32_0, _sumfp32_1, _sumfp32_2, _sumfp32_3, vl);
                    // transpose 4x8
                    // float32x4x4_t _sumfp32_0;
                    // _sumfp32_0.val[0] = _sumfp32_00;
                    // _sumfp32_0.val[1] = _sumfp32_10;
                    // _sumfp32_0.val[2] = _sumfp32_20;
                    // _sumfp32_0.val[3] = _sumfp32_30;
                    // float32x4x4_t _sumfp32_1;
                    // _sumfp32_1.val[0] = _sumfp32_01;
                    // _sumfp32_1.val[1] = _sumfp32_11;
                    // _sumfp32_1.val[2] = _sumfp32_21;
                    // _sumfp32_1.val[3] = _sumfp32_31;

                    // vst4q_f32(outptr, _sumfp32_0);
                    // vst4q_f32(outptr + 16, _sumfp32_1);

                    outptr += 32;
                }
            }
        }

        if (num_output_elempack == 1 && out_elempack == 4)
        {
            fprintf(stderr, "in 1363\n");
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int j = 0; j < outh; j++)
            {
                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output; p++)
                {
                    const signed char* kptr = weight_data_tm.row<const signed char>(p);
                    const signed char* m0 = bottom_blob_int8_unpacked.row<const signed char>(j * 4);
                    const signed char* m1 = bottom_blob_int8_unpacked.row<const signed char>(j * 4 + 1);
                    const signed char* m2 = bottom_blob_int8_unpacked.row<const signed char>(j * 4 + 2);
                    const signed char* m3 = bottom_blob_int8_unpacked.row<const signed char>(j * 4 + 3);

                    int sum0 = 0;
                    int sum1 = 0;
                    int sum2 = 0;
                    int sum3 = 0;

                    int i = 0;

                    int n = num_input;

                    vl = vsetvlmax_e32m4();
                    vint32m4_t _sum0 = vmv_v_x_i32m4(0, vl);
                    vint32m4_t _sum1 = vmv_v_x_i32m4(0, vl);
                    vint32m4_t _sum2 = vmv_v_x_i32m4(0, vl);
                    vint32m4_t _sum3 = vmv_v_x_i32m4(0, vl);

                    while (n > 0)
                    {
                        vl = vsetvl_e32m4(n);
                        vint8m1_t _val0 = vle8_v_i8m1(m0, vl);
                        vint8m1_t _val1 = vle8_v_i8m1(m1, vl);
                        vint8m1_t _val2 = vle8_v_i8m1(m2, vl);
                        vint8m1_t _val3 = vle8_v_i8m1(m3, vl);

                        vint8m1_t _w = vle8_v_i8m1(kptr, vl);

                        vint16m2_t _s0 = vwmul_vv_i16m2(_val0, _w, vl);
                        vint16m2_t _s1 = vwmul_vv_i16m2(_val1, _w, vl);
                        vint16m2_t _s2 = vwmul_vv_i16m2(_val2, _w, vl);
                        vint16m2_t _s3 = vwmul_vv_i16m2(_val3, _w, vl);

                        _sum0 = vwadd_wv_i32m4(_sum0, _s0, vl);
                        _sum1 = vwadd_wv_i32m4(_sum1, _s1, vl);
                        _sum2 = vwadd_wv_i32m4(_sum2, _s2, vl);
                        _sum3 = vwadd_wv_i32m4(_sum3, _s3, vl);
                    }

                    vint32m1_t _sum0_scala = vmv_v_x_i32m1(0, vl);
                    vint32m1_t _sum1_scala = vmv_v_x_i32m1(0, vl);
                    vint32m1_t _sum2_scala = vmv_v_x_i32m1(0, vl);
                    vint32m1_t _sum3_scala = vmv_v_x_i32m1(0, vl);

                    vl = vsetvlmax_e32m4();
                    _sum0_scala = vredsum_vs_i32m4_i32m1(_sum0_scala, _sum0, _sum0_scala, vl);
                    _sum1_scala = vredsum_vs_i32m4_i32m1(_sum1_scala, _sum1, _sum1_scala, vl);
                    _sum2_scala = vredsum_vs_i32m4_i32m1(_sum2_scala, _sum2, _sum2_scala, vl);
                    _sum3_scala = vredsum_vs_i32m4_i32m1(_sum3_scala, _sum3, _sum3_scala, vl);
                    sum0 = vmv_x_s_i32m1_i32(_sum0_scala);
                    sum1 = vmv_x_s_i32m1_i32(_sum1_scala);
                    sum2 = vmv_x_s_i32m1_i32(_sum2_scala);
                    sum3 = vmv_x_s_i32m1_i32(_sum3_scala);

                    // int32x4_t _sum0 = vdupq_n_s32(0);
                    // int32x4_t _sum1 = vdupq_n_s32(0);
                    // int32x4_t _sum2 = vdupq_n_s32(0);
                    // int32x4_t _sum3 = vdupq_n_s32(0);
                    // for (; i + 7 < num_input; i += 8)
                    // {
                    //     int8x8_t _val0 = vld1_s8(m0);
                    //     int8x8_t _val1 = vld1_s8(m1);
                    //     int8x8_t _val2 = vld1_s8(m2);
                    //     int8x8_t _val3 = vld1_s8(m3);
                    //     int8x8_t _w = vld1_s8(kptr);

                    //     int16x8_t _s0 = vmull_s8(_val0, _w);
                    //     int16x8_t _s1 = vmull_s8(_val1, _w);
                    //     int16x8_t _s2 = vmull_s8(_val2, _w);
                    //     int16x8_t _s3 = vmull_s8(_val3, _w);
                    //     _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                    //     _sum1 = vaddw_s16(_sum1, vget_low_s16(_s1));
                    //     _sum2 = vaddw_s16(_sum2, vget_low_s16(_s2));
                    //     _sum3 = vaddw_s16(_sum3, vget_low_s16(_s3));
                    //     _sum0 = vaddw_s16(_sum0, vget_high_s16(_s0));
                    //     _sum1 = vaddw_s16(_sum1, vget_high_s16(_s1));
                    //     _sum2 = vaddw_s16(_sum2, vget_high_s16(_s2));
                    //     _sum3 = vaddw_s16(_sum3, vget_high_s16(_s3));

                    //     m0 += 8;
                    //     m1 += 8;
                    //     m2 += 8;
                    //     m3 += 8;
                    //     kptr += 8;
                    // }

                    // sum0 = vaddvq_s32(_sum0);
                    // sum1 = vaddvq_s32(_sum1);
                    // sum2 = vaddvq_s32(_sum2);
                    // sum3 = vaddvq_s32(_sum3);

                    // for (; i < num_input; i++)
                    // {
                    //     sum0 += *m0++ * kptr[0];
                    //     sum1 += *m1++ * kptr[0];
                    //     sum2 += *m2++ * kptr[0];
                    //     sum3 += *m3++ * kptr[0];
                    //     kptr += 1;
                    // }

                    // dequantize and relu
                    float sumfp32_0 = sum0 * scale_in_data[p];
                    float sumfp32_1 = sum1 * scale_in_data[p];
                    float sumfp32_2 = sum2 * scale_in_data[p];
                    float sumfp32_3 = sum3 * scale_in_data[p];

                    if (bias_term)
                    {
                        sumfp32_0 += bias_data[p];
                        sumfp32_1 += bias_data[p];
                        sumfp32_2 += bias_data[p];
                        sumfp32_3 += bias_data[p];
                    }

                    outptr[0] = activation_ss(sumfp32_0, activation_type, activation_params);
                    outptr[1] = activation_ss(sumfp32_1, activation_type, activation_params);
                    outptr[2] = activation_ss(sumfp32_2, activation_type, activation_params);
                    outptr[3] = activation_ss(sumfp32_3, activation_type, activation_params);
                    outptr += 4;
                }
            }
        }

        if (num_output_elempack == 8 && out_elempack == 1)
        {
            fprintf(stderr, "in 1499\n");
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int j = 0; j < outh; j++)
            {
                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output / num_output_elempack; p++)
                {
                    const signed char* kptr = weight_data_tm.row<const signed char>(p);
                    const signed char* m = bottom_blob_int8_unpacked.row<const signed char>(j);

                    // int32x4_t _sum0 = vdupq_n_s32(0);
                    // int32x4_t _sum1 = vdupq_n_s32(0);
                    vint32m2_t _sum = vmv_v_x_i32m2(0, vl);

                    int i = 0;
                    // for (; i + 3 < num_input; i += 4)
                    // {
                    //     int8x8_t _val0 = vdup_n_s8(m[0]);
                    //     int8x8_t _val1 = vdup_n_s8(m[1]);
                    //     int8x8_t _val2 = vdup_n_s8(m[2]);
                    //     int8x8_t _val3 = vdup_n_s8(m[3]);

                    //     int8x16_t _w0 = vld1q_s8(kptr);
                    //     int8x16_t _w1 = vld1q_s8(kptr + 16);

                    //     int16x8_t _s0 = vmull_s8(_val0, vget_low_s8(_w0));
                    //     int16x8_t _s1 = vmull_s8(_val2, vget_low_s8(_w1));
                    //     _s0 = vmlal_s8(_s0, _val1, vget_high_s8(_w0));
                    //     _s1 = vmlal_s8(_s1, _val3, vget_high_s8(_w1));

                    //     _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                    //     _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));
                    //     _sum0 = vaddw_s16(_sum0, vget_low_s16(_s1));
                    //     _sum1 = vaddw_s16(_sum1, vget_high_s16(_s1));

                    //     m += 4;
                    //     kptr += 32;
                    // }
                    for (; i < num_input; i++)
                    {
                        vl = 8;
                        vint8m1_t _val = vmv_v_x_i8m1(m[0], vl);
                        vint8m1_t _w = vmv_v_x_i8m1(kptr[0], vl);

                        // int8x8_t _val = vld1_dup_s8(m);
                        // int8x8_t _w = vld1_s8(kptr);

                        vint16m2_t _s = vwmul_vv_i16m2(_val, _w, vl);
                        vint16m1_t _s0 = vget_v_i16m2_i16m1(_s, 0);

                        _sum = vwadd_wv_i32m2(_sum, _s0, vl);

                        // int16x8_t _s0 = vmull_s8(_val, _w);
                        // _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                        // _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));

                        m++;
                        kptr += 8;
                    }

                    // dequantize and relu
                    // float32x4_t _scale_in0 = vld1q_f32((const float*)scale_in_data + p * 8);
                    // float32x4_t _scale_in1 = vld1q_f32((const float*)scale_in_data + p * 8 + 4);
                    vfloat32m2_t _scale_in = vle32_v_f32m2((const float*)scale_in_data + p * 8, vl);

                    vfloat32m2_t _sumfp32 = vfcvt_f_x_v_f32m2(_sum, vl);

                    // float32x4_t _sumfp32_0 = vcvtq_f32_s32(_sum0);
                    // float32x4_t _sumfp32_1 = vcvtq_f32_s32(_sum1);

                    if (bias_term)
                    {
                        vfloat32m2_t _bias = vle32_v_f32m2((const float*)bias_data + p * 8, vl);
                        _sumfp32 = vfmacc_vv_f32m2(_bias, _sumfp32, _scale_in, vl);
                        // float32x4_t _bias0 = vld1q_f32((const float*)bias_data + p * 8);
                        // float32x4_t _bias1 = vld1q_f32((const float*)bias_data + p * 8 + 4);
                        // _sumfp32_0 = vmlaq_f32(_bias0, _sumfp32_0, _scale_in0);
                        // _sumfp32_1 = vmlaq_f32(_bias1, _sumfp32_1, _scale_in1);
                    }
                    else
                    {
                        _sumfp32 = vfmul_vv_f32m2(_sumfp32, _scale_in, vl);
                        // _sumfp32_0 = vmulq_f32(_sumfp32_0, _scale_in0);
                        // _sumfp32_1 = vmulq_f32(_sumfp32_1, _scale_in1);
                    }

                    // _sumfp32_0 = activation_ps(_sumfp32_0, activation_type, activation_params, vl);
                    _sumfp32 = activation_ps(_sumfp32, activation_type, activation_params, vl);

                    vse32_v_f32m2(outptr, _sumfp32, vl);
                    // vst1q_f32(outptr, _sumfp32_0);
                    // vst1q_f32(outptr + 4, _sumfp32_1);
                    outptr += 8;
                }
            }
        }
#endif // __riscv_vector

        if (num_output_elempack == 1 && out_elempack == 1)
        {
            fprintf(stderr, "I think I am here\n");
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int j = 0; j < outh; j++)
            {
                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output; p++)
                {
                    const signed char* kptr = weight_data_tm.row<const signed char>(p);
                    const signed char* m = bottom_blob_int8_unpacked.row<const signed char>(j);

                    int sum = 0;

                    int i = 0;
#if __riscv_vector

                    int n = num_input;
                    vint32m4_t _sum = vmv_v_x_i32m4(0, vsetvlmax_e32m4());
                    while (n > 0)
                    {
                        vl = vsetvl_e32m4(n);
                        vint8m1_t _val = vle8_v_i8m1(m, vl);
                        vint8m1_t _w = vle8_v_i8m1(kptr, vl);

                        vint16m2_t _s = vwmul_vv_i16m2(_val, _w, vl);
                        _sum = vwadd_wv_i32m4(_sum, _s, vl);

                        // sum += vfmv_f_s_f32m1_f32(_sum);

                        m += vl;
                        kptr += vl;
                        n -= vl;
                    }

                    vint32m1_t _sum_scala = vmv_v_x_i32m1(0, vl);
                    _sum_scala = vredsum_vs_i32m4_i32m1(_sum_scala, _sum, _sum_scala, vl);
                    sum = vmv_x_s_i32m1_i32(_sum_scala);

                    // int32x4_t _sum0 = vdupq_n_s32(0);
                    // int32x4_t _sum1 = vdupq_n_s32(0);
                    // for (; i + 7 < num_input; i += 8)
                    // {
                    //     int8x8_t _val = vld1_s8(m);
                    //     int8x8_t _w = vld1_s8(kptr);

                    //     int16x8_t _s0 = vmull_s8(_val, _w);
                    //     _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                    //     _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));

                    //     m += 8;
                    //     kptr += 8;
                    // }

                    // _sum0 = vaddq_s32(_sum0, _sum1);
                    // sum = vaddvq_s32(_sum0);
#endif // __riscv_vector         

// for (; i < num_input; i++) \
// {                          \
//     sum += *m++ * *kptr++; \
// }

                    // dequantize and relu
                    float sumfp32 = sum * scale_in_data[p];

                    if (bias_term)
                        sumfp32 += bias_data[p];

                    outptr[0] = activation_ss(sumfp32, activation_type, activation_params);
                    outptr += 1;
                }
            }
        }

        return 0;
    }

    Mat bottom_blob_int8_flattened = bottom_blob_int8;
    if (bottom_blob_int8.dims != 1)
    {
        Option opt_flatten = opt;
        opt_flatten.blob_allocator = opt.workspace_allocator;
        flatten->forward(bottom_blob_int8, bottom_blob_int8_flattened, opt_flatten);
    }

    //     int elempack = bottom_blob_int8_flattened.elempack;

    int out_elempack = 1;
#if __riscv_vector
    if (opt.use_packing_layout)
    {
        out_elempack = num_output % 8 == 0 ? 8 : 1;
    }
#endif
    fprintf(stderr, "num_output = %d\n", num_output);
    fprintf(stderr, "out_elempack = %d\n", out_elempack);

    top_blob.create(num_output / out_elempack, (size_t)(4u * out_elempack), out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

#if __riscv_vector
    if (out_elempack == 8)
    {
        fprintf(stderr, "in 1703\n");
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            const signed char* kptr = weight_data_tm.row<const signed char>(p);
            const signed char* sptr = bottom_blob_int8_flattened;

            vl = 8;

            vint32m2_t _sum0 = vmv_v_x_i32m2(0, vl);
            // int32x4_t _sum0 = vdupq_n_s32(0);
            // int32x4_t _sum1 = vdupq_n_s32(0);

            int i = 0;
            // for (; i + 1 < num_input; i += 2)
            // {
            //     int8x8_t _val0 = vdup_n_s8(sptr[0]);
            //     int8x8_t _val1 = vdup_n_s8(sptr[1]);

            //     int8x8_t _w0 = vld1_s8(kptr);
            //     int8x8_t _w1 = vld1_s8(kptr + 8);

            //     int16x8_t _s0 = vmull_s8(_val0, _w0);
            //     _s0 = vmlal_s8(_s0, _val1, _w1);

            //     _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
            //     _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));

            //     sptr += 2;
            //     kptr += 16;
            // }
            for (; i < num_input; i++)
            {
                vint8m1_t _val = vmv_v_x_i8m1(sptr[0], vl);
                vint8m1_t _w = vle8_v_i8m1(kptr, vl);
                print_vint8m1(_w, 8);
                fprintf(stderr, "====================\n");


                vint16m1_t _s = vget_v_i16m2_i16m1(vwmul_vv_i16m2(_val, _w, vl), 0);
                _sum0 = vwadd_wv_i32m2(_sum0, _s, vl);

                // int8x8_t _val = vdup_n_s8(sptr[0]);

                // int8x8_t _w = vld1_s8(kptr);

                // int16x8_t _s0 = vmull_s8(_val, _w);
                // _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                // _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));

                sptr += 1;
                kptr += 8;
            }

            // dequantize and relu

            // float32x4_t _scale_in0 = vld1q_f32((const float*)scale_in_data + p * 8);
            // float32x4_t _scale_in1 = vld1q_f32((const float*)scale_in_data + p * 8 + 4);
            print_vint32m2(_sum0, 8);

            vfloat32m2_t _scale_in = vle32_v_f32m2((const float*)scale_in_data + p * 8, vl);

            vfloat32m2_t _sumfp32 = vfcvt_f_x_v_f32m2(_sum0, vl);
            // float32x4_t _sumfp32_0 = vcvtq_f32_s32(_sum0);
            // float32x4_t _sumfp32_1 = vcvtq_f32_s32(_sum1);

            if (bias_term)
            {
                fprintf(stderr, "Has bias\n");
                vfloat32m2_t _bias = vle32_v_f32m2((const float*)bias_data + p * 8, vl);
                fprintf(stderr, "======print _bias======\n");
                print_vfloat32m2(_bias, 8);

                fprintf(stderr, "======print _scale_in======\n");
                print_vfloat32m2(_scale_in, 8);

                _sumfp32 = vfmacc_vv_f32m2(_bias, _sumfp32, _scale_in, vl);
                // float32x4_t _bias0 = vld1q_f32((const float*)bias_data + p * 8);
                // float32x4_t _bias1 = vld1q_f32((const float*)bias_data + p * 8 + 4);
                // _sumfp32_0 = vmlaq_f32(_bias0, _sumfp32_0, _scale_in0);
                // _sumfp32_1 = vmlaq_f32(_bias1, _sumfp32_1, _scale_in1);
            }
            else
            {
                _sumfp32 = vfmul_vv_f32m2(_sumfp32, _scale_in, vl);
                // _sumfp32_0 = vmulq_f32(_sumfp32_0, _scale_in0);
                // _sumfp32_1 = vmulq_f32(_sumfp32_1, _scale_in1);
            }
            fprintf(stderr, "print _sumfp32\n");
            print_vfloat32m2(_sumfp32);
            fprintf(stderr, "activation_type: %d\n", activation_type);

            _sumfp32 = activation_ps(_sumfp32, activation_type, activation_params, vl);
            fprintf(stderr, "print _sumfp32 after activate\n");
            print_vfloat32m2(_sumfp32);
            // _sumfp32_1 = activation_ps(_sumfp32_1, activation_type, activation_params, vl);

            float* outptr = (float*)top_blob + p * 8;
            vse32_v_f32m2(outptr, _sumfp32, vl);
            // vst1q_f32(outptr, _sumfp32_0);
            // vst1q_f32(outptr + 4, _sumfp32_1);
        }
    }
#endif // __riscv_vector

    if (out_elempack == 1)
    {
        
        fprintf(stderr, "in 1795\n");
        fprintf(stderr, "out_elempack == 1 version\n");
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            const signed char* kptr = weight_data_tm.row<const signed char>(p);
            const signed char* sptr = bottom_blob_int8_flattened;

            int sum = 0;

            int i = 0;
            for (; i < num_input; i++)
            {
                signed char val = sptr[0];

                signed char w = kptr[0];
                fprintf(stderr, "val = %d, w = %d\n", val, w);

                sum += val * w;

                sptr += 1;
                kptr += 1;
            }
            fprintf(stderr, "sum = %d\n", sum);

            // dequantize and relu
            float sumfp32 = sum * scale_in_data[p];

            if (bias_term)
                sumfp32 += bias_data[p];
            fprintf(stderr, "sumfp32 = %f\n", sumfp32);

            sumfp32 = activation_ss(sumfp32, activation_type, activation_params);
            fprintf(stderr, "sumfp32 after activation = %f\n", sumfp32);

            top_blob[p] = sumfp32;

        }
    }

    return 0;
}
#endif // NCNN_INT8

} // namespace ncnn
