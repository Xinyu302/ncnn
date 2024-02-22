// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
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

static void convolution_gemm_transB_packed_tile_fp16sa(const Mat& AT_tile, const Mat& BT_tile, const Mat& CT_tile, Mat& topT_tile, Mat& top_blob, int i, int max_ii, int j, int max_jj, int k, int max_kk, bool k_end)
{
    int vl = 8;
    // NCNN_LOGE("convolution_gemm_transB_packed_tile_fp16sa %d %d %d %d %d %d", i, max_ii, j, max_jj, k, max_kk);

    const int out_elempack = top_blob.elempack;
    const int out_hstep = (int)top_blob.cstep;

    const __fp16* pAT = AT_tile;
    const __fp16* pBT = BT_tile;
    const __fp16* pC = CT_tile;

    __fp16* outptr = topT_tile;

    int ii = 0;
    for (; ii + 7 < max_ii; ii += 8)
    {
        __fp16* outptr0 = (__fp16*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        const __fp16* pB = pBT;

        if (pC)
        {
            pC = (const __fp16*)CT_tile + i + ii;
        }

        int jj = 0;
        for (; jj + 11 < max_jj; jj += 12)
        {
            vl = 8;
            const __fp16* pA = pAT;

            vfloat16m1_t _sum0;
            vfloat16m1_t _sum1;
            vfloat16m1_t _sum2;
            vfloat16m1_t _sum3;
            vfloat16m1_t _sum4;
            vfloat16m1_t _sum5;
            vfloat16m1_t _sum6;
            vfloat16m1_t _sum7;
            vfloat16m1_t _sum8;
            vfloat16m1_t _sum9;
            vfloat16m1_t _suma;
            vfloat16m1_t _sumb;

            // float16x8_t _sum0;
            // float16x8_t _sum1;
            // float16x8_t _sum2;
            // float16x8_t _sum3;
            // float16x8_t _sum4;
            // float16x8_t _sum5;
            // float16x8_t _sum6;
            // float16x8_t _sum7;
            // float16x8_t _sum8;
            // float16x8_t _sum9;
            // float16x8_t _suma;
            // float16x8_t _sumb;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = vle16_v_f16m1(pC, vl);
                    _sum1 = _sum0;
                    _sum2 = _sum0;
                    _sum3 = _sum0;
                    _sum4 = _sum0;
                    _sum5 = _sum0;
                    _sum6 = _sum0;
                    _sum7 = _sum0;
                    _sum8 = _sum0;
                    _sum9 = _sum0;
                    _suma = _sum0;
                    _sumb = _sum0;
                }
                else
                {
                    _sum0 = vfmv_v_f_f16m1(0.f, vl);
                    _sum1 = vfmv_v_f_f16m1(0.f, vl);
                    _sum2 = vfmv_v_f_f16m1(0.f, vl);
                    _sum3 = vfmv_v_f_f16m1(0.f, vl);
                    _sum4 = vfmv_v_f_f16m1(0.f, vl);
                    _sum5 = vfmv_v_f_f16m1(0.f, vl);
                    _sum6 = vfmv_v_f_f16m1(0.f, vl);
                    _sum7 = vfmv_v_f_f16m1(0.f, vl);
                    _sum8 = vfmv_v_f_f16m1(0.f, vl);
                    _sum9 = vfmv_v_f_f16m1(0.f, vl);
                    _suma = vfmv_v_f_f16m1(0.f, vl);
                    _sumb = vfmv_v_f_f16m1(0.f, vl);
                }
            }
            else
            {
                _sum0 = vle16_v_f16m1(outptr, vl);
                _sum1 = vle16_v_f16m1(outptr + 8, vl);
                _sum2 = vle16_v_f16m1(outptr + 8 * 2, vl);
                _sum3 = vle16_v_f16m1(outptr + 8 * 3, vl);
                _sum4 = vle16_v_f16m1(outptr + 8 * 4, vl);
                _sum5 = vle16_v_f16m1(outptr + 8 * 5, vl);
                _sum6 = vle16_v_f16m1(outptr + 8 * 6, vl);
                _sum7 = vle16_v_f16m1(outptr + 8 * 7, vl);
                _sum8 = vle16_v_f16m1(outptr + 8 * 8, vl);
                _sum9 = vle16_v_f16m1(outptr + 8 * 9, vl);
                _suma = vle16_v_f16m1(outptr + 8 * 10, vl);
                _sumb = vle16_v_f16m1(outptr + 8 * 11, vl);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                // float16x8_t _pA = vld1q_f16(pA);
                vfloat16m1_t _pA = vle16_v_f16m1(pA, vl);

                // float16x4_t _pB0 = vld1_f16(pB);
                // float16x4_t _pB1 = vld1_f16(pB + 4);
                // float16x4_t _pB2 = vld1_f16(pB + 8);
                _sum0 = vfmacc_vf_f16m1(_sum0, pB[0], _pA, vl);
                _sum1 = vfmacc_vf_f16m1(_sum1, pB[1], _pA, vl);
                _sum2 = vfmacc_vf_f16m1(_sum2, pB[2], _pA, vl);
                _sum3 = vfmacc_vf_f16m1(_sum3, pB[3], _pA, vl);
                _sum4 = vfmacc_vf_f16m1(_sum4, pB[4], _pA, vl);
                _sum5 = vfmacc_vf_f16m1(_sum5, pB[5], _pA, vl);
                _sum6 = vfmacc_vf_f16m1(_sum6, pB[6], _pA, vl);
                _sum7 = vfmacc_vf_f16m1(_sum7, pB[7], _pA, vl);
                _sum8 = vfmacc_vf_f16m1(_sum8, pB[8], _pA, vl);
                _sum9 = vfmacc_vf_f16m1(_sum9, pB[9], _pA, vl);
                _suma = vfmacc_vf_f16m1(_suma, pB[10], _pA, vl);
                _sumb = vfmacc_vf_f16m1(_sumb, pB[11], _pA, vl);

                pA += 8;
                pB += 12;
            }

            if (k_end)
            {
                if (out_elempack == 8)
                {
                    vl = 8;
                    vse16_v_f16m1(outptr0, _sum0, vl);
                    vse16_v_f16m1(outptr0 + 8, _sum1, vl);
                    vse16_v_f16m1(outptr0 + 8 * 2, _sum2, vl);
                    vse16_v_f16m1(outptr0 + 8 * 3, _sum3, vl);
                    vse16_v_f16m1(outptr0 + 8 * 4, _sum4, vl);
                    vse16_v_f16m1(outptr0 + 8 * 5, _sum5, vl);
                    vse16_v_f16m1(outptr0 + 8 * 6, _sum6, vl);
                    vse16_v_f16m1(outptr0 + 8 * 7, _sum7, vl);
                    vse16_v_f16m1(outptr0 + 8 * 8, _sum8, vl);
                    vse16_v_f16m1(outptr0 + 8 * 9, _sum9, vl);
                    vse16_v_f16m1(outptr0 + 8 * 10, _suma, vl);
                    vse16_v_f16m1(outptr0 + 8 * 11, _sumb, vl);
                    // vst1q_f16(outptr0, _sum0);
                    // vst1q_f16(outptr0 + 8, _sum1);
                    // vst1q_f16(outptr0 + 8 * 2, _sum2);
                    // vst1q_f16(outptr0 + 8 * 3, _sum3);
                    // vst1q_f16(outptr0 + 8 * 4, _sum4);
                    // vst1q_f16(outptr0 + 8 * 5, _sum5);
                    // vst1q_f16(outptr0 + 8 * 6, _sum6);
                    // vst1q_f16(outptr0 + 8 * 7, _sum7);
                    // vst1q_f16(outptr0 + 8 * 8, _sum8);
                    // vst1q_f16(outptr0 + 8 * 9, _sum9);
                    // vst1q_f16(outptr0 + 8 * 10, _suma);
                    // vst1q_f16(outptr0 + 8 * 11, _sumb);
                    outptr0 += 96;
                }
                if (out_elempack == 4)
                {
                    vl = 4;
                    vse16_v_f16m1(outptr0, _sum0, vl);
                    vse16_v_f16m1(outptr0 + 4, _sum1, vl);
                    vse16_v_f16m1(outptr0 + 4 * 2, _sum2, vl);
                    vse16_v_f16m1(outptr0 + 4 * 3, _sum3, vl);
                    vse16_v_f16m1(outptr0 + 4 * 4, _sum4, vl);
                    vse16_v_f16m1(outptr0 + 4 * 5, _sum5, vl);
                    vse16_v_f16m1(outptr0 + 4 * 6, _sum6, vl);
                    vse16_v_f16m1(outptr0 + 4 * 7, _sum7, vl);
                    vse16_v_f16m1(outptr0 + 4 * 8, _sum8, vl);
                    vse16_v_f16m1(outptr0 + 4 * 9, _sum9, vl);
                    vse16_v_f16m1(outptr0 + 4 * 10, _suma, vl);
                    vse16_v_f16m1(outptr0 + 4 * 11, _sumb, vl);

                    vse16_v_f16m1(outptr0 + out_hstep * 4, vslidedown_vx_f16m1(_sum0, _sum0, 4, vl), vl);
                    vse16_v_f16m1(outptr0 + out_hstep * 4 + 4, vslidedown_vx_f16m1(_sum1, _sum1, 4, vl), vl);
                    vse16_v_f16m1(outptr0 + out_hstep * 4 + 4 * 2, vslidedown_vx_f16m1(_sum2, _sum2, 4, vl), vl);
                    vse16_v_f16m1(outptr0 + out_hstep * 4 + 4 * 3, vslidedown_vx_f16m1(_sum3, _sum3, 4, vl), vl);
                    vse16_v_f16m1(outptr0 + out_hstep * 4 + 4 * 4, vslidedown_vx_f16m1(_sum4, _sum4, 4, vl), vl);
                    vse16_v_f16m1(outptr0 + out_hstep * 4 + 4 * 5, vslidedown_vx_f16m1(_sum5, _sum5, 4, vl), vl);
                    vse16_v_f16m1(outptr0 + out_hstep * 4 + 4 * 6, vslidedown_vx_f16m1(_sum6, _sum6, 4, vl), vl);
                    vse16_v_f16m1(outptr0 + out_hstep * 4 + 4 * 7, vslidedown_vx_f16m1(_sum7, _sum7, 4, vl), vl);
                    vse16_v_f16m1(outptr0 + out_hstep * 4 + 4 * 8, vslidedown_vx_f16m1(_sum8, _sum8, 4, vl), vl);
                    vse16_v_f16m1(outptr0 + out_hstep * 4 + 4 * 9, vslidedown_vx_f16m1(_sum9, _sum9, 4, vl), vl);
                    vse16_v_f16m1(outptr0 + out_hstep * 4 + 4 * 10, vslidedown_vx_f16m1(_suma, _suma, 4, vl), vl);
                    vse16_v_f16m1(outptr0 + out_hstep * 4 + 4 * 11, vslidedown_vx_f16m1(_sumb, _sumb, 4, vl), vl);

                    // vst1_f16(outptr0, vget_low_f16(_sum0));
                    // vst1_f16(outptr0 + 4, vget_low_f16(_sum1));
                    // vst1_f16(outptr0 + 4 * 2, vget_low_f16(_sum2));
                    // vst1_f16(outptr0 + 4 * 3, vget_low_f16(_sum3));
                    // vst1_f16(outptr0 + 4 * 4, vget_low_f16(_sum4));
                    // vst1_f16(outptr0 + 4 * 5, vget_low_f16(_sum5));
                    // vst1_f16(outptr0 + 4 * 6, vget_low_f16(_sum6));
                    // vst1_f16(outptr0 + 4 * 7, vget_low_f16(_sum7));
                    // vst1_f16(outptr0 + 4 * 8, vget_low_f16(_sum8));
                    // vst1_f16(outptr0 + 4 * 9, vget_low_f16(_sum9));
                    // vst1_f16(outptr0 + 4 * 10, vget_low_f16(_suma));
                    // vst1_f16(outptr0 + 4 * 11, vget_low_f16(_sumb));

                    // vst1_f16(outptr0 + out_hstep * 4, vget_high_f16(_sum0));
                    // vst1_f16(outptr0 + out_hstep * 4 + 4, vget_high_f16(_sum1));
                    // vst1_f16(outptr0 + out_hstep * 4 + 4 * 2, vget_high_f16(_sum2));
                    // vst1_f16(outptr0 + out_hstep * 4 + 4 * 3, vget_high_f16(_sum3));
                    // vst1_f16(outptr0 + out_hstep * 4 + 4 * 4, vget_high_f16(_sum4));
                    // vst1_f16(outptr0 + out_hstep * 4 + 4 * 5, vget_high_f16(_sum5));
                    // vst1_f16(outptr0 + out_hstep * 4 + 4 * 6, vget_high_f16(_sum6));
                    // vst1_f16(outptr0 + out_hstep * 4 + 4 * 7, vget_high_f16(_sum7));
                    // vst1_f16(outptr0 + out_hstep * 4 + 4 * 8, vget_high_f16(_sum8));
                    // vst1_f16(outptr0 + out_hstep * 4 + 4 * 9, vget_high_f16(_sum9));
                    // vst1_f16(outptr0 + out_hstep * 4 + 4 * 10, vget_high_f16(_suma));
                    // vst1_f16(outptr0 + out_hstep * 4 + 4 * 11, vget_high_f16(_sumb));

                    outptr0 += 48;
                }
                if (out_elempack == 1)
                {
                    // transpose8x12_ph(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7, _sum8, _sum9, _suma, _sumb);
                    vl = 8;
                    vsse16_v_f16m1(outptr0, out_hstep * sizeof(__fp16), _sum0, vl);
                    vsse16_v_f16m1(outptr0 + 1, out_hstep * sizeof(__fp16), _sum1, vl);
                    vsse16_v_f16m1(outptr0 + 2, out_hstep * sizeof(__fp16), _sum2, vl);
                    vsse16_v_f16m1(outptr0 + 3, out_hstep * sizeof(__fp16), _sum3, vl);
                    vsse16_v_f16m1(outptr0 + 4, out_hstep * sizeof(__fp16), _sum4, vl);
                    vsse16_v_f16m1(outptr0 + 5, out_hstep * sizeof(__fp16), _sum5, vl);
                    vsse16_v_f16m1(outptr0 + 6, out_hstep * sizeof(__fp16), _sum6, vl);
                    vsse16_v_f16m1(outptr0 + 7, out_hstep * sizeof(__fp16), _sum7, vl);
                    vsse16_v_f16m1(outptr0 + 8, out_hstep * sizeof(__fp16), _sum8, vl);
                    vsse16_v_f16m1(outptr0 + 9, out_hstep * sizeof(__fp16), _sum9, vl);
                    vsse16_v_f16m1(outptr0 + 10, out_hstep * sizeof(__fp16), _suma, vl);
                    vsse16_v_f16m1(outptr0 + 11, out_hstep * sizeof(__fp16), _sumb, vl);
                    // vst1_f16(outptr0, vget_low_f16(_sum0));
                    // vst1_f16(outptr0 + 4, vget_high_f16(_sum0));
                    // vst1_f16(outptr0 + 8, vget_low_f16(_sum1));
                    // vst1_f16(outptr0 + out_hstep, vget_high_f16(_sum1));
                    // vst1_f16(outptr0 + out_hstep + 4, vget_low_f16(_sum2));
                    // vst1_f16(outptr0 + out_hstep + 8, vget_high_f16(_sum2));
                    // vst1_f16(outptr0 + out_hstep * 2, vget_low_f16(_sum3));
                    // vst1_f16(outptr0 + out_hstep * 2 + 4, vget_high_f16(_sum3));
                    // vst1_f16(outptr0 + out_hstep * 2 + 8, vget_low_f16(_sum4));
                    // vst1_f16(outptr0 + out_hstep * 3, vget_high_f16(_sum4));
                    // vst1_f16(outptr0 + out_hstep * 3 + 4, vget_low_f16(_sum5));
                    // vst1_f16(outptr0 + out_hstep * 3 + 8, vget_high_f16(_sum5));
                    // vst1_f16(outptr0 + out_hstep * 4, vget_low_f16(_sum6));
                    // vst1_f16(outptr0 + out_hstep * 4 + 4, vget_high_f16(_sum6));
                    // vst1_f16(outptr0 + out_hstep * 4 + 8, vget_low_f16(_sum7));
                    // vst1_f16(outptr0 + out_hstep * 5, vget_high_f16(_sum7));
                    // vst1_f16(outptr0 + out_hstep * 5 + 4, vget_low_f16(_sum8));
                    // vst1_f16(outptr0 + out_hstep * 5 + 8, vget_high_f16(_sum8));
                    // vst1_f16(outptr0 + out_hstep * 6, vget_low_f16(_sum9));
                    // vst1_f16(outptr0 + out_hstep * 6 + 4, vget_high_f16(_sum9));
                    // vst1_f16(outptr0 + out_hstep * 6 + 8, vget_low_f16(_suma));
                    // vst1_f16(outptr0 + out_hstep * 7, vget_high_f16(_suma));
                    // vst1_f16(outptr0 + out_hstep * 7 + 4, vget_low_f16(_sumb));
                    // vst1_f16(outptr0 + out_hstep * 7 + 8, vget_high_f16(_sumb));

                    outptr0 += 12;
                }
            }
            else
            {
                vl = 8;
                vse16_v_f16m1(outptr, _sum0, vl);
                vse16_v_f16m1(outptr + 8, _sum1, vl);
                vse16_v_f16m1(outptr + 8 * 2, _sum2, vl);
                vse16_v_f16m1(outptr + 8 * 3, _sum3, vl);
                vse16_v_f16m1(outptr + 8 * 4, _sum4, vl);
                vse16_v_f16m1(outptr + 8 * 5, _sum5, vl);
                vse16_v_f16m1(outptr + 8 * 6, _sum6, vl);
                vse16_v_f16m1(outptr + 8 * 7, _sum7, vl);
                vse16_v_f16m1(outptr + 8 * 8, _sum8, vl);
                vse16_v_f16m1(outptr + 8 * 9, _sum9, vl);
                vse16_v_f16m1(outptr + 8 * 10, _suma, vl);
                vse16_v_f16m1(outptr + 8 * 11, _sumb, vl);

                // vst1q_f16(outptr, _sum0);
                // vst1q_f16(outptr + 8, _sum1);
                // vst1q_f16(outptr + 8 * 2, _sum2);
                // vst1q_f16(outptr + 8 * 3, _sum3);
                // vst1q_f16(outptr + 8 * 4, _sum4);
                // vst1q_f16(outptr + 8 * 5, _sum5);
                // vst1q_f16(outptr + 8 * 6, _sum6);
                // vst1q_f16(outptr + 8 * 7, _sum7);
                // vst1q_f16(outptr + 8 * 8, _sum8);
                // vst1q_f16(outptr + 8 * 9, _sum9);
                // vst1q_f16(outptr + 8 * 10, _suma);
                // vst1q_f16(outptr + 8 * 11, _sumb);
            }

            outptr += 96;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            vl = 8;
            const __fp16* pA = pAT;

            vfloat16m1_t _sum0;
            vfloat16m1_t _sum1;
            vfloat16m1_t _sum2;
            vfloat16m1_t _sum3;
            vfloat16m1_t _sum4;
            vfloat16m1_t _sum5;
            vfloat16m1_t _sum6;
            vfloat16m1_t _sum7;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = vle16_v_f16m1(pC, vl);
                    _sum1 = _sum0;
                    _sum2 = _sum0;
                    _sum3 = _sum0;
                    _sum4 = _sum0;
                    _sum5 = _sum0;
                    _sum6 = _sum0;
                    _sum7 = _sum0;
                    // _sum0 = vld1q_f16(pC);
                    // _sum1 = _sum0;
                    // _sum2 = _sum0;
                    // _sum3 = _sum0;
                    // _sum4 = _sum0;
                    // _sum5 = _sum0;
                    // _sum6 = _sum0;
                    // _sum7 = _sum0;
                }
                else
                {
                    _sum0 = vfmv_v_f_f16m1(0.f, vl);
                    _sum1 = vfmv_v_f_f16m1(0.f, vl);
                    _sum2 = vfmv_v_f_f16m1(0.f, vl);
                    _sum3 = vfmv_v_f_f16m1(0.f, vl);
                    _sum4 = vfmv_v_f_f16m1(0.f, vl);
                    _sum5 = vfmv_v_f_f16m1(0.f, vl);
                    _sum6 = vfmv_v_f_f16m1(0.f, vl);
                    _sum7 = vfmv_v_f_f16m1(0.f, vl);
                    // _sum0 = vdupq_n_f16(0.f);
                    // _sum1 = vdupq_n_f16(0.f);
                    // _sum2 = vdupq_n_f16(0.f);
                    // _sum3 = vdupq_n_f16(0.f);
                    // _sum4 = vdupq_n_f16(0.f);
                    // _sum5 = vdupq_n_f16(0.f);
                    // _sum6 = vdupq_n_f16(0.f);
                    // _sum7 = vdupq_n_f16(0.f);
                }
            }
            else
            {
                _sum0 = vle16_v_f16m1(outptr, vl);
                _sum1 = vle16_v_f16m1(outptr + 8, vl);
                _sum2 = vle16_v_f16m1(outptr + 8 * 2, vl);
                _sum3 = vle16_v_f16m1(outptr + 8 * 3, vl);
                _sum4 = vle16_v_f16m1(outptr + 8 * 4, vl);
                _sum5 = vle16_v_f16m1(outptr + 8 * 5, vl);
                _sum6 = vle16_v_f16m1(outptr + 8 * 6, vl);
                _sum7 = vle16_v_f16m1(outptr + 8 * 7, vl);
                // _sum0 = vld1q_f16(outptr);
                // _sum1 = vld1q_f16(outptr + 8);
                // _sum2 = vld1q_f16(outptr + 8 * 2);
                // _sum3 = vld1q_f16(outptr + 8 * 3);
                // _sum4 = vld1q_f16(outptr + 8 * 4);
                // _sum5 = vld1q_f16(outptr + 8 * 5);
                // _sum6 = vld1q_f16(outptr + 8 * 6);
                // _sum7 = vld1q_f16(outptr + 8 * 7);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                // float16x8_t _pA = vld1q_f16(pA);
                vfloat16m1_t _pA = vle16_v_f16m1(pA, vl);

                _sum0 = vfmacc_vf_f16m1(_sum0, pB[0], _pA, vl);
                _sum1 = vfmacc_vf_f16m1(_sum1, pB[1], _pA, vl);
                _sum2 = vfmacc_vf_f16m1(_sum2, pB[2], _pA, vl);
                _sum3 = vfmacc_vf_f16m1(_sum3, pB[3], _pA, vl);
                _sum4 = vfmacc_vf_f16m1(_sum4, pB[4], _pA, vl);
                _sum5 = vfmacc_vf_f16m1(_sum5, pB[5], _pA, vl);
                _sum6 = vfmacc_vf_f16m1(_sum6, pB[6], _pA, vl);
                _sum7 = vfmacc_vf_f16m1(_sum7, pB[7], _pA, vl);

                // float16x8_t _pB = vld1q_f16(pB);

                // _sum0 = vfmaq_laneq_f16(_sum0, _pA, _pB, 0);
                // _sum1 = vfmaq_laneq_f16(_sum1, _pA, _pB, 1);
                // _sum2 = vfmaq_laneq_f16(_sum2, _pA, _pB, 2);
                // _sum3 = vfmaq_laneq_f16(_sum3, _pA, _pB, 3);
                // _sum4 = vfmaq_laneq_f16(_sum4, _pA, _pB, 4);
                // _sum5 = vfmaq_laneq_f16(_sum5, _pA, _pB, 5);
                // _sum6 = vfmaq_laneq_f16(_sum6, _pA, _pB, 6);
                // _sum7 = vfmaq_laneq_f16(_sum7, _pA, _pB, 7);

                pA += 8;
                pB += 8;
            }

            if (k_end)
            {
                if (out_elempack == 8)
                {
                    vl = 8;
                    vse16_v_f16m1(outptr0, _sum0, vl);
                    vse16_v_f16m1(outptr0 + 8, _sum1, vl);
                    vse16_v_f16m1(outptr0 + 8 * 2, _sum2, vl);
                    vse16_v_f16m1(outptr0 + 8 * 3, _sum3, vl);
                    vse16_v_f16m1(outptr0 + 8 * 4, _sum4, vl);
                    vse16_v_f16m1(outptr0 + 8 * 5, _sum5, vl);
                    vse16_v_f16m1(outptr0 + 8 * 6, _sum6, vl);
                    vse16_v_f16m1(outptr0 + 8 * 7, _sum7, vl);

                    // vst1q_f16(outptr0, _sum0);
                    // vst1q_f16(outptr0 + 8, _sum1);
                    // vst1q_f16(outptr0 + 8 * 2, _sum2);
                    // vst1q_f16(outptr0 + 8 * 3, _sum3);
                    // vst1q_f16(outptr0 + 8 * 4, _sum4);
                    // vst1q_f16(outptr0 + 8 * 5, _sum5);
                    // vst1q_f16(outptr0 + 8 * 6, _sum6);
                    // vst1q_f16(outptr0 + 8 * 7, _sum7);
                    outptr0 += 64;
                }
                if (out_elempack == 4)
                {
                    vl = 4;
                    vse16_v_f16m1(outptr0, _sum0, vl);
                    vse16_v_f16m1(outptr0 + 4, _sum1, vl);
                    vse16_v_f16m1(outptr0 + 4 * 2, _sum2, vl);
                    vse16_v_f16m1(outptr0 + 4 * 3, _sum3, vl);
                    vse16_v_f16m1(outptr0 + 4 * 4, _sum4, vl);
                    vse16_v_f16m1(outptr0 + 4 * 5, _sum5, vl);
                    vse16_v_f16m1(outptr0 + 4 * 6, _sum6, vl);
                    vse16_v_f16m1(outptr0 + 4 * 7, _sum7, vl);

                    vse16_v_f16m1(outptr0 + out_hstep * 4, vslidedown_vx_f16m1(_sum0, _sum0, 4, vl), vl);
                    vse16_v_f16m1(outptr0 + out_hstep * 4 + 4, vslidedown_vx_f16m1(_sum1, _sum1, 4, vl), vl);
                    vse16_v_f16m1(outptr0 + out_hstep * 4 + 4 * 2, vslidedown_vx_f16m1(_sum2, _sum2, 4, vl), vl);
                    vse16_v_f16m1(outptr0 + out_hstep * 4 + 4 * 3, vslidedown_vx_f16m1(_sum3, _sum3, 4, vl), vl);
                    vse16_v_f16m1(outptr0 + out_hstep * 4 + 4 * 4, vslidedown_vx_f16m1(_sum4, _sum4, 4, vl), vl);
                    vse16_v_f16m1(outptr0 + out_hstep * 4 + 4 * 5, vslidedown_vx_f16m1(_sum5, _sum5, 4, vl), vl);
                    vse16_v_f16m1(outptr0 + out_hstep * 4 + 4 * 6, vslidedown_vx_f16m1(_sum6, _sum6, 4, vl), vl);
                    vse16_v_f16m1(outptr0 + out_hstep * 4 + 4 * 7, vslidedown_vx_f16m1(_sum7, _sum7, 4, vl), vl);

                    // vst1_f16(outptr0, vget_low_f16(_sum0));
                    // vst1_f16(outptr0 + 4, vget_low_f16(_sum1));
                    // vst1_f16(outptr0 + 4 * 2, vget_low_f16(_sum2));
                    // vst1_f16(outptr0 + 4 * 3, vget_low_f16(_sum3));
                    // vst1_f16(outptr0 + 4 * 4, vget_low_f16(_sum4));
                    // vst1_f16(outptr0 + 4 * 5, vget_low_f16(_sum5));
                    // vst1_f16(outptr0 + 4 * 6, vget_low_f16(_sum6));
                    // vst1_f16(outptr0 + 4 * 7, vget_low_f16(_sum7));

                    // vst1_f16(outptr0 + out_hstep * 4, vget_high_f16(_sum0));
                    // vst1_f16(outptr0 + out_hstep * 4 + 4, vget_high_f16(_sum1));
                    // vst1_f16(outptr0 + out_hstep * 4 + 4 * 2, vget_high_f16(_sum2));
                    // vst1_f16(outptr0 + out_hstep * 4 + 4 * 3, vget_high_f16(_sum3));
                    // vst1_f16(outptr0 + out_hstep * 4 + 4 * 4, vget_high_f16(_sum4));
                    // vst1_f16(outptr0 + out_hstep * 4 + 4 * 5, vget_high_f16(_sum5));
                    // vst1_f16(outptr0 + out_hstep * 4 + 4 * 6, vget_high_f16(_sum6));
                    // vst1_f16(outptr0 + out_hstep * 4 + 4 * 7, vget_high_f16(_sum7));

                    outptr0 += 32;
                }
                if (out_elempack == 1)
                {
                    vl = 8;
                    vsse16_v_f16m1(outptr0, out_hstep * sizeof(__fp16), _sum0, vl);
                    vsse16_v_f16m1(outptr0 + 1, out_hstep * sizeof(__fp16), _sum1, vl);
                    vsse16_v_f16m1(outptr0 + 2, out_hstep * sizeof(__fp16), _sum2, vl);
                    vsse16_v_f16m1(outptr0 + 3, out_hstep * sizeof(__fp16), _sum3, vl);
                    vsse16_v_f16m1(outptr0 + 4, out_hstep * sizeof(__fp16), _sum4, vl);
                    vsse16_v_f16m1(outptr0 + 5, out_hstep * sizeof(__fp16), _sum5, vl);
                    vsse16_v_f16m1(outptr0 + 6, out_hstep * sizeof(__fp16), _sum6, vl);
                    vsse16_v_f16m1(outptr0 + 7, out_hstep * sizeof(__fp16), _sum7, vl);

                    // transpose8x8_ph(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7);

                    // vst1q_f16(outptr0, _sum0);
                    // vst1q_f16(outptr0 + out_hstep, _sum1);
                    // vst1q_f16(outptr0 + out_hstep * 2, _sum2);
                    // vst1q_f16(outptr0 + out_hstep * 3, _sum3);
                    // vst1q_f16(outptr0 + out_hstep * 4, _sum4);
                    // vst1q_f16(outptr0 + out_hstep * 5, _sum5);
                    // vst1q_f16(outptr0 + out_hstep * 6, _sum6);
                    // vst1q_f16(outptr0 + out_hstep * 7, _sum7);

                    outptr0 += 8;
                }
            }
            else
            {
                vse16_v_f16m1(outptr, _sum0, vl);
                vse16_v_f16m1(outptr + 8, _sum1, vl);
                vse16_v_f16m1(outptr + 8 * 2, _sum2, vl);
                vse16_v_f16m1(outptr + 8 * 3, _sum3, vl);
                vse16_v_f16m1(outptr + 8 * 4, _sum4, vl);
                vse16_v_f16m1(outptr + 8 * 5, _sum5, vl);
                vse16_v_f16m1(outptr + 8 * 6, _sum6, vl);
                vse16_v_f16m1(outptr + 8 * 7, _sum7, vl);

                // vst1q_f16(outptr, _sum0);
                // vst1q_f16(outptr + 8, _sum1);
                // vst1q_f16(outptr + 8 * 2, _sum2);
                // vst1q_f16(outptr + 8 * 3, _sum3);
                // vst1q_f16(outptr + 8 * 4, _sum4);
                // vst1q_f16(outptr + 8 * 5, _sum5);
                // vst1q_f16(outptr + 8 * 6, _sum6);
                // vst1q_f16(outptr + 8 * 7, _sum7);
            }

            outptr += 64;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            vl = 8;
            const __fp16* pA = pAT;

            // float16x8_t _sum0;
            // float16x8_t _sum1;
            // float16x8_t _sum2;
            // float16x8_t _sum3;
            vfloat16m1_t _sum0;
            vfloat16m1_t _sum1;
            vfloat16m1_t _sum2;
            vfloat16m1_t _sum3;

            if (k == 0)
            {
                if (pC)
                {
                    // _sum0 = vld1q_f16(pC);
                    // _sum1 = _sum0;
                    // _sum2 = _sum0;
                    // _sum3 = _sum0;
                    _sum0 = vle16_v_f16m1(pC, vl);
                    _sum1 = _sum0;
                    _sum2 = _sum0;
                    _sum3 = _sum0;
                }
                else
                {
                    // _sum0 = vdupq_n_f16(0.f);
                    // _sum1 = vdupq_n_f16(0.f);
                    // _sum2 = vdupq_n_f16(0.f);
                    // _sum3 = vdupq_n_f16(0.f);
                    _sum0 = vfmv_v_f_f16m1(0.f, vl);
                    _sum1 = vfmv_v_f_f16m1(0.f, vl);
                    _sum2 = vfmv_v_f_f16m1(0.f, vl);
                    _sum3 = vfmv_v_f_f16m1(0.f, vl);
                }
            }
            else
            {
                // _sum0 = vld1q_f16(outptr);
                // _sum1 = vld1q_f16(outptr + 8);
                // _sum2 = vld1q_f16(outptr + 8 * 2);
                // _sum3 = vld1q_f16(outptr + 8 * 3);
                _sum0 = vle16_v_f16m1(outptr, vl);
                _sum1 = vle16_v_f16m1(outptr + 8, vl);
                _sum2 = vle16_v_f16m1(outptr + 8 * 2, vl);
                _sum3 = vle16_v_f16m1(outptr + 8 * 3, vl);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat16m1_t _pA = vle16_v_f16m1(pA, vl);

                _sum0 = vfmacc_vf_f16m1(_sum0, pB[0], _pA, vl);
                _sum1 = vfmacc_vf_f16m1(_sum1, pB[1], _pA, vl);
                _sum2 = vfmacc_vf_f16m1(_sum2, pB[2], _pA, vl);
                _sum3 = vfmacc_vf_f16m1(_sum3, pB[3], _pA, vl);
                // float16x8_t _pA = vld1q_f16(pA);

                // float16x4_t _pB = vld1_f16(pB);

                // _sum0 = vfmaq_lane_f16(_sum0, _pA, _pB, 0);
                // _sum1 = vfmaq_lane_f16(_sum1, _pA, _pB, 1);
                // _sum2 = vfmaq_lane_f16(_sum2, _pA, _pB, 2);
                // _sum3 = vfmaq_lane_f16(_sum3, _pA, _pB, 3);

                pA += 8;
                pB += 4;
            }

            if (k_end)
            {
                if (out_elempack == 8)
                {
                    vse16_v_f16m1(outptr0, _sum0, vl);
                    vse16_v_f16m1(outptr0 + 8, _sum1, vl);
                    vse16_v_f16m1(outptr0 + 8 * 2, _sum2, vl);
                    vse16_v_f16m1(outptr0 + 8 * 3, _sum3, vl);
                    // vst1q_f16(outptr0, _sum0);
                    // vst1q_f16(outptr0 + 8, _sum1);
                    // vst1q_f16(outptr0 + 8 * 2, _sum2);
                    // vst1q_f16(outptr0 + 8 * 3, _sum3);
                    outptr0 += 32;
                }
                if (out_elempack == 4)
                {
                    vl = 4;
                    vse16_v_f16m1(outptr0, _sum0, vl);
                    vse16_v_f16m1(outptr0 + 4, _sum1, vl);
                    vse16_v_f16m1(outptr0 + 4 * 2, _sum2, vl);
                    vse16_v_f16m1(outptr0 + 4 * 3, _sum3, vl);

                    vse16_v_f16m1(outptr0 + out_hstep * 4, vslidedown_vx_f16m1(_sum0, _sum0, 4, vl), vl);
                    vse16_v_f16m1(outptr0 + out_hstep * 4 + 4, vslidedown_vx_f16m1(_sum1, _sum1, 4, vl), vl);
                    vse16_v_f16m1(outptr0 + out_hstep * 4 + 4 * 2, vslidedown_vx_f16m1(_sum2, _sum2, 4, vl), vl);
                    vse16_v_f16m1(outptr0 + out_hstep * 4 + 4 * 3, vslidedown_vx_f16m1(_sum3, _sum3, 4, vl), vl);

                    // vst1_f16(outptr0, vget_low_f16(_sum0));
                    // vst1_f16(outptr0 + 4, vget_low_f16(_sum1));
                    // vst1_f16(outptr0 + 4 * 2, vget_low_f16(_sum2));
                    // vst1_f16(outptr0 + 4 * 3, vget_low_f16(_sum3));

                    // vst1_f16(outptr0 + out_hstep * 4, vget_high_f16(_sum0));
                    // vst1_f16(outptr0 + out_hstep * 4 + 4, vget_high_f16(_sum1));
                    // vst1_f16(outptr0 + out_hstep * 4 + 4 * 2, vget_high_f16(_sum2));
                    // vst1_f16(outptr0 + out_hstep * 4 + 4 * 3, vget_high_f16(_sum3));

                    outptr0 += 16;
                }
                if (out_elempack == 1)
                {
                    vsse16_v_f16m1(outptr0, out_hstep * sizeof(__fp16), _sum0, vl);
                    vsse16_v_f16m1(outptr0 + 1, out_hstep * sizeof(__fp16), _sum1, vl);
                    vsse16_v_f16m1(outptr0 + 2, out_hstep * sizeof(__fp16), _sum2, vl);
                    vsse16_v_f16m1(outptr0 + 3, out_hstep * sizeof(__fp16), _sum3, vl);

                    // transpose8x4_ph(_sum0, _sum1, _sum2, _sum3);

                    // vst1_f16(outptr0, vget_low_f16(_sum0));
                    // vst1_f16(outptr0 + out_hstep * 1, vget_high_f16(_sum0));
                    // vst1_f16(outptr0 + out_hstep * 2, vget_low_f16(_sum1));
                    // vst1_f16(outptr0 + out_hstep * 3, vget_high_f16(_sum1));
                    // vst1_f16(outptr0 + out_hstep * 4, vget_low_f16(_sum2));
                    // vst1_f16(outptr0 + out_hstep * 5, vget_high_f16(_sum2));
                    // vst1_f16(outptr0 + out_hstep * 6, vget_low_f16(_sum3));
                    // vst1_f16(outptr0 + out_hstep * 7, vget_high_f16(_sum3));

                    outptr0 += 4;
                }
            }
            else
            {
                vse16_v_f16m1(outptr, _sum0, vl);
                vse16_v_f16m1(outptr + 8, _sum1, vl);
                vse16_v_f16m1(outptr + 8 * 2, _sum2, vl);
                vse16_v_f16m1(outptr + 8 * 3, _sum3, vl);
                // vst1q_f16(outptr, _sum0);
                // vst1q_f16(outptr + 8, _sum1);
                // vst1q_f16(outptr + 8 * 2, _sum2);
                // vst1q_f16(outptr + 8 * 3, _sum3);
            }

            outptr += 32;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const __fp16* pA = pAT;

            // float16x8_t _sum0;
            // float16x8_t _sum1;
            vl = 8;
            vfloat16m1_t _sum0;
            vfloat16m1_t _sum1;

            if (k == 0)
            {
                if (pC)
                {
                    // _sum0 = vld1q_f16(pC);
                    // _sum1 = _sum0;
                    _sum0 = vle16_v_f16m1(pC, vl);
                    _sum1 = _sum0;
                }
                else
                {
                    // _sum0 = vdupq_n_f16(0.f);
                    // _sum1 = vdupq_n_f16(0.f);
                    _sum0 = vfmv_v_f_f16m1(0.f, vl);
                    _sum1 = vfmv_v_f_f16m1(0.f, vl);
                }
            }
            else
            {
                // _sum0 = vld1q_f16(outptr);
                // _sum1 = vld1q_f16(outptr + 8);
                _sum0 = vle16_v_f16m1(outptr, vl);
                _sum1 = vle16_v_f16m1(outptr + 8, vl);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat16m1_t _pA = vle16_v_f16m1(pA, vl);

                _sum0 = vfmacc_vf_f16m1(_sum0, pB[0], _pA, vl);
                _sum1 = vfmacc_vf_f16m1(_sum1, pB[1], _pA, vl);
                // float16x8_t _pA = vld1q_f16(pA);

                // float16x4_t _pB = vld1_f16(pB);

                // _sum0 = vfmaq_lane_f16(_sum0, _pA, _pB, 0);
                // _sum1 = vfmaq_lane_f16(_sum1, _pA, _pB, 1);

                pA += 8;
                pB += 2;
            }

            if (k_end)
            {
                if (out_elempack == 8)
                {
                    // vst1q_f16(outptr0, _sum0);
                    // vst1q_f16(outptr0 + 8, _sum1);
                    vse16_v_f16m1(outptr0, _sum0, vl);
                    vse16_v_f16m1(outptr0 + 8, _sum1, vl);
                    outptr0 += 16;
                }
                if (out_elempack == 4)
                {
                    vl = 4;
                    vse16_v_f16m1(outptr0, _sum0, vl);
                    vse16_v_f16m1(outptr0 + 4, _sum1, vl);

                    vse16_v_f16m1(outptr0 + out_hstep * 4, vslidedown_vx_f16m1(_sum0, _sum0, 4, vl), vl);
                    vse16_v_f16m1(outptr0 + out_hstep * 4 + 4, vslidedown_vx_f16m1(_sum1, _sum1, 4, vl), vl);
                    // vst1_f16(outptr0, vget_low_f16(_sum0));
                    // vst1_f16(outptr0 + 4, vget_low_f16(_sum1));

                    // vst1_f16(outptr0 + out_hstep * 4, vget_high_f16(_sum0));
                    // vst1_f16(outptr0 + out_hstep * 4 + 4, vget_high_f16(_sum1));
                    outptr0 += 8;
                }
                if (out_elempack == 1)
                {
                    vsse16_v_f16m1(outptr0, out_hstep * sizeof(__fp16), _sum0, vl);
                    vsse16_v_f16m1(outptr0 + 1, out_hstep * sizeof(__fp16), _sum1, vl);
                    // __fp16 sum0[8];
                    // __fp16 sum1[8];
                    // vst1q_f16(sum0, _sum0);
                    // vst1q_f16(sum1, _sum1);

                    // outptr0[0] = sum0[0];
                    // outptr0[out_hstep] = sum0[1];
                    // outptr0[out_hstep * 2] = sum0[2];
                    // outptr0[out_hstep * 3] = sum0[3];
                    // outptr0[out_hstep * 4] = sum0[4];
                    // outptr0[out_hstep * 5] = sum0[5];
                    // outptr0[out_hstep * 6] = sum0[6];
                    // outptr0[out_hstep * 7] = sum0[7];

                    // outptr0[1] = sum1[0];
                    // outptr0[out_hstep + 1] = sum1[1];
                    // outptr0[out_hstep * 2 + 1] = sum1[2];
                    // outptr0[out_hstep * 3 + 1] = sum1[3];
                    // outptr0[out_hstep * 4 + 1] = sum1[4];
                    // outptr0[out_hstep * 5 + 1] = sum1[5];
                    // outptr0[out_hstep * 6 + 1] = sum1[6];
                    // outptr0[out_hstep * 7 + 1] = sum1[7];
                    outptr0 += 2;
                }
            }
            else
            {
                vse16_v_f16m1(outptr, _sum0, vl);
                vse16_v_f16m1(outptr + 8, _sum1, vl);
                // vst1q_f16(outptr, _sum0);
                // vst1q_f16(outptr + 8, _sum1);
            }

            outptr += 16;
        }
        for (; jj < max_jj; jj += 1)
        {
            const __fp16* pA = pAT;

            // float16x8_t _sum0;
            vfloat16m1_t _sum0;

            if (k == 0)
            {
                if (pC)
                {
                    // _sum0 = vld1q_f16(pC);
                    _sum0 = vle16_v_f16m1(pC, vl);
                }
                else
                {
                    // _sum0 = vdupq_n_f16(0.f);
                    _sum0 = vfmv_v_f_f16m1(0.f, vl);
                }
            }
            else
            {
                // _sum0 = vld1q_f16(outptr);
                _sum0 = vle16_v_f16m1(outptr, vl);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                // float16x8_t _pA = vld1q_f16(pA);

                // float16x8_t _pB = vld1q_dup_f16(pB);

                // _sum0 = vfmaq_f16(_sum0, _pA, _pB);
                vfloat16m1_t _pA = vle16_v_f16m1(pA, vl);
                _sum0 = vfmacc_vf_f16m1(_sum0, pB[0], _pA, vl);

                pA += 8;
                pB += 1;
            }

            if (k_end)
            {
                if (out_elempack == 8)
                {
                    // vst1q_f16(outptr0, _sum0);
                    vse16_v_f16m1(outptr0, _sum0, vl);
                    outptr0 += 8;
                }
                if (out_elempack == 4)
                {
                    // vst1_f16(outptr0, vget_low_f16(_sum0));
                    // vst1_f16(outptr0 + out_hstep * 4, vget_high_f16(_sum0));
                    vl = 4;
                    vse16_v_f16m1(outptr0, _sum0, vl);
                    vse16_v_f16m1(outptr0 + out_hstep * 4, vslidedown_vx_f16m1(_sum0, _sum0, 4, vl), vl);
                    outptr0 += 4;
                }
                if (out_elempack == 1)
                {
                    vsse16_v_f16m1(outptr0, out_hstep * sizeof(__fp16), _sum0, vl);
                    // __fp16 sum0[8];
                    // vst1q_f16(sum0, _sum0);

                    // outptr0[0] = sum0[0];
                    // outptr0[out_hstep * 1] = sum0[1];
                    // outptr0[out_hstep * 2] = sum0[2];
                    // outptr0[out_hstep * 3] = sum0[3];
                    // outptr0[out_hstep * 4] = sum0[4];
                    // outptr0[out_hstep * 5] = sum0[5];
                    // outptr0[out_hstep * 6] = sum0[6];
                    // outptr0[out_hstep * 7] = sum0[7];
                    outptr0++;
                }
            }
            else
            {
                // vst1q_f16(outptr, _sum0);
                vse16_v_f16m1(outptr, _sum0, vl);
            }

            outptr += 8;
        }

        pAT += max_kk * 8;
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        vl = 4;
        __fp16* outptr0 = (__fp16*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        const __fp16* pB = pBT;

        if (pC)
        {
            pC = (const __fp16*)CT_tile + i + ii;
        }

        int jj = 0;
        for (; jj + 11 < max_jj; jj += 12)
        {
            vl = 4;
            vfloat16m1_t _sum0;
            vfloat16m1_t _sum1;
            vfloat16m1_t _sum2;
            vfloat16m1_t _sum3;
            vfloat16m1_t _sum4;
            vfloat16m1_t _sum5;
            vfloat16m1_t _sum6;
            vfloat16m1_t _sum7;
            vfloat16m1_t _sum8;
            vfloat16m1_t _sum9;
            vfloat16m1_t _suma;
            vfloat16m1_t _sumb;
            // float16x4_t _sum0;
            // float16x4_t _sum1;
            // float16x4_t _sum2;
            // float16x4_t _sum3;
            // float16x4_t _sum4;
            // float16x4_t _sum5;
            // float16x4_t _sum6;
            // float16x4_t _sum7;
            // float16x4_t _sum8;
            // float16x4_t _sum9;
            // float16x4_t _suma;
            // float16x4_t _sumb;

            if (k == 0)
            {
                if (pC)
                {
                    // _sum0 = vld1_f16(pC);
                    // _sum1 = _sum0;
                    // _sum2 = _sum0;
                    // _sum3 = _sum0;
                    // _sum4 = _sum0;
                    // _sum5 = _sum0;
                    // _sum6 = _sum0;
                    // _sum7 = _sum0;
                    // _sum8 = _sum0;
                    // _sum9 = _sum0;
                    // _suma = _sum0;
                    // _sumb = _sum0;
                    _sum0 = vle16_v_f16m1(pC, vl);
                    _sum1 = _sum0;
                    _sum2 = _sum0;
                    _sum3 = _sum0;
                    _sum4 = _sum0;
                    _sum5 = _sum0;
                    _sum6 = _sum0;
                    _sum7 = _sum0;
                    _sum8 = _sum0;
                    _sum9 = _sum0;
                    _suma = _sum0;
                    _sumb = _sum0;
                }
                else
                {
                    // _sum0 = vdup_n_f16(0.f);
                    // _sum1 = vdup_n_f16(0.f);
                    // _sum2 = vdup_n_f16(0.f);
                    // _sum3 = vdup_n_f16(0.f);
                    // _sum4 = vdup_n_f16(0.f);
                    // _sum5 = vdup_n_f16(0.f);
                    // _sum6 = vdup_n_f16(0.f);
                    // _sum7 = vdup_n_f16(0.f);
                    // _sum8 = vdup_n_f16(0.f);
                    // _sum9 = vdup_n_f16(0.f);
                    // _suma = vdup_n_f16(0.f);
                    // _sumb = vdup_n_f16(0.f);
                    _sum0 = vfmv_v_f_f16m1(0.f, vl);
                    _sum1 = vfmv_v_f_f16m1(0.f, vl);
                    _sum2 = vfmv_v_f_f16m1(0.f, vl);
                    _sum3 = vfmv_v_f_f16m1(0.f, vl);
                    _sum4 = vfmv_v_f_f16m1(0.f, vl);
                    _sum5 = vfmv_v_f_f16m1(0.f, vl);
                    _sum6 = vfmv_v_f_f16m1(0.f, vl);
                    _sum7 = vfmv_v_f_f16m1(0.f, vl);
                    _sum8 = vfmv_v_f_f16m1(0.f, vl);
                    _sum9 = vfmv_v_f_f16m1(0.f, vl);
                    _suma = vfmv_v_f_f16m1(0.f, vl);
                    _sumb = vfmv_v_f_f16m1(0.f, vl);
                }
            }
            else
            {
                // _sum0 = vld1_f16(outptr);
                // _sum1 = vld1_f16(outptr + 4 * 1);
                // _sum2 = vld1_f16(outptr + 4 * 2);
                // _sum3 = vld1_f16(outptr + 4 * 3);
                // _sum4 = vld1_f16(outptr + 4 * 4);
                // _sum5 = vld1_f16(outptr + 4 * 5);
                // _sum6 = vld1_f16(outptr + 4 * 6);
                // _sum7 = vld1_f16(outptr + 4 * 7);
                // _sum8 = vld1_f16(outptr + 4 * 8);
                // _sum9 = vld1_f16(outptr + 4 * 9);
                // _suma = vld1_f16(outptr + 4 * 10);
                // _sumb = vld1_f16(outptr + 4 * 11);
                _sum0 = vle16_v_f16m1(outptr, vl);
                _sum1 = vle16_v_f16m1(outptr + 4, vl);
                _sum2 = vle16_v_f16m1(outptr + 4 * 2, vl);
                _sum3 = vle16_v_f16m1(outptr + 4 * 3, vl);
                _sum4 = vle16_v_f16m1(outptr + 4 * 4, vl);
                _sum5 = vle16_v_f16m1(outptr + 4 * 5, vl);
                _sum6 = vle16_v_f16m1(outptr + 4 * 6, vl);
                _sum7 = vle16_v_f16m1(outptr + 4 * 7, vl);
                _sum8 = vle16_v_f16m1(outptr + 4 * 8, vl);
                _sum9 = vle16_v_f16m1(outptr + 4 * 9, vl);
                _suma = vle16_v_f16m1(outptr + 4 * 10, vl);
                _sumb = vle16_v_f16m1(outptr + 4 * 11, vl);
            }

            const __fp16* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                // float16x4_t _pA = vld1_f16(pA);
                // float16x4_t _pB0 = vld1_f16(pB);
                // float16x4_t _pB1 = vld1_f16(pB + 4);
                // float16x4_t _pB2 = vld1_f16(pB + 8);

                // _sum0 = vfma_lane_f16(_sum0, _pA, _pB0, 0);
                // _sum1 = vfma_lane_f16(_sum1, _pA, _pB0, 1);
                // _sum2 = vfma_lane_f16(_sum2, _pA, _pB0, 2);
                // _sum3 = vfma_lane_f16(_sum3, _pA, _pB0, 3);
                // _sum4 = vfma_lane_f16(_sum4, _pA, _pB1, 0);
                // _sum5 = vfma_lane_f16(_sum5, _pA, _pB1, 1);
                // _sum6 = vfma_lane_f16(_sum6, _pA, _pB1, 2);
                // _sum7 = vfma_lane_f16(_sum7, _pA, _pB1, 3);
                // _sum8 = vfma_lane_f16(_sum8, _pA, _pB2, 0);
                // _sum9 = vfma_lane_f16(_sum9, _pA, _pB2, 1);
                // _suma = vfma_lane_f16(_suma, _pA, _pB2, 2);
                // _sumb = vfma_lane_f16(_sumb, _pA, _pB2, 3);
                vfloat16m1_t _pA = vle16_v_f16m1(pA, vl);
                _sum0 = vfmacc_vf_f16m1(_sum0, pB[0], _pA, vl);
                _sum1 = vfmacc_vf_f16m1(_sum1, pB[1], _pA, vl);
                _sum2 = vfmacc_vf_f16m1(_sum2, pB[2], _pA, vl);
                _sum3 = vfmacc_vf_f16m1(_sum3, pB[3], _pA, vl);
                _sum4 = vfmacc_vf_f16m1(_sum4, pB[4], _pA, vl);
                _sum5 = vfmacc_vf_f16m1(_sum5, pB[5], _pA, vl);
                _sum6 = vfmacc_vf_f16m1(_sum6, pB[6], _pA, vl);
                _sum7 = vfmacc_vf_f16m1(_sum7, pB[7], _pA, vl);
                _sum8 = vfmacc_vf_f16m1(_sum8, pB[8], _pA, vl);
                _sum9 = vfmacc_vf_f16m1(_sum9, pB[9], _pA, vl);
                _suma = vfmacc_vf_f16m1(_suma, pB[10], _pA, vl);
                _sumb = vfmacc_vf_f16m1(_sumb, pB[11], _pA, vl);

                pA += 4;
                pB += 12;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    // vst1_f16(outptr0, _sum0);
                    // vst1_f16(outptr0 + 4, _sum1);
                    // vst1_f16(outptr0 + 4 * 2, _sum2);
                    // vst1_f16(outptr0 + 4 * 3, _sum3);
                    // vst1_f16(outptr0 + 4 * 4, _sum4);
                    // vst1_f16(outptr0 + 4 * 5, _sum5);
                    // vst1_f16(outptr0 + 4 * 6, _sum6);
                    // vst1_f16(outptr0 + 4 * 7, _sum7);
                    // vst1_f16(outptr0 + 4 * 8, _sum8);
                    // vst1_f16(outptr0 + 4 * 9, _sum9);
                    // vst1_f16(outptr0 + 4 * 10, _suma);
                    // vst1_f16(outptr0 + 4 * 11, _sumb);
                    vse16_v_f16m1(outptr0, _sum0, vl);
                    vse16_v_f16m1(outptr0 + 4, _sum1, vl);
                    vse16_v_f16m1(outptr0 + 4 * 2, _sum2, vl);
                    vse16_v_f16m1(outptr0 + 4 * 3, _sum3, vl);
                    vse16_v_f16m1(outptr0 + 4 * 4, _sum4, vl);
                    vse16_v_f16m1(outptr0 + 4 * 5, _sum5, vl);
                    vse16_v_f16m1(outptr0 + 4 * 6, _sum6, vl);
                    vse16_v_f16m1(outptr0 + 4 * 7, _sum7, vl);
                    vse16_v_f16m1(outptr0 + 4 * 8, _sum8, vl);
                    vse16_v_f16m1(outptr0 + 4 * 9, _sum9, vl);
                    vse16_v_f16m1(outptr0 + 4 * 10, _suma, vl);
                    vse16_v_f16m1(outptr0 + 4 * 11, _sumb, vl);
                    outptr0 += 48;
                }
                if (out_elempack == 1)
                {
                    // transpose4x12_ph(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7, _sum8, _sum9, _suma, _sumb);

                    // vst1_f16(outptr0, _sum0);
                    // vst1_f16(outptr0 + 4, _sum1);
                    // vst1_f16(outptr0 + 8, _sum2);
                    // vst1_f16(outptr0 + out_hstep, _sum3);
                    // vst1_f16(outptr0 + out_hstep + 4, _sum4);
                    // vst1_f16(outptr0 + out_hstep + 8, _sum5);
                    // vst1_f16(outptr0 + out_hstep * 2, _sum6);
                    // vst1_f16(outptr0 + out_hstep * 2 + 4, _sum7);
                    // vst1_f16(outptr0 + out_hstep * 2 + 8, _sum8);
                    // vst1_f16(outptr0 + out_hstep * 3, _sum9);
                    // vst1_f16(outptr0 + out_hstep * 3 + 4, _suma);
                    // vst1_f16(outptr0 + out_hstep * 3 + 8, _sumb);
                    vsse16_v_f16m1(outptr0, out_hstep * sizeof(__fp16), _sum0, vl);
                    vsse16_v_f16m1(outptr0 + 1, out_hstep * sizeof(__fp16), _sum1, vl);
                    vsse16_v_f16m1(outptr0 + 2, out_hstep * sizeof(__fp16), _sum2, vl);
                    vsse16_v_f16m1(outptr0 + 3, out_hstep * sizeof(__fp16), _sum3, vl);
                    vsse16_v_f16m1(outptr0 + 4, out_hstep * sizeof(__fp16), _sum4, vl);
                    vsse16_v_f16m1(outptr0 + 5, out_hstep * sizeof(__fp16), _sum5, vl);
                    vsse16_v_f16m1(outptr0 + 6, out_hstep * sizeof(__fp16), _sum6, vl);
                    vsse16_v_f16m1(outptr0 + 7, out_hstep * sizeof(__fp16), _sum7, vl);
                    vsse16_v_f16m1(outptr0 + 8, out_hstep * sizeof(__fp16), _sum8, vl);
                    vsse16_v_f16m1(outptr0 + 9, out_hstep * sizeof(__fp16), _sum9, vl);
                    vsse16_v_f16m1(outptr0 + 10, out_hstep * sizeof(__fp16), _suma, vl);
                    vsse16_v_f16m1(outptr0 + 11, out_hstep * sizeof(__fp16), _sumb, vl);
                    outptr0 += 12;
                }
            }
            else
            {
                // vst1_f16(outptr, _sum0);
                // vst1_f16(outptr + 4, _sum1);
                // vst1_f16(outptr + 4 * 2, _sum2);
                // vst1_f16(outptr + 4 * 3, _sum3);
                // vst1_f16(outptr + 4 * 4, _sum4);
                // vst1_f16(outptr + 4 * 5, _sum5);
                // vst1_f16(outptr + 4 * 6, _sum6);
                // vst1_f16(outptr + 4 * 7, _sum7);
                // vst1_f16(outptr + 4 * 8, _sum8);
                // vst1_f16(outptr + 4 * 9, _sum9);
                // vst1_f16(outptr + 4 * 10, _suma);
                // vst1_f16(outptr + 4 * 11, _sumb);
                vse16_v_f16m1(outptr, _sum0, vl);
                vse16_v_f16m1(outptr + 4, _sum1, vl);
                vse16_v_f16m1(outptr + 4 * 2, _sum2, vl);
                vse16_v_f16m1(outptr + 4 * 3, _sum3, vl);
                vse16_v_f16m1(outptr + 4 * 4, _sum4, vl);
                vse16_v_f16m1(outptr + 4 * 5, _sum5, vl);
                vse16_v_f16m1(outptr + 4 * 6, _sum6, vl);
                vse16_v_f16m1(outptr + 4 * 7, _sum7, vl);
                vse16_v_f16m1(outptr + 4 * 8, _sum8, vl);
                vse16_v_f16m1(outptr + 4 * 9, _sum9, vl);
                vse16_v_f16m1(outptr + 4 * 10, _suma, vl);
                vse16_v_f16m1(outptr + 4 * 11, _sumb, vl);
            }

            outptr += 48;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            // float16x4_t _sum0;
            // float16x4_t _sum1;
            // float16x4_t _sum2;
            // float16x4_t _sum3;
            // float16x4_t _sum4;
            // float16x4_t _sum5;
            // float16x4_t _sum6;
            // float16x4_t _sum7;
            vfloat16m1_t _sum0;
            vfloat16m1_t _sum1;
            vfloat16m1_t _sum2;
            vfloat16m1_t _sum3;
            vfloat16m1_t _sum4;
            vfloat16m1_t _sum5;
            vfloat16m1_t _sum6;
            vfloat16m1_t _sum7;

            if (k == 0)
            {
                if (pC)
                {
                    // _sum0 = vld1_f16(pC);
                    // _sum1 = _sum0;
                    // _sum2 = _sum0;
                    // _sum3 = _sum0;
                    // _sum4 = _sum0;
                    // _sum5 = _sum0;
                    // _sum6 = _sum0;
                    // _sum7 = _sum0;
                    _sum0 = vle16_v_f16m1(pC, vl);
                    _sum1 = _sum0;
                    _sum2 = _sum0;
                    _sum3 = _sum0;
                    _sum4 = _sum0;
                    _sum5 = _sum0;
                    _sum6 = _sum0;
                    _sum7 = _sum0;
                }
                else
                {
                    // _sum0 = vdup_n_f16(0.f);
                    // _sum1 = vdup_n_f16(0.f);
                    // _sum2 = vdup_n_f16(0.f);
                    // _sum3 = vdup_n_f16(0.f);
                    // _sum4 = vdup_n_f16(0.f);
                    // _sum5 = vdup_n_f16(0.f);
                    // _sum6 = vdup_n_f16(0.f);
                    // _sum7 = vdup_n_f16(0.f);
                    _sum0 = vfmv_v_f_f16m1(0.f, vl);
                    _sum1 = vfmv_v_f_f16m1(0.f, vl);
                    _sum2 = vfmv_v_f_f16m1(0.f, vl);
                    _sum3 = vfmv_v_f_f16m1(0.f, vl);
                    _sum4 = vfmv_v_f_f16m1(0.f, vl);
                    _sum5 = vfmv_v_f_f16m1(0.f, vl);
                    _sum6 = vfmv_v_f_f16m1(0.f, vl);
                    _sum7 = vfmv_v_f_f16m1(0.f, vl);
                }
            }
            else
            {
                // _sum0 = vld1_f16(outptr);
                // _sum1 = vld1_f16(outptr + 4 * 1);
                // _sum2 = vld1_f16(outptr + 4 * 2);
                // _sum3 = vld1_f16(outptr + 4 * 3);
                // _sum4 = vld1_f16(outptr + 4 * 4);
                // _sum5 = vld1_f16(outptr + 4 * 5);
                // _sum6 = vld1_f16(outptr + 4 * 6);
                // _sum7 = vld1_f16(outptr + 4 * 7);
                _sum0 = vle16_v_f16m1(outptr, vl);
                _sum1 = vle16_v_f16m1(outptr + 4, vl);
                _sum2 = vle16_v_f16m1(outptr + 4 * 2, vl);
                _sum3 = vle16_v_f16m1(outptr + 4 * 3, vl);
                _sum4 = vle16_v_f16m1(outptr + 4 * 4, vl);
                _sum5 = vle16_v_f16m1(outptr + 4 * 5, vl);
                _sum6 = vle16_v_f16m1(outptr + 4 * 6, vl);
                _sum7 = vle16_v_f16m1(outptr + 4 * 7, vl);
            }

            const __fp16* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                // float16x4_t _pA = vld1_f16(pA);
                // float16x4_t _pB0 = vld1_f16(pB);
                // float16x4_t _pB1 = vld1_f16(pB + 4);

                // _sum0 = vfma_lane_f16(_sum0, _pA, _pB0, 0);
                // _sum1 = vfma_lane_f16(_sum1, _pA, _pB0, 1);
                // _sum2 = vfma_lane_f16(_sum2, _pA, _pB0, 2);
                // _sum3 = vfma_lane_f16(_sum3, _pA, _pB0, 3);
                // _sum4 = vfma_lane_f16(_sum4, _pA, _pB1, 0);
                // _sum5 = vfma_lane_f16(_sum5, _pA, _pB1, 1);
                // _sum6 = vfma_lane_f16(_sum6, _pA, _pB1, 2);
                // _sum7 = vfma_lane_f16(_sum7, _pA, _pB1, 3);
                vfloat16m1_t _pA = vle16_v_f16m1(pA, vl);
                _sum0 = vfmacc_vf_f16m1(_sum0, pB[0], _pA, vl);
                _sum1 = vfmacc_vf_f16m1(_sum1, pB[1], _pA, vl);
                _sum2 = vfmacc_vf_f16m1(_sum2, pB[2], _pA, vl);
                _sum3 = vfmacc_vf_f16m1(_sum3, pB[3], _pA, vl);
                _sum4 = vfmacc_vf_f16m1(_sum4, pB[4], _pA, vl);
                _sum5 = vfmacc_vf_f16m1(_sum5, pB[5], _pA, vl);
                _sum6 = vfmacc_vf_f16m1(_sum6, pB[6], _pA, vl);
                _sum7 = vfmacc_vf_f16m1(_sum7, pB[7], _pA, vl);

                pA += 4;
                pB += 8;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    // vst1_f16(outptr0, _sum0);
                    // vst1_f16(outptr0 + 4, _sum1);
                    // vst1_f16(outptr0 + 4 * 2, _sum2);
                    // vst1_f16(outptr0 + 4 * 3, _sum3);
                    // vst1_f16(outptr0 + 4 * 4, _sum4);
                    // vst1_f16(outptr0 + 4 * 5, _sum5);
                    // vst1_f16(outptr0 + 4 * 6, _sum6);
                    // vst1_f16(outptr0 + 4 * 7, _sum7);
                    vse16_v_f16m1(outptr0, _sum0, vl);
                    vse16_v_f16m1(outptr0 + 4, _sum1, vl);
                    vse16_v_f16m1(outptr0 + 4 * 2, _sum2, vl);
                    vse16_v_f16m1(outptr0 + 4 * 3, _sum3, vl);
                    vse16_v_f16m1(outptr0 + 4 * 4, _sum4, vl);
                    vse16_v_f16m1(outptr0 + 4 * 5, _sum5, vl);
                    vse16_v_f16m1(outptr0 + 4 * 6, _sum6, vl);
                    vse16_v_f16m1(outptr0 + 4 * 7, _sum7, vl);
                    outptr0 += 32;
                }
                if (out_elempack == 1)
                {
                    vsse16_v_f16m1(outptr0, out_hstep * sizeof(__fp16), _sum0, vl);
                    vsse16_v_f16m1(outptr0 + 1, out_hstep * sizeof(__fp16), _sum1, vl);
                    vsse16_v_f16m1(outptr0 + 2, out_hstep * sizeof(__fp16), _sum2, vl);
                    vsse16_v_f16m1(outptr0 + 3, out_hstep * sizeof(__fp16), _sum3, vl);
                    vsse16_v_f16m1(outptr0 + 4, out_hstep * sizeof(__fp16), _sum4, vl);
                    vsse16_v_f16m1(outptr0 + 5, out_hstep * sizeof(__fp16), _sum5, vl);
                    vsse16_v_f16m1(outptr0 + 6, out_hstep * sizeof(__fp16), _sum6, vl);
                    vsse16_v_f16m1(outptr0 + 7, out_hstep * sizeof(__fp16), _sum7, vl);
                    // transpose4x8_ph(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7);

                    // vst1_f16(outptr0, _sum0);
                    // vst1_f16(outptr0 + 4, _sum1);
                    // vst1_f16(outptr0 + out_hstep, _sum2);
                    // vst1_f16(outptr0 + out_hstep + 4, _sum3);
                    // vst1_f16(outptr0 + out_hstep * 2, _sum4);
                    // vst1_f16(outptr0 + out_hstep * 2 + 4, _sum5);
                    // vst1_f16(outptr0 + out_hstep * 3, _sum6);
                    // vst1_f16(outptr0 + out_hstep * 3 + 4, _sum7);
                    outptr0 += 8;
                }
            }
            else
            {
                vse16_v_f16m1(outptr, _sum0, vl);
                vse16_v_f16m1(outptr + 4, _sum1, vl);
                vse16_v_f16m1(outptr + 4 * 2, _sum2, vl);
                vse16_v_f16m1(outptr + 4 * 3, _sum3, vl);
                vse16_v_f16m1(outptr + 4 * 4, _sum4, vl);
                vse16_v_f16m1(outptr + 4 * 5, _sum5, vl);
                vse16_v_f16m1(outptr + 4 * 6, _sum6, vl);
                vse16_v_f16m1(outptr + 4 * 7, _sum7, vl);
                // vst1_f16(outptr, _sum0);
                // vst1_f16(outptr + 4, _sum1);
                // vst1_f16(outptr + 4 * 2, _sum2);
                // vst1_f16(outptr + 4 * 3, _sum3);
                // vst1_f16(outptr + 4 * 4, _sum4);
                // vst1_f16(outptr + 4 * 5, _sum5);
                // vst1_f16(outptr + 4 * 6, _sum6);
                // vst1_f16(outptr + 4 * 7, _sum7);
            }

            outptr += 32;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            // float16x4_t _sum0;
            // float16x4_t _sum1;
            // float16x4_t _sum2;
            // float16x4_t _sum3;
            vfloat16m1_t _sum0;
            vfloat16m1_t _sum1;
            vfloat16m1_t _sum2;
            vfloat16m1_t _sum3;

            if (k == 0)
            {
                if (pC)
                {
                    // _sum0 = vld1_f16(pC);
                    // _sum1 = _sum0;
                    // _sum2 = _sum0;
                    // _sum3 = _sum0;
                    _sum0 = vle16_v_f16m1(pC, vl);
                    _sum1 = _sum0;
                    _sum2 = _sum0;
                    _sum3 = _sum0;
                }
                else
                {
                    // _sum0 = vdup_n_f16(0.f);
                    // _sum1 = vdup_n_f16(0.f);
                    // _sum2 = vdup_n_f16(0.f);
                    // _sum3 = vdup_n_f16(0.f);
                    _sum0 = vfmv_v_f_f16m1(0.f, vl);
                    _sum1 = vfmv_v_f_f16m1(0.f, vl);
                    _sum2 = vfmv_v_f_f16m1(0.f, vl);
                    _sum3 = vfmv_v_f_f16m1(0.f, vl);
                }
            }
            else
            {
                // _sum0 = vld1_f16(outptr);
                // _sum1 = vld1_f16(outptr + 4);
                // _sum2 = vld1_f16(outptr + 4 * 2);
                // _sum3 = vld1_f16(outptr + 4 * 3);
                _sum0 = vle16_v_f16m1(outptr, vl);
                _sum1 = vle16_v_f16m1(outptr + 4, vl);
                _sum2 = vle16_v_f16m1(outptr + 4 * 2, vl);
                _sum3 = vle16_v_f16m1(outptr + 4 * 3, vl);
            }

            const __fp16* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                // float16x4_t _pA = vld1_f16(pA);
                // float16x4_t _pB = vld1_f16(pB);

                // _sum0 = vfma_lane_f16(_sum0, _pA, _pB, 0);
                // _sum1 = vfma_lane_f16(_sum1, _pA, _pB, 1);
                // _sum2 = vfma_lane_f16(_sum2, _pA, _pB, 2);
                // _sum3 = vfma_lane_f16(_sum3, _pA, _pB, 3);
                vfloat16m1_t _pA = vle16_v_f16m1(pA, vl);
                _sum0 = vfmacc_vf_f16m1(_sum0, pB[0], _pA, vl);
                _sum1 = vfmacc_vf_f16m1(_sum1, pB[1], _pA, vl);
                _sum2 = vfmacc_vf_f16m1(_sum2, pB[2], _pA, vl);
                _sum3 = vfmacc_vf_f16m1(_sum3, pB[3], _pA, vl);

                pA += 4;
                pB += 4;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    // vst1_f16(outptr0, _sum0);
                    // vst1_f16(outptr0 + 4, _sum1);
                    // vst1_f16(outptr0 + 4 * 2, _sum2);
                    // vst1_f16(outptr0 + 4 * 3, _sum3);
                    vse16_v_f16m1(outptr0, _sum0, vl);
                    vse16_v_f16m1(outptr0 + 4, _sum1, vl);
                    vse16_v_f16m1(outptr0 + 4 * 2, _sum2, vl);
                    vse16_v_f16m1(outptr0 + 4 * 3, _sum3, vl);
                    outptr0 += 16;
                }
                if (out_elempack == 1)
                {
                    vsse16_v_f16m1(outptr0, out_hstep * sizeof(__fp16), _sum0, vl);
                    vsse16_v_f16m1(outptr0 + 1, out_hstep * sizeof(__fp16), _sum1, vl);
                    vsse16_v_f16m1(outptr0 + 2, out_hstep * sizeof(__fp16), _sum2, vl);
                    vsse16_v_f16m1(outptr0 + 3, out_hstep * sizeof(__fp16), _sum3, vl);
                    // transpose4x4_ph(_sum0, _sum1, _sum2, _sum3);

                    // vst1_f16(outptr0, _sum0);
                    // vst1_f16(outptr0 + out_hstep, _sum1);
                    // vst1_f16(outptr0 + out_hstep * 2, _sum2);
                    // vst1_f16(outptr0 + out_hstep * 3, _sum3);
                    outptr0 += 4;
                }
            }
            else
            {
                // vst1_f16(outptr, _sum0);
                // vst1_f16(outptr + 4, _sum1);
                // vst1_f16(outptr + 4 * 2, _sum2);
                // vst1_f16(outptr + 4 * 3, _sum3);
                vse16_v_f16m1(outptr, _sum0, vl);
                vse16_v_f16m1(outptr + 4, _sum1, vl);
                vse16_v_f16m1(outptr + 4 * 2, _sum2, vl);
                vse16_v_f16m1(outptr + 4 * 3, _sum3, vl);
            }

            outptr += 16;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            // float16x4_t _sum0;
            // float16x4_t _sum1;
            vfloat16m1_t _sum0;
            vfloat16m1_t _sum1;

            if (k == 0)
            {
                if (pC)
                {
                    // _sum0 = vld1_f16(pC);
                    // _sum1 = _sum0;
                    _sum0 = vle16_v_f16m1(pC, vl);
                    _sum1 = _sum0;
                }
                else
                {
                    // _sum0 = vdup_n_f16(0.f);
                    // _sum1 = vdup_n_f16(0.f);
                    _sum0 = vfmv_v_f_f16m1(0.f, vl);
                    _sum1 = vfmv_v_f_f16m1(0.f, vl);
                }
            }
            else
            {
                // _sum0 = vld1_f16(outptr);
                // _sum1 = vld1_f16(outptr + 4);
                _sum0 = vle16_v_f16m1(outptr, vl);
                _sum1 = vle16_v_f16m1(outptr + 4, vl);
            }

            const __fp16* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                // float16x4_t _pA = vld1_f16(pA);

                // _sum0 = vfma_n_f16(_sum0, _pA, pB[0]);
                // _sum1 = vfma_n_f16(_sum1, _pA, pB[1]);
                vfloat16m1_t _pA = vle16_v_f16m1(pA, vl);
                _sum0 = vfmacc_vf_f16m1(_sum0, pB[0], _pA, vl);
                _sum1 = vfmacc_vf_f16m1(_sum1, pB[1], _pA, vl);

                pA += 4;
                pB += 2;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    // vst1_f16(outptr0, _sum0);
                    // vst1_f16(outptr0 + 4, _sum1);
                    vse16_v_f16m1(outptr0, _sum0, vl);
                    vse16_v_f16m1(outptr0 + 4, _sum1, vl);
                    outptr0 += 8;
                }
                if (out_elempack == 1)
                {
                    // __fp16 sum0[4];
                    // __fp16 sum1[4];
                    // vst1_f16(sum0, _sum0);
                    // vst1_f16(sum1, _sum1);

                    // outptr0[0] = sum0[0];
                    // outptr0[out_hstep] = sum0[1];
                    // outptr0[out_hstep * 2] = sum0[2];
                    // outptr0[out_hstep * 3] = sum0[3];
                    // outptr0[1] = sum1[0];
                    // outptr0[out_hstep + 1] = sum1[1];
                    // outptr0[out_hstep * 2 + 1] = sum1[2];
                    // outptr0[out_hstep * 3 + 1] = sum1[3];
                    vsse16_v_f16m1(outptr0, out_hstep * sizeof(__fp16), _sum0, vl);
                    vsse16_v_f16m1(outptr0 + 1, out_hstep * sizeof(__fp16), _sum1, vl);
                    outptr0 += 2;
                }
            }
            else
            {
                // vst1_f16(outptr, _sum0);
                // vst1_f16(outptr + 4, _sum1);
                vse16_v_f16m1(outptr, _sum0, vl);
                vse16_v_f16m1(outptr + 4, _sum1, vl);
            }

            outptr += 8;
        }
        for (; jj < max_jj; jj += 1)
        {
            // float16x4_t _sum0;
            vfloat16m1_t _sum0;

            if (k == 0)
            {
                if (pC)
                {
                    // _sum0 = vld1_f16(pC);
                    _sum0 = vle16_v_f16m1(pC, vl);
                }
                else
                {
                    // _sum0 = vdup_n_f16(0.f);
                    _sum0 = vfmv_v_f_f16m1(0.f, vl);
                }
            }
            else
            {
                // _sum0 = vld1_f16(outptr);
                _sum0 = vle16_v_f16m1(outptr, vl);
            }

            const __fp16* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat16m1_t _pA = vle16_v_f16m1(pA, vl);
                _sum0 = vfmacc_vf_f16m1(_sum0, pB[0], _pA, vl);
                // float16x4_t _pA = vld1_f16(pA);
                // float16x4_t _pB = vdup_n_f16(pB[0]);

                // _sum0 = vfma_f16(_sum0, _pA, _pB);

                pA += 4;
                pB += 1;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    // vst1_f16(outptr0, _sum0);
                    vse16_v_f16m1(outptr0, _sum0, vl);
                    outptr0 += 4;
                }
                if (out_elempack == 1)
                {
                    vsse16_v_f16m1(outptr0, out_hstep * sizeof(__fp16), _sum0, vl);
                    // __fp16 sum0[4];
                    // vst1_f16(sum0, _sum0);

                    // outptr0[0] = sum0[0];
                    // outptr0[out_hstep] = sum0[1];
                    // outptr0[out_hstep * 2] = sum0[2];
                    // outptr0[out_hstep * 3] = sum0[3];
                    outptr0++;
                }
            }
            else
            {
                // vst1_f16(outptr, _sum0);
                vse16_v_f16m1(outptr, _sum0, vl);
            }

            outptr += 4;
        }

        pAT += max_kk * 4;
    }
    for (; ii + 1 < max_ii; ii += 2)
    {
        vl = 4;
        __fp16* outptr0 = (__fp16*)top_blob + (i + ii) * out_hstep + j;

        const __fp16* pB = pBT;

        if (pC)
        {
            pC = (const __fp16*)CT_tile + i + ii;
        }

        int jj = 0;
        for (; jj + 11 < max_jj; jj += 12)
        {
            // float16x4_t _sum00;
            // float16x4_t _sum01;
            // float16x4_t _sum02;
            // float16x4_t _sum10;
            // float16x4_t _sum11;
            // float16x4_t _sum12;
            vfloat16m1_t _sum00;
            vfloat16m1_t _sum01;
            vfloat16m1_t _sum02;
            vfloat16m1_t _sum10;
            vfloat16m1_t _sum11;
            vfloat16m1_t _sum12;

            if (k == 0)
            {
                if (pC)
                {
                    // _sum00 = vdup_n_f16(pC[0]);
                    // _sum01 = vdup_n_f16(pC[0]);
                    // _sum02 = vdup_n_f16(pC[0]);
                    // _sum10 = vdup_n_f16(pC[1]);
                    // _sum11 = vdup_n_f16(pC[1]);
                    // _sum12 = vdup_n_f16(pC[1]);
                    _sum00 = vfmv_v_f_f16m1(pC[0], vl);
                    _sum01 = vfmv_v_f_f16m1(pC[0], vl);
                    _sum02 = vfmv_v_f_f16m1(pC[0], vl);
                    _sum10 = vfmv_v_f_f16m1(pC[1], vl);
                    _sum11 = vfmv_v_f_f16m1(pC[1], vl);
                    _sum12 = vfmv_v_f_f16m1(pC[1], vl);
                }
                else
                {
                    // _sum00 = vdup_n_f16(0.f);
                    // _sum01 = vdup_n_f16(0.f);
                    // _sum02 = vdup_n_f16(0.f);
                    // _sum10 = vdup_n_f16(0.f);
                    // _sum11 = vdup_n_f16(0.f);
                    // _sum12 = vdup_n_f16(0.f);
                    _sum00 = vfmv_v_f_f16m1(0.f, vl);
                    _sum01 = vfmv_v_f_f16m1(0.f, vl);
                    _sum02 = vfmv_v_f_f16m1(0.f, vl);
                    _sum10 = vfmv_v_f_f16m1(0.f, vl);
                    _sum11 = vfmv_v_f_f16m1(0.f, vl);
                    _sum12 = vfmv_v_f_f16m1(0.f, vl);
                }
            }
            else
            {
                // float16x4x2_t _tmp01 = vld2_f16(outptr);
                // float16x4x2_t _tmp23 = vld2_f16(outptr + 8);
                // float16x4x2_t _tmp45 = vld2_f16(outptr + 16);
                // _sum00 = _tmp01.val[0];
                // _sum01 = _tmp23.val[0];
                // _sum02 = _tmp45.val[0];
                // _sum10 = _tmp01.val[1];
                // _sum11 = _tmp23.val[1];
                // _sum12 = _tmp45.val[1];
                vlseg2e16_v_f16m1(&_sum00, &_sum10, outptr, vl);
                vlseg2e16_v_f16m1(&_sum01, &_sum11, outptr + 8, vl);
                vlseg2e16_v_f16m1(&_sum02, &_sum12, outptr + 16, vl);
            }

            const __fp16* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                // float16x4_t _pB0 = vld1_f16(pB);
                // float16x4_t _pB1 = vld1_f16(pB + 4);
                // float16x4_t _pB2 = vld1_f16(pB + 8);

                // float16x4_t _pA0 = vld1_dup_f16(pA);
                // float16x4_t _pA1 = vld1_dup_f16(pA + 1);

                // _sum00 = vfma_f16(_sum00, _pB0, _pA0);
                // _sum01 = vfma_f16(_sum01, _pB1, _pA0);
                // _sum02 = vfma_f16(_sum02, _pB2, _pA0);
                // _sum10 = vfma_f16(_sum10, _pB0, _pA1);
                // _sum11 = vfma_f16(_sum11, _pB1, _pA1);
                // _sum12 = vfma_f16(_sum12, _pB2, _pA1);
                vfloat16m1_t _pB0 = vle16_v_f16m1(pB, vl);
                vfloat16m1_t _pB1 = vle16_v_f16m1(pB + 4, vl);
                vfloat16m1_t _pB2 = vle16_v_f16m1(pB + 8, vl);

                _sum00 = vfmacc_vf_f16m1(_sum00, pA[0], _pB0, vl);
                _sum01 = vfmacc_vf_f16m1(_sum01, pA[0], _pB1, vl);
                _sum02 = vfmacc_vf_f16m1(_sum02, pA[0], _pB2, vl);
                _sum10 = vfmacc_vf_f16m1(_sum10, pA[1], _pB0, vl);
                _sum11 = vfmacc_vf_f16m1(_sum11, pA[1], _pB1, vl);
                _sum12 = vfmacc_vf_f16m1(_sum12, pA[1], _pB2, vl);

                pA += 2;
                pB += 12;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    // vst1_f16(outptr0, _sum00);
                    // vst1_f16(outptr0 + 4, _sum01);
                    // vst1_f16(outptr0 + 8, _sum02);
                    // vst1_f16(outptr0 + out_hstep, _sum10);
                    // vst1_f16(outptr0 + out_hstep + 4, _sum11);
                    // vst1_f16(outptr0 + out_hstep + 8, _sum12);
                    vse16_v_f16m1(outptr0, _sum00, vl);
                    vse16_v_f16m1(outptr0 + 4, _sum01, vl);
                    vse16_v_f16m1(outptr0 + 8, _sum02, vl);
                    vse16_v_f16m1(outptr0 + out_hstep, _sum10, vl);
                    vse16_v_f16m1(outptr0 + out_hstep + 4, _sum11, vl);
                    vse16_v_f16m1(outptr0 + out_hstep + 8, _sum12, vl);
                    outptr0 += 12;
                }
            }
            else
            {
                vsseg2e16_v_f16m1(outptr, _sum00, _sum10, vl);
                vsseg2e16_v_f16m1(outptr + 8, _sum01, _sum11, vl);
                vsseg2e16_v_f16m1(outptr + 16, _sum02, _sum12, vl);
                // float16x4x2_t _tmp01;
                // _tmp01.val[0] = _sum00;
                // _tmp01.val[1] = _sum10;
                // float16x4x2_t _tmp23;
                // _tmp23.val[0] = _sum01;
                // _tmp23.val[1] = _sum11;
                // float16x4x2_t _tmp45;
                // _tmp45.val[0] = _sum02;
                // _tmp45.val[1] = _sum12;
                // vst2_f16(outptr, _tmp01);
                // vst2_f16(outptr + 8, _tmp23);
                // vst2_f16(outptr + 16, _tmp45);
            }

            outptr += 24;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            // float16x4_t _sum00;
            // float16x4_t _sum01;
            // float16x4_t _sum10;
            // float16x4_t _sum11;
            vfloat16m1_t _sum00;
            vfloat16m1_t _sum01;
            vfloat16m1_t _sum10;
            vfloat16m1_t _sum11;

            if (k == 0)
            {
                if (pC)
                {
                    // _sum00 = vdup_n_f16(pC[0]);
                    // _sum01 = vdup_n_f16(pC[0]);
                    // _sum10 = vdup_n_f16(pC[1]);
                    // _sum11 = vdup_n_f16(pC[1]);
                    _sum00 = vfmv_v_f_f16m1(pC[0], vl);
                    _sum01 = vfmv_v_f_f16m1(pC[0], vl);
                    _sum10 = vfmv_v_f_f16m1(pC[1], vl);
                    _sum11 = vfmv_v_f_f16m1(pC[1], vl);
                }
                else
                {
                    // _sum00 = vdup_n_f16(0.f);
                    // _sum01 = vdup_n_f16(0.f);
                    // _sum10 = vdup_n_f16(0.f);
                    // _sum11 = vdup_n_f16(0.f);
                    _sum00 = vfmv_v_f_f16m1(0.f, vl);
                    _sum01 = vfmv_v_f_f16m1(0.f, vl);
                    _sum10 = vfmv_v_f_f16m1(0.f, vl);
                    _sum11 = vfmv_v_f_f16m1(0.f, vl);
                }
            }
            else
            {
                // float16x4x2_t _tmp01 = vld2_f16(outptr);
                // float16x4x2_t _tmp23 = vld2_f16(outptr + 8);
                // _sum00 = _tmp01.val[0];
                // _sum01 = _tmp23.val[0];
                // _sum10 = _tmp01.val[1];
                // _sum11 = _tmp23.val[1];
                vlseg2e16_v_f16m1(&_sum00, &_sum10, outptr, vl);
                vlseg2e16_v_f16m1(&_sum01, &_sum11, outptr + 8, vl);
            }

            const __fp16* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                // float16x4_t _pB0 = vld1_f16(pB);
                // float16x4_t _pB1 = vld1_f16(pB + 4);

                // float16x4_t _pA0 = vld1_dup_f16(pA);
                // float16x4_t _pA1 = vld1_dup_f16(pA + 1);

                // _sum00 = vfma_f16(_sum00, _pB0, _pA0);
                // _sum01 = vfma_f16(_sum01, _pB1, _pA0);
                // _sum10 = vfma_f16(_sum10, _pB0, _pA1);
                // _sum11 = vfma_f16(_sum11, _pB1, _pA1);

                vfloat16m1_t _pB0 = vle16_v_f16m1(pB, vl);
                vfloat16m1_t _pB1 = vle16_v_f16m1(pB + 4, vl);

                _sum00 = vfmacc_vf_f16m1(_sum00, pA[0], _pB0, vl);
                _sum01 = vfmacc_vf_f16m1(_sum01, pA[0], _pB1, vl);
                _sum10 = vfmacc_vf_f16m1(_sum10, pA[1], _pB0, vl);
                _sum11 = vfmacc_vf_f16m1(_sum11, pA[1], _pB1, vl);

                pA += 2;
                pB += 8;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    // vst1_f16(outptr0, _sum00);
                    // vst1_f16(outptr0 + 4, _sum01);
                    // vst1_f16(outptr0 + out_hstep, _sum10);
                    // vst1_f16(outptr0 + out_hstep + 4, _sum11);
                    vse16_v_f16m1(outptr0, _sum00, vl);
                    vse16_v_f16m1(outptr0 + 4, _sum01, vl);
                    vse16_v_f16m1(outptr0 + out_hstep, _sum10, vl);
                    vse16_v_f16m1(outptr0 + out_hstep + 4, _sum11, vl);
                    outptr0 += 8;
                }
            }
            else
            {
                vsseg2e16_v_f16m1(outptr, _sum00, _sum10, vl);
                vsseg2e16_v_f16m1(outptr + 8, _sum01, _sum11, vl);
                // float16x4x2_t _tmp01;
                // _tmp01.val[0] = _sum00;
                // _tmp01.val[1] = _sum10;
                // float16x4x2_t _tmp23;
                // _tmp23.val[0] = _sum01;
                // _tmp23.val[1] = _sum11;
                // vst2_f16(outptr, _tmp01);
                // vst2_f16(outptr + 8, _tmp23);
            }

            outptr += 16;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            // float16x4_t _sum0;
            // float16x4_t _sum1;
            vfloat16m1_t _sum0;
            vfloat16m1_t _sum1;

            if (k == 0)
            {
                if (pC)
                {
                    // _sum0 = vdup_n_f16(pC[0]);
                    // _sum1 = vdup_n_f16(pC[1]);
                    _sum0 = vfmv_v_f_f16m1(pC[0], vl);
                    _sum1 = vfmv_v_f_f16m1(pC[1], vl);
                }
                else
                {
                    // _sum0 = vdup_n_f16(0.f);
                    // _sum1 = vdup_n_f16(0.f);
                    _sum0 = vfmv_v_f_f16m1(0.f, vl);
                    _sum1 = vfmv_v_f_f16m1(0.f, vl);
                }
            }
            else
            {
                vlseg2e16_v_f16m1(&_sum0, &_sum1, outptr, vl);
                // float16x4x2_t _tmp01 = vld2_f16(outptr);
                // _sum0 = _tmp01.val[0];
                // _sum1 = _tmp01.val[1];
            }

            const __fp16* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                // float16x4_t _pB = vld1_f16(pB);

                // _sum0 = vfma_n_f16(_sum0, _pB, pA[0]);
                // _sum1 = vfma_n_f16(_sum1, _pB, pA[1]);
                vfloat16m1_t _pB = vle16_v_f16m1(pB, vl);
                _sum0 = vfmacc_vf_f16m1(_sum0, pA[0], _pB, vl);
                _sum1 = vfmacc_vf_f16m1(_sum1, pA[1], _pB, vl);

                pA += 2;
                pB += 4;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    // vst1_f16(outptr0, (_sum0));
                    // vst1_f16(outptr0 + out_hstep, (_sum1));
                    vse16_v_f16m1(outptr0, _sum0, vl);
                    vse16_v_f16m1(outptr0 + out_hstep, _sum1, vl);
                    outptr0 += 4;
                }
            }
            else
            {
                vsseg2e16_v_f16m1(outptr, _sum0, _sum1, vl);
                // float16x4x2_t _tmp01;
                // _tmp01.val[0] = _sum0;
                // _tmp01.val[1] = _sum1;
                // vst2_f16(outptr, _tmp01);
            }

            outptr += 8;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            __fp16 sum00;
            __fp16 sum01;
            __fp16 sum10;
            __fp16 sum11;

            if (k == 0)
            {
                if (pC)
                {
                    sum00 = pC[0];
                    sum01 = pC[1];
                    sum10 = pC[0];
                    sum11 = pC[1];
                }
                else
                {
                    sum00 = 0.f;
                    sum01 = 0.f;
                    sum10 = 0.f;
                    sum11 = 0.f;
                }
            }
            else
            {
                sum00 = outptr[0];
                sum01 = outptr[1];
                sum10 = outptr[2];
                sum11 = outptr[3];
            }

            const __fp16* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum00 += pA[0] * pB[0];
                sum01 += pA[1] * pB[0];
                sum10 += pA[0] * pB[1];
                sum11 += pA[1] * pB[1];

                pA += 2;
                pB += 2;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = sum00;
                    outptr0[1] = sum10;
                    outptr0[out_hstep] = sum01;
                    outptr0[out_hstep + 1] = sum11;
                    outptr0 += 2;
                }
            }
            else
            {
                outptr[0] = sum00;
                outptr[1] = sum01;
                outptr[2] = sum10;
                outptr[3] = sum11;
            }

            outptr += 4;
        }
        for (; jj < max_jj; jj += 1)
        {
            __fp16 sum0;
            __fp16 sum1;

            if (k == 0)
            {
                if (pC)
                {
                    sum0 = pC[0];
                    sum1 = pC[1];
                }
                else
                {
                    sum0 = 0.f;
                    sum1 = 0.f;
                }
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
            }

            const __fp16* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum0 += pA[0] * pB[0];
                sum1 += pA[1] * pB[0];
                pA += 2;
                pB += 1;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = sum0;
                    outptr0[out_hstep] = sum1;
                    outptr0++;
                }
            }
            else
            {
                outptr[0] = sum0;
                outptr[1] = sum1;
            }

            outptr += 2;
        }

        pAT += max_kk * 2;
    }
    for (; ii < max_ii; ii += 1)
    {
        vl = 4;
        __fp16* outptr0 = (__fp16*)top_blob + (i + ii) * out_hstep + j;

        const __fp16* pB = pBT;

        if (pC)
        {
            pC = (const __fp16*)CT_tile + i + ii;
        }

        int jj = 0;
        for (; jj + 11 < max_jj; jj += 12)
        {
            // float16x4_t _sum0;
            // float16x4_t _sum1;
            // float16x4_t _sum2;
            vfloat16m1_t _sum0;
            vfloat16m1_t _sum1;
            vfloat16m1_t _sum2;

            if (k == 0)
            {
                if (pC)
                {
                    // _sum0 = vdup_n_f16(pC[0]);
                    // _sum1 = vdup_n_f16(pC[0]);
                    // _sum2 = vdup_n_f16(pC[0]);
                    _sum0 = vfmv_v_f_f16m1(pC[0], vl);
                    _sum1 = vfmv_v_f_f16m1(pC[0], vl);
                    _sum2 = vfmv_v_f_f16m1(pC[0], vl);
                }
                else
                {
                    // _sum0 = vdup_n_f16(0.f);
                    // _sum1 = vdup_n_f16(0.f);
                    // _sum2 = vdup_n_f16(0.f);
                    _sum0 = vfmv_v_f_f16m1(0.f, vl);
                    _sum1 = vfmv_v_f_f16m1(0.f, vl);
                    _sum2 = vfmv_v_f_f16m1(0.f, vl);
                }
            }
            else
            {
                // _sum0 = vld1_f16(outptr);
                // _sum1 = vld1_f16(outptr + 4);
                // _sum2 = vld1_f16(outptr + 8);
                _sum0 = vle16_v_f16m1(outptr, vl);
                _sum1 = vle16_v_f16m1(outptr + 4, vl);
                _sum2 = vle16_v_f16m1(outptr + 8, vl);
            }

            const __fp16* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                // float16x4_t _pB0 = vld1_f16(pB);
                // float16x4_t _pB1 = vld1_f16(pB + 4);
                // float16x4_t _pB2 = vld1_f16(pB + 8);

                // float16x4_t _pA0 = vdup_n_f16(pA[0]);
                vfloat16m1_t _pB0 = vle16_v_f16m1(pB, vl);
                vfloat16m1_t _pB1 = vle16_v_f16m1(pB + 4, vl);
                vfloat16m1_t _pB2 = vle16_v_f16m1(pB + 8, vl);

                // _sum0 = vfma_f16(_sum0, _pA0, _pB0);
                // _sum1 = vfma_f16(_sum1, _pA0, _pB1);
                // _sum2 = vfma_f16(_sum2, _pA0, _pB2);
                _sum0 = vfmacc_vf_f16m1(_sum0, pA[0], _pB0, vl);
                _sum1 = vfmacc_vf_f16m1(_sum1, pA[0], _pB1, vl);
                _sum2 = vfmacc_vf_f16m1(_sum2, pA[0], _pB2, vl);

                pA += 1;
                pB += 12;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    // vst1_f16(outptr0, _sum0);
                    // vst1_f16(outptr0 + 4, _sum1);
                    // vst1_f16(outptr0 + 8, _sum2);
                    vse16_v_f16m1(outptr0, _sum0, vl);
                    vse16_v_f16m1(outptr0 + 4, _sum1, vl);
                    vse16_v_f16m1(outptr0 + 8, _sum2, vl);
                    outptr0 += 12;
                }
            }
            else
            {
                // vst1_f16(outptr, _sum0);
                // vst1_f16(outptr + 4, _sum1);
                // vst1_f16(outptr + 8, _sum2);
                vse16_v_f16m1(outptr, _sum0, vl);
                vse16_v_f16m1(outptr + 4, _sum1, vl);
                vse16_v_f16m1(outptr + 8, _sum2, vl);
            }

            outptr += 12;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            // float16x4_t _sum0;
            // float16x4_t _sum1;
            vfloat16m1_t _sum0;
            vfloat16m1_t _sum1;

            if (k == 0)
            {
                if (pC)
                {
                    // _sum0 = vdup_n_f16(pC[0]);
                    // _sum1 = vdup_n_f16(pC[0]);
                    _sum0 = vfmv_v_f_f16m1(pC[0], vl);
                    _sum1 = vfmv_v_f_f16m1(pC[0], vl);
                }
                else
                {
                    // _sum0 = vdup_n_f16(0.f);
                    // _sum1 = vdup_n_f16(0.f);
                    _sum0 = vfmv_v_f_f16m1(0.f, vl);
                    _sum1 = vfmv_v_f_f16m1(0.f, vl);
                }
            }
            else
            {
                // _sum0 = vld1_f16(outptr);
                // _sum1 = vld1_f16(outptr + 4);
                _sum0 = vle16_v_f16m1(outptr, vl);
                _sum1 = vle16_v_f16m1(outptr + 4, vl);
            }

            const __fp16* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                // float16x4_t _pB0 = vld1_f16(pB);
                // float16x4_t _pB1 = vld1_f16(pB + 4);

                // float16x4_t _pA0 = vdup_n_f16(pA[0]);

                // _sum0 = vfma_f16(_sum0, _pA0, _pB0);
                // _sum1 = vfma_f16(_sum1, _pA0, _pB1);
                vfloat16m1_t _pB0 = vle16_v_f16m1(pB, vl);
                vfloat16m1_t _pB1 = vle16_v_f16m1(pB + 4, vl);

                _sum0 = vfmacc_vf_f16m1(_sum0, pA[0], _pB0, vl);
                _sum1 = vfmacc_vf_f16m1(_sum1, pA[0], _pB1, vl);

                pA += 1;
                pB += 8;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    // vst1_f16(outptr0, _sum0);
                    // vst1_f16(outptr0 + 4, _sum1);
                    vse16_v_f16m1(outptr0, _sum0, vl);
                    vse16_v_f16m1(outptr0 + 4, _sum1, vl);
                    outptr0 += 8;
                }
            }
            else
            {
                vse16_v_f16m1(outptr, _sum0, vl);
                vse16_v_f16m1(outptr + 4, _sum1, vl);
                // vst1_f16(outptr, _sum0);
                // vst1_f16(outptr + 4, _sum1);
            }

            outptr += 8;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            // float16x4_t _sum;
            vfloat16m1_t _sum;

            if (k == 0)
            {
                if (pC)
                {
                    // _sum = vdup_n_f16(pC[0]);
                    _sum = vfmv_v_f_f16m1(pC[0], vl);
                }
                else
                {
                    // _sum = vdup_n_f16(0.f);
                    _sum = vfmv_v_f_f16m1(0.f, vl);
                }
            }
            else
            {
                // _sum = vld1_f16(outptr);
                _sum = vle16_v_f16m1(outptr, vl);
            }

            const __fp16* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                // float16x4_t _pB = vld1_f16(pB);
                // float16x4_t _pA = vdup_n_f16(pA[0]);
                vfloat16m1_t _pB = vle16_v_f16m1(pB, vl);
                _sum = vfmacc_vf_f16m1(_sum, pA[0], _pB, vl);
                // _sum = vfma_f16(_sum, _pA, _pB);

                pA += 1;
                pB += 4;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    // vst1_f16(outptr0, _sum);
                    vse16_v_f16m1(outptr0, _sum, vl);
                    outptr0 += 4;
                }
            }
            else
            {
                // vst1_f16(outptr, _sum);
                vse16_v_f16m1(outptr, _sum, vl);
            }

            outptr += 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            __fp16 sum0;
            __fp16 sum1;

            if (k == 0)
            {
                if (pC)
                {
                    sum0 = pC[0];
                    sum1 = pC[0];
                }
                else
                {
                    sum0 = 0.f;
                    sum1 = 0.f;
                }
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
            }

            const __fp16* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum0 += pA[0] * pB[0];
                sum1 += pA[0] * pB[1];

                pA += 1;
                pB += 2;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = sum0;
                    outptr0[1] = sum1;
                    outptr0 += 2;
                }
            }
            else
            {
                outptr[0] = sum0;
                outptr[1] = sum1;
            }

            outptr += 2;
        }
        for (; jj < max_jj; jj += 1)
        {
            __fp16 sum;

            if (k == 0)
            {
                if (pC)
                {
                    sum = pC[0];
                }
                else
                {
                    sum = 0.f;
                }
            }
            else
            {
                sum = outptr[0];
            }

            const __fp16* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum += pA[0] * pB[0];

                pA += 1;
                pB += 1;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = sum;
                    outptr0++;
                }
            }
            else
            {
                outptr[0] = sum;
            }

            outptr += 1;
        }

        pAT += max_kk;
    }
}

static void convolution_im2col_gemm_get_optimal_tile_mnk_fp16sa(int M, int N, int K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    // resolve optimal tile size from cache size
    // const int l2_cache_size_fp16 = (int)(get_cpu_level2_cache_size() / sizeof(unsigned short));
    const int l2_cache_size_fp16 = 64 * 1024 / sizeof(unsigned short); // 64 kb

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    // solve K
    {
        // try not to split K
        int tile_size = (l2_cache_size_fp16 - 32) / 12;

        TILE_K = std::max(8, tile_size / 8 * 8);

        int nn_K = (K + TILE_K - 1) / TILE_K;
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 7) / 8 * 8);
    }

    // solve M
    {
        int nn_M = (M + 63) / 64;

        TILE_M = std::max(8, ((M + nn_M - 1) / nn_M + 7) / 8 * 8);
    }

    {
        TILE_M *= std::min(nT, get_physical_cpu_count());

        int nn_M = (M + TILE_M - 1) / TILE_M;
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 7) / 8 * 8);

        if (nT > 1)
        {
            TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 7) / 8 * 8);
        }
    }

    if (N > 0)
    {
        int tile_size;
        if (TILE_K >= K)
        {
            tile_size = (l2_cache_size_fp16 - TILE_M * TILE_K) / TILE_K;
        }
        else
        {
            tile_size = (l2_cache_size_fp16 - TILE_M * TILE_K) / (TILE_M + TILE_K);
        }

        TILE_N = std::max(4, tile_size / 4 * 4);

        int nn_N = (N + TILE_N - 1) / TILE_N;
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 3) / 4 * 4);
    }
    // TILE_M = 64;
    // TILE_N = 128;
    // TILE_K = 512;
}

static void convolution_im2col_gemm_transform_kernel_fp16sa(const Mat& kernel, Mat& AT, int inch, int outch, int kernel_w, int kernel_h, const Option& opt)
{
    // NCNN_LOGE("convolution_im2col_gemm_transform_kernel_fp16sa %p", kernel.data);
    const int maxk = kernel_w * kernel_h;

    const int M = outch;
    const int K = inch * maxk;

    int TILE_M, TILE_N, TILE_K;
    convolution_im2col_gemm_get_optimal_tile_mnk_fp16sa(M, 0, K, TILE_M, TILE_N, TILE_K, opt.num_threads);

    const int nn_M = (M + TILE_M - 1) / TILE_M;

    int elempack = 1;
    if (opt.use_packing_layout)
    {
        elempack = inch % 8 == 0 ? 8 : 1;
    }

    // maxk-inch-outch to pa-maxk-inch/pa-outch
    Mat A_data;
    if (maxk == 1)
    {
        cast_float32_to_float16(kernel, A_data);
        A_data = A_data.reshape(maxk * inch, outch);
    }
    else
    {
        Mat weight_data_r2 = kernel.reshape(maxk, inch, outch);

        A_data.create(maxk * inch, outch, (size_t)2u);

        for (int q = 0; q < outch; q += 1)
        {
            __fp16* g00 = A_data.row<__fp16>(q);

            for (int p = 0; p + (elempack - 1) < inch; p += elempack)
            {
                for (int k = 0; k < maxk; k++)
                {
                    for (int i = 0; i < elempack; i++)
                    {
                        const float* k00 = weight_data_r2.channel(q).row(p + i);
                        g00[0] = (__fp16)k00[k];
                        g00++;
                    }
                }
            }
        }
    }

    AT.create(TILE_K * TILE_M, (K + TILE_K - 1) / TILE_K, (M + TILE_M - 1) / TILE_M, (size_t)2u);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        const int max_ii = std::min((M - i), TILE_M);

        for (int k = 0; k < K; k += TILE_K)
        {
            const int max_kk = std::min((K - k), TILE_K);

            Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

            convolution_im2col_pack_A_tile_bf16_fp16(A_data, AT_tile, i, max_ii, k, max_kk);
        }
    }
}

static void convolution_im2col_gemm_fp16sa(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, const Mat& bias, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
{
    int nT = 1;
    // NCNN_LOGE("convolution_im2col_gemm_fp16sa %p %p %p %p", bottom_blob.data, top_blob.data, AT.data, bias.data);
    const int maxk = kernel_w * kernel_h;

    const int M = top_blob.c * top_blob.elempack;
    const int N = top_blob.w * top_blob.h;
    const int K = bottom_blob.c * bottom_blob.elempack * maxk;

    int TILE_M, TILE_N, TILE_K;
    convolution_im2col_gemm_get_optimal_tile_mnk_fp16sa(M, N, K, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    NCNN_LOGE("TILE M/N/K = %d %d %d -> %d %d %d", M, N, K, TILE_M, TILE_N, TILE_K);
    NCNN_LOGE("NT = %d", nT);

    Mat BT(TILE_K * TILE_N, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N, 2u, opt.workspace_allocator);

    const int nn_NK = nn_N * nn_K;

    struct timeval start, end;

    gettimeofday(&start, NULL);

    #pragma omp parallel for num_threads(nT)
    for (int ppjk = 0; ppjk < nn_NK; ppjk++)
    {
        const int ppj = ppjk / nn_K;
        const int ppk = ppjk % nn_K;

        const int j = ppj * TILE_N;
        const int k = ppk * TILE_K;

        const int max_jj = std::min((N - j), TILE_N);
        const int max_kk = std::min((K - k), TILE_K);

        Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

        // im2col
        convolution_im2col_input_tile_bf16_fp16(bottom_blob, BT_tile, j, max_jj, k, max_kk, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h);
    }

    gettimeofday(&end, NULL);
    // time in ms
    NCNN_LOGE("im2col time = %f", (end.tv_sec - start.tv_sec) * 1000.f + (end.tv_usec - start.tv_usec) / 1000.f);

    Mat topT_tileX;
    if (K > TILE_K)
        topT_tileX.create(TILE_N * TILE_M, 1, nT, 2u, opt.workspace_allocator);

    gettimeofday(&start, NULL);

    #pragma omp parallel for num_threads(nT)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        Mat topT_tile;
        if (K > TILE_K)
            topT_tile = topT_tileX.channel(0);

        const int max_ii = std::min((M - i), TILE_M);

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min((N - j), TILE_N);

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min((K - k), TILE_K);

                const Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

                const Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                bool k_end = k + TILE_K >= K;

                convolution_gemm_transB_packed_tile_fp16sa(AT_tile, BT_tile, bias, topT_tile, top_blob, i, max_ii, j, max_jj, k, max_kk, k_end);
            }
        }
    }
    gettimeofday(&end, NULL);

    // time in ms
    NCNN_LOGE("gemm time = %f", (end.tv_sec - start.tv_sec) * 1000.f + (end.tv_usec - start.tv_usec) / 1000.f);
}
