// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

static void conv3x3s1_winograd43_transform_input_packn_int8_rvv(const Mat& bottom_blob, Mat& bottom_blob_tm, const Option& opt)
{
    const int packn = csrr_vlenb() / 2;
    const size_t vl = vsetvl_e16m1(packn);

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int inch = bottom_blob.c;

    const int w_tiles = (w - 2) / 4;
    const int h_tiles = (h - 2) / 4;
    const int tiles = w_tiles * h_tiles;

    const float sq2 = 1.41421356237;
    const float sq2_d2 = 1.41421356237 / 2;

    // const float itm[6][6] = {
    //     {1.0f,  0.0f,  -2.5f,  0.0f,  1.0f, 0.0f},
    //     {0.0f, -sq2,   -2.0f,  sq2/2, 1.0f, 0.0f},
    //     {0.0f,  sq2,   -2.0f, -sq2/2, 1.0f, 0.0f},
    //     {0.0f, -sq2/2, -0.5f,  sq2,   1.0f, 0.0f},
    //     {0.0f,  sq2/2, -0.5f, -sq2,   1.0f, 0.0f},
    //     {0.0f,  1.0f,   0.0f,  -2.5f, 0.0f, 1.0f}
    // };

    // 0 =  r00 - 2.5f * r02 + r04
    // 1 = -(sq2 * r01 - sq2_d2 * r03) + (r04 - 2 * r02)
    // 2 =  (sq2 * r01 - sq2_d2 * r03) + (r04 - 2 * r02)
    // 3 = -(sq2_d2 * r01 - sq2 * r03) + (r04 - 0.5f * r02)
    // 4 =  (sq2_d2 * r01 - sq2 * r03) + (r04 - 0.5f * r02)
    // 5 =  r01 - 2.5f * r03 + r05

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < inch; q++)
    {
        const Mat img0 = bottom_blob.channel(q);
        Mat img0_tm = bottom_blob_tm.channel(q);

        // NOTE c99 variable length array
        int8_t tmp[6][6][packn];

        // tile
        for (int i = 0; i < h_tiles; i++)
        {
            for (int j = 0; j < w_tiles; j++)
            {
                const int8_t* r0 = img0.row<const int8_t>(i * 4) + (j * 4) * packn;

                for (int m = 0; m < 6; m++)
                {
                    vfloat16m1_t _r00 = vle16_v_f16m1(r0, vl);
                    vfloat16m1_t _r01 = vle16_v_f16m1(r0 + packn, vl);
                    vfloat16m1_t _r02 = vle16_v_f16m1(r0 + packn * 2, vl);
                    vfloat16m1_t _r03 = vle16_v_f16m1(r0 + packn * 3, vl);
                    vfloat16m1_t _r04 = vle16_v_f16m1(r0 + packn * 4, vl);
                    vfloat16m1_t _r05 = vle16_v_f16m1(r0 + packn * 5, vl);

                    vfloat16m1_t _tmp01a = vfmacc_vf_f16m1(vfmul_vf_f16m1(_r01, sq2, vl), -sq2_d2, _r03, vl);
                    vfloat16m1_t _tmp01b = vfmacc_vf_f16m1(_r04, -2.f, _r02, vl);
                    vfloat16m1_t _tmp23a = vfmacc_vf_f16m1(vfmul_vf_f16m1(_r01, sq2_d2, vl), -sq2, _r03, vl);
                    vfloat16m1_t _tmp23b = vfmacc_vf_f16m1(_r04, -0.5f, _r02, vl);

                    vfloat16m1_t _tmp0m = vfmacc_vf_f16m1(vfadd_vv_f16m1(_r00, _r04, vl), -2.5f, _r02, vl);
                    vfloat16m1_t _tmp1m = vfsub_vv_f16m1(_tmp01b, _tmp01a, vl);
                    vfloat16m1_t _tmp2m = vfadd_vv_f16m1(_tmp01b, _tmp01a, vl);
                    vfloat16m1_t _tmp3m = vfsub_vv_f16m1(_tmp23b, _tmp23a, vl);
                    vfloat16m1_t _tmp4m = vfadd_vv_f16m1(_tmp23b, _tmp23a, vl);
                    vfloat16m1_t _tmp5m = vfmacc_vf_f16m1(vfadd_vv_f16m1(_r01, _r05, vl), -2.5f, _r03, vl);

                    vse16_v_f16m1(tmp[0][m], _tmp0m, vl);
                    vse16_v_f16m1(tmp[1][m], _tmp1m, vl);
                    vse16_v_f16m1(tmp[2][m], _tmp2m, vl);
                    vse16_v_f16m1(tmp[3][m], _tmp3m, vl);
                    vse16_v_f16m1(tmp[4][m], _tmp4m, vl);
                    vse16_v_f16m1(tmp[5][m], _tmp5m, vl);

                    r0 += w * packn;
                }

                int8_t* r0_tm_0 = (int8_t*)img0_tm + (i * w_tiles + j) * packn;
                int8_t* r0_tm_1 = r0_tm_0 + tiles * packn;
                int8_t* r0_tm_2 = r0_tm_0 + tiles * packn * 2;
                int8_t* r0_tm_3 = r0_tm_0 + tiles * packn * 3;
                int8_t* r0_tm_4 = r0_tm_0 + tiles * packn * 4;
                int8_t* r0_tm_5 = r0_tm_0 + tiles * packn * 5;

                for (int m = 0; m < 6; m++)
                {
                    vfloat16m1_t _r00 = vle16_v_f16m1(tmp[m][0], vl);
                    vfloat16m1_t _r01 = vle16_v_f16m1(tmp[m][1], vl);
                    vfloat16m1_t _r02 = vle16_v_f16m1(tmp[m][2], vl);
                    vfloat16m1_t _r03 = vle16_v_f16m1(tmp[m][3], vl);
                    vfloat16m1_t _r04 = vle16_v_f16m1(tmp[m][4], vl);
                    vfloat16m1_t _r05 = vle16_v_f16m1(tmp[m][5], vl);

                    vfloat16m1_t _tmp01a = vfmacc_vf_f16m1(vfmul_vf_f16m1(_r01, sq2, vl), -sq2_d2, _r03, vl);
                    vfloat16m1_t _tmp01b = vfmacc_vf_f16m1(_r04, -2.f, _r02, vl);
                    vfloat16m1_t _tmp23a = vfmacc_vf_f16m1(vfmul_vf_f16m1(_r01, sq2_d2, vl), -sq2, _r03, vl);
                    vfloat16m1_t _tmp23b = vfmacc_vf_f16m1(_r04, -0.5f, _r02, vl);

                    vfloat16m1_t _tmp0m = vfmacc_vf_f16m1(vfadd_vv_f16m1(_r00, _r04, vl), -2.5f, _r02, vl);
                    vfloat16m1_t _tmp1m = vfsub_vv_f16m1(_tmp01b, _tmp01a, vl);
                    vfloat16m1_t _tmp2m = vfadd_vv_f16m1(_tmp01b, _tmp01a, vl);
                    vfloat16m1_t _tmp3m = vfsub_vv_f16m1(_tmp23b, _tmp23a, vl);
                    vfloat16m1_t _tmp4m = vfadd_vv_f16m1(_tmp23b, _tmp23a, vl);
                    vfloat16m1_t _tmp5m = vfmacc_vf_f16m1(vfadd_vv_f16m1(_r01, _r05, vl), -2.5f, _r03, vl);

                    vse16_v_f16m1(r0_tm_0, _tmp0m, vl);
                    vse16_v_f16m1(r0_tm_1, _tmp1m, vl);
                    vse16_v_f16m1(r0_tm_2, _tmp2m, vl);
                    vse16_v_f16m1(r0_tm_3, _tmp3m, vl);
                    vse16_v_f16m1(r0_tm_4, _tmp4m, vl);
                    vse16_v_f16m1(r0_tm_5, _tmp5m, vl);

                    r0_tm_0 += tiles * packn * 6;
                    r0_tm_1 += tiles * packn * 6;
                    r0_tm_2 += tiles * packn * 6;
                    r0_tm_3 += tiles * packn * 6;
                    r0_tm_4 += tiles * packn * 6;
                    r0_tm_5 += tiles * packn * 6;
                }
            }
        }
    }
}

static void conv3x3s1_winograd43_transform_output_packn_int8_rvv(const Mat& top_blob_tm, Mat& top_blob, const Mat& bias, const Option& opt)
{
    const int packn = csrr_vlenb() / 2;
    const size_t vl = vsetvl_e16m1(packn);

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int outch = top_blob.c;

    const int w_tiles = outw / 4;
    const int h_tiles = outh / 4;
    const int tiles = w_tiles * h_tiles;

    const int8_t* biasptr = bias;

    const float sq2 = 1.41421356237;
    const float sq2_m2 = 1.41421356237 * 2;
    const float sq2_d2 = 1.41421356237 / 2;
    const float sq2_d4 = 1.41421356237 / 4;

    // const float otm[4][6] = {
    //     {1.0f, 1.0f,   1.0f,  1.0f,  1.0f,   0.0f},
    //     {0.0f, sq2/2, -sq2/2, sq2,   -sq2,   0.0f},
    //     {0.0f, 0.5f,   0.5f,  2.0f,  2.0f,   0.0f},
    //     {0.0f, sq2/4, -sq2/4, sq2*2, -sq2*2, 1.0f}
    // };

    // 0 = r00 + (r01 + r02) + (r03 + r04)
    // 1 =       (r01 - r02) * sq2_d2 + (r03 - r04) * sq2
    // 2 =       (r01 + r02) * 0.5f + (r03 + r04) * 2
    // 3 = r05 + (r01 - r02) * sq2_d4 + (r03 - r04) * sq2_m2

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        const Mat out0_tm = top_blob_tm.channel(p);
        Mat out0 = top_blob.channel(p);

        vfloat16m1_t _bias0 = biasptr ? vle16_v_f16m1(biasptr + p * packn, vl) : vfmv_v_f_f16m1(0.f, vl);

        // NOTE variable length array
        int8_t tmp[4][6][packn];

        // tile
        for (int i = 0; i < h_tiles; i++)
        {
            for (int j = 0; j < w_tiles; j++)
            {
                const int8_t* output0_tm_0 = (const int8_t*)out0_tm + (i * w_tiles + j) * packn;
                const int8_t* output0_tm_1 = output0_tm_0 + tiles * packn;
                const int8_t* output0_tm_2 = output0_tm_0 + tiles * packn * 2;
                const int8_t* output0_tm_3 = output0_tm_0 + tiles * packn * 3;
                const int8_t* output0_tm_4 = output0_tm_0 + tiles * packn * 4;
                const int8_t* output0_tm_5 = output0_tm_0 + tiles * packn * 5;

                int8_t* output0 = out0.row<int8_t>(i * 4) + (j * 4) * packn;

                for (int m = 0; m < 6; m++)
                {
                    vfloat16m1_t _r00 = vle16_v_f16m1(output0_tm_0, vl);
                    vfloat16m1_t _r01 = vle16_v_f16m1(output0_tm_1, vl);
                    vfloat16m1_t _r02 = vle16_v_f16m1(output0_tm_2, vl);
                    vfloat16m1_t _r03 = vle16_v_f16m1(output0_tm_3, vl);
                    vfloat16m1_t _r04 = vle16_v_f16m1(output0_tm_4, vl);
                    vfloat16m1_t _r05 = vle16_v_f16m1(output0_tm_5, vl);

                    vfloat16m1_t _tmp02a = vfadd_vv_f16m1(_r01, _r02, vl);
                    vfloat16m1_t _tmp02b = vfadd_vv_f16m1(_r03, _r04, vl);
                    vfloat16m1_t _tmp13a = vfsub_vv_f16m1(_r01, _r02, vl);
                    vfloat16m1_t _tmp13b = vfsub_vv_f16m1(_r03, _r04, vl);

                    vfloat16m1_t _tmp0m = vfadd_vv_f16m1(vfadd_vv_f16m1(_r00, _tmp02a, vl), _tmp02b, vl);
                    vfloat16m1_t _tmp1m = vfmacc_vf_f16m1(vfmul_vf_f16m1(_tmp13a, sq2_d2, vl), sq2, _tmp13b, vl);
                    vfloat16m1_t _tmp2m = vfmacc_vf_f16m1(vfmul_vf_f16m1(_tmp02a, 0.5f, vl), 2.f, _tmp02b, vl);
                    vfloat16m1_t _tmp3m = vfmacc_vf_f16m1(vfmacc_vf_f16m1(_r05, sq2_d4, _tmp13a, vl), sq2_m2, _tmp13b, vl);

                    vse16_v_f16m1(tmp[0][m], _tmp0m, vl);
                    vse16_v_f16m1(tmp[1][m], _tmp1m, vl);
                    vse16_v_f16m1(tmp[2][m], _tmp2m, vl);
                    vse16_v_f16m1(tmp[3][m], _tmp3m, vl);

                    output0_tm_0 += tiles * packn * 6;
                    output0_tm_1 += tiles * packn * 6;
                    output0_tm_2 += tiles * packn * 6;
                    output0_tm_3 += tiles * packn * 6;
                    output0_tm_4 += tiles * packn * 6;
                    output0_tm_5 += tiles * packn * 6;
                }

                for (int m = 0; m < 4; m++)
                {
                    vfloat16m1_t _r00 = vle16_v_f16m1(tmp[m][0], vl);
                    vfloat16m1_t _r01 = vle16_v_f16m1(tmp[m][1], vl);
                    vfloat16m1_t _r02 = vle16_v_f16m1(tmp[m][2], vl);
                    vfloat16m1_t _r03 = vle16_v_f16m1(tmp[m][3], vl);
                    vfloat16m1_t _r04 = vle16_v_f16m1(tmp[m][4], vl);
                    vfloat16m1_t _r05 = vle16_v_f16m1(tmp[m][5], vl);

                    vfloat16m1_t _tmp02a = vfadd_vv_f16m1(_r01, _r02, vl);
                    vfloat16m1_t _tmp02b = vfadd_vv_f16m1(_r03, _r04, vl);
                    vfloat16m1_t _tmp13a = vfsub_vv_f16m1(_r01, _r02, vl);
                    vfloat16m1_t _tmp13b = vfsub_vv_f16m1(_r03, _r04, vl);

                    vfloat16m1_t _out00 = vfadd_vv_f16m1(_bias0, vfadd_vv_f16m1(vfadd_vv_f16m1(_r00, _tmp02a, vl), _tmp02b, vl), vl);
                    vfloat16m1_t _out01 = vfadd_vv_f16m1(_bias0, vfmacc_vf_f16m1(vfmul_vf_f16m1(_tmp13a, sq2_d2, vl), sq2, _tmp13b, vl), vl);
                    vfloat16m1_t _out02 = vfadd_vv_f16m1(_bias0, vfmacc_vf_f16m1(vfmul_vf_f16m1(_tmp02a, 0.5f, vl), 2.f, _tmp02b, vl), vl);
                    vfloat16m1_t _out03 = vfadd_vv_f16m1(_bias0, vfmacc_vf_f16m1(vfmacc_vf_f16m1(_r05, sq2_d4, _tmp13a, vl), sq2_m2, _tmp13b, vl), vl);

                    vse16_v_f16m1(output0, _out00, vl);
                    vse16_v_f16m1(output0 + packn, _out01, vl);
                    vse16_v_f16m1(output0 + packn * 2, _out02, vl);
                    vse16_v_f16m1(output0 + packn * 3, _out03, vl);

                    output0 += outw * packn;
                }
            }
        }
    }
}

static void conv3x3s1_winograd23_transform_input_packn_int8_rvv(const Mat& bottom_blob, Mat& bottom_blob_tm, const Option& opt)
{
    const int packn = csrr_vlenb() / 2;
    const size_t vl = vsetvl_e16m1(packn);

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int inch = bottom_blob.c;

    const int w_tiles = (w - 2) / 2;
    const int h_tiles = (h - 2) / 2;
    const int tiles = w_tiles * h_tiles;

    // const float itm[4][4] = {
    //     {1.0f,  0.0f, -1.0f,  0.0f},
    //     {0.0f,  1.0f,  1.00f, 0.0f},
    //     {0.0f, -1.0f,  1.00f, 0.0f},
    //     {0.0f, -1.0f,  0.00f, 1.0f}
    // };

    // 0 = r00 - r02
    // 1 = r01 + r02
    // 2 = r02 - r01
    // 3 = r03 - r01

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < inch; q++)
    {
        const Mat img0 = bottom_blob.channel(q);
        Mat img0_tm = bottom_blob_tm.channel(q);

        // NOTE c99 variable length array
        int8_t tmp[4][4][packn];

        // tile
        for (int i = 0; i < h_tiles; i++)
        {
            for (int j = 0; j < w_tiles; j++)
            {
                const int8_t* r0 = img0.row<const int8_t>(i * 2) + (j * 2) * packn;

                for (int m = 0; m < 4; m++)
                {
                    vfloat16m1_t _r00 = vle16_v_f16m1(r0, vl);
                    vfloat16m1_t _r01 = vle16_v_f16m1(r0 + packn, vl);
                    vfloat16m1_t _r02 = vle16_v_f16m1(r0 + packn * 2, vl);
                    vfloat16m1_t _r03 = vle16_v_f16m1(r0 + packn * 3, vl);

                    vfloat16m1_t _tmp0m = vfsub_vv_f16m1(_r00, _r02, vl);
                    vfloat16m1_t _tmp1m = vfadd_vv_f16m1(_r01, _r02, vl);
                    vfloat16m1_t _tmp2m = vfsub_vv_f16m1(_r02, _r01, vl);
                    vfloat16m1_t _tmp3m = vfsub_vv_f16m1(_r03, _r01, vl);

                    vse16_v_f16m1(tmp[0][m], _tmp0m, vl);
                    vse16_v_f16m1(tmp[1][m], _tmp1m, vl);
                    vse16_v_f16m1(tmp[2][m], _tmp2m, vl);
                    vse16_v_f16m1(tmp[3][m], _tmp3m, vl);

                    r0 += w * packn;
                }

                int8_t* r0_tm_0 = (int8_t*)img0_tm + (i * w_tiles + j) * packn;
                int8_t* r0_tm_1 = r0_tm_0 + tiles * packn;
                int8_t* r0_tm_2 = r0_tm_0 + tiles * packn * 2;
                int8_t* r0_tm_3 = r0_tm_0 + tiles * packn * 3;

                for (int m = 0; m < 4; m++)
                {
                    vfloat16m1_t _tmp00 = vle16_v_f16m1(tmp[m][0], vl);
                    vfloat16m1_t _tmp01 = vle16_v_f16m1(tmp[m][1], vl);
                    vfloat16m1_t _tmp02 = vle16_v_f16m1(tmp[m][2], vl);
                    vfloat16m1_t _tmp03 = vle16_v_f16m1(tmp[m][3], vl);

                    vfloat16m1_t _r0tm0 = vfsub_vv_f16m1(_tmp00, _tmp02, vl);
                    vfloat16m1_t _r0tm1 = vfadd_vv_f16m1(_tmp01, _tmp02, vl);
                    vfloat16m1_t _r0tm2 = vfsub_vv_f16m1(_tmp02, _tmp01, vl);
                    vfloat16m1_t _r0tm3 = vfsub_vv_f16m1(_tmp03, _tmp01, vl);

                    vse16_v_f16m1(r0_tm_0, _r0tm0, vl);
                    vse16_v_f16m1(r0_tm_1, _r0tm1, vl);
                    vse16_v_f16m1(r0_tm_2, _r0tm2, vl);
                    vse16_v_f16m1(r0_tm_3, _r0tm3, vl);

                    r0_tm_0 += tiles * packn * 4;
                    r0_tm_1 += tiles * packn * 4;
                    r0_tm_2 += tiles * packn * 4;
                    r0_tm_3 += tiles * packn * 4;
                }
            }
        }
    }
}

static void conv3x3s1_winograd23_transform_output_packn_int8_rvv(const Mat& top_blob_tm, Mat& top_blob, const Mat& bias, const Option& opt)
{
    const int packn = csrr_vlenb() / 2;
    const size_t vl = vsetvl_e16m1(packn);

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int outch = top_blob.c;

    const int w_tiles = outw / 2;
    const int h_tiles = outh / 2;
    const int tiles = w_tiles * h_tiles;

    const int8_t* biasptr = bias;

    // const float otm[2][4] = {
    //     {1.0f,  1.0f,  1.0f,  0.0f},
    //     {0.0f,  1.0f, -1.0f,  1.0f}
    // };

    // 0 = r00 + r01 + r02
    // 1 = r01 - r02 + r03

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        const Mat out0_tm = top_blob_tm.channel(p);
        Mat out0 = top_blob.channel(p);

        vfloat16m1_t _bias0 = biasptr ? vle16_v_f16m1(biasptr + p * packn, vl) : vfmv_v_f_f16m1(0.f, vl);

        // NOTE variable length array
        int8_t tmp[2][4][packn];

        // tile
        for (int i = 0; i < h_tiles; i++)
        {
            for (int j = 0; j < w_tiles; j++)
            {
                const int8_t* output0_tm_0 = (const int8_t*)out0_tm + (i * w_tiles + j) * packn;
                const int8_t* output0_tm_1 = output0_tm_0 + tiles * packn;
                const int8_t* output0_tm_2 = output0_tm_0 + tiles * packn * 2;
                const int8_t* output0_tm_3 = output0_tm_0 + tiles * packn * 3;

                int8_t* output0 = out0.row<int8_t>(i * 2) + (j * 2) * packn;

                for (int m = 0; m < 4; m++)
                {
                    vfloat16m1_t _out0tm0 = vle16_v_f16m1(output0_tm_0, vl);
                    vfloat16m1_t _out0tm1 = vle16_v_f16m1(output0_tm_1, vl);
                    vfloat16m1_t _out0tm2 = vle16_v_f16m1(output0_tm_2, vl);
                    vfloat16m1_t _out0tm3 = vle16_v_f16m1(output0_tm_3, vl);

                    vfloat16m1_t _tmp0m = vfadd_vv_f16m1(vfadd_vv_f16m1(_out0tm0, _out0tm1, vl), _out0tm2, vl);
                    vfloat16m1_t _tmp1m = vfadd_vv_f16m1(vfsub_vv_f16m1(_out0tm1, _out0tm2, vl), _out0tm3, vl);

                    vse16_v_f16m1(tmp[0][m], _tmp0m, vl);
                    vse16_v_f16m1(tmp[1][m], _tmp1m, vl);

                    output0_tm_0 += tiles * packn * 4;
                    output0_tm_1 += tiles * packn * 4;
                    output0_tm_2 += tiles * packn * 4;
                    output0_tm_3 += tiles * packn * 4;
                }

                for (int m = 0; m < 2; m++)
                {
                    vfloat16m1_t _tmp00 = vle16_v_f16m1(tmp[m][0], vl);
                    vfloat16m1_t _tmp01 = vle16_v_f16m1(tmp[m][1], vl);
                    vfloat16m1_t _tmp02 = vle16_v_f16m1(tmp[m][2], vl);
                    vfloat16m1_t _tmp03 = vle16_v_f16m1(tmp[m][3], vl);

                    vfloat16m1_t _out00 = vfadd_vv_f16m1(_bias0, vfadd_vv_f16m1(vfadd_vv_f16m1(_tmp00, _tmp01, vl), _tmp02, vl), vl);
                    vfloat16m1_t _out01 = vfadd_vv_f16m1(_bias0, vfadd_vv_f16m1(vfsub_vv_f16m1(_tmp01, _tmp02, vl), _tmp03, vl), vl);

                    vse16_v_f16m1(output0, _out00, vl);
                    vse16_v_f16m1(output0 + packn, _out01, vl);

                    output0 += outw * packn;
                }
            }
        }
    }
}
