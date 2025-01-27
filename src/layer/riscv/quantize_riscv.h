// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
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

#ifndef LAYER_QUANTIZE_RISCV_H
#define LAYER_QUANTIZE_RISCV_H

#include "quantize.h"

namespace ncnn {

class Quantize_riscv : public Quantize
{
public:
    Quantize_riscv();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

#if __riscv_vector && __riscv_zfh
    int forward_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
    int forward_fp16sa(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
#endif // __riscv_vector && __riscv_zfh
};

} // namespace ncnn

#endif // LAYER_QUANTIZE_RISCV_H
