// Copyright (c) 2026 Lawry Sorenson
// SPDX-License-Identifier: MIT

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

#include <torch/csrc/stable/c/shim.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "../helpers.h"

using torch::stable::Tensor;
using torch::headeronly::ScalarType;
using torch::headeronly::DeviceType;

using namespace std;

namespace hirschberg_viterbi {

    Tensor viterbi_cuda(
        const Tensor& log_probs,
        const Tensor& targets,
        const int32_t blank=0) {

        auto [text_holder, text_len, cont_log_probs, ans] =
            common_setup<DeviceType::CUDA, int32_t>(log_probs, targets, blank);

        cout << "POINT C" << endl;

        auto run = [&]<typename scalar_t>() {
            // call the base alignment function here
            // dummy function here just returns 0s for all the alignment
            cudaMemset(ans.mutable_data_ptr<int32_t>(), 0, ans.size(0)*4);
        };

        if (log_probs.scalar_type() == torch::headeronly::ScalarType::Double) {
            run.template operator()<double>();
        } else {
            run.template operator()<float>();
        }

        return ans;
    }

    Tensor hirschberg_viterbi_cuda(
        const Tensor& log_probs,
        const Tensor& targets,
        const int32_t blank=0,
        const int64_t soft_mem_limit=1000) {

        STD_TORCH_CHECK(false, "CUDA alignment has not been implemented yet. Please move tensors to cpu.");

        return log_probs;
    }

    STABLE_TORCH_LIBRARY_IMPL(hirschberg_viterbi, CUDA, m) {
        m.impl("viterbi", TORCH_BOX(&viterbi_cuda));
        m.impl("hirschberg_viterbi", TORCH_BOX(&hirschberg_viterbi_cuda));
    }

} // end namespace