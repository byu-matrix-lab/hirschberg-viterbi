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

using namespace std;

namespace hirschberg_viterbi {

    Tensor viterbi_cuda(
        const Tensor& log_probs,
        const Tensor& targets,
        const int32_t blank=0) {
        STD_TORCH_CHECK(false, "CUDA alignment has not been implemented yet. Please move tensors to cpu.");

        return log_probs;
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

}