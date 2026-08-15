#pragma once

#include <Python.h>

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

// namespace py = pybind11;

template<typename backtrack_t, typename scalar_t, typename target_t>
void _normal_viterbi_helper(
    const torch::stable::Tensor& logits,
    target_t* text,
    target_t* const ans,
    int logits_left,
    int logits_right,
    int text_left,
    int text_right
);

template<typename backtrack_t, typename scalar_t, typename target_t>
void _hirschberg_helper(
    const torch::stable::Tensor& logits,
    target_t* text,
    target_t* const ans,
    int logits_left,
    int logits_right,
    int text_left,
    int text_right,
    int64_t soft_mem_limit
);

torch::stable::Tensor viterbi_cpu(
    const torch::stable::Tensor& log_probs,
    const torch::stable::Tensor& targets,
    const int32_t blank=0
);

torch::stable::Tensor hirschberg_viterbi_cpu(
    const torch::stable::Tensor& log_probs,
    const torch::stable::Tensor& targets,
    const int32_t blank = 0,
    const int64_t soft_mem_limit=1000LL
);
