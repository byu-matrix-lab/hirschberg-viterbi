#pragma once

#include <Python.h>

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

namespace hirschberg_viterbi {

    template<typename scalar_t>
    inline std::pair<scalar_t, scalar_t> get_bounds(
        int i,
        int n,
        int duration,
        double var_rat,
        double stds,
        int num_dels,
        int num_inser
    );

    template<typename scalar_t>
    std::pair<scalar_t*, scalar_t*> calculate_bounds(
        int len_logits,
        int len_text,
        double var_rat,
        double confidence,
        double accuracy,
        double precision,
        double recall
    );

    torch::stable::Tensor pruned_viterbi_cpu(
        const torch::stable::Tensor& log_probs,
        const torch::stable::Tensor& targets,
        const int32_t blank = 0,
        const double var_rat = 3.7,
        const double confidence = 0.99,
        const double accuracy = 0.97,
        const double precision = -1.0,
        const double recall = -1.0,
        const double padding = 5.0
    );

    torch::stable::Tensor pruned_hirschberg_viterbi_cpu(
        const torch::stable::Tensor& log_probs,
        const torch::stable::Tensor& targets,
        const int32_t blank = 0,
        const double var_rat = 3.7,
        const double confidence = 0.99,
        const double accuracy = 0.97,
        const double precision = -1.0,
        const double recall = -1.0,
        const double padding = 5.0,
        const int64_t soft_mem_limit=1000LL
    );

}