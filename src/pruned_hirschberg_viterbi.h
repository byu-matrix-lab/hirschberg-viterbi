#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

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
    double conf,
    double accuracy,
    double precision,
    double recall
);

py::array_t<int32_t> pruned_viterbi(
    const py::array_t<float>& log_probs,
    const py::array_t<int32_t>& targets,
    const int32_t blank = 0,
    const double var_rat = 3.7,
    const double conf = 0.99,
    const double accuracy = 0.97,
    const double precision = -1.0,
    const double recall = -1.0
);

py::array_t<int32_t> pruned_hirschberg_viterbi(
    const py::array_t<float>& log_probs,
    const py::array_t<int32_t>& targets,
    const int32_t blank = 0,
    const double var_rat = 3.7,
    const double conf = 0.99,
    const double accuracy = 0.97,
    const double precision = -1.0,
    const double recall = -1.0,
    const int64_t soft_mem_limit=1000LL
);