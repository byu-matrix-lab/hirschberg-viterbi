///#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

template<typename backtrack_t, typename scalar_t, typename target_t>
void _normal_viterbi_helper(
    const py::array_t<scalar_t>& logits,
    target_t* text,
    py::array_t<target_t>& ans,
    int logits_left,
    int logits_right,
    int text_left,
    int text_right
);

template<typename backtrack_t, typename scalar_t, typename target_t>
void _hirschberg_helper(
    const py::array_t<scalar_t>& logits,
    target_t* text,
    py::array_t<target_t>& ans,
    int logits_left,
    int logits_right,
    int text_left,
    int text_right,
    int64_t soft_mem_limi
);

py::array_t<int32_t> viterbi(
    const py::array_t<float>& log_probs,
    const py::array_t<int32_t>& targets,
    const int32_t blank = 0
);

py::array_t<int32_t> hirschberg_viterbi(
    const py::array_t<float>& log_probs,
    const py::array_t<int32_t>& targets,
    const int32_t blank = 0,
    const int64_t soft_mem_limit=1000LL
);

// Add star parameters (maybe)
// map to torchaudio signature