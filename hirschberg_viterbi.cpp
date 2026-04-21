#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

#include "helpers.h"

namespace py = pybind11;

using namespace std; // TODO: remove this

template<typename backtrack_t, typename scalar_t, typename target_t>
void _normal_viterbi_helper(
    const py::array_t<scalar_t>& logits,
    target_t* text,
    py::array_t<target_t>& ans,
    int logits_left,
    int logits_right,
    int text_left,
    int text_right) {
    
    const scalar_t mask_val = -std::numeric_limits<scalar_t>::infinity();

    auto logits_view = logits.template unchecked<2>();

    int max_width = text_right - text_left;
    scalar_t* cur_probs = new scalar_t[2+max_width];
    scalar_t* prev_probs = new scalar_t[2+max_width];

    for (int i=0;i<max_width+2;++i) cur_probs[i] = mask_val;
    prev_probs[0] = prev_probs[1] = mask_val;
    cur_probs[2] = 0;

    backtrack_t backedges(logits_right - logits_left, max_width);

    text += text_left; // for easier math now

    cur_probs+=2;
    prev_probs+=2;
    for(int time=logits_left; time < logits_right; ++time) {
        swap(cur_probs, prev_probs);

        for (int ci=0;ci<max_width;++ci) {
            scalar_t& val=cur_probs[ci];
            uint8_t back = 0;
            val = prev_probs[ci];

            if (prev_probs[ci-1] > val) val=prev_probs[ci-1], back=1;
            // text padding allows us to look back past the start of text
            if (text[ci] != text[ci-2] && prev_probs[ci-2] > val) val=prev_probs[ci-2], back=2;

            val += logits_view(time, text[ci]);
            backedges.set(time-logits_left, ci, back);     
        }

    }
    text -= text_left;

    int cur = text_right-1;
    scalar_t cur_val = cur_probs[max_width-1];
    if (cur_probs[max_width-2] > cur_val) cur_val=cur_probs[max_width-2],cur=text_right-2;
    // this one is needed for hirschberg calls
    assert (text_right > 0);
    if (text[text_right-1]!=text[text_right-3] && cur_probs[max_width-3] > cur_val) cur_val=cur_probs[max_width-3],cur=text_right-3;

    cur_probs-=2;
    prev_probs-=2;

    delete[] cur_probs;
    delete[] prev_probs;

    auto ans_data = ans.template mutable_unchecked<1>();

    for (int time=logits_right; --time >= logits_left; ) {
        ans_data(time) = cur;
        cur -= backedges.get(time-logits_left, cur-text_left);
    }

}

template<typename backtrack_t, typename scalar_t, typename target_t>
void _hirschberg_helper(
    const py::array_t<scalar_t>& logits,
    target_t* text,
    py::array_t<target_t>& ans,
    int logits_left,
    int logits_right,
    int text_left,
    int text_right,
    int64_t soft_mem_limit) {

    if (logits_left >= logits_right) return;

    if (text_left + 1 == text_right) {
        auto ans_data = ans.template mutable_unchecked<1>();
        for (int time=logits_left;time<logits_right;++time) ans_data(time) = text_left;
        return;
    }

    if (backtrack_t::needed_size(logits_right-logits_left, text_right-text_left) <= soft_mem_limit) {
        _normal_viterbi_helper<backtrack_t, scalar_t, target_t>(logits, text, ans, logits_left, logits_right, text_left, text_right);
        return;
    }
    
    const scalar_t mask_val = -std::numeric_limits<scalar_t>::infinity();

    auto logits_view = logits.template unchecked<2>();

    int max_width = text_right - text_left;
    scalar_t* cur_left_probs = new scalar_t[2+max_width];
    scalar_t* prev_probs = new scalar_t[2+max_width];

    for (int i=0;i<max_width+2;++i) cur_left_probs[i] = mask_val;
    prev_probs[0] = prev_probs[1] = mask_val;
    cur_left_probs[2] = 0;

    // for easier math now
    text += text_left;
    cur_left_probs+=2;
    prev_probs+=2;

    // split-1 is not needed, but done to make sure it matches the python impl during testing
    int split = logits_left + logits_right-1 >> 1;

    for(int time=logits_left; time <= split; ++time) {
        swap(cur_left_probs, prev_probs);

        for (int ci=0;ci<max_width;++ci) {
            scalar_t& val=cur_left_probs[ci];
            val = prev_probs[ci];

            if (prev_probs[ci-1] > val) val=prev_probs[ci-1];
            // text padding allows us to look back past the start of text
            if (text[ci] != text[ci-2] && prev_probs[ci-2] > val) val=prev_probs[ci-2];

            val += logits_view(time, text[ci]);

        }
    }

    scalar_t* cur_right_probs = new scalar_t[2+max_width];
    for (int i=0;i<max_width+2;++i) cur_right_probs[i] = mask_val;
    cur_right_probs[max_width-1] = 0;
    prev_probs-=2;
    prev_probs[max_width] = prev_probs[max_width+1] = mask_val;

    for (int time=logits_right; --time >= split;) {
        swap(cur_right_probs, prev_probs);

        for (int ci=0;ci<max_width;++ci) {
            scalar_t& val=cur_right_probs[ci];
            val = prev_probs[ci];

            if (prev_probs[ci+1] > val) val=prev_probs[ci+1];
            // text padding allows us to look back past the start of text
            if (text[ci] != text[ci+2] && prev_probs[ci+2] > val) val=prev_probs[ci+2];

            val += logits_view(time, text[ci]);

        }
    }

    int pivot=0;
    scalar_t pivot_prob = mask_val;
    for (int ci=0;ci<max_width;++ci) {
        scalar_t cur = cur_left_probs[ci] + cur_right_probs[ci] - logits_view(split, text[ci]);
        if (cur > pivot_prob) pivot_prob = cur, pivot=ci;
    }
    pivot += text_left;

    auto ans_data = ans.template mutable_unchecked<1>();
    ans_data(split) = pivot;

    cur_left_probs-=2;
    delete[] cur_left_probs;
    delete[] cur_right_probs;
    delete[] prev_probs;

    text -= text_left;

    // recurse left
    _hirschberg_helper<backtrack_t, scalar_t, target_t>(
        logits,
        text,
        ans,
        logits_left,
        split,
        text_left,
        pivot+1,
        soft_mem_limit
    );

    // recurse right
    _hirschberg_helper<backtrack_t, scalar_t, target_t>(
        logits,
        text,
        ans,
        split+1,
        logits_right,
        pivot,
        text_right,
        soft_mem_limit
    );
}

py::array_t<int32_t> viterbi(
    const py::array_t<float>& log_probs,
    const py::array_t<int32_t>& targets,
    const int32_t blank = 0) {

    if (log_probs.ndim() != 2) throw std::runtime_error("log_probs must be a 2-D array.");
    if (targets.ndim() != 1) throw std::runtime_error("targets must be a 1-D array.");

    auto [text, text_len] = add_blanks<int32_t>(targets, blank);

    const auto T = log_probs.shape(0);

    if (text_len > T) throw std::runtime_error("log_probs must be longer than the input text for this implementation.");

    auto ans = py::array_t<int32_t>({T});

    _normal_viterbi_helper<bt_full_byte, float, int32_t>(
        log_probs,
        text,
        ans,
        0,
        T,
        0,
        text_len
    );

    text-=2; // remove the original padding
    delete[] text;

    return ans;
}

py::array_t<int32_t> hirschberg_viterbi(
    const py::array_t<float>& log_probs,
    const py::array_t<int32_t>& targets,
    const int32_t blank = 0,
    const int64_t soft_mem_limit=1000LL) {

    if (log_probs.ndim() != 2) throw std::runtime_error("log_probs must be a 2-D array.");
    if (targets.ndim() != 1) throw std::runtime_error("targets must be a 1-D array.");

    auto [text, text_len] = add_blanks<int32_t>(targets, blank);

    const auto T = log_probs.shape(0);

    if (text_len > T) throw std::runtime_error("log_probs must be longer than the input text for this implementation.");

    auto ans = py::array_t<int32_t>({T});

    _hirschberg_helper<bt_full_byte, float, int32_t>(
        log_probs,
        text,
        ans,
        0,
        T,
        0,
        text_len,
        soft_mem_limit
    );

    text-=2; // remove the original padding
    delete[] text;

    return ans;
}