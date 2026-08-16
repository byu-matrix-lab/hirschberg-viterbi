
#include <type_traits>

#include <Python.h>

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

#include "helpers.h"

using torch::stable::Tensor;
using torch::headeronly::ScalarType;
using torch::headeronly::DeviceType;

namespace hirschberg_viterbi {

    template<typename backtrack_t, typename scalar_t, typename target_t>
    void _normal_viterbi_helper(
        const Tensor& logits,
        target_t* text,
        target_t* const ans,
        int logits_left,
        int logits_right,
        int text_left,
        int text_right) {
        
        const scalar_t mask_val = -std::numeric_limits<scalar_t>::infinity();

        auto* const logits_ptr = logits.const_data_ptr<scalar_t>();
        // 64-bit to avoid potential overflow for audio past 17.4 hours
        int64_t logits_stride = logits.stride(0);

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
        const scalar_t* logits_view=logits_ptr+logits_left*logits_stride;
        for(int time=logits_left; time < logits_right; ++time, logits_view+=logits_stride) {
            std::swap(cur_probs, prev_probs);
            
            for (int ci=0;ci<max_width;++ci) {
                scalar_t& val=cur_probs[ci];
                uint8_t back = 0;
                val = prev_probs[ci];
                
                if (prev_probs[ci-1] > val) val=prev_probs[ci-1], back=1;
                // text padding allows us to look back past the start of text
                if (text[ci] != text[ci-2] && prev_probs[ci-2] > val) val=prev_probs[ci-2], back=2;
                
                val += logits_view[text[ci]];
                backedges.set(time-logits_left, ci, back);     
            }
            
            // if float shift max to 0 for numerical stability on long inputs
            if constexpr (std::is_same_v<scalar_t, float>) {
                if (time % RESCALE_MAX_FREQ == 0) rescale_max(cur_probs, max_width);
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

        for (int time=logits_right; --time >= logits_left; ) {
            ans[time] = cur;
            cur -= backedges.get(time-logits_left, cur-text_left);
        }
    }

    template<typename backtrack_t, typename scalar_t, typename target_t>
    void _hirschberg_helper(
        const Tensor& logits,
        target_t* text,
        target_t* const ans,
        int logits_left,
        int logits_right,
        int text_left,
        int text_right,
        int64_t soft_mem_limit) {

        if (logits_left >= logits_right) return;

        if (text_left + 1 == text_right) {
            for (int time=logits_left;time<logits_right;++time) ans[time] = text_left;
            return;
        }

        if (backtrack_t::needed_size(logits_right-logits_left, text_right-text_left) <= soft_mem_limit) {
            _normal_viterbi_helper<backtrack_t, scalar_t, target_t>(logits, text, ans, logits_left, logits_right, text_left, text_right);
            return;
        }
        
        const scalar_t mask_val = -std::numeric_limits<scalar_t>::infinity();

        auto* const logits_ptr = logits.const_data_ptr<scalar_t>();
        // 64-bit to avoid potential overflow for audio past 17.4 hours
        int64_t logits_stride = logits.stride(0);

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
        int split = (logits_left + logits_right-1) >> 1;

        const scalar_t* logits_view=logits_ptr+logits_left*logits_stride;
        for(int time=logits_left; time <= split; ++time, logits_view+=logits_stride) {
            std::swap(cur_left_probs, prev_probs);

            for (int ci=0;ci<max_width;++ci) {
                scalar_t& val=cur_left_probs[ci];
                val = prev_probs[ci];

                if (prev_probs[ci-1] > val) val=prev_probs[ci-1];
                // text padding allows us to look back past the start of text
                if (text[ci] != text[ci-2] && prev_probs[ci-2] > val) val=prev_probs[ci-2];

                val += logits_view[text[ci]];

            }

            // if float shift max to 0 for numerical stability on long inputs
            if constexpr (std::is_same_v<scalar_t, float>) {
                if (time % RESCALE_MAX_FREQ == 0) rescale_max(cur_left_probs, max_width);
            }
        }

        scalar_t* cur_right_probs = new scalar_t[2+max_width];
        for (int i=0;i<max_width+2;++i) cur_right_probs[i] = mask_val;
        cur_right_probs[max_width-1] = 0;
        prev_probs-=2;
        prev_probs[max_width] = prev_probs[max_width+1] = mask_val;

        logits_view=logits_ptr+logits_right*logits_stride;
        for (int time=logits_right;--time >= split;) {
            std::swap(cur_right_probs, prev_probs);
            logits_view-=logits_stride;

            for (int ci=0;ci<max_width;++ci) {
                scalar_t& val=cur_right_probs[ci];
                val = prev_probs[ci];

                if (prev_probs[ci+1] > val) val=prev_probs[ci+1];
                // text padding allows us to look back past the start of text
                if (text[ci] != text[ci+2] && prev_probs[ci+2] > val) val=prev_probs[ci+2];

                val += logits_view[text[ci]];
            }

            // if float shift max to 0 for numerical stability on long inputs
            if constexpr (std::is_same_v<scalar_t, float>) {
                if (time % RESCALE_MAX_FREQ == 0) rescale_max(cur_right_probs, max_width);
            }
        }

        assert (logits_view == logits_ptr+split*logits_stride);

        int pivot=0;
        scalar_t pivot_prob = mask_val;
        for (int ci=0;ci<max_width;++ci) {
            scalar_t cur = cur_left_probs[ci] + cur_right_probs[ci] - logits_view[text[ci]];
            if (cur > pivot_prob) pivot_prob = cur, pivot=ci;
        }
        pivot += text_left;

        ans[split] = pivot;

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

    Tensor viterbi_cpu(
        const Tensor& log_probs,
        const Tensor& targets,
        const int32_t blank) {

        auto [text, text_len, cont_log_probs, ans] =
            common_setup<DeviceType::CPU, int32_t>(log_probs, targets, blank);
        
        // allow double for log_prob data type
        // but I think ints can be used for all reasonable character sets and times
        auto run = [&]<typename T>() {
            _normal_viterbi_helper<bt_full_byte, T, int32_t>(
                cont_log_probs,
                text,
                ans.mutable_data_ptr<int32_t>(),
                0,
                cont_log_probs.size(0),
                0,
                text_len
            );
        };

        if (log_probs.scalar_type() == torch::headeronly::ScalarType::Double) {
            run.template operator()<double>();
        } else {
            run.template operator()<float>();
        }

        text-=2; // remove the original padding
        delete[] text;
        
        return ans;
    }

    Tensor hirschberg_viterbi_cpu(
        const Tensor& log_probs,
        const Tensor& targets,
        const int32_t blank,
        const int64_t soft_mem_limit) {

        auto [text, text_len, cont_log_probs, ans] =
            common_setup<DeviceType::CPU, int32_t>(log_probs, targets, blank);
            
        // allow double for log_prob data type
        // but I think ints can be used for all reasonable character sets and times
        auto run = [&]<typename T>() {
            _hirschberg_helper<bt_full_byte, T, int32_t>(
                cont_log_probs,
                text,
                ans.mutable_data_ptr<int32_t>(),
                0,
                cont_log_probs.size(0),
                0,
                text_len,
                soft_mem_limit
            );
        };

        if (log_probs.scalar_type() == torch::headeronly::ScalarType::Double) {
            run.template operator()<double>();
        } else {
            run.template operator()<float>();
        }

        text-=2; // remove the original padding
        delete[] text;
        
        return ans;
    }

}