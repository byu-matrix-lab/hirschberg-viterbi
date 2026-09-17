
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

using namespace std;

namespace hirschberg_viterbi {

    template<typename scalar_t>
    inline pair<scalar_t, scalar_t> get_bounds(
        int i,
        int n,
        int duration,
        double var_rat,
        double stds,
        int num_dels,
        int num_inser,
        double padding) {

        int left, right;

        int curn = max(1, n-num_dels+num_inser);

        double mean = (double)duration / curn;
        double var = var_rat * mean * mean;

        if (num_dels >= i-1) {
            left = 0;
        } else {
            int curi = i-num_dels;

            double comb_var = var / (1.0 / curi + 1.0 / (curn - curi));
            left = ceil(max(0.0, mean*curi - max(padding, stds * sqrt(comb_var))));
        }

        if (num_dels >= n-1-i) {
            right = duration;
        } else {
            int curi = i+num_inser;

            double comb_var = var / (1.0 / curi + 1.0 / (curn - curi));
            right = (int)(min((double)duration, mean*curi + max(padding, stds * sqrt(comb_var))));
        }

        return {left, right};
    }

    template<typename scalar_t>
    std::pair<scalar_t*, scalar_t*> calculate_bounds(
        int duration,
        int n,
        double var_rat,
        double confidence,
        double accuracy,
        double precision,
        double recall,
        double padding) {
        if (precision == -1) precision = accuracy;
        if (recall == -1) recall = accuracy;

        // Compute the inverse normal cdf for confidence adjustment
        // double p = 1 - (1 - confidence) / 2 / n;
        // double stds = erfinv(2*p-1) * sqrt(2);
        double stds = sqrt(2) * xsf::cephes::erfcinv((1 - confidence) / n);

        scalar_t* lower_bounds = new scalar_t[duration];
        scalar_t* upper_bounds = new scalar_t[duration];

        for (int i=0;i<duration;++i) {
            lower_bounds[i] = 2*n+1;
            upper_bounds[i] = -1;
        }

        int dels = ceil(n * (1 - precision));
        int inser = ceil(n * precision * (1 / recall - 1));

        for (int ci=0;ci<n;++ci) {
            auto [left, right] = get_bounds<scalar_t>(ci+1, n+1, duration-1, var_rat, stds, dels, inser, padding);

            int true_i = 2*ci+1;

            lower_bounds[right] = min(lower_bounds[right], true_i-1);
            upper_bounds[left] = max(upper_bounds[left], true_i+2); // +2 to be exclusive
        }

        for (int i=1;i<duration;++i) upper_bounds[i] = max(upper_bounds[i], upper_bounds[i-1]);
        for (int i=duration-1;i--;) lower_bounds[i] = min(lower_bounds[i], lower_bounds[i+1]);

        for (int i=0;i<duration; ++i) assert(lower_bounds[i] < upper_bounds[i]);

        return {lower_bounds, upper_bounds};
    }

    template<typename backtrack_t, typename scalar_t, typename target_t>
    void _pruned_normal_viterbi_helper(
        const Tensor& logits,
        target_t* text,
        target_t* const ans,
        int logits_left,
        int logits_right,
        int text_left,
        int text_right,
        target_t* lower_bounds,
        target_t* upper_bounds,
        target_t* widths,
        int max_width) {

        constexpr scalar_t mask_val = -std::numeric_limits<scalar_t>::infinity();

        auto* const logits_ptr = logits.const_data_ptr<scalar_t>();
        // 64-bit to avoid potential overflow for audio past 17.4 hours
        int64_t logits_stride = logits.stride(0);

        scalar_t* cur_probs = new scalar_t[4+max_width];
        scalar_t* prev_probs = new scalar_t[4+max_width];

        for (int i=0;i<5;++i) cur_probs[i] = mask_val;
        prev_probs[0] = prev_probs[1] = mask_val;
        cur_probs[2] = 0;

        backtrack_t backedges(logits_right - logits_left, max_width);

        cur_probs+=2;
        prev_probs+=2;
        const scalar_t* logits_view=logits_ptr+logits_left*logits_stride;
        for(int time=logits_left; time < logits_right; ++time, logits_view+=logits_stride) {
            swap(cur_probs, prev_probs);

            // TODO: move these outside of the loop
            int pstart = time>logits_left ? lower_bounds[time-1] : text_left;
            int pwidth = time>logits_left ? widths[time-1] : 1;

            // THIS IS WRONG HERE
            int shift = lower_bounds[time] - pstart;
            assert (shift >= 0); // check monotonic, TODO: remove

            // ci + shift >= 0
            // int cstart = max(0, -shift);

            // ci + shift - 2 < pwidth
            int cend = min(widths[time], pwidth + 2 - shift);

            // TODO: optimize this out to just the sides
            // excluding what will be done later
            for (int ci=cend;ci<widths[time]+2;++ci) cur_probs[ci] = mask_val;

            text += lower_bounds[time];
            prev_probs+=shift;
            for (int ci=0;ci<cend;++ci) {
                scalar_t& val=cur_probs[ci];
                uint8_t back = 0;

                val = prev_probs[ci];

                if (prev_probs[ci-1] > val) val=prev_probs[ci-1], back=1;
                // text padding allows us to look back past the start of text
                if (text[ci] != text[ci-2] && prev_probs[ci-2] > val) val=prev_probs[ci-2], back=2;

                val += logits_view[text[ci]];
                backedges.set(time-logits_left, ci, back);     
            }
            prev_probs-=shift;
            text -= lower_bounds[time];

            if constexpr (std::is_same_v<scalar_t, float> && RESCALE_MAX_FREQ!=-1) {
                if (time % RESCALE_MAX_FREQ == 0) rescale_max(cur_probs, cend);
            }
        }

        int cur = text_right-1;
        scalar_t* end_pointer = cur_probs + widths[logits_right-1];
        scalar_t cur_val = end_pointer[-1];
        if (end_pointer[-2] > cur_val) cur_val=end_pointer[-2],cur=text_right-2;
        // this one is needed for hirschberg calls
        assert (text_right > 0);
        if (text[text_right-1]!=text[text_right-3] && end_pointer[-3] > cur_val) cur_val=end_pointer[-3],cur=text_right-3;

        assert (cur_val != mask_val);

        cur_probs-=2;
        prev_probs-=2;

        delete[] cur_probs;
        delete[] prev_probs;

        for (int time=logits_right; --time >= logits_left; ) {
            ans[time] = cur;
            cur -= backedges.get(time-logits_left, cur-lower_bounds[time]);
        }

    }

    template<typename backtrack_t, typename scalar_t, typename target_t>
    void _pruned_hirschberg_helper(
        const Tensor& logits,
        target_t* text,
        target_t* const ans,
        int logits_left,
        int logits_right,
        int text_left,
        int text_right,
        target_t* lower_bounds,
        target_t* upper_bounds,
        target_t* widths,
        int64_t soft_mem_limit) {

        if (logits_left >= logits_right) return;

        if (text_left + 1 == text_right) {
            for (int time=logits_left;time<logits_right;++time) ans[time] = text_left;
            return;
        }

        // TODO: consider recalculating bounds with each recursion
        int max_width = 0;
        for (int time=logits_left; time<logits_right; ++time) {
            lower_bounds[time] = max(lower_bounds[time], text_left);
            upper_bounds[time] = min(upper_bounds[time], text_right);
            assert (lower_bounds[time] < upper_bounds[time]);
            widths[time] = upper_bounds[time] - lower_bounds[time];
            max_width = max(max_width, widths[time]);
        }

        if (backtrack_t::needed_size(logits_right-logits_left, max_width) <= soft_mem_limit) {
            _pruned_normal_viterbi_helper<backtrack_t, scalar_t, target_t>(
                logits,
                text,
                ans,
                logits_left,
                logits_right,
                text_left,
                text_right,
                lower_bounds,
                upper_bounds,
                widths,
                max_width
            );
            return;
        }
        
        constexpr scalar_t mask_val = -std::numeric_limits<scalar_t>::infinity();

        auto* const logits_ptr = logits.const_data_ptr<scalar_t>();
        // 64-bit to avoid potential overflow for audio past 17.4 hours
        int64_t logits_stride = logits.stride(0);

        scalar_t* cur_left_probs = new scalar_t[4+max_width];
        scalar_t* prev_probs = new scalar_t[4+max_width];

        for (int i=0;i<5;++i) cur_left_probs[i] = mask_val;
        prev_probs[0] = prev_probs[1] = mask_val;
        cur_left_probs[2] = 0;

        // for easier math now
        cur_left_probs+=2;
        prev_probs+=2;

        // split-1 is not needed, but done to make sure it matches the python impl during testing
        int split = (logits_left + logits_right-1) >> 1;

        const scalar_t* logits_view=logits_ptr+logits_left*logits_stride;
        for(int time=logits_left; time <= split; ++time, logits_view+=logits_stride) {
            swap(cur_left_probs, prev_probs);

            int pstart = time>logits_left ? lower_bounds[time-1] : text_left;
            int pwidth = time>logits_left ? widths[time-1] : 1;

            int shift = lower_bounds[time] - pstart;
            assert (shift >= 0); // check monotonic

            // ci + shift >= 0
            // int cstart = max(0, -shift);

            // ci + shift - 2 < pwidth
            int cend = min(widths[time], pwidth + 2 - shift);

            for (int ci=cend;ci<widths[time]+2;++ci) cur_left_probs[ci] = mask_val;

            text += lower_bounds[time];
            prev_probs+=shift;
            for (int ci=0;ci<cend;++ci) {
                scalar_t& val=cur_left_probs[ci];

                val = prev_probs[ci];

                if (prev_probs[ci-1] > val) val=prev_probs[ci-1];
                // text padding allows us to look back past the start of text
                if (text[ci] != text[ci-2] && prev_probs[ci-2] > val) val=prev_probs[ci-2];

                val += logits_view[text[ci]];
            }
            prev_probs-=shift;
            text -= lower_bounds[time];

            if constexpr (std::is_same_v<scalar_t, float> && RESCALE_MAX_FREQ!=-1) {
                if (time % RESCALE_MAX_FREQ == 0) rescale_max(cur_left_probs, cend);
            }
        }

        scalar_t* cur_right_probs = new scalar_t[4+max_width];
        for (int i=0;i<5;++i) cur_right_probs[i] = mask_val;
        cur_right_probs[2] = 0;

        cur_right_probs += 2;
        logits_view=logits_ptr+logits_right*logits_stride;
        for (int time=logits_right; --time >= split;) {
            swap(cur_right_probs, prev_probs);
            logits_view-=logits_stride;

            int pstart = time<logits_right-1 ? lower_bounds[time+1] : text_right-1;
            int pwidth = time<logits_right-1 ? widths[time+1] : 1;

            int shift = lower_bounds[time] - pstart;
            assert (shift <= 0); // check monotonic

            // ci + shift + 2 >= 0
            int cstart = max(0, -shift-2);

            // ci + shift < pwidth
            int cend = min(widths[time], pwidth - shift);

            for (int ci=0;ci<cstart;++ci) cur_right_probs[ci] = mask_val;
            for (int ci=cend;ci<widths[time]+2;++ci) cur_right_probs[ci] = mask_val;

            text += lower_bounds[time];
            prev_probs+=shift;
            for (int ci=cstart;ci<cend;++ci) {
                scalar_t& val=cur_right_probs[ci];

                val = prev_probs[ci];

                if (prev_probs[ci+1] > val) val=prev_probs[ci+1];
                // text padding allows us to look back past the start of text
                if (text[ci] != text[ci+2] && prev_probs[ci+2] > val) val=prev_probs[ci+2];

                val += logits_view[text[ci]];
            }
            prev_probs-=shift;
            text -= lower_bounds[time];

            if constexpr (std::is_same_v<scalar_t, float> && RESCALE_MAX_FREQ!=-1) {
                if (time % RESCALE_MAX_FREQ == 0) rescale_max(cur_right_probs+cstart, cend-cstart);
            }
        }

        int pivot=0;
        scalar_t pivot_prob = mask_val;
        text += lower_bounds[split];
        for (int ci=0;ci<widths[split];++ci) {
            scalar_t cur = cur_left_probs[ci] + cur_right_probs[ci] - logits_view[text[ci]];
            if (cur > pivot_prob) pivot_prob = cur, pivot=ci;
        }
        text -= lower_bounds[split];
        pivot += lower_bounds[split];

        ans[split] = pivot;

        cur_right_probs-=2;
        cur_left_probs-=2;
        prev_probs-=2;
        delete[] cur_left_probs;
        delete[] cur_right_probs;
        delete[] prev_probs;

        // recurse left
        _pruned_hirschberg_helper<backtrack_t, scalar_t, target_t>(
            logits,
            text,
            ans,
            logits_left,
            split,
            text_left,
            pivot+1,
            lower_bounds,
            upper_bounds,
            widths,
            soft_mem_limit
        );

        // recurse right
        _pruned_hirschberg_helper<backtrack_t, scalar_t, target_t>(
            logits,
            text,
            ans,
            split+1,
            logits_right,
            pivot,
            text_right,
            lower_bounds,
            upper_bounds,
            widths,
            soft_mem_limit
        );
    }

    Tensor pruned_viterbi_cpu(
        const Tensor& log_probs,
        const Tensor& targets,
        const int32_t blank = 0,
        const double var_rat = 3.7,
        const double confidence = 0.99,
        const double accuracy = 0.97,
        const double precision = -1.0,
        const double recall = -1.0,
        const double padding = 375.0) {

        pruning_check(var_rat, confidence, accuracy, precision, recall, padding);

        auto [text, text_len, cont_log_probs, ans] =
            common_setup<DeviceType::CPU, int32_t>(log_probs, targets, blank);
        
        int32_t T = cont_log_probs.size(0);

        auto [lower_bounds, upper_bounds] = calculate_bounds<int32_t>(
            T,
            targets.size(0),
            var_rat,
            confidence,
            accuracy,
            precision,
            recall,
            padding
        );

        int32_t* widths = new int32_t[T];
        int32_t max_width = 0;

        for (int i=0;i<T;++i) {
            widths[i] = upper_bounds[i] - lower_bounds[i];
            max_width = max(max_width, widths[i]);
        }
        
        // allow double for log_prob data type
        // but I think ints can be used for all reasonable character sets and times
        auto run = [&]<typename scalar_t>() {
            _pruned_normal_viterbi_helper<bt_full_byte, scalar_t, int32_t>(
                cont_log_probs,
                text,
                ans.mutable_data_ptr<int32_t>(),
                0,
                T,
                0,
                text_len,
                lower_bounds,
                upper_bounds,
                widths,
                max_width
            );
        };

        if (log_probs.scalar_type() == torch::headeronly::ScalarType::Double) {
            run.template operator()<double>();
        } else {
            run.template operator()<float>();
        }

        text-=2; // remove the original padding
        delete[] text;
        delete[] lower_bounds;
        delete[] upper_bounds;
        delete[] widths;

        return ans;
    }

    Tensor pruned_hirschberg_viterbi_cpu(
        const Tensor& log_probs,
        const Tensor& targets,
        const int32_t blank = 0,
        const double var_rat = 3.7,
        const double confidence = 0.99,
        const double accuracy = 0.97,
        const double precision = -1.0,
        const double recall = -1.0,
        const double padding = 375.0,
        const int64_t soft_mem_limit=1000LL) {

        pruning_check(var_rat, confidence, accuracy, precision, recall, padding);

        auto [text, text_len, cont_log_probs, ans] =
            common_setup<DeviceType::CPU, int32_t>(log_probs, targets, blank);
        
        int32_t T = cont_log_probs.size(0);
        
        auto [lower_bounds, upper_bounds] = calculate_bounds<int32_t>(
            T,
            targets.size(0),
            var_rat,
            confidence,
            accuracy,
            precision,
            recall,
            padding
        );

        int32_t* widths = new int32_t[T];
        int32_t max_width = 0;

        for (int i=0;i<T;++i) {
            widths[i] = upper_bounds[i] - lower_bounds[i];
            max_width = max(max_width, widths[i]);
        }

        // allow double for log_prob data type
        // but I think ints can be used for all reasonable character sets and times
        auto run = [&]<typename scalar_t>() {
            _pruned_hirschberg_helper<bt_full_byte, scalar_t, int32_t>(
                cont_log_probs,
                text,
                ans.mutable_data_ptr<int32_t>(),
                0,
                T,
                0,
                text_len,
                lower_bounds,
                upper_bounds,
                widths,
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
        delete[] lower_bounds;
        delete[] upper_bounds;
        delete[] widths;

        return ans;

    }

}