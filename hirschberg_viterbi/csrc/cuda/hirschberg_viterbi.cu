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
#include <cuda/std/utility>

#include "../helpers.h"
#include "helpers.cuh"

using torch::stable::Tensor;
using torch::headeronly::ScalarType;
using torch::headeronly::DeviceType;

using namespace std;

namespace hirschberg_viterbi {

    template<typename backtrack_t, typename scalar_t, typename target_t, bool single_warp_align = false>
    __global__ __launch_bounds__(1024) void _normal_viterbi_kernel(
        alignment_task<backtrack_t, scalar_t, target_t>* task_pool,
        int num_tasks
    ) {
        // should this be templated to also have the base case for one warp per task? (consider that as a design option)

        // compute the task index

        int task_ind;
        int work_ind;
        int threads_per_task;
        int tasks_per_block;
        if constexpr (single_warp_align) {
            threads_per_task = warpSize;
            tasks_per_block = blockDim.x / threads_per_task;
            task_ind = blockIdx.x * tasks_per_block + threadIdx.x / threads_per_task;
            work_ind = threadIdx.x % threads_per_task;
        } else {
            threads_per_task = blockDim.x;
            tasks_per_block = 1;
            task_ind = blockIdx.x;
            work_ind = threadIdx.x;
        }
        
        // for pascal compatability, not threads can exit untill all sync threads are passed

        if (task_ind < num_tasks) {
            // run normal alignment

            constexpr scalar_t mask_val = -std::numeric_limits<scalar_t>::infinity();

            auto& task = task_pool[task_ind];

            scalar_t* cur_probs = task.cur_probs;
            scalar_t* prev_probs = task.prev_probs;

            int text_left = task.text_left, text_right = task.text_right;
            int logits_left = task.logits_left, logits_right = task.logits_right;
            
            int max_width = text_right - text_left;
            for (int i=work_ind-2;i<max_width;i+=threads_per_task) cur_probs[i] = mask_val;

            // can this be done any better?
            if (threadIdx.x == 0) {
                prev_probs[-2] = prev_probs[-1] = mask_val;
                // same warp as above, so this should be fine to be overwritten
                cur_probs[0] = 0;
            }
            
            const target_t* text = task.text + text_left;

            // printf("%d %d\n", threadIdx.x, task.text_right);

            int logits_stride = task.logits_stride;
            const scalar_t* logits_view=task.logits_ptr+logits_left*logits_stride;

            // make a shallow register copy to reduce pointer indirection in the hot loop
            backtrack_t backedges = task.backtrack;

            // all threads done with their setup now
            if constexpr (!single_warp_align) __syncthreads();

            for(int time=logits_left; time < logits_right; ++time, logits_view+=logits_stride) {
                cuda::std::swap(cur_probs, prev_probs);
                
                for (int ci=work_ind;ci<max_width;ci+=threads_per_task) {
                    scalar_t& val=cur_probs[ci];
                    uint8_t back = 0;
                    val = prev_probs[ci];
                    
                    // text padding allows us to look back past the start of text
                    if (prev_probs[ci-1] > val) val=prev_probs[ci-1], back=1;
                    if (text[ci] != text[ci-2] && prev_probs[ci-2] > val) val=prev_probs[ci-2], back=2;
                    
                    val += logits_view[text[ci]];
                    backedges.set(time-logits_left, ci, back);     
                }

                if constexpr (!single_warp_align) __syncthreads();
                
                // warp reduce vs block reduce for the shifting here, probably just overallocate memory and use num_valid
                // otherwise consider templating the other function but not this function.
                // if float shift max to 0 for numerical stability on long inputs
                if constexpr (std::is_same_v<scalar_t, float> && RESCALE_MAX_FREQ!=-1) {
                    // TODO: need to be able to select the number of threads
                    if (time % RESCALE_MAX_FREQ == 0) cuda_rescale_max_dispatch<scalar_t, single_warp_align>(cur_probs, max_width, threads_per_task);
                }
            }

            // how to handle syncing (I might want the number of threads )
        }

        if constexpr (single_warp_align) __syncthreads();

        // now run backtracking
        if (threadIdx.x < tasks_per_block) {
            task_ind = blockIdx.x * tasks_per_block + threadIdx.x;
            if (task_ind < num_tasks) {
                // this thread will now backtrack for task_ind

                auto& task = task_pool[task_ind];

                int text_left=task.text_left;

                int cur = task.text_right-1;

                // this cur probs isn't the real cur probs output from the task
                scalar_t* cur_probs = (task.logits_right - task.logits_left)%2==0 ? task.cur_probs : task.prev_probs;
                const target_t* text = task.text;

                // won't want to recompute this with the pruned version
                int max_width = task.text_right - text_left;
                scalar_t cur_val = cur_probs[max_width-1];
                if (cur_probs[max_width-2] > cur_val) cur_val=cur_probs[max_width-2],cur=task.text_right-2;
                // this one is needed for hirschberg calls
                if (text[task.text_right-1]!=text[task.text_right-3] && cur_probs[max_width-3] > cur_val) cur_val=cur_probs[max_width-3],cur=task.text_right-3;

                int logits_left = task.logits_left;
                target_t* ans = task.ans;
                backtrack_t backedges = task.backtrack;
                for (int time=task.logits_right; --time >= logits_left; ) {
                    ans[time] = cur;
                    cur -= backedges.get(time-logits_left, cur-text_left);
                }
                
            }
        }
    }

    template<typename backtrack_t, typename scalar_t, typename target_t>
    void _normal_viterbi_helper(
        const Tensor& logits,
        target_t* text,
        target_t* const ans,
        int logits_left,
        int logits_right,
        int text_left,
        int text_right) {
        // this exists to make it match the format of the cpu code more

        // dummy function here just returns 0s for all the alignment
        cudaMemset(ans, 0, (logits_right - logits_left)*4);

        using task_t = alignment_task<backtrack_t, scalar_t, target_t>;

        int max_width = text_right - text_left;
        int step = (max_width + 2)*sizeof(scalar_t);
        int64_t mem_needed = sizeof(task_t) + step*2 + backtrack_t::needed_size(logits_right - logits_left, max_width);

        // use a tensor to allocate the needed memory pool?
        auto device_holder = make_cuda_unique<uint8_t>(mem_needed);
        auto* data = device_holder.get();

        task_t* task_pool = (task_t*)data;
        data += sizeof(task_t);

        // otherwise need alignup and padded space
        static_assert(sizeof(task_t) % alignof(scalar_t) == 0);

        task_t task{
            .logits_ptr = logits.mutable_data_ptr<scalar_t>(),
            .logits_stride = logits.stride(0),
            .text = text,
            .ans = ans,
            .logits_left = logits_left,
            .logits_right = logits_right,
            .text_left = text_left,
            .text_right = text_right,

            .cur_probs = (scalar_t*)(data) + 2,
            .prev_probs = (scalar_t*)(data + step) + 2,
            .backtrack = backtrack_t{max_width, data + step*2},
        };

        cudaMemcpy(task_pool, &task, sizeof(task_t), cudaMemcpyHostToDevice);

        // launch the kernel now
        // don't launch more threads than needed (round up to next multiple of 32)
        // cap at 1024 for newer gpus, but 512 for p100 and doubles
        // 1024 hits the register limit for the sms (can we make it a bit smaller) - make small changes to optimize double usage
        int num_threads = clamp((text_right - text_left + 1 + 31)&~int{31}, 32, 1024);
        _normal_viterbi_kernel<backtrack_t, scalar_t, target_t><<<1, num_threads>>>(task_pool, 1);
        
        // basecase for hv will be single-warp blocks to remove inter warp syncing

        // make sure that all pointers are aligned to their scopes
        // align biggest to smallest?
        // keep this in mind later too

    }

    Tensor viterbi_cuda(
        const Tensor& log_probs,
        const Tensor& targets,
        const int32_t blank=0) {

        auto [text_holder, text_len, cont_log_probs, ans] =
            common_setup<DeviceType::CUDA, int32_t>(log_probs, targets, blank);

        auto text = text_holder.mutable_data_ptr<int32_t>()+2;

        auto run = [&]<typename scalar_t>() {
            _normal_viterbi_helper<bt_full_byte_device, scalar_t, int32_t>(
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