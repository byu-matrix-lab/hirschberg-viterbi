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
        // TODO: switch this to the task pool setup as well
        task_pool<alignment_task<scalar_t, target_t>>* pool
    ) {
        // should this be templated to also have the base case for one warp per task? (consider that as a design option)

        // assumption made in my code
        assert(blockDim.x % 32 == 0);

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

        if (task_ind < pool->num_tasks) {
            // run normal alignment

            constexpr scalar_t mask_val = -std::numeric_limits<scalar_t>::infinity();

            auto& task = pool->data()[task_ind];

            scalar_t* cur_probs = task.cur_probs;
            scalar_t* prev_probs = task.prev_probs;

            int text_left = task.text_left, text_right = task.text_right;
            int logits_left = task.logits_left, logits_right = task.logits_right;
            
            int max_width = text_right - text_left;
            for (int i=work_ind-2;i<max_width;i+=threads_per_task) cur_probs[i] = mask_val;

            // can this be done any better?
            if (work_ind == 2) {
                prev_probs[-2] = prev_probs[-1] = mask_val;
                // same warp as above, so this should be fine to be overwritten
                cur_probs[0] = 0;
            }
            
            const target_t* text = task.text + text_left;

            int64_t logits_stride = task.logits_stride;
            const scalar_t* logits_view=task.logits_ptr+logits_left*logits_stride;

            // make a shallow register copy to reduce pointer indirection in the hot loop
            backtrack_t backedges{task.max_width, (uint8_t*)task.data1};

            // all threads done with their setup now
            if constexpr (single_warp_align) __syncwarp();
            else __syncthreads();

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

                if constexpr (single_warp_align) __syncwarp();
                else __syncthreads();
                
                // warp reduce vs block reduce for the shifting here, probably just overallocate memory and use num_valid
                // otherwise consider templating the other function but not this function.
                // if float shift max to 0 for numerical stability on long inputs
                if constexpr (std::is_same_v<scalar_t, float> && RESCALE_MAX_FREQ!=-1) {
                    // TODO: need to be able to select the number of threads
                    if (time % RESCALE_MAX_FREQ == 0) cuda_rescale_max_dispatch<scalar_t, single_warp_align>(cur_probs, max_width, blockDim.x);
                }
            }

            // how to handle syncing (I might want the number of threads )
        }

        if constexpr (single_warp_align) __syncthreads();

        // now run backtracking
        if (threadIdx.x < tasks_per_block) {
            task_ind = blockIdx.x * tasks_per_block + threadIdx.x;
            if (task_ind < pool->num_tasks) {
                // this thread will now backtrack for task_ind

                auto& task = pool->data()[task_ind];

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
                backtrack_t backedges{task.max_width, (uint8_t*)task.data1};
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
        int logits_right,
        int text_right) {

        using task_t = alignment_task<scalar_t, target_t>;
        using pool_t = task_pool<task_t>;

        int max_width = text_right;
        int step = (max_width + 2)*sizeof(scalar_t);
        // 256 to satifsy the memory aligner requirements
        int64_t mem_needed = 256 + 2* sizeof(task_t)+ step*2 + backtrack_t::needed_size(logits_right, max_width);

        // use a tensor to allocate the needed memory pool?
        auto device_holder = make_cuda_unique<uint8_t>(mem_needed);
        auto* data = device_holder.get();

        pool_t* pool = (pool_t*)data;
        data += 2*sizeof(task_t);

        // align up to 128

        // also have this call the sharding kernel, if only to have a single source of truth, though it will add latency otherwize

        // otherwise need alignup and padded space
        static_assert(sizeof(task_t) % alignof(scalar_t) == 0);

        pool_t seed_pool{
            .num_tasks = 1,
            .capacity = 1
        };

        cudaMemcpy(pool, &seed_pool, sizeof(pool_t), cudaMemcpyHostToDevice);

        task_t task{
            .logits_ptr = logits.mutable_data_ptr<scalar_t>(),
            .logits_stride = logits.stride(0),
            .text = text,
            .ans = ans,
            .logits_left = 0,
            .logits_right = logits_right,
            .text_left = 0,
            .text_right = text_right,
            .max_width = max_width,

            .cur_probs = (scalar_t*)(data) + 2,
            .prev_probs = (scalar_t*)(data + step) + 2,
            .data1 = (void*)(data + step*2),
            .data2 = nullptr
        };

        cudaMemcpy(pool->data(), &task, sizeof(task_t), cudaMemcpyHostToDevice);

        // launch the kernel now
        // don't launch more threads than needed (round up to next multiple of 32)
        // cap at 1024 for newer gpus, but 512 for p100 and doubles
        // 1024 hits the register limit for the sms (can we make it a bit smaller) - make small changes to optimize double usage
        int num_threads = clamp((text_right + 1 + 31)&~int{31}, 32, 1024);
        _normal_viterbi_kernel<backtrack_t, scalar_t, target_t><<<1, num_threads>>>(pool);
    }

    template<typename scalar_t, typename target_t>
    __global__ __launch_bounds__(1024) void _hirshberg_viterbi_prework_kernel(
        task_pool<alignment_task<scalar_t, target_t>>* pool
    ) {

        // assumption made in my code
        assert(blockDim.x % 32 == 0);

        int task_ind = blockIdx.x / 2;
        int work_ind = threadIdx.x;
        int threads_per_task = blockDim.x;
        
        // for pascal compatability, not threads can exit untill all sync threads are passed
        if (task_ind >= pool->num_tasks) return;

        constexpr scalar_t mask_val = -std::numeric_limits<scalar_t>::infinity();

        auto& task = pool->data()[task_ind];

        int text_left = task.text_left, text_right = task.text_right;
        int logits_left = task.logits_left, logits_right = task.logits_right;
        
        int max_width = task.max_width;

        const target_t* text = task.text + text_left;

        // this could probably be an int as long as multiplication is upcast
        int64_t logits_stride = task.logits_stride;

        int split = (logits_left + logits_right - 1) >> 1;

        if (blockIdx.x % 2 == 0) {
            // forward pass
            scalar_t* cur_left_probs = task.cur_probs;
            scalar_t* prev_probs = task.prev_probs;

            for (int i=work_ind-2;i<max_width;i+=threads_per_task) cur_left_probs[i] = mask_val;

            if (threadIdx.x == 2) {
                prev_probs[-2] = prev_probs[-1] = mask_val;
                // same thread as above, so this should be fine to be overwritten
                cur_left_probs[0] = 0;
            }

            __syncthreads();

            const scalar_t* logits_view=task.logits_ptr+logits_left*logits_stride;
            for(int time=logits_left; time <= split; ++time, logits_view+=logits_stride) {
                cuda::std::swap(cur_left_probs, prev_probs);

                for (int ci=work_ind;ci<max_width;ci+=threads_per_task) {
                    scalar_t& val=cur_left_probs[ci];
                    val = prev_probs[ci];

                    if (prev_probs[ci-1] > val) val=prev_probs[ci-1];
                    // text padding allows us to look back past the start of text
                    if (text[ci] != text[ci-2] && prev_probs[ci-2] > val) val=prev_probs[ci-2];

                    val += logits_view[text[ci]];
                }

                __syncthreads();

                // if float shift max to 0 for numerical stability on long inputs
                if constexpr (std::is_same_v<scalar_t, float> && RESCALE_MAX_FREQ!=-1) {
                    if (time % RESCALE_MAX_FREQ == 0) cuda_rescale_max_dispatch<scalar_t>(cur_left_probs, max_width, threads_per_task);
                }
            }

        } else {
            // backward pass
            // they get padded on the right, not on the left???
            scalar_t* cur_right_probs = (scalar_t*)task.data1;
            scalar_t* prev_probs = (scalar_t*)task.data2;

            for (int i=work_ind;i<max_width+2;i+=threads_per_task) cur_right_probs[i] = mask_val;
            // which warp should set these (ideally same that sets cur_right_probs first)
            // that way there is no race condition here
            if ((max_width-1)%threads_per_task == work_ind) {
                cur_right_probs[max_width-1] = 0;
                prev_probs[max_width] = prev_probs[max_width+1] = mask_val;
            }

            __syncthreads();

            const scalar_t* logits_view=task.logits_ptr+logits_right*logits_stride;
            for (int time=logits_right;--time >= split;) {
                cuda::std::swap(cur_right_probs, prev_probs);
                logits_view-=logits_stride;

                for (int ci=work_ind;ci<max_width;ci+=threads_per_task) {
                    scalar_t& val=cur_right_probs[ci];
                    val = prev_probs[ci];

                    if (prev_probs[ci+1] > val) val=prev_probs[ci+1];
                    // text padding allows us to look back past the end of text
                    if (text[ci] != text[ci+2] && prev_probs[ci+2] > val) val=prev_probs[ci+2];

                    val += logits_view[text[ci]];
                }

                __syncthreads();

                // if float shift max to 0 for numerical stability on long inputs
                if constexpr (std::is_same_v<scalar_t, float> && RESCALE_MAX_FREQ!=-1) {
                    if (time % RESCALE_MAX_FREQ == 0) cuda_rescale_max_dispatch<scalar_t>(cur_right_probs, max_width, threads_per_task);
                }
            }
        }
    }

    template<typename scalar_t, typename target_t>
    __global__ __launch_bounds__(1024) void _hirshberg_viterbi_postwork_kernel(
        task_pool<alignment_task<scalar_t, target_t>>* in_pool,
        task_pool<alignment_task<scalar_t, target_t>>* out_pool
    ) {
        int task_ind = blockIdx.x;
        int work_ind = threadIdx.x;
        int threads_per_task = blockDim.x;

        if (task_ind >= in_pool->num_tasks) return;

        auto& task = in_pool->data()[task_ind];

        int text_left = task.text_left, text_right = task.text_right;
        int logits_left = task.logits_left, logits_right = task.logits_right;
        
        int max_width = task.max_width;
        int split = (logits_left + logits_right - 1) >> 1;

        const target_t* text = task.text + text_left;

        // this could probably be an int as long as multiplication is upcast
        int64_t logits_stride = task.logits_stride;
        const scalar_t* logits_view = task.logits_ptr+split*logits_stride;

        // should this be here or in the prework kernel?
        scalar_t* cur_left_probs = (split + 1 - logits_left) % 2 == 0 ? task.cur_probs : task.prev_probs;
        scalar_t* cur_right_probs = (scalar_t*)((logits_right - split) % 2 == 0 ? task.data1 : task.data2);

        using psi = cuda::std::pair<scalar_t, int>;
        psi local = {-std::numeric_limits<scalar_t>::infinity(), 0};

        auto op = max_op{};
        for (int ci=work_ind;ci<max_width;ci+=threads_per_task) {
            scalar_t cur = cur_left_probs[ci] + cur_right_probs[ci] - logits_view[text[ci]];
            local = op(local, {cur, ci});
        }

        using BlockReduceT = cub::BlockReduce<psi, 1024>;
        __shared__ typename BlockReduceT::TempStorage temp_storage;
        psi block_max = BlockReduceT(temp_storage).Reduce(local, op);

        if (threadIdx.x == 0) {
            int pivot = block_max.second + text_left;

            task.ans[split] = pivot;

            out_pool->data()[2*task_ind] = {
                .logits_ptr = task.logits_ptr,
                .logits_stride = logits_stride,
                .text = task.text,
                .ans = task.ans,
                .logits_left = logits_left,
                .logits_right = split,
                .text_left = text_left,
                .text_right = pivot+1,
                .max_width = pivot+1-text_left,
            };

            out_pool->data()[2*task_ind+1] = {
                .logits_ptr = task.logits_ptr,
                .logits_stride = logits_stride,
                .text = task.text,
                .ans = task.ans,
                .logits_left = split+1,
                .logits_right = logits_right,
                .text_left = pivot,
                .text_right = text_right,
                .max_width = text_right-pivot,
            };
        }

    }

    template<typename backtrack_t, typename scalar_t, typename target_t>
    void _hirschberg_viterbi_helper(
        const Tensor& logits,
        target_t* text,
        target_t* const ans,
        int logits_right,
        int text_right) {

        using task_t = alignment_task<scalar_t, target_t>;
        using pool_t = task_pool<task_t>;

        // how many divide by 2, rounding up needed until <= 15
        // in practice the cutoff could be a few times greater than this
        int max_tasks = compute_max_tasks(logits_right);
        int max_width = text_right;

        size_t mem_needed = 0;
        
        // could definitely be smaller than this (down to 50% or so)
        // but this is easy and only uses an extra 1.5MB for 3-hour inputs
        // (maximum number of tasks ever +1) * 3 * sizeof(task_t)
        // current tasks, next tasks, base tasks
        size_t mem_per_task_pool = (max_tasks + 1) * sizeof(task_t);
        mem_needed += mem_per_task_pool * 3;
        
        // 128 to pad to cache line
        mem_needed += 128;

        // tasks memory pool (left, right, back1, back2, backpointers)
        // full length included, with one overlap between adjacent tasks
        // each task pads 2 to the left
        size_t mem_per_text_cut = (size_t)(max_width + (2+1) * max_tasks) * sizeof(scalar_t);
        // left/right/prev1/prev2 - mem_per_text_cut * 4
        // left/right/backpointer - mem_per_text_cut * 2 + warpSize * logits size * uint8_t
        size_t branching_tasks = 128 * max_tasks + std::max(mem_per_text_cut*4, mem_per_text_cut*2 + (uintptr_t)backtrack_t::needed_size(logits_right, 32));

        // can these be different, or will they be the same?
        // I might not need to make two calls here
        branching_tasks = max(branching_tasks, compute_reshard_memory_space<task_t, alignment_type::viterbi, backtrack_t>(max_tasks));
        branching_tasks = max(branching_tasks, compute_reshard_memory_space<task_t, alignment_type::hirschberg_viterbi>(max_tasks));

        mem_needed += branching_tasks;

        // use a tensor to allocate the needed memory pool?
        auto device_holder = make_cuda_unique<uint8_t>(mem_needed);
        auto* data = device_holder.get();

        pool_t* cur_pool = (pool_t*)data;
        data += mem_per_task_pool;

        pool_t* next_pool = (pool_t*)data;
        data += mem_per_task_pool;

        pool_t* basecase_pool = (pool_t*)data;
        data += mem_per_task_pool;

        // put working memory on different cache line from input
        align_up_address(data);

        // data is now the memory pool address

        // use this to initalize all pools
        pool_t seed_pool{
            .num_tasks = 0,
            .capacity = max_tasks
        };

        cudaMemcpy(next_pool, &seed_pool, sizeof(pool_t), cudaMemcpyHostToDevice);
        cudaMemcpy(basecase_pool, &seed_pool, sizeof(pool_t), cudaMemcpyHostToDevice);
        
        seed_pool.num_tasks = 1;
        cudaMemcpy(cur_pool, &seed_pool, sizeof(pool_t), cudaMemcpyHostToDevice);

        task_t task{
            .logits_ptr = logits.mutable_data_ptr<scalar_t>(),
            .logits_stride = logits.stride(0),
            .text = text,
            .ans = ans,
            .logits_left = 0,
            .logits_right = logits_right,
            .text_left = 0,
            .text_right = text_right,
            .max_width = max_width,
        };

        // copy the seed task over the the current pool
        cudaMemcpy(cur_pool->data(), &task, sizeof(task_t), cudaMemcpyHostToDevice);

        // loop until everything is solved now

        // number of tasks will double each time until we are close to stopping

        int num_tasks = 1;
        for(int width = logits_right;width > 15;width=width+1>>1) {
            
            // maybe scale number of threads to estimated max_width given current bounds

            reshard_memory<task_t, alignment_type::hirschberg_viterbi>(cur_pool, data, branching_tasks, num_tasks);
            _hirshberg_viterbi_prework_kernel<scalar_t, target_t><<<2*num_tasks, 1024>>>(cur_pool);
            
            // set the number of tasks in the next pool
            int next_tasks = 2*num_tasks;
            cudaMemcpy(&next_pool->num_tasks, &next_tasks, sizeof(int), cudaMemcpyHostToDevice);
            _hirshberg_viterbi_postwork_kernel<scalar_t, target_t><<<num_tasks, 1024>>>(cur_pool, next_pool);

            // TODO: use cub::DevicePartition::If to filter out the base cases early (instead of over recursing)
            // if width < 100 or so
            // filter out any max width <= warpSize for the basecase
            // for a sequence with a lot of silence, maybe still run hirschberg, even if the text is short

            num_tasks = next_tasks;

            swap(cur_pool, next_pool);
        }

        reshard_memory<task_t, alignment_type::viterbi, backtrack_t>(cur_pool, data, branching_tasks, num_tasks);
        _normal_viterbi_kernel<backtrack_t, scalar_t, target_t><<<num_tasks, 1024>>>(cur_pool);
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
                cont_log_probs.size(0),
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
        
        auto [text_holder, text_len, cont_log_probs, ans] =
            common_setup<DeviceType::CUDA, int32_t>(log_probs, targets, blank);

        auto text = text_holder.mutable_data_ptr<int32_t>()+2;

        auto run = [&]<typename scalar_t>() {
            _hirschberg_viterbi_helper<bt_full_byte_device, scalar_t, int32_t>(
                cont_log_probs,
                text,
                ans.mutable_data_ptr<int32_t>(),
                cont_log_probs.size(0),
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

    STABLE_TORCH_LIBRARY_IMPL(hirschberg_viterbi, CUDA, m) {
        m.impl("viterbi", TORCH_BOX(&viterbi_cuda));
        m.impl("hirschberg_viterbi", TORCH_BOX(&hirschberg_viterbi_cuda));
    }

} // end namespace