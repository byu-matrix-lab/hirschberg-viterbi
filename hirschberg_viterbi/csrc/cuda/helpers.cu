// Copyright (c) 2026 Lawry Sorenson
// SPDX-License-Identifier: MIT

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/device.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

#include <torch/csrc/stable/c/shim.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/functional>
#include <cub/cub.cuh> // for prefix sums (useful for calculating placement with scans or memory partitions)

#include <thrust/iterator/iterator_facade.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <cstddef>

#include "../helpers.h"
#include "helpers.cuh"

using torch::stable::Tensor;
using torch::headeronly::ScalarType;
using torch::headeronly::DeviceType;

using namespace std;

namespace hirschberg_viterbi {

    // maybe remove print helpers, I don't know how helpful they are or what format I would want them to take

    template<typename T>
    void print_flat_cpu_pointer(T* const data, int64_t n, int64_t offset) {
        for (int i=0;i<n;++i) cout << data[offset+i] << ' ';
        cout << endl;
    }

    // maybe add a print flat cuda pointer here too that will make a copy for me

    void print_flat_tensor(const Tensor& data, int64_t n=-1, int64_t offset=0) {
        if (n == -1) n = data.numel();
        auto cpu_device = torch::stable::Device("cpu");
        // this could be done in one to() call
        auto cont = torch::stable::contiguous(torch::stable::to(data, cpu_device));

        if (cont.scalar_type() == torch::headeronly::ScalarType::Double) {
            print_flat_cpu_pointer(cont.const_data_ptr<double>(), n, offset);
        } else if (cont.scalar_type() == torch::headeronly::ScalarType::Float) {
            print_flat_cpu_pointer(cont.const_data_ptr<float>(), n, offset);
        } else if (cont.scalar_type() == torch::headeronly::ScalarType::Int) {
            print_flat_cpu_pointer(cont.const_data_ptr<int32_t>(), n, offset);
        }
    }

    template<typename target_t>
    __global__ void add_blanks_kernel(target_t* ans, target_t* const view, int n, target_t blank) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= n) return;

        // edge cases
        if (idx < 3 || idx+3 >= n) {
            ans[idx] = blank;
            return;
        }

        // consider this warp divergence vs having each thread write 2 values
        ans[idx] = (idx&1) ? view[(idx-3)>>1] : blank;
    }
    
    template<typename target_t>
    std::pair<Tensor, int> add_blanks(
        const Tensor& targets,
        const target_t blank,
        device_tag<DeviceType::CUDA>) {
        
        int n = 2*targets.size(0)+5;
        Tensor ans_holder = torch::stable::empty(
            {n},
            torch::headeronly::CppTypeToScalarType<target_t>::value,
            torch::headeronly::Layout::Strided,
            targets.device(),
            false // don't pin memory
        );

        target_t* ans = ans_holder.mutable_data_ptr<target_t>();
        
        // why is this an int and not an int32_t
        auto* const targ_view = targets.const_data_ptr<target_t>();

        int num_blocks = (n+255)/256;

        add_blanks_kernel<target_t><<<num_blocks, 256>>>(ans, (target_t*)targ_view, n, blank);

        n-=4;

        return {std::move(ans_holder), n};
    }

    template std::pair<Tensor, int> add_blanks<int32_t>(
        const Tensor& targets,
        const int32_t blank,
        device_tag<DeviceType::CUDA>);

    struct check_target_response {
        int bad_ind;
        int bad_val;
        int count_repeats;
    };

    // this one is just a sanity check

    template<typename target_t>
    __global__ void check_targets_kernel(target_t* const targets, int target_length, target_t charset_size, target_t blank, check_target_response* ans) {
        // each thread computes first bad index and total number of repeats (will need logic for thread 0)
        int bad_ind = target_length;
        int count = 0;

        for(int i=blockIdx.x * blockDim.x + threadIdx.x;i<target_length;i+=blockDim.x) {
            if (targets[i] <= 0 || targets[i] >= charset_size) {
                bad_ind = i;
                break;
            }
            count += i>0 && targets[i] == targets[i-1];
        }

        // TODO: I think a device scan with thrust could be faster here by sharding across SMs
        // but this is good enough for now

        using BlockReduceT = cub::BlockReduce<int, 1024>;
        __shared__ typename BlockReduceT::TempStorage temp_storage;

        bad_ind = BlockReduceT(temp_storage).Reduce(bad_ind, min_op{});
        __syncthreads(); // could be removed if I use another temp storage
        count = BlockReduceT(temp_storage).Sum(count);

        // accumulate at the end 
        if (threadIdx.x == 0){
            ans->bad_ind = bad_ind;
            if (bad_ind < target_length) ans->bad_val = targets[bad_ind];
            else ans->bad_val = 0;
            ans->count_repeats = count;
        }
    }
    
    template<typename target_t>
    void check_targets(
        const Tensor& targets, 
        int sequence_length,
        target_t charset_size,
        target_t blank,
        device_tag<DeviceType::CUDA>) {

            // targets must be contiguous here
        target_t* const targ_pointer = (target_t*)targets.const_data_ptr<target_t>();

        // will want a struct back from the kernel
        // consider using a tensor holder here for consistency
        auto res_device_holder = make_cuda_unique<check_target_response>(1);

        check_targets_kernel<target_t><<<1, 1024>>>(targ_pointer, targets.size(0), charset_size, blank, res_device_holder.get());

        // TODO: propagate these error checkers through all of the cuda
        auto err = cudaGetLastError();
        STD_TORCH_CHECK(err == cudaSuccess, "check_targets launch failed: ", cudaGetErrorString(err));

        check_target_response res;
        err = cudaMemcpy(&res, res_device_holder.get(), sizeof(check_target_response), cudaMemcpyDeviceToHost);

        STD_TORCH_CHECK(err == cudaSuccess,
                        "check_targets result copy failed: ", cudaGetErrorString(err));
        
        if (res.bad_ind < targets.size(0)) {
            STD_TORCH_CHECK(res.bad_val != blank, "Blank should not occur in targets (index ", res.bad_ind,")");
            STD_TORCH_CHECK(0 < res.bad_val && res.bad_val < charset_size, "Target value ", res.bad_val, "is outside character set [1, ",charset_size,")");
        }

        STD_TORCH_CHECK(
            res.count_repeats + targets.size(0) <= sequence_length,
            "Target sequence too long for CTC. Found log_probs length ",
            sequence_length,
            " < target length ",
            targets.size(0),
            " + ",
            res.count_repeats,
            " repeats."
        );
    }

    template void check_targets<int32_t>(
        const Tensor& targets,
        int sequence_length,
        int32_t charset_size,
        int32_t blank,
        device_tag<DeviceType::CUDA>);
    
    __host__ int compute_max_tasks(int logits_length) {
        int ans = 0;
        // 15 logits can have at most 2*15+1 text positions
        // this ensures the alignment will fit in a single warp
        // repeatedly divide by 2, rounding up, until 15 is reached
        while (logits_length > 15) ++ans, logits_length = (logits_length+1)>>1;
        return 1<<ans;
    }

    template<typename task_t, alignment_type tag, typename backtrack_t = void>
    struct MemoryRef {
        using scalar_t = typename task_t::scalar_type;
        task_t* p;
        uint8_t* mem_pool;
        size_t mem_size; // could be size_t instead

        __host__ __device__
        operator size_t() const {
            size_t ans;
            if constexpr (tag == alignment_type::viterbi) {
                ans = (p->max_width + 2)*sizeof(scalar_t)*2 + backtrack_t::needed_size(p->logits_right - p->logits_left, p->max_width);
            } else {
                // in hirschberg_viterbi
                ans = (p->max_width + 2)*sizeof(scalar_t)*4;
            }
            // pruned ones will need more padding here

            // align up to next cache line
            ans = (ans + size_t{127}) & ~size_t{127};
            return ans;
        }

        __host__ __device__
        MemoryRef& operator=(size_t value) {
            // must change only this task here
            // fill in the pointers correctly

            // could be removed after debug?
            assert(value + (size_t)(*this) <= mem_size);

            uint8_t* allocation = mem_pool + value;

            if constexpr (tag == alignment_type::viterbi) {
                size_t size_per_row = (p->max_width + 2)*sizeof(scalar_t);

                p->cur_probs = (scalar_t*)(allocation) + 2;
                p->prev_probs = (scalar_t*)(allocation + size_per_row) + 2;
                p->data1 = (void*)(allocation + size_per_row*2);
                p->data2 = nullptr;

            } else {
                // in hirschberg_viterbi
                size_t size_per_row = (p->max_width + 2)*sizeof(scalar_t);

                p->cur_probs = (scalar_t*)(allocation) + 2;
                p->prev_probs = (scalar_t*)(allocation + size_per_row) + 2;

                // important that these two are right padded instead of left padded
                p->data1 = (void*)(allocation + size_per_row*2);
                p->data2 = (void*)(allocation + size_per_row*3);
            }
            return *this;
        }

        MemoryRef& operator=(const MemoryRef&) = delete;
    };

    template<typename task_t, alignment_type tag, typename backtrack_t = void>
    struct MakeMemoryRef {
        task_t* tasks;
        uint8_t* mem_pool;
        size_t mem_size;

        __host__ __device__
        MemoryRef<task_t, tag, backtrack_t> operator()(size_t index) const {
            // is this safe?
            return {tasks + index, mem_pool, mem_size};
        }
    };

    template<typename task_t, alignment_type tag, typename backtrack_t = void>
    using MemoryScanIterator = thrust::transform_iterator<
        MakeMemoryRef<task_t, tag, backtrack_t>,
        thrust::counting_iterator<size_t>,
        MemoryRef<task_t, tag, backtrack_t>,
        std::size_t
    >;

    template<typename task_t, alignment_type tag, typename backtrack_t>
    void reshard_memory(task_pool<task_t>* tasks, uint8_t* mem_pool, size_t mem_size, int n) {
        // add path where it copies the size from device if not included
        assert(n != -1);

        assert((uintptr_t)mem_pool % 128 == 0);

        using iterator_t = MemoryScanIterator<task_t, tag, backtrack_t>;
        using maker_t = MakeMemoryRef<task_t, tag, backtrack_t>;

        // // d_tasks points to the first task in device memory.
        iterator_t it(thrust::counting_iterator<size_t>{0}, maker_t{tasks->data(), mem_pool, mem_size});

        auto err = cub::DeviceScan::ExclusiveSum(mem_pool, mem_size, it, it, n);

        STD_TORCH_CHECK(err == cudaSuccess, "failed to partition memory pool between tasks", cudaGetErrorString(err));
    }

    template<typename task_t, alignment_type tag, typename backtrack_t>
    size_t compute_reshard_memory_space(int n) {

        using iterator_t = MemoryScanIterator<task_t, tag, backtrack_t>;
        using maker_t = MakeMemoryRef<task_t, tag, backtrack_t>;

        // // d_tasks points to the first task in device memory.
        iterator_t it(thrust::counting_iterator<size_t>{0}, maker_t{(task_t*)nullptr, (uint8_t*)nullptr, 0});

        size_t ans = 0;

        auto err = cub::DeviceScan::ExclusiveSum(nullptr, ans, it, it, n);

        // TODO: add more of these
        STD_TORCH_CHECK(err == cudaSuccess, "failed to compute memory needed for partitioning kernel", cudaGetErrorString(err));

        return ans;
    }

    template void reshard_memory<
            alignment_task<double, int32_t>,
            alignment_type::viterbi,
            bt_full_byte_device>(
        task_pool<alignment_task<double, int32_t>>* tasks,
        uint8_t* mem_pool,
        size_t mem_size,
        int n);

    template void reshard_memory<
            alignment_task<float, int32_t>,
            alignment_type::viterbi,
            bt_full_byte_device>(
        task_pool<alignment_task<float, int32_t>>* tasks,
        uint8_t* mem_pool,
        size_t mem_size,
        int n);

    template void reshard_memory<
            alignment_task<double, int32_t>,
            alignment_type::hirschberg_viterbi>(
        task_pool<alignment_task<double, int32_t>>* tasks,
        uint8_t* mem_pool,
        size_t mem_size,
        int n);

    template void reshard_memory<
            alignment_task<float, int32_t>,
            alignment_type::hirschberg_viterbi>(
        task_pool<alignment_task<float, int32_t>>* tasks,
        uint8_t* mem_pool,
        size_t mem_size,
        int n);

    template size_t compute_reshard_memory_space<
            alignment_task<double, int32_t>,
            alignment_type::hirschberg_viterbi>(int n);

    template size_t compute_reshard_memory_space<
            alignment_task<double, int32_t>,
            alignment_type::viterbi,
            bt_full_byte_device>(int n);

    template size_t compute_reshard_memory_space<
            alignment_task<float, int32_t>,
            alignment_type::hirschberg_viterbi>(int n);

    template size_t compute_reshard_memory_space<
            alignment_task<float, int32_t>,
            alignment_type::viterbi,
            bt_full_byte_device>(int n);

};
