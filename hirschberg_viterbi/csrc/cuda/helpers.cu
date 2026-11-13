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

#include "../helpers.h"

using torch::stable::Tensor;
using torch::headeronly::ScalarType;
using torch::headeronly::DeviceType;

using namespace std;

namespace hirschberg_viterbi {

    struct MinOp {
    template <typename T>
    __host__ __device__ constexpr T operator()(const T& a, const T& b) const {
        return (a < b) ? a : b;
    }
    };

    struct CudaDeleter {
        void operator()(void* ptr) const noexcept {
            if (ptr) {
                cudaFree(ptr);
            }
        }
    };

    // 2. Create a convenient type alias (supports both single items and arrays T[])
    template <typename T>
    using cuda_unique_ptr = std::unique_ptr<T, CudaDeleter>;

    // 3. Helper factory function (like std::make_unique)
    template <typename T>
    cuda_unique_ptr<T[]> make_cuda_unique(size_t count) {
        T* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, count * sizeof(T));
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMalloc failed: ") + cudaGetErrorString(err));
        }
        return cuda_unique_ptr<T[]>(ptr);
    }

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

    struct CheckTargetResponse{
        int bad_ind;
        int bad_val;
        int count_repeats;
    };

    // this one is just a sanity check

    template<typename target_t>
    __global__ void check_targets_kernel(target_t* const targets, int target_length, target_t charset_size, target_t blank, CheckTargetResponse* ans) {
        // each thread computes first bad index and total number of repeats (will need logic for thread 0)
        int bad_ind = target_length;
        int count = 0;

        for(int i=blockIdx.x * blockDim.x + threadIdx.x;i<target_length;i+=blockDim.x) {
            if (targets[i] <= 0 || targets[i] >= charset_size) bad_ind = i;
            // with the assumption that data is passed correctly, does this sync slow down the kernel (likely yes)
            if (__any_sync(0xffffffffu, bad_ind < target_length)) break;
            count += i>0 && targets[i] == targets[i-1];
        }

        // TODO: I think a device scan with thrust could be faster here by sharding across SMS
        // but this is good enough for now

        using BlockReduceT = cub::BlockReduce<int, 1024>;
        __shared__ typename BlockReduceT::TempStorage temp_storage;

        bad_ind = BlockReduceT(temp_storage).Reduce(bad_ind, MinOp{});
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
        auto res_device_holder = make_cuda_unique<CheckTargetResponse>(1);

        check_targets_kernel<target_t><<<1, 1024>>>(targ_pointer, targets.size(0), charset_size, blank, res_device_holder.get());

        CheckTargetResponse res;
        cudaMemcpy(&res, res_device_holder.get(), sizeof(CheckTargetResponse), cudaMemcpyDeviceToHost);
        
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
};
