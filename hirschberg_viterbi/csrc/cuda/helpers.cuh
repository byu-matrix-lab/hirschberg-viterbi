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

using torch::stable::Tensor;

using namespace std;

namespace hirschberg_viterbi {

    // rounds up an address to 128-bytes
    // I think this might avoid cache conflicts between sms
    // but I would want to profile it (against a no-op version)
    template<uintptr_t A = 128, typename T>
    __host__ __device__ void align_up_address(T*& addr) {
        static_assert((A & (A - 1)) == 0 && A > 0, "Alignment A must be a power of 2");
        addr = (T*)(((uintptr_t)addr + A-1) & ~uintptr_t{A-1});
    }
    
    template<typename S, typename T>
    __host__ __device__ void align_up_address(T*& addr) {
        align_up_address<alignof(S)>(addr);
    }

    // will need a new backtracking data structure that handles pooled device memory
    struct bt_full_byte_device {
        int max_width;
        uint8_t* data;

        static __host__  __device__ int64_t needed_size(int n, int width) {
            return (int64_t)width*n;
        }

        __host__ __device__ bt_full_byte_device(int width, uint8_t* buffer) {
            max_width = width;
            data = (uint8_t*)buffer;
        }

        __device__ uint8_t get(int ti, int ci) {
            return data[(int64_t)max_width*ti + ci];
        }

        __device__ void set(int ti, int ci, uint8_t val) {
            data[(int64_t)max_width*ti + ci] = val;
        }
    };

    template<typename backtrack_t, typename scalar_t, typename target_t>
    struct alignment_task {
        // these four fields are const across recursion levels. Potentially remove them
        // but they could be useful for alignment multiple sequences at once
        const scalar_t* logits_ptr;
        int64_t logits_stride;
        const target_t* text;
        target_t* ans;
        int logits_left;
        int logits_right;
        int text_left;
        int text_right;

        // no soft mem limit because I am using a different recursive ending (and it doesn't depend on each subtask)

        // how should these be allocated (can I use one as a holding position before calculating the others)
        // need to consider if it will be included or not
        scalar_t* cur_probs;
        scalar_t* prev_probs;
        backtrack_t backtrack;
    };

    struct min_op {
    template <typename T>
    __host__ __device__ constexpr T operator()(const T& a, const T& b) const {
        return (a < b) ? a : b;
    }
    };

    struct cuda_deleter {
        void operator()(void* ptr) const noexcept {
            if (ptr) {
                cudaFree(ptr);
            }
        }
    };

    // 2. Create a convenient type alias (supports both single items and arrays T[])
    template <typename T>
    using cuda_unique_ptr = std::unique_ptr<T, cuda_deleter>;

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

    // TODO: add a print tensor function here

};
