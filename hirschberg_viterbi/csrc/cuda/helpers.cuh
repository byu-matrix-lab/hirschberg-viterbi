// Copyright (c) 2026 Lawry Sorenson
// SPDX-License-Identifier: MIT

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/shim_utils.h>

#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

using torch::stable::Tensor;

using namespace std;

namespace hirschberg_viterbi {

    enum class alignment_type {
        viterbi,
        hirschberg_viterbi
    };

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

    template<typename scalar_t, typename target_t>
    struct alignment_task {
        using scalar_type = scalar_t;
        using target_type = target_t;

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
        int max_width;

        // no soft mem limit because I am using a different recursive ending (and it doesn't depend on each subtask)

        // how should these be allocated (can I use one as a holding position before calculating the others)
        // need to consider if it will be included or not
        scalar_t* cur_probs;
        scalar_t* prev_probs;
        // maybe add two pointers here instead (unified memory, but different params)
        // replace backtrack with another void*
        void* data1;
        void* data2;
    };

    // trailing allocator removes one level of pointer indirection
    template<typename task_t>
    struct alignas(task_t) task_pool {
        static_assert(alignof(task_t) >= alignof(int));
        int num_tasks;
        int capacity;

        // maybe just overwrite the [] operator to return specific indices
        __host__ __device__ task_t* data() {
            static_assert(sizeof(task_t) > sizeof(this));
            return reinterpret_cast<task_t*>(this) + 1;
        }
    };

    struct min_op {
    template <typename T>
    __host__ __device__ constexpr T operator()(const T& a, const T& b) const {
        return (a < b) ? a : b;
    }
    };

    struct max_op {
    template <typename T>
    __host__ __device__ constexpr T operator()(const T& a, const T& b) const {
        return (a > b) ? a : b;
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

    // maybe add some template programming here to select the number of threads
    
    template<typename scalar_t, int threads, bool single_warp>
    __device__ void cuda_rescale_max(scalar_t* array, int size) {
        if constexpr (single_warp || threads == 32) {
            using WarpReduceT = cub::WarpReduce<scalar_t>;
            __shared__ typename WarpReduceT::TempStorage temp_storage[threads / 32];

            auto op = max_op{};

            scalar_t local = -std::numeric_limits<scalar_t>::infinity();
            for (int i=threadIdx.x%32; i < size; i += 32) local = op(local, array[i]);

            int warpInd = threadIdx.x / 32;
            scalar_t block_max = WarpReduceT(temp_storage[warpInd]).Reduce(local, op);
            __shared__ scalar_t block_high[threads / 32];
            if (threadIdx.x%32 == 0) block_high[warpInd] = block_max;

            __syncwarp();

            const scalar_t offset = block_high[warpInd];
            if (offset > -std::numeric_limits<scalar_t>::infinity()) {
                for (int i=threadIdx.x%32; i<size; i += 32) array[i] -= offset;
            }

            __syncwarp();
        } else {
            using BlockReduceT = cub::BlockReduce<scalar_t, threads>;
            __shared__ typename BlockReduceT::TempStorage temp_storage;

            auto op = max_op{};

            scalar_t local = -std::numeric_limits<scalar_t>::infinity();
            for (int i=threadIdx.x; i < size; i += threads) local = op(local, array[i]);

            scalar_t block_max = BlockReduceT(temp_storage).Reduce(local, op);

            __shared__ scalar_t block_high;

            if (threadIdx.x == 0) block_high = block_max;

            __syncthreads();

            const scalar_t offset = block_high;

            if (offset > -std::numeric_limits<scalar_t>::infinity()) {
                for (int i=threadIdx.x; i<size; i += threads) array[i] -= offset;
            }

            // is this needed?
            __syncthreads();
        }
    }

    template<typename scalar_t, bool single_warp = false, int candidate = 32>
    __device__ void cuda_rescale_max_dispatch(scalar_t* array, int size, int threads) {
        if (threads == candidate) {
            cuda_rescale_max<scalar_t, candidate, single_warp>(array, size);
        } else if constexpr (candidate < 1024) {
            cuda_rescale_max_dispatch<scalar_t, single_warp, candidate + 32>(
                array, size, threads);
        } else {
            assert(false);
        }
    }

    template<typename task_t, alignment_type tag, typename backtrack_t = void>
    void reshard_memory(task_pool<task_t>* tasks, uint8_t* mem_pool, size_t mem_size, int n = -1);

    template<typename task_t, alignment_type tag, typename backtrack_t = void>
    size_t compute_reshard_memory_space(int n);

    // TODO: add this to all operations
    inline cudaStream_t get_torch_cuda_stream(int32_t device_index) {
        void* stream = nullptr;
        TORCH_ERROR_CODE_CHECK(
            aoti_torch_get_current_cuda_stream(device_index, &stream));
        return static_cast<cudaStream_t>(stream);
    }

    __host__ int compute_max_tasks(int logits_length);
};
