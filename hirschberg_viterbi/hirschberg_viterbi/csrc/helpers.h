#pragma once

#include <torch/csrc/stable/tensor.h>

// how often should this happen?
#define RESCALE_MAX_FREQ 5000

namespace hirschberg_viterbi {

    // I did not see any gains from using this with hirschberg viterbi
    struct bt_two_bits {
        int max_width;
        uint8_t* data;
        static int64_t needed_size(int n, int width) {
            int64_t ans = (width+3)>>2;
            return ans*n;
        }

        bt_two_bits(int n, int width) {
            max_width = (width+3)>>2;
            int64_t size = (int64_t)max_width * n;
            data = new uint8_t[size];
            for (int64_t i=0;i<size;++i) data[i] = 0;
        }

        uint8_t get(int ti, int ci) {
            uint8_t ans = data[(int64_t)max_width*ti + (ci>>2)];
            ans >>= ci%4*2;
            return ans&3;
        }

        void set(int ti, int ci, uint8_t val) {
            uint8_t& reg = data[(int64_t)max_width*ti + (ci>>2)];
            // can only be called once, otherwise problems
            ci=ci%4*2;
            assert(!(reg&(3<<ci)));
            reg|=val<<ci;
        }

        ~bt_two_bits() {
            delete[] data;
        }
    };

    struct bt_full_byte {
        int max_width;
        uint8_t* data;
        static int64_t needed_size(int n, int width) {
            return (int64_t)width*n;
        }

        bt_full_byte(int n, int width) {
            max_width = width;
            int64_t size = (int64_t)width * n;
            data = new uint8_t[size];
            // initialization not needed because compute will fill it in
            // for (int64_t i=0;i<size;++i) data[i] = 0;
        }

        uint8_t get(int ti, int ci) {
            return data[(int64_t)max_width*ti + ci];
        }

        void set(int ti, int ci, uint8_t val) {
            data[(int64_t)max_width*ti + ci] = val;
        }

        ~bt_full_byte() {
            delete[] data;
        }
    };

    template<typename target_t>
    std::pair<target_t*, int> add_blanks(
        const torch::stable::Tensor& targets,
        const target_t blank);

    template<torch::headeronly::DeviceType device, typename target_t>
    std::tuple<target_t*, int, torch::stable::Tensor, torch::stable::Tensor> common_setup(
        const torch::stable::Tensor& log_probs,
        const torch::stable::Tensor& targets,
        const target_t blank = 0);

    template<typename scalar_t>
    inline void rescale_max(scalar_t* array, int size) {
        scalar_t high = -std::numeric_limits<scalar_t>::infinity();
        for (int i=size;i--;) high=std::max(high, array[i]);
        for (int i=size;i--;) array[i]-=high;
    }

    void pruning_check(
        double var_rat,
        double confidence, 
        double accuracy,
        double precision,
        double recall,
        double padding);
}


namespace xsf {
    namespace cephes {
        double erfcinv(double y);
    }
}