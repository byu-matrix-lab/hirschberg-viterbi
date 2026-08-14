#pragma once

#include <torch/csrc/stable/tensor.h>

using torch::stable::Tensor;
using torch::headeronly::DeviceType;

// I did not see any gains from using this with hirschberg viterbi
struct bt_two_bits {
    int max_width;
    uint8_t* data;
    static int64_t needed_size(int n, int width);
    bt_two_bits(int n, int width);
    uint8_t get(int ti, int ci);
    void set(int ti, int ci, uint8_t val);
    ~bt_two_bits();
};

struct bt_full_byte {
    int max_width;
    uint8_t* data;
    static int64_t needed_size(int n, int width);
    bt_full_byte(int n, int width);
    uint8_t get(int ti, int ci);
    void set(int ti, int ci, uint8_t val);
    ~bt_full_byte();
};

template<typename target_t>
std::pair<target_t*, int> add_blanks(
    const Tensor& targets,
    const target_t blank);

template<DeviceType device, typename target_t>
std::tuple<target_t*, int, Tensor, Tensor> common_setup(
    const Tensor& log_probs,
    const Tensor& targets,
    const target_t blank = 0);

namespace xsf {
    namespace cephes {
        double erfcinv(double y);
    }
}