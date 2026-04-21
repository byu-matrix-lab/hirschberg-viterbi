#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

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
    const py::array_t<target_t>& targets,
    const target_t blank);

namespace xsf {
    namespace cephes {
        double erfcinv(double y);
    }
}