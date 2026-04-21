#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <cmath>

#include "helpers.h"

namespace py = pybind11;

using namespace std;

int64_t bt_two_bits::needed_size(int n, int width) {
    int64_t ans = (width+3)>>2;
    return ans*n;
}

bt_two_bits::bt_two_bits(int n, int width) {
    max_width = (width+3)>>2;
    int64_t size = (int64_t)max_width * n;
    data = new uint8_t[size];
    for (int64_t i=0;i<size;++i) data[i] = 0;
}

uint8_t bt_two_bits::get(int ti, int ci) {
    uint8_t ans = data[(int64_t)max_width*ti + (ci>>2)];
    ans >>= ci%4*2;
    return ans&3;
}

void bt_two_bits::set(int ti, int ci, uint8_t val) {
    uint8_t& reg = data[(int64_t)max_width*ti + (ci>>2)];
    // can only be called once, otherwise problems
    ci=ci%4*2;
    assert(!(reg&(3<<ci)));
    reg|=val<<ci;
}

bt_two_bits::~bt_two_bits() {
    delete[] data;
}

int64_t bt_full_byte::needed_size(int n, int width) {
    return (int64_t)width*n;
}

bt_full_byte::bt_full_byte(int n, int width) {
    max_width = width;
    int64_t size = (int64_t)width * n;
    data = new uint8_t[size];
    for (int64_t i=0;i<size;++i) data[i] = 0;
}

uint8_t bt_full_byte::get(int ti, int ci) {
    return data[(int64_t)max_width*ti + ci];
}

void bt_full_byte::set(int ti, int ci, uint8_t val) {
    data[(int64_t)max_width*ti + ci] = val;
}

bt_full_byte::~bt_full_byte() {
    delete[] data;
}

template<typename target_t>
std::pair<target_t*, int> add_blanks(
    const py::array_t<target_t>& targets,
    const target_t blank) {
    int n = targets.shape(0);
    target_t* ans = new target_t[2*n+5];


    ans += 2;
    auto targ_view = targets.template unchecked<1>();
    for (int i=0;i<n;++i) {
        ans[2*i] = blank;
        ans[2*i+1] = targ_view(i);
    }
    ans[2*n] = blank;

    // add padding
    ans[-2]=ans[-1]=blank;
    ans[2*n+1] = ans[2*n+2] = blank;

    return {ans, 2*n+1};
}

template std::pair<int32_t*, int> add_blanks<int32_t>(
    const py::array_t<int32_t>& targets,
    const int32_t blank);

// not used, but here in case I choose to later
template std::pair<int64_t*, int> add_blanks<int64_t>(
    const py::array_t<int64_t>& targets,
    const int64_t blank);


// adapted from scipy/xsf
// https://github.com/scipy/xsf/blob/33768a09623689efdf7bcaf0afa167341dda0758/include/xsf/cephes/erfinv.h#L56
// LICENSE AT https://github.com/scipy/xsf/blob/33768a09623689efdf7bcaf0afa167341dda0758/LICENSE
// copied here because I was too lazy to figure out adding it as a dependency for now
// TODO: add it as a dependency
namespace xsf {
    namespace cephes {

        namespace detail {
            constexpr double SQRTPI = 2.50662827463100050242E0;

            /* approximation for 0 <= |y - 0.5| <= 3/8 */
            constexpr double ndtri_P0[5] = {
                -5.99633501014107895267E1, 9.80010754185999661536E1,  -5.66762857469070293439E1,
                1.39312609387279679503E1,  -1.23916583867381258016E0,
            };

            constexpr double ndtri_Q0[8] = {
                /* 1.00000000000000000000E0, */
                1.95448858338141759834E0, 4.67627912898881538453E0,  8.63602421390890590575E1, -2.25462687854119370527E2,
                2.00260212380060660359E2, -8.20372256168333339912E1, 1.59056225126211695515E1, -1.18331621121330003142E0,
            };

            /* Approximation for interval z = sqrt(-2 log y ) between 2 and 8
                * i.e., y between exp(-2) = .135 and exp(-32) = 1.27e-14.
                */
            constexpr double ndtri_P1[9] = {
                4.05544892305962419923E0,   3.15251094599893866154E1,   5.71628192246421288162E1,
                4.40805073893200834700E1,   1.46849561928858024014E1,   2.18663306850790267539E0,
                -1.40256079171354495875E-1, -3.50424626827848203418E-2, -8.57456785154685413611E-4,
            };

            constexpr double ndtri_Q1[8] = {
                /*  1.00000000000000000000E0, */
                1.57799883256466749731E1,   4.53907635128879210584E1,   4.13172038254672030440E1,
                1.50425385692907503408E1,   2.50464946208309415979E0,   -1.42182922854787788574E-1,
                -3.80806407691578277194E-2, -9.33259480895457427372E-4,
            };

            /* Approximation for interval z = sqrt(-2 log y ) between 8 and 64
                * i.e., y between exp(-32) = 1.27e-14 and exp(-2048) = 3.67e-890.
                */

            constexpr double ndtri_P2[9] = {
                3.23774891776946035970E0,  6.91522889068984211695E0,  3.93881025292474443415E0,
                1.33303460815807542389E0,  2.01485389549179081538E-1, 1.23716634817820021358E-2,
                3.01581553508235416007E-4, 2.65806974686737550832E-6, 6.23974539184983293730E-9,
            };

            constexpr double ndtri_Q2[8] = {
                /*  1.00000000000000000000E0, */
                6.02427039364742014255E0,  3.67983563856160859403E0,  1.37702099489081330271E0,  2.16236993594496635890E-1,
                1.34204006088543189037E-2, 3.28014464682127739104E-4, 2.89247864745380683936E-6, 6.79019408009981274425E-9,
            };

        } // namespace detail

        inline double polevl(double x, const double coef[], int N) {
            double ans;
            int i;
            const double *p;

            p = coef;
            ans = *p++;
            i = N;

            do {
                ans = ans * x + *p++;
            } while (--i);

            return (ans);
        }

        inline double p1evl(double x, const double coef[], int N) {
            double ans;
            const double *p;
            int i;

            p = coef;
            ans = x + *p++;
            i = N - 1;

            do
                ans = ans * x + *p++;
            while (--i);

            return (ans);
        }

        inline double ndtri(double y0) {
            double x, y, z, y2, x0, x1;
            int code;

            if (y0 == 0.0) {
                return -std::numeric_limits<double>::infinity();
            }
            if (y0 == 1.0) {
                return std::numeric_limits<double>::infinity();
            }

            code = 1;
            y = y0;
            if (y > (1.0 - 0.13533528323661269189)) { /* 0.135... = exp(-2) */
                y = 1.0 - y;
                code = 0;
            }

            if (y > 0.13533528323661269189) {
                y = y - 0.5;
                y2 = y * y;
                x = y + y * (y2 * polevl(y2, detail::ndtri_P0, 4) / p1evl(y2, detail::ndtri_Q0, 8));
                x = x * detail::SQRTPI;
                return (x);
            }

            x = std::sqrt(-2.0 * std::log(y));
            x0 = x - std::log(x) / x;

            z = 1.0 / x;
            if (x < 8.0) { /* y > exp(-32) = 1.2664165549e-14 */
                x1 = z * polevl(z, detail::ndtri_P1, 8) / p1evl(z, detail::ndtri_Q1, 8);
            } else {
                x1 = z * polevl(z, detail::ndtri_P2, 8) / p1evl(z, detail::ndtri_Q2, 8);
            }
            x = x0 - x1;
            if (code != 0) {
                x = -x;
            }
            return (x);
        }

        double erfcinv(double y) {
            constexpr double domain_lb = 0;
            constexpr double domain_ub = 2;

            if ((domain_lb < y) && (y < domain_ub)) {
                return -ndtri(0.5 * y) * M_SQRT1_2;
            } else if (y == domain_lb) {
                return std::numeric_limits<double>::infinity();
            } else if (y == domain_ub) {
                return -std::numeric_limits<double>::infinity();
            } else assert (false);
        }
    }
}
