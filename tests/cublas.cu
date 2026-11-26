/*
 * CUDA for CPU allows you to compile and execute CUDA kernels on CPUs.
 *
 * Copyright (C) 2014 Javier Cabezas
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// cuBLAS (cublas_v2.h) against a reference: every routine, in each of its
// types, with every operation, triangle, side and diagonal it takes,
// strides (negative ones in level 1 and 2), leading dimensions larger than
// the rows, and NaN where the BLAS must not read (the other triangle, a
// unit diagonal, C when beta is 0). The reference is written apart from the
// library: it builds each matrix in full from its storage (band, packed,
// a triangle) and computes in long double. Also the mixed-precision
// products, the device's pointers for scalars, calls captured in a graph
// (scalars read when the graph runs, or when captured), the copies, the
// management functions and the errors. CTest runs it with cuda4cpu's own
// BLAS (CUDA4CPU_BLAS=builtin) and with the CBLAS library it finds.

#include <cublas_v2.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

namespace {

using cf = std::complex<float>;
using cd = std::complex<double>;
using lc = std::complex<long double>;

int failures = 0;

void check(bool ok, const char *what, int line)
{
    if (!ok) {
        if (++failures <= 40)
            std::printf("line %d: %s\n", line, what);
    }
}

#define CHECK(cond) check((cond), #cond, __LINE__)
#define OK(call) check((call) == CUBLAS_STATUS_SUCCESS, #call, __LINE__)

template <typename T> constexpr bool is_complex = !std::is_same_v<T, float> && !std::is_same_v<T, double>;
template <typename T> struct real_of { using type = T; };
template <typename T> struct real_of<std::complex<T>> { using type = T; };
template <typename T> using real_t = typename real_of<T>::type;

// The API's types for the test's
inline float *api(float *p) { return p; }
inline const float *api(const float *p) { return p; }
inline double *api(double *p) { return p; }
inline const double *api(const double *p) { return p; }
inline cuComplex *api(cf *p) { return reinterpret_cast<cuComplex *>(p); }
inline const cuComplex *api(const cf *p) { return reinterpret_cast<const cuComplex *>(p); }
inline cuDoubleComplex *api(cd *p) { return reinterpret_cast<cuDoubleComplex *>(p); }
inline const cuDoubleComplex *api(const cd *p) { return reinterpret_cast<const cuDoubleComplex *>(p); }

template <typename T, typename S, typename D, typename C, typename Z>
constexpr auto pick(S s, D d, C c, Z z)
{
    if constexpr (std::is_same_v<T, float>)
        return s;
    else if constexpr (std::is_same_v<T, double>)
        return d;
    else if constexpr (std::is_same_v<T, cf>)
        return c;
    else
        return z;
}

#define BLAS(name) pick<T>(cublasS##name, cublasD##name, cublasC##name, cublasZ##name)

std::mt19937 generator(12345);

template <typename T>
T random_value()
{
    std::uniform_real_distribution<double> d(-1, 1);
    if constexpr (is_complex<T>)
        return T(real_t<T>(d(generator)), real_t<T>(d(generator)));
    else
        return T(d(generator));
}

template <typename T>
std::vector<T> random_vector(std::size_t n)
{
    std::vector<T> v(n);
    for (T &e : v)
        e = random_value<T>();
    return v;
}

template <typename T>
T nan_value()
{
    const real_t<T> nan = std::numeric_limits<real_t<T>>::quiet_NaN();
    if constexpr (is_complex<T>)
        return T(nan, nan);
    else
        return nan;
}

template <typename T>
bool is_nan(T x)
{
    if constexpr (is_complex<T>)
        return std::isnan(x.real()) || std::isnan(x.imag());
    else
        return std::isnan(x);
}

template <typename T>
lc as_ld(T x)
{
    if constexpr (is_complex<T>)
        return lc(x.real(), x.imag());
    else
        return lc(x, 0);
}

template <typename T>
long double tolerance()
{
    return std::is_same_v<real_t<T>, float> ? 2e-5L : 1e-13L;
}

//! Whether got is want to within the type's precision, relative to the
//! magnitude the terms of a sum of n products could have
template <typename T>
bool close(T got, lc want, long double n = 1)
{
    return std::abs(as_ld(got) - want) <= tolerance<T>() * (n + 1) * (1 + std::abs(want));
}

template <typename T>
bool close_all(const std::vector<T> &got, const std::vector<lc> &want, long double n)
{
    if (got.size() != want.size())
        return false;
    for (std::size_t i = 0; i < got.size(); ++i)
        if (!close(got[i], want[i], n))
            return false;
    return true;
}

//! The BLAS's element k of a strided vector: a negative increment starts at
//! the end
inline std::ptrdiff_t at(int k, int n, int inc)
{
    return inc >= 0 ? std::ptrdiff_t(k) * inc : std::ptrdiff_t(n - 1 - k) * -inc;
}

inline std::size_t length(int n, int inc)
{
    return n == 0 ? 0 : 1 + std::size_t(n - 1) * std::abs(inc);
}

lc conj_if(lc x, bool c) { return c ? std::conj(x) : x; }

//! A full matrix (column-major, rows x cols) of long doubles
struct full {
    int rows, cols;
    std::vector<lc> e;

    full(int r, int c) : rows(r), cols(c), e(std::size_t(r) * c) {}
    lc &operator()(int i, int j) { return e[i + std::size_t(j) * rows]; }
    lc operator()(int i, int j) const { return e[i + std::size_t(j) * rows]; }
};

template <typename T>
full dense(const std::vector<T> &a, int rows, int cols, int lda)
{
    full f(rows, cols);
    for (int j = 0; j < cols; ++j)
        for (int i = 0; i < rows; ++i)
            f(i, j) = as_ld(a[i + std::size_t(j) * lda]);
    return f;
}

//! op(A) of a full matrix
full apply(const full &a, cublasOperation_t o)
{
    if (o == CUBLAS_OP_N)
        return a;
    full r(a.cols, a.rows);
    for (int j = 0; j < a.cols; ++j)
        for (int i = 0; i < a.rows; ++i)
            r(j, i) = conj_if(a(i, j), o == CUBLAS_OP_C);
    return r;
}

full multiply(const full &a, const full &b)
{
    full r(a.rows, b.cols);
    for (int j = 0; j < b.cols; ++j)
        for (int i = 0; i < a.rows; ++i) {
            lc s = 0;
            for (int l = 0; l < a.cols; ++l)
                s += a(i, l) * b(l, j);
            r(i, j) = s;
        }
    return r;
}

//! The index of (i, j), i <= j (upper) or i >= j (lower), in packed storage
inline std::size_t packed(int i, int j, int n, bool upper)
{
    return upper ? i + std::size_t(j) * (j + 1) / 2 : i + std::size_t(j) * (2 * n - j - 1) / 2;
}

//! A matrix with storage for its upper or lower triangle: the stored
//! triangle random, the other NaN (never read), the diagonal real for
//! hermitian matrices, and NaN when unit (triangular)
struct triangle_case {
    bool upper;
    bool unit;
};

constexpr cublasOperation_t ops[] = {CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C};
constexpr cublasFillMode_t fills[] = {CUBLAS_FILL_MODE_UPPER, CUBLAS_FILL_MODE_LOWER};
constexpr cublasDiagType_t diags[] = {CUBLAS_DIAG_NON_UNIT, CUBLAS_DIAG_UNIT};
constexpr cublasSideMode_t sides[] = {CUBLAS_SIDE_LEFT, CUBLAS_SIDE_RIGHT};

// ----------------------------------------------------------------------------
// Level 1

template <typename T>
void level1(cublasHandle_t h)
{
    const int n = 13;
    const T alpha = random_value<T>();
    for (const auto [incx, incy] : {std::pair{1, 1}, std::pair{2, -3}, std::pair{-1, 2}}) {
        const auto x = random_vector<T>(length(n, incx));
        const auto y0 = random_vector<T>(length(n, incy));

        // axpy, copy, swap
        auto y = y0;
        OK(BLAS(axpy)(h, n, api(&alpha), api(x.data()), incx, api(y.data()), incy));
        bool ok = true;
        for (int k = 0; k < n; ++k)
            ok &= close(y[at(k, n, incy)], as_ld(alpha) * as_ld(x[at(k, n, incx)]) + as_ld(y0[at(k, n, incy)]), 2);
        CHECK(ok);
        y = y0;
        OK(BLAS(copy)(h, n, api(x.data()), incx, api(y.data()), incy));
        ok = true;
        for (int k = 0; k < n; ++k)
            ok &= y[at(k, n, incy)] == x[at(k, n, incx)];
        CHECK(ok);
        auto xs = x;
        y = y0;
        OK(BLAS(swap)(h, n, api(xs.data()), incx, api(y.data()), incy));
        ok = true;
        for (int k = 0; k < n; ++k)
            ok &= y[at(k, n, incy)] == x[at(k, n, incx)] && xs[at(k, n, incx)] == y0[at(k, n, incy)];
        CHECK(ok);

        // dot products
        lc u = 0, cj = 0;
        for (int k = 0; k < n; ++k) {
            u += as_ld(x[at(k, n, incx)]) * as_ld(y0[at(k, n, incy)]);
            cj += std::conj(as_ld(x[at(k, n, incx)])) * as_ld(y0[at(k, n, incy)]);
        }
        T result{};
        if constexpr (is_complex<T>) {
            const auto dotu = pick<T>(0, 0, cublasCdotu, cublasZdotu);
            const auto dotc = pick<T>(0, 0, cublasCdotc, cublasZdotc);
            OK(dotu(h, n, api(x.data()), incx, api(y0.data()), incy, api(&result)));
            CHECK(close(result, u, n));
            OK(dotc(h, n, api(x.data()), incx, api(y0.data()), incy, api(&result)));
            CHECK(close(result, cj, n));
        } else {
            const auto dot = pick<T>(cublasSdot, cublasDdot, 0, 0);
            OK(dot(h, n, x.data(), incx, y0.data(), incy, &result));
            CHECK(close(result, u, n));
        }

        // rot: real c, and real or complex s
        const real_t<T> c = 0.6;
        const T s = is_complex<T> ? random_value<T>() : T(0.8);
        auto xr = x;
        y = y0;
        OK(BLAS(rot)(h, n, api(xr.data()), incx, api(y.data()), incy, &c, api(&s)));
        ok = true;
        for (int k = 0; k < n; ++k) {
            const lc xi = as_ld(x[at(k, n, incx)]), yi = as_ld(y0[at(k, n, incy)]);
            ok &= close(xr[at(k, n, incx)], lc(c) * xi + as_ld(s) * yi, 2) &&
                  close(y[at(k, n, incy)], lc(c) * yi - std::conj(as_ld(s)) * xi, 2);
        }
        CHECK(ok);
        if constexpr (is_complex<T>) {
            const real_t<T> sr = 0.8;
            const auto srot = pick<T>(0, 0, cublasCsrot, cublasZdrot);
            xr = x;
            y = y0;
            OK(srot(h, n, api(xr.data()), incx, api(y.data()), incy, &c, &sr));
            ok = true;
            for (int k = 0; k < n; ++k) {
                const lc xi = as_ld(x[at(k, n, incx)]), yi = as_ld(y0[at(k, n, incy)]);
                ok &= close(xr[at(k, n, incx)], lc(c) * xi + lc(sr) * yi, 2) &&
                      close(y[at(k, n, incy)], lc(c) * yi - lc(sr) * xi, 2);
            }
            CHECK(ok);
        }

        // rotm, with each kind of H
        if constexpr (!is_complex<T>) {
            const auto rotm = pick<T>(cublasSrotm, cublasDrotm, 0, 0);
            for (T flag : {T(-2), T(-1), T(0), T(1)}) {
                const T param[5] = {flag, T(0.5), T(-0.25), T(0.75), T(1.5)};
                T h11 = 1, h21 = 0, h12 = 0, h22 = 1;
                if (flag == -1)
                    h11 = param[1], h21 = param[2], h12 = param[3], h22 = param[4];
                else if (flag == 0)
                    h21 = param[2], h12 = param[3];
                else if (flag == 1)
                    h11 = param[1], h21 = -1, h12 = 1, h22 = param[4];
                xr = x;
                y = y0;
                OK(rotm(h, n, xr.data(), incx, y.data(), incy, param));
                ok = true;
                for (int k = 0; k < n; ++k) {
                    const lc xi = as_ld(x[at(k, n, incx)]), yi = as_ld(y0[at(k, n, incy)]);
                    ok &= close(xr[at(k, n, incx)], lc(h11) * xi + lc(h12) * yi, 2) &&
                          close(y[at(k, n, incy)], lc(h21) * xi + lc(h22) * yi, 2);
                }
                CHECK(ok);
            }
        }
    }

    // Reductions, which take positive increments
    for (int incx : {1, 3}) {
        auto x = random_vector<T>(length(n, incx));
        x[at(7, n, incx)] = T(3);
        x[at(9, n, incx)] = T(-3); // a tie: the first
        x[at(4, n, incx)] = T(0);
        int index = -1;
        OK(pick<T>(cublasIsamax, cublasIdamax, cublasIcamax, cublasIzamax)(h, n, api(x.data()), incx, &index));
        CHECK(index == 8);
        OK(pick<T>(cublasIsamin, cublasIdamin, cublasIcamin, cublasIzamin)(h, n, api(x.data()), incx, &index));
        CHECK(index == 5);
        long double sum = 0, squares = 0;
        for (int k = 0; k < n; ++k) {
            const lc e = as_ld(x[at(k, n, incx)]);
            sum += std::abs(e.real()) + std::abs(e.imag());
            squares += std::norm(e);
        }
        real_t<T> r = 0;
        OK(pick<T>(cublasSasum, cublasDasum, cublasScasum, cublasDzasum)(h, n, api(x.data()), incx, &r));
        CHECK(close(r, sum, n));
        OK(pick<T>(cublasSnrm2, cublasDnrm2, cublasScnrm2, cublasDznrm2)(h, n, api(x.data()), incx, &r));
        CHECK(close(r, std::sqrt(squares), n));

        // scal, and complex vectors' real scal
        auto xs = x;
        const T alpha2 = random_value<T>();
        OK(BLAS(scal)(h, n, api(&alpha2), api(xs.data()), incx));
        bool ok = true;
        for (int k = 0; k < n; ++k)
            ok &= close(xs[at(k, n, incx)], as_ld(alpha2) * as_ld(x[at(k, n, incx)]));
        CHECK(ok);
        if constexpr (is_complex<T>) {
            const real_t<T> ra = -1.5;
            xs = x;
            OK(pick<T>(0, 0, cublasCsscal, cublasZdscal)(h, n, &ra, api(xs.data()), incx));
            ok = true;
            for (int k = 0; k < n; ++k)
                ok &= close(xs[at(k, n, incx)], lc(ra) * as_ld(x[at(k, n, incx)]));
            CHECK(ok);
        }
    }

    // Norms of numbers whose squares overflow
    const real_t<T> big = std::is_same_v<real_t<T>, float> ? real_t<T>(3e30) : real_t<T>(3e300);
    const T large[2] = {T(big), T(big * 4 / 3)};
    real_t<T> norm = 0;
    OK(pick<T>(cublasSnrm2, cublasDnrm2, cublasScnrm2, cublasDznrm2)(h, 2, api(large), 1, &norm));
    CHECK(close(norm, as_ld(big) * 5.0L / 3.0L, 2));

    // Empty vectors: zero results
    int index = 7;
    OK(pick<T>(cublasIsamax, cublasIdamax, cublasIcamax, cublasIzamax)(h, 0, api(large), 1, &index));
    CHECK(index == 0);

    // rotg: r = (a b) rotated, with b zeroed
    for (int trial = 0; trial < 5; ++trial) {
        T a = random_value<T>(), b = random_value<T>();
        const T a0 = a, b0 = b;
        real_t<T> c = 0;
        T s{};
        OK(BLAS(rotg)(h, api(&a), api(&b), &c, api(&s)));
        CHECK(close(a, lc(c) * as_ld(a0) + as_ld(s) * as_ld(b0), 2));
        CHECK(std::abs(-std::conj(as_ld(s)) * as_ld(a0) + lc(c) * as_ld(b0)) < tolerance<T>() * 10);
        CHECK(std::abs(lc(c) * lc(c) + std::norm(as_ld(s)) - 1.0L) < tolerance<T>() * 10);
    }
    if constexpr (!is_complex<T>) {
        T a = 3, b = 4, c = 0, s = 0;
        OK(BLAS(rotg)(h, &a, &b, &c, &s));
        CHECK(close(a, 5) && close(c, 0.6L) && close(s, 0.8L) && close(b, 1 / 0.6L));
        a = 0, b = 0;
        OK(BLAS(rotg)(h, &a, &b, &c, &s));
        CHECK(c == 1 && s == 0 && a == 0 && b == 0);

        // rotmg: H takes (x1, y1) to (x1', 0), and d1' x1'^2 = d1 x1^2 + d2 y1^2
        const auto rotmg = pick<T>(cublasSrotmg, cublasDrotmg, 0, 0);
        for (const auto [d1v, d2v, x1v, y1v] : {std::array<T, 4>{2, 3, 4, 5}, std::array<T, 4>{1, 1, T(1e-5), 2},
                                                 std::array<T, 4>{5, 0.5, 2, -1}, std::array<T, 4>{1e8, 1, 1, 1e-4}}) {
            T d1 = d1v, d2 = d2v, x1 = x1v, param[5] = {};
            OK(rotmg(h, &d1, &d2, &x1, &y1v, param));
            T h11 = 1, h21 = 0, h12 = 0, h22 = 1;
            if (param[0] == -1)
                h11 = param[1], h21 = param[2], h12 = param[3], h22 = param[4];
            else if (param[0] == 0)
                h21 = param[2], h12 = param[3];
            else if (param[0] == 1)
                h11 = param[1], h21 = -1, h12 = 1, h22 = param[4];
            const long double xn = (long double)h11 * x1v + (long double)h12 * y1v;
            const long double yn = (long double)h21 * x1v + (long double)h22 * y1v;
            CHECK(std::abs(yn) <= tolerance<T>() * 10 * (std::abs(xn) + 1));
            CHECK(std::abs(xn - x1) <= tolerance<T>() * 10 * (std::abs(xn) + 1));
            const long double before = (long double)d1v * x1v * x1v + (long double)d2v * y1v * y1v;
            CHECK(std::abs((long double)d1 * x1 * x1 - before) <= tolerance<T>() * 10 * before);
        }
        T d1 = 2, d2 = 3, x1 = 4, y1 = 0, param[5] = {};
        OK(rotmg(h, &d1, &d2, &x1, &y1, param));
        CHECK(param[0] == -2);
    }
}

// ----------------------------------------------------------------------------
// Level 2

template <typename T>
void level2_general(cublasHandle_t h)
{
    const int m = 7, n = 5, lda = 9;
    const T alpha = random_value<T>(), beta = random_value<T>();
    const auto A = random_vector<T>(std::size_t(lda) * n);
    const full a = dense(A, m, n, lda);
    for (cublasOperation_t o : ops) {
        const full oa = apply(a, o);
        for (const auto [incx, incy] : {std::pair{1, 1}, std::pair{2, -1}}) {
            const auto x = random_vector<T>(length(oa.cols, incx));
            const auto y0 = random_vector<T>(length(oa.rows, incy));
            auto y = y0;
            OK(BLAS(gemv)(h, o, m, n, api(&alpha), api(A.data()), lda, api(x.data()), incx, api(&beta),
                          api(y.data()), incy));
            bool ok = true;
            for (int i = 0; i < oa.rows; ++i) {
                lc s = 0;
                for (int j = 0; j < oa.cols; ++j)
                    s += oa(i, j) * as_ld(x[at(j, oa.cols, incx)]);
                ok &= close(y[at(i, oa.rows, incy)], as_ld(alpha) * s + as_ld(beta) * as_ld(y0[at(i, oa.rows, incy)]), n);
            }
            CHECK(ok);
        }
        // beta 0: y isn't read
        const T zero{};
        std::vector<T> x = random_vector<T>(oa.cols), y(oa.rows, nan_value<T>());
        OK(BLAS(gemv)(h, o, m, n, api(&alpha), api(A.data()), lda, api(x.data()), 1, api(&zero), api(y.data()), 1));
        bool ok = true;
        for (int i = 0; i < oa.rows; ++i) {
            lc s = 0;
            for (int j = 0; j < oa.cols; ++j)
                s += oa(i, j) * as_ld(x[j]);
            ok &= close(y[i], as_ld(alpha) * s, n);
        }
        CHECK(ok);

        // Band storage: kl subdiagonals and ku superdiagonals
        const int kl = 2, ku = 1, ldb = kl + ku + 2;
        std::vector<T> band(std::size_t(ldb) * n, nan_value<T>());
        full b(m, n);
        for (int j = 0; j < n; ++j)
            for (int i = std::max(0, j - ku); i <= std::min(m - 1, j + kl); ++i) {
                band[(ku + i - j) + std::size_t(j) * ldb] = random_value<T>();
                b(i, j) = as_ld(band[(ku + i - j) + std::size_t(j) * ldb]);
            }
        const full ob = apply(b, o);
        const auto xb = random_vector<T>(ob.cols);
        const auto yb0 = random_vector<T>(ob.rows);
        auto yb = yb0;
        OK(BLAS(gbmv)(h, o, m, n, kl, ku, api(&alpha), api(band.data()), ldb, api(xb.data()), 1, api(&beta),
                      api(yb.data()), 1));
        ok = true;
        for (int i = 0; i < ob.rows; ++i) {
            lc s = 0;
            for (int j = 0; j < ob.cols; ++j)
                s += ob(i, j) * as_ld(xb[j]);
            ok &= close(yb[i], as_ld(alpha) * s + as_ld(beta) * as_ld(yb0[i]), n);
        }
        CHECK(ok);
    }

    // Batches
    const int batch = 3;
    std::vector<std::vector<T>> As, xs, ys, y0s;
    std::vector<const T *> Ap, xp;
    std::vector<T *> yp;
    for (int i = 0; i < batch; ++i) {
        As.push_back(random_vector<T>(std::size_t(lda) * n));
        xs.push_back(random_vector<T>(n));
        y0s.push_back(random_vector<T>(m));
    }
    ys = y0s;
    for (int i = 0; i < batch; ++i) {
        Ap.push_back(As[i].data());
        xp.push_back(xs[i].data());
        yp.push_back(ys[i].data());
    }
    using P = decltype(api(Ap[0]));
    OK(BLAS(gemvBatched)(h, CUBLAS_OP_N, m, n, api(&alpha), reinterpret_cast<const P *>(Ap.data()), lda,
                         reinterpret_cast<const P *>(xp.data()), 1, api(&beta),
                         reinterpret_cast<decltype(api(yp[0])) *>(yp.data()), 1, batch));
    bool ok = true;
    for (int b = 0; b < batch; ++b) {
        const full ab = dense(As[b], m, n, lda);
        for (int i = 0; i < m; ++i) {
            lc s = 0;
            for (int j = 0; j < n; ++j)
                s += ab(i, j) * as_ld(xs[b][j]);
            ok &= close(ys[b][i], as_ld(alpha) * s + as_ld(beta) * as_ld(y0s[b][i]), n);
        }
    }
    CHECK(ok);
    std::vector<T> As2(std::size_t(lda) * n * batch), xs2(std::size_t(n) * batch), ys2(std::size_t(m) * batch);
    for (int b = 0; b < batch; ++b) {
        std::copy(As[b].begin(), As[b].end(), As2.begin() + std::size_t(lda) * n * b);
        std::copy(xs[b].begin(), xs[b].end(), xs2.begin() + std::size_t(n) * b);
        std::copy(y0s[b].begin(), y0s[b].end(), ys2.begin() + std::size_t(m) * b);
    }
    OK(BLAS(gemvStridedBatched)(h, CUBLAS_OP_N, m, n, api(&alpha), api(As2.data()), lda, (long long)lda * n,
                                api(xs2.data()), 1, n, api(&beta), api(ys2.data()), 1, m, batch));
    ok = true;
    for (int b = 0; b < batch; ++b)
        for (int i = 0; i < m; ++i)
            ok &= close(ys2[i + std::size_t(m) * b], as_ld(ys[b][i]), n);
    CHECK(ok);

    // Rank-1 updates of general matrices: ger (real), geru and gerc
    const auto x = random_vector<T>(length(m, 2));
    const auto y = random_vector<T>(length(n, -1));
    for (bool conjugate : {false, true}) {
        if (conjugate && !is_complex<T>)
            continue;
        auto B = A;
        if constexpr (is_complex<T>) {
            const auto f = conjugate ? pick<T>(0, 0, cublasCgerc, cublasZgerc) : pick<T>(0, 0, cublasCgeru, cublasZgeru);
            OK(f(h, m, n, api(&alpha), api(x.data()), 2, api(y.data()), -1, api(B.data()), lda));
        } else {
            OK(pick<T>(cublasSger, cublasDger, 0, 0)(h, m, n, &alpha, x.data(), 2, y.data(), -1, B.data(), lda));
        }
        ok = true;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i)
                ok &= close(B[i + std::size_t(j) * lda],
                            a(i, j) + as_ld(alpha) * as_ld(x[at(i, m, 2)]) * conj_if(as_ld(y[at(j, n, -1)]), conjugate), 2);
        CHECK(ok);
    }
}

//! A symmetric (or hermitian) or triangular matrix's storage, and the full
//! matrix: dense (lda), band (k diagonals, lda >= k + 1) or packed
enum class storage { dense, band, packed };

template <typename T>
struct stored_matrix {
    std::vector<T> data;
    full m;
    int lda;
};

//! The stored triangle random (the diagonal real when hermitian, and NaN
//! when unit), everything else NaN; m is the full symmetric/hermitian
//! matrix (triangular: the triangle, with ones on a unit diagonal)
template <typename T>
stored_matrix<T> make_stored(int n, int k, storage s, bool upper, bool herm, bool triangular, bool unit)
{
    const int lda = s == storage::dense ? n + 1 : s == storage::band ? k + 2 : 1;
    const std::size_t size = s == storage::packed ? std::size_t(n) * (n + 1) / 2 : std::size_t(lda) * n;
    stored_matrix<T> r{std::vector<T>(size, nan_value<T>()), full(n, n), lda};
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            if (upper ? i > j : i < j)
                continue;
            if (s == storage::band && std::abs(i - j) > k)
                continue;
            const std::size_t index = s == storage::dense  ? i + std::size_t(j) * lda
                                      : s == storage::band ? (upper ? (k + i - j) : (i - j)) + std::size_t(j) * lda
                                                           : packed(i, j, n, upper);
            T value = random_value<T>();
            if (i == j && herm)
                value = T(std::real(value));
            if (i == j && triangular)
                value += T(4); // well conditioned
            if (i == j && unit) {
                r.data[index] = nan_value<T>();
                r.m(i, j) = 1;
                continue;
            }
            r.data[index] = value;
            r.m(i, j) = as_ld(value);
            if (!triangular && i != j)
                r.m(j, i) = conj_if(as_ld(value), herm);
        }
    }
    return r;
}

template <typename T>
void level2_symmetric(cublasHandle_t h)
{
    const int n = 6, k = 2;
    const T alpha = random_value<T>(), beta = random_value<T>();
    for (cublasFillMode_t uplo : fills) {
        const bool upper = uplo == CUBLAS_FILL_MODE_UPPER;
        for (bool herm : {false, true}) {
            if (herm && !is_complex<T>)
                continue;
            for (storage s : {storage::dense, storage::band, storage::packed}) {
                // Real band and packed: sbmv and spmv; complex: hbmv and hpmv
                if (s != storage::dense && is_complex<T> != herm)
                    continue;
                const auto S = make_stored<T>(n, k, s, upper, herm, false, false);
                const auto x = random_vector<T>(length(n, 2));
                const auto y0 = random_vector<T>(length(n, -1));
                auto y = y0;
                const T *a = S.data.data();
                if (s == storage::dense) {
                    if constexpr (is_complex<T>) {
                        if (herm)
                            OK(pick<T>(0, 0, cublasChemv, cublasZhemv)(h, uplo, n, api(&alpha), api(a), S.lda,
                                                                       api(x.data()), 2, api(&beta), api(y.data()), -1));
                        else
                            OK(BLAS(symv)(h, uplo, n, api(&alpha), api(a), S.lda, api(x.data()), 2, api(&beta),
                                          api(y.data()), -1));
                    } else {
                        OK(BLAS(symv)(h, uplo, n, &alpha, a, S.lda, x.data(), 2, &beta, y.data(), -1));
                    }
                } else if (s == storage::band) {
                    if constexpr (is_complex<T>)
                        OK(pick<T>(0, 0, cublasChbmv, cublasZhbmv)(h, uplo, n, k, api(&alpha), api(a), S.lda,
                                                                   api(x.data()), 2, api(&beta), api(y.data()), -1));
                    else
                        OK(pick<T>(cublasSsbmv, cublasDsbmv, 0, 0)(h, uplo, n, k, &alpha, a, S.lda, x.data(), 2, &beta,
                                                                   y.data(), -1));
                } else {
                    if constexpr (is_complex<T>)
                        OK(pick<T>(0, 0, cublasChpmv, cublasZhpmv)(h, uplo, n, api(&alpha), api(a), api(x.data()), 2,
                                                                   api(&beta), api(y.data()), -1));
                    else
                        OK(pick<T>(cublasSspmv, cublasDspmv, 0, 0)(h, uplo, n, &alpha, a, x.data(), 2, &beta,
                                                                   y.data(), -1));
                }
                bool ok = true;
                for (int i = 0; i < n; ++i) {
                    lc sum = 0;
                    for (int j = 0; j < n; ++j)
                        sum += S.m(i, j) * as_ld(x[at(j, n, 2)]);
                    ok &= close(y[at(i, n, -1)], as_ld(alpha) * sum + as_ld(beta) * as_ld(y0[at(i, n, -1)]), n);
                }
                check(ok, herm ? "hermitian matrix times vector" : "symmetric matrix times vector", __LINE__);
            }

            // Rank-1 and rank-2 updates: dense (syr, her, syr2, her2) and
            // packed (spr, hpr, spr2, hpr2, real or hermitian)
            for (storage s : {storage::dense, storage::packed}) {
                if (s == storage::packed && is_complex<T> != herm)
                    continue;
                for (bool two : {false, true}) {
                    auto S = make_stored<T>(n, 0, s, upper, herm, false, false);
                    const auto before = S.data;
                    const auto x = random_vector<T>(length(n, 2));
                    const auto y = random_vector<T>(n);
                    const real_t<T> ralpha = 0.75;
                    T *a = S.data.data();
                    if constexpr (is_complex<T>) {
                        if (herm && s == storage::dense && !two)
                            OK(pick<T>(0, 0, cublasCher, cublasZher)(h, uplo, n, &ralpha, api(x.data()), 2, api(a), S.lda));
                        else if (herm && s == storage::dense)
                            OK(pick<T>(0, 0, cublasCher2, cublasZher2)(h, uplo, n, api(&alpha), api(x.data()), 2,
                                                                       api(y.data()), 1, api(a), S.lda));
                        else if (herm && !two)
                            OK(pick<T>(0, 0, cublasChpr, cublasZhpr)(h, uplo, n, &ralpha, api(x.data()), 2, api(a)));
                        else if (herm)
                            OK(pick<T>(0, 0, cublasChpr2, cublasZhpr2)(h, uplo, n, api(&alpha), api(x.data()), 2,
                                                                       api(y.data()), 1, api(a)));
                        else if (!two)
                            OK(BLAS(syr)(h, uplo, n, api(&alpha), api(x.data()), 2, api(a), S.lda));
                        else
                            OK(BLAS(syr2)(h, uplo, n, api(&alpha), api(x.data()), 2, api(y.data()), 1, api(a), S.lda));
                    } else {
                        if (s == storage::dense && !two)
                            OK(BLAS(syr)(h, uplo, n, &alpha, x.data(), 2, a, S.lda));
                        else if (s == storage::dense)
                            OK(BLAS(syr2)(h, uplo, n, &alpha, x.data(), 2, y.data(), 1, a, S.lda));
                        else if (!two)
                            OK(pick<T>(cublasSspr, cublasDspr, 0, 0)(h, uplo, n, &ralpha, x.data(), 2, a));
                        else
                            OK(pick<T>(cublasSspr2, cublasDspr2, 0, 0)(h, uplo, n, &alpha, x.data(), 2, y.data(), 1, a));
                    }
                    const lc al = herm && !two ? lc(ralpha) : (!is_complex<T> && s == storage::packed && !two)
                                                                  ? lc(ralpha)
                                                                  : as_ld(alpha);
                    bool ok = true;
                    for (int j = 0; j < n; ++j) {
                        for (int i = 0; i < n; ++i) {
                            const bool in = upper ? i <= j : i >= j;
                            const std::size_t index = s == storage::dense ? i + std::size_t(j) * S.lda
                                                                          : (in ? packed(i, j, n, upper) : 0);
                            if (!in) {
                                if (s == storage::dense)
                                    ok &= is_nan(S.data[index]);
                                continue;
                            }
                            const lc xi = as_ld(x[at(i, n, 2)]), xj = as_ld(x[at(j, n, 2)]);
                            const lc yi = as_ld(y[i]), yj = as_ld(y[j]);
                            lc want = as_ld(before[index]);
                            if (!two)
                                want += al * xi * conj_if(xj, herm);
                            else
                                want += al * xi * conj_if(yj, herm) + conj_if(al, herm) * yi * conj_if(xj, herm);
                            if (herm && i == j)
                                want = lc(want.real());
                            ok &= close(S.data[index], want, 3);
                            if (herm && i == j)
                                ok &= std::imag(S.data[index]) == 0;
                        }
                    }
                    check(ok, two ? "rank-2 update" : "rank-1 update", __LINE__);
                }
            }
        }
    }
}

template <typename T>
void level2_triangular(cublasHandle_t h)
{
    const int n = 6, k = 2;
    for (cublasFillMode_t uplo : fills)
        for (cublasOperation_t o : ops)
            for (cublasDiagType_t d : diags)
                for (storage s : {storage::dense, storage::band, storage::packed}) {
                    const bool upper = uplo == CUBLAS_FILL_MODE_UPPER;
                    const auto S = make_stored<T>(n, k, s, upper, false, true, d == CUBLAS_DIAG_UNIT);
                    const full oa = apply(S.m, o);
                    for (int incx : {1, -2}) {
                        const auto x0 = random_vector<T>(length(n, incx));
                        // x = op(A) x
                        auto x = x0;
                        const T *a = S.data.data();
                        if (s == storage::dense)
                            OK(BLAS(trmv)(h, uplo, o, d, n, api(a), S.lda, api(x.data()), incx));
                        else if (s == storage::band)
                            OK(BLAS(tbmv)(h, uplo, o, d, n, k, api(a), S.lda, api(x.data()), incx));
                        else
                            OK(BLAS(tpmv)(h, uplo, o, d, n, api(a), api(x.data()), incx));
                        bool ok = true;
                        for (int i = 0; i < n; ++i) {
                            lc sum = 0;
                            for (int j = 0; j < n; ++j)
                                sum += oa(i, j) * as_ld(x0[at(j, n, incx)]);
                            ok &= close(x[at(i, n, incx)], sum, n);
                        }
                        check(ok, "triangular matrix times vector", __LINE__);
                        // op(A) x = b
                        x = x0;
                        if (s == storage::dense)
                            OK(BLAS(trsv)(h, uplo, o, d, n, api(a), S.lda, api(x.data()), incx));
                        else if (s == storage::band)
                            OK(BLAS(tbsv)(h, uplo, o, d, n, k, api(a), S.lda, api(x.data()), incx));
                        else
                            OK(BLAS(tpsv)(h, uplo, o, d, n, api(a), api(x.data()), incx));
                        ok = true;
                        for (int i = 0; i < n; ++i) {
                            lc sum = 0;
                            for (int j = 0; j < n; ++j)
                                sum += oa(i, j) * as_ld(x[at(j, n, incx)]);
                            ok &= close(x0[at(i, n, incx)], sum, 4 * n);
                        }
                        check(ok, "triangular solve", __LINE__);
                    }
                }
}

// ----------------------------------------------------------------------------
// Level 3

template <typename T>
void level3_products(cublasHandle_t h)
{
    const T alpha = random_value<T>(), beta = random_value<T>();
    // gemm with every pair of operations, and a larger one (in parallel)
    for (const auto [m, n, k] : {std::array<int, 3>{5, 4, 3}, std::array<int, 3>{67, 45, 39}}) {
        for (cublasOperation_t oa : ops) {
            for (cublasOperation_t ob : ops) {
                const int ar = oa == CUBLAS_OP_N ? m : k, ac = oa == CUBLAS_OP_N ? k : m;
                const int br = ob == CUBLAS_OP_N ? k : n, bc = ob == CUBLAS_OP_N ? n : k;
                const int lda = ar + 1, ldb = br + 2, ldc = m + 3;
                const auto A = random_vector<T>(std::size_t(lda) * ac);
                const auto B = random_vector<T>(std::size_t(ldb) * bc);
                const auto C0 = random_vector<T>(std::size_t(ldc) * n);
                auto C = C0;
                OK(BLAS(gemm)(h, oa, ob, m, n, k, api(&alpha), api(A.data()), lda, api(B.data()), ldb, api(&beta),
                              api(C.data()), ldc));
                const full p = multiply(apply(dense(A, ar, ac, lda), oa), apply(dense(B, br, bc, ldb), ob));
                bool ok = true;
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < m; ++i)
                        ok &= close(C[i + std::size_t(j) * ldc],
                                    as_ld(alpha) * p(i, j) + as_ld(beta) * as_ld(C0[i + std::size_t(j) * ldc]), k);
                check(ok, "gemm", __LINE__);
                if (m > 10 || oa != ob)
                    continue;
                // beta 0: C isn't read
                std::vector<T> Cn(std::size_t(ldc) * n, nan_value<T>());
                const T zero{};
                OK(BLAS(gemm)(h, oa, ob, m, n, k, api(&alpha), api(A.data()), lda, api(B.data()), ldb, api(&zero),
                              api(Cn.data()), ldc));
                ok = true;
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < m; ++i)
                        ok &= close(Cn[i + std::size_t(j) * ldc], as_ld(alpha) * p(i, j), k);
                check(ok, "gemm with beta 0", __LINE__);
                if constexpr (is_complex<T>) {
                    C = C0;
                    OK(pick<T>(0, 0, cublasCgemm3m, cublasZgemm3m)(h, oa, ob, m, n, k, api(&alpha), api(A.data()), lda,
                                                                   api(B.data()), ldb, api(&beta), api(C.data()), ldc));
                    ok = true;
                    for (int j = 0; j < n; ++j)
                        for (int i = 0; i < m; ++i)
                            ok &= close(C[i + std::size_t(j) * ldc],
                                        as_ld(alpha) * p(i, j) + as_ld(beta) * as_ld(C0[i + std::size_t(j) * ldc]), k);
                    check(ok, "gemm3m", __LINE__);
                }
            }
        }
    }

    // Batches of products
    const int m = 6, n = 5, k = 4, batch = 5;
    const auto A = random_vector<T>(std::size_t(m) * k * batch);
    const auto B = random_vector<T>(std::size_t(n) * k * batch);
    const auto C0 = random_vector<T>(std::size_t(m) * n * batch);
    auto C = C0;
    OK(BLAS(gemmStridedBatched)(h, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, api(&alpha), api(A.data()), m, (long long)m * k,
                                api(B.data()), n, (long long)n * k, api(&beta), api(C.data()), m, (long long)m * n,
                                batch));
    std::vector<lc> want(C.size());
    for (int b = 0; b < batch; ++b) {
        std::vector<T> Ab(A.begin() + std::size_t(m) * k * b, A.begin() + std::size_t(m) * k * (b + 1));
        std::vector<T> Bb(B.begin() + std::size_t(n) * k * b, B.begin() + std::size_t(n) * k * (b + 1));
        const full p = multiply(dense(Ab, m, k, m), apply(dense(Bb, n, k, n), CUBLAS_OP_T));
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) {
                const std::size_t e = i + std::size_t(j) * m + std::size_t(m) * n * b;
                want[e] = as_ld(alpha) * p(i, j) + as_ld(beta) * as_ld(C0[e]);
            }
    }
    CHECK(close_all(C, want, k));
    C = C0;
    std::vector<const T *> Ap, Bp;
    std::vector<T *> Cp;
    for (int b = 0; b < batch; ++b) {
        Ap.push_back(A.data() + std::size_t(m) * k * b);
        Bp.push_back(B.data() + std::size_t(n) * k * b);
        Cp.push_back(C.data() + std::size_t(m) * n * b);
    }
    using P = decltype(api(Ap[0]));
    using Q = decltype(api(Cp[0]));
    OK(BLAS(gemmBatched)(h, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, api(&alpha), reinterpret_cast<const P *>(Ap.data()), m,
                         reinterpret_cast<const P *>(Bp.data()), n, api(&beta), reinterpret_cast<const Q *>(Cp.data()),
                         m, batch));
    CHECK(close_all(C, want, k));
}

template <typename T>
void level3_symmetric(cublasHandle_t h)
{
    const T alpha = random_value<T>(), beta = random_value<T>();
    const int m = 5, n = 4;
    // symm and hemm
    for (cublasSideMode_t sd : sides)
        for (cublasFillMode_t uplo : fills)
            for (bool herm : {false, true}) {
                if (herm && !is_complex<T>)
                    continue;
                const int ka = sd == CUBLAS_SIDE_LEFT ? m : n;
                const auto S = make_stored<T>(ka, 0, storage::dense, uplo == CUBLAS_FILL_MODE_UPPER, herm, false, false);
                const int ldb = m + 1, ldc = m + 2;
                const auto B = random_vector<T>(std::size_t(ldb) * n);
                const auto C0 = random_vector<T>(std::size_t(ldc) * n);
                auto C = C0;
                if constexpr (is_complex<T>) {
                    const auto f = herm ? pick<T>(0, 0, cublasChemm, cublasZhemm) : pick<T>(0, 0, cublasCsymm, cublasZsymm);
                    OK(f(h, sd, uplo, m, n, api(&alpha), api(S.data.data()), S.lda, api(B.data()), ldb, api(&beta),
                         api(C.data()), ldc));
                } else {
                    OK(BLAS(symm)(h, sd, uplo, m, n, &alpha, S.data.data(), S.lda, B.data(), ldb, &beta, C.data(), ldc));
                }
                const full b = dense(B, m, n, ldb);
                const full p = sd == CUBLAS_SIDE_LEFT ? multiply(S.m, b) : multiply(b, S.m);
                bool ok = true;
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < m; ++i)
                        ok &= close(C[i + std::size_t(j) * ldc],
                                    as_ld(alpha) * p(i, j) + as_ld(beta) * as_ld(C0[i + std::size_t(j) * ldc]), ka);
                check(ok, herm ? "hemm" : "symm", __LINE__);
            }

    // syrk, herk, syr2k, her2k, syrkx and herkx
    const int k = 3;
    const real_t<T> ralpha = 0.5, rbeta = -0.25;
    for (cublasFillMode_t uplo : fills) {
        const bool upper = uplo == CUBLAS_FILL_MODE_UPPER;
        for (cublasOperation_t o : ops) {
            for (bool herm : {false, true}) {
                if (herm && !is_complex<T>)
                    continue;
                // Complex symmetric: N or T; hermitian: N or C
                if (is_complex<T> && (herm ? o == CUBLAS_OP_T : o == CUBLAS_OP_C))
                    continue;
                for (int kind = 0; kind < 3; ++kind) { // k, 2k, kx
                    const int rows = o == CUBLAS_OP_N ? n : k, cols = o == CUBLAS_OP_N ? k : n;
                    const int lda = rows + 1, ldc = n + 1;
                    const auto A = random_vector<T>(std::size_t(lda) * cols);
                    const auto B = random_vector<T>(std::size_t(lda) * cols);
                    std::vector<T> C0 = random_vector<T>(std::size_t(ldc) * n);
                    for (int j = 0; j < n; ++j)
                        for (int i = 0; i < n; ++i)
                            if (upper ? i > j : i < j)
                                C0[i + std::size_t(j) * ldc] = nan_value<T>();
                            else if (herm && i == j)
                                C0[i + std::size_t(j) * ldc] = T(std::real(C0[i + std::size_t(j) * ldc]));
                    auto C = C0;
                    const T *a = A.data(), *b = B.data();
                    if constexpr (is_complex<T>) {
                        if (herm && kind == 0)
                            OK(pick<T>(0, 0, cublasCherk, cublasZherk)(h, uplo, o, n, k, &ralpha, api(a), lda, &rbeta,
                                                                       api(C.data()), ldc));
                        else if (herm && kind == 1)
                            OK(pick<T>(0, 0, cublasCher2k, cublasZher2k)(h, uplo, o, n, k, api(&alpha), api(a), lda,
                                                                         api(b), lda, &rbeta, api(C.data()), ldc));
                        else if (herm)
                            OK(pick<T>(0, 0, cublasCherkx, cublasZherkx)(h, uplo, o, n, k, api(&alpha), api(a), lda,
                                                                         api(b), lda, &rbeta, api(C.data()), ldc));
                        else if (kind == 0)
                            OK(BLAS(syrk)(h, uplo, o, n, k, api(&alpha), api(a), lda, api(&beta), api(C.data()), ldc));
                        else if (kind == 1)
                            OK(BLAS(syr2k)(h, uplo, o, n, k, api(&alpha), api(a), lda, api(b), lda, api(&beta),
                                           api(C.data()), ldc));
                        else
                            OK(BLAS(syrkx)(h, uplo, o, n, k, api(&alpha), api(a), lda, api(b), lda, api(&beta),
                                           api(C.data()), ldc));
                    } else {
                        if (kind == 0)
                            OK(BLAS(syrk)(h, uplo, o, n, k, &alpha, a, lda, &beta, C.data(), ldc));
                        else if (kind == 1)
                            OK(BLAS(syr2k)(h, uplo, o, n, k, &alpha, a, lda, b, lda, &beta, C.data(), ldc));
                        else
                            OK(BLAS(syrkx)(h, uplo, o, n, k, &alpha, a, lda, b, lda, &beta, C.data(), ldc));
                    }
                    // P = op(A), Q = op(B): op(A) is n x k
                    const cublasOperation_t real_op = !is_complex<T> && o == CUBLAS_OP_C ? CUBLAS_OP_T : o;
                    const full P = apply(dense(A, rows, cols, lda), real_op);
                    const full Q = apply(dense(B, rows, cols, lda), real_op);
                    const lc al = herm && kind == 0 ? lc(ralpha) : as_ld(alpha);
                    const lc be = herm ? lc(rbeta) : as_ld(beta);
                    bool ok = true;
                    for (int j = 0; j < n; ++j)
                        for (int i = 0; i < n; ++i) {
                            const lc c0 = as_ld(C0[i + std::size_t(j) * ldc]);
                            const T got = C[i + std::size_t(j) * ldc];
                            if (upper ? i > j : i < j) {
                                ok &= is_nan(got);
                                continue;
                            }
                            lc s1 = 0, s2 = 0;
                            for (int l = 0; l < k; ++l) {
                                const full &second = kind == 0 ? P : Q;
                                s1 += P(i, l) * conj_if(second(j, l), herm);
                                s2 += Q(i, l) * conj_if(P(j, l), herm);
                            }
                            lc want = al * s1 + be * c0;
                            if (kind == 1)
                                want += conj_if(al, herm) * s2;
                            if (herm && i == j)
                                want = lc(want.real());
                            ok &= close(got, want, 2 * k);
                        }
                    check(ok, kind == 0 ? "syrk/herk" : kind == 1 ? "syr2k/her2k" : "syrkx/herkx", __LINE__);
                }
            }
        }
    }
}

template <typename T>
void level3_triangular(cublasHandle_t h)
{
    const T alpha = random_value<T>();
    const int m = 5, n = 4;
    for (cublasSideMode_t sd : sides)
        for (cublasFillMode_t uplo : fills)
            for (cublasOperation_t o : ops)
                for (cublasDiagType_t d : diags) {
                    const int ka = sd == CUBLAS_SIDE_LEFT ? m : n;
                    const auto S = make_stored<T>(ka, 0, storage::dense, uplo == CUBLAS_FILL_MODE_UPPER, false, true,
                                                  d == CUBLAS_DIAG_UNIT);
                    const full oa = apply(S.m, o);
                    const int ldb = m + 1, ldc = m + 2;
                    const auto B0 = random_vector<T>(std::size_t(ldb) * n);
                    const full b = dense(B0, m, n, ldb);
                    const full p = sd == CUBLAS_SIDE_LEFT ? multiply(oa, b) : multiply(b, oa);
                    // trmm, out of place and in place
                    for (bool in_place : {false, true}) {
                        auto B = B0;
                        std::vector<T> C(std::size_t(ldc) * n, nan_value<T>());
                        T *c = in_place ? B.data() : C.data();
                        const int ld = in_place ? ldb : ldc;
                        OK(BLAS(trmm)(h, sd, uplo, o, d, m, n, api(&alpha), api(S.data.data()), S.lda, api(B.data()),
                                      ldb, api(c), ld));
                        bool ok = true;
                        for (int j = 0; j < n; ++j)
                            for (int i = 0; i < m; ++i)
                                ok &= close(c[i + std::size_t(j) * ld], as_ld(alpha) * p(i, j), ka);
                        check(ok, in_place ? "trmm in place" : "trmm", __LINE__);
                    }
                    // trsm: op(A) X = alpha B, or X op(A) = alpha B
                    auto X = B0;
                    OK(BLAS(trsm)(h, sd, uplo, o, d, m, n, api(&alpha), api(S.data.data()), S.lda, api(X.data()), ldb));
                    const full x = dense(X, m, n, ldb);
                    const full back = sd == CUBLAS_SIDE_LEFT ? multiply(oa, x) : multiply(x, oa);
                    bool ok = true;
                    for (int j = 0; j < n; ++j)
                        for (int i = 0; i < m; ++i)
                            ok &= std::abs(back(i, j) - as_ld(alpha) * b(i, j)) <=
                                      tolerance<T>() * 8 * ka * (1 + std::abs(as_ld(alpha) * b(i, j)));
                    check(ok, "trsm", __LINE__);
                    if (sd != CUBLAS_SIDE_LEFT || o != CUBLAS_OP_N)
                        continue;
                    // A batch: each solve as the single one's
                    std::vector<T> X1 = B0, X2 = B0;
                    const T *Ap[2] = {S.data.data(), S.data.data()};
                    T *Bp[2] = {X1.data(), X2.data()};
                    using P = decltype(api(Ap[0]));
                    using Q = decltype(api(Bp[0]));
                    OK(BLAS(trsmBatched)(h, sd, uplo, o, d, m, n, api(&alpha), reinterpret_cast<const P *>(Ap), S.lda,
                                         reinterpret_cast<const Q *>(Bp), ldb, 2));
                    ok = true;
                    for (std::size_t e = 0; e < X.size(); ++e)
                        ok &= (is_nan(X[e]) && is_nan(X1[e])) || close(X1[e], as_ld(X[e]), 4 * ka);
                    ok &= X1 == X2 || std::equal(X1.begin(), X1.end(), X2.begin(), [](T u, T w) {
                              return u == w || (is_nan(u) && is_nan(w));
                          });
                    check(ok, "trsmBatched", __LINE__);
                }

    // geam: C = alpha op(A) + beta op(B), also in place
    const T beta = random_value<T>();
    for (cublasOperation_t oa : ops)
        for (cublasOperation_t ob : ops) {
            const int ar = oa == CUBLAS_OP_N ? m : n, br = ob == CUBLAS_OP_N ? m : n;
            const int ac = oa == CUBLAS_OP_N ? n : m, bc = ob == CUBLAS_OP_N ? n : m;
            const auto A = random_vector<T>(std::size_t(ar + 1) * ac);
            const auto B = random_vector<T>(std::size_t(br + 1) * bc);
            std::vector<T> C(std::size_t(m + 2) * n);
            OK(BLAS(geam)(h, oa, ob, m, n, api(&alpha), api(A.data()), ar + 1, api(&beta), api(B.data()), br + 1,
                          api(C.data()), m + 2));
            const full fa = apply(dense(A, ar, ac, ar + 1), oa), fb = apply(dense(B, br, bc, br + 1), ob);
            bool ok = true;
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < m; ++i)
                    ok &= close(C[i + std::size_t(j) * (m + 2)], as_ld(alpha) * fa(i, j) + as_ld(beta) * fb(i, j), 2);
            check(ok, "geam", __LINE__);
        }
    auto A = random_vector<T>(std::size_t(m) * n);
    const auto A0 = A;
    const auto B = random_vector<T>(std::size_t(m) * n);
    OK(BLAS(geam)(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, api(&alpha), api(A.data()), m, api(&beta), api(B.data()), m,
                  api(A.data()), m));
    bool ok = true;
    for (std::size_t e = 0; e < A.size(); ++e)
        ok &= close(A[e], as_ld(alpha) * as_ld(A0[e]) + as_ld(beta) * as_ld(B[e]), 2);
    CHECK(ok);
    // In place with a transposed operand isn't allowed
    CHECK(BLAS(geam)(h, CUBLAS_OP_T, CUBLAS_OP_N, m, m, api(&alpha), api(A.data()), m, api(&beta), api(B.data()), m,
                     api(A.data()), m) == CUBLAS_STATUS_INVALID_VALUE);

    // dgmm: diag(x) A (left) or A diag(x) (right)
    for (cublasSideMode_t sd : sides)
        for (int incx : {1, 2}) {
            const auto M = random_vector<T>(std::size_t(m + 1) * n);
            const auto x = random_vector<T>(length(sd == CUBLAS_SIDE_LEFT ? m : n, incx));
            std::vector<T> C(std::size_t(m) * n);
            OK(BLAS(dgmm)(h, sd, m, n, api(M.data()), m + 1, api(x.data()), incx, api(C.data()), m));
            ok = true;
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < m; ++i)
                    ok &= close(C[i + std::size_t(j) * m],
                                as_ld(M[i + std::size_t(j) * (m + 1)]) *
                                    as_ld(x[std::size_t(sd == CUBLAS_SIDE_LEFT ? i : j) * incx]));
            check(ok, "dgmm", __LINE__);
        }
}

//! alpha 0: the BLAS doesn't read the operands it would multiply (NaN
//! here), and a product's result is beta C, or zero; and infinite norms
template <typename T>
void alpha_zero(cublasHandle_t h)
{
    const T zero{}, beta = T(2);
    const int n = 4;
    std::vector<T> nan(16, nan_value<T>());
    std::vector<T> y(n, T(1));
    OK(BLAS(axpy)(h, n, api(&zero), api(nan.data()), 1, api(y.data()), 1));
    CHECK(y[0] == T(1) && y[3] == T(1));
    std::vector<T> A(16, T(3));
    if constexpr (is_complex<T>)
        OK(pick<T>(0, 0, cublasCgeru, cublasZgeru)(h, n, n, api(&zero), api(nan.data()), 1, api(nan.data()), 1,
                                                   api(A.data()), n));
    else
        OK(pick<T>(cublasSger, cublasDger, 0, 0)(h, n, n, &zero, nan.data(), 1, nan.data(), 1, A.data(), n));
    OK(BLAS(syr)(h, CUBLAS_FILL_MODE_UPPER, n, api(&zero), api(nan.data()), 1, api(A.data()), n));
    OK(BLAS(syr2)(h, CUBLAS_FILL_MODE_LOWER, n, api(&zero), api(nan.data()), 1, api(nan.data()), 1, api(A.data()), n));
    CHECK(A[5] == T(3) && A[15] == T(3));
    // Products: C = beta C
    std::vector<T> C(16, T(1));
    OK(BLAS(gemm)(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, api(&zero), api(nan.data()), n, api(nan.data()), n, api(&beta),
                  api(C.data()), n));
    CHECK(C[0] == T(2) && C[15] == T(2));
    OK(BLAS(symm)(h, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, n, n, api(&zero), api(nan.data()), n, api(nan.data()), n,
                  api(&beta), api(C.data()), n));
    CHECK(C[0] == T(4) && C[15] == T(4));
    OK(BLAS(syrk)(h, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, n, api(&zero), api(nan.data()), n, api(&beta), api(C.data()),
                  n));
    CHECK(C[0] == T(8) && C[1] == T(4)); // the lower triangle untouched
    std::vector<T> yv(n, T(1));
    OK(BLAS(gemv)(h, CUBLAS_OP_T, n, n, api(&zero), api(nan.data()), n, api(nan.data()), 1, api(&beta), api(yv.data()),
                  1));
    CHECK(yv[0] == T(2) && yv[3] == T(2));
    OK(BLAS(symv)(h, CUBLAS_FILL_MODE_LOWER, n, api(&zero), api(nan.data()), n, api(nan.data()), 1, api(&beta),
                  api(yv.data()), 1));
    CHECK(yv[0] == T(4));
    // Triangular: zero, without reading A or B
    std::vector<T> B = nan;
    OK(BLAS(trsm)(h, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, n, api(&zero),
                  api(nan.data()), n, api(B.data()), n));
    CHECK(B[0] == T(0) && B[15] == T(0));
    B = nan;
    std::vector<T> D(16, T(5));
    OK(BLAS(trmm)(h, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_UNIT, n, n, api(&zero),
                  api(nan.data()), n, api(B.data()), n, api(D.data()), n));
    CHECK(D[0] == T(0) && D[15] == T(0));
    // trmm in place needs ldb == ldc
    std::vector<T> E(64, T(1));
    const T one = T(1);
    CHECK(BLAS(trmm)(h, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, n, api(&one),
                     api(A.data()), n, api(E.data()), 6, api(E.data()), 8) == CUBLAS_STATUS_INVALID_VALUE);
    // Norms of infinities
    const real_t<T> inf = std::numeric_limits<real_t<T>>::infinity();
    const T infinite[2] = {T(inf), T(-inf)};
    real_t<T> norm = 0;
    OK(pick<T>(cublasSnrm2, cublasDnrm2, cublasScnrm2, cublasDznrm2)(h, 2, api(infinite), 1, &norm));
    CHECK(norm == inf);
    // Batches with nothing to compute don't read their arrays
    OK(BLAS(gemmBatched)(h, CUBLAS_OP_N, CUBLAS_OP_N, 0, n, n, api(&one), nullptr, 1, nullptr, n, api(&one), nullptr, 1,
                         3));
    // A rank-2 update of nothing needs no y
    OK(BLAS(syr2)(h, CUBLAS_FILL_MODE_UPPER, 0, api(&one), api(A.data()), 1, nullptr, 1, api(A.data()), 1));
}

template <typename T>
void every_routine(cublasHandle_t h)
{
    alpha_zero<T>(h);
    level1<T>(h);
    level2_general<T>(h);
    level2_symmetric<T>(h);
    level2_triangular<T>(h);
    level3_products<T>(h);
    level3_symmetric<T>(h);
    level3_triangular<T>(h);
}

// ----------------------------------------------------------------------------
// Products of other types

void mixed_precision(cublasHandle_t h)
{
    const int m = 6, n = 5, k = 7;
    // Half A and B, float C, computed in float
    std::vector<__half> Ah(m * k), Bh(k * n);
    full a(m, k), b(k, n);
    for (int i = 0; i < m * k; ++i) {
        Ah[i] = __float2half(float(random_value<double>()));
        a.e[i] = __half2float(Ah[i]);
    }
    for (int i = 0; i < k * n; ++i) {
        Bh[i] = __float2half(float(random_value<double>()));
        b.e[i] = __half2float(Bh[i]);
    }
    const full p = multiply(a, b);
    std::vector<float> C(m * n, std::nanf(""));
    const float one = 1, zero = 0;
    OK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one, Ah.data(), CUDA_R_16F, m, Bh.data(), CUDA_R_16F, k,
                    &zero, C.data(), CUDA_R_32F, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    bool ok = true;
    for (int i = 0; i < m * n; ++i)
        ok &= close(C[i], p.e[i], k);
    CHECK(ok);
    // cuBLAS 10's computeType, a cudaDataType
    std::fill(C.begin(), C.end(), 0.0f);
    OK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one, Ah.data(), CUDA_R_16F, m, Bh.data(), CUDA_R_16F, k,
                    &zero, C.data(), CUDA_R_32F, m, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
    ok = true;
    for (int i = 0; i < m * n; ++i)
        ok &= close(C[i], p.e[i], k);
    CHECK(ok);
    // Half everything: Hgemm, rounded to half
    std::vector<__half> Ch(m * n);
    const __half hone = __float2half(1.0f), hzero = __float2half(0.0f);
    OK(cublasHgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &hone, Ah.data(), m, Bh.data(), k, &hzero, Ch.data(), m));
    ok = true;
    for (int i = 0; i < m * n; ++i)
        ok &= std::abs(__half2float(Ch[i]) - p.e[i].real()) <= 2e-3L * (1 + std::abs(p.e[i]));
    CHECK(ok);
    // 8-bit integers into 32-bit ones, exactly
    std::vector<int8_t> Ai(m * k), Bi(k * n);
    for (auto &e : Ai)
        e = int8_t(std::uniform_int_distribution<int>(-128, 127)(generator));
    for (auto &e : Bi)
        e = int8_t(std::uniform_int_distribution<int>(-128, 127)(generator));
    std::vector<int32_t> Ci(m * n, 5);
    const int32_t ione = 1, ithree = 3;
    OK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &ione, Ai.data(), CUDA_R_8I, k, Bi.data(), CUDA_R_8I, k,
                    &ithree, Ci.data(), CUDA_R_32I, m, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
    ok = true;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            int32_t s = 15;
            for (int l = 0; l < k; ++l)
                s += int32_t(Ai[l + i * k]) * Bi[l + j * k];
            ok &= Ci[i + j * m] == s;
        }
    CHECK(ok);
    // Terms that overflow 32 bits, in a result that doesn't
    const int8_t a8[2] = {1, -1}, b8[2] = {100, 100};
    int32_t c32 = 7;
    const int32_t big = 100000000, izero = 0;
    OK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 2, &big, a8, CUDA_R_8I, 1, b8, CUDA_R_8I, 2, &izero, &c32,
                    CUDA_R_32I, 1, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
    CHECK(c32 == 0);
    // A strided batch of floats, as single products
    const int batch = 3;
    const auto Af = random_vector<float>(m * k * batch), Bf = random_vector<float>(k * n * batch);
    std::vector<float> Cf(m * n * batch), Cs(m * n * batch);
    OK(cublasGemmStridedBatchedEx(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one, Af.data(), CUDA_R_32F, m, m * k,
                                  Bf.data(), CUDA_R_32F, k, k * n, &zero, Cf.data(), CUDA_R_32F, m, m * n, batch,
                                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    OK(cublasSgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one, Af.data(), m, m * k, Bf.data(), k, k * n,
                                 &zero, Cs.data(), m, m * n, batch));
    CHECK(Cf == Cs);
    // Types it doesn't multiply
    CHECK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one, Ah.data(), CUDA_R_16F, m, Bh.data(), CUDA_C_32F, k,
                       &zero, C.data(), CUDA_R_32F, m, CUBLAS_COMPUTE_32F,
                       CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_NOT_SUPPORTED);
}

// ----------------------------------------------------------------------------
// Pointer modes, streams and graphs

void pointers_and_graphs(cublasHandle_t h)
{
    const int n = 10;
    float *x, *y, *alpha, *result;
    CHECK(cudaMalloc(&x, n * sizeof(float)) == cudaSuccess);
    CHECK(cudaMalloc(&y, n * sizeof(float)) == cudaSuccess);
    CHECK(cudaMalloc(&alpha, sizeof(float)) == cudaSuccess);
    CHECK(cudaMalloc(&result, sizeof(float)) == cudaSuccess);
    for (int i = 0; i < n; ++i)
        x[i] = float(i), y[i] = 1;

    // The device's pointers: alpha and the result in device memory
    cublasPointerMode_t mode;
    OK(cublasGetPointerMode(h, &mode));
    CHECK(mode == CUBLAS_POINTER_MODE_HOST);
    OK(cublasSetPointerMode(h, CUBLAS_POINTER_MODE_DEVICE));
    *alpha = 2;
    OK(cublasSaxpy(h, n, alpha, x, 1, y, 1));
    CHECK(y[3] == 7);
    OK(cublasSdot(h, n, x, 1, x, 1, result));
    CHECK(*result == 285);

    // Captured into a graph: device scalars are read when the graph runs
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream) == cudaSuccess);
    OK(cublasSetStream(h, stream));
    cudaStream_t got;
    OK(cublasGetStream(h, &got));
    CHECK(got == stream);
    for (int i = 0; i < n; ++i)
        y[i] = 1;
    cudaGraph_t graph;
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal) == cudaSuccess);
    OK(cublasSaxpy(h, n, alpha, x, 1, y, 1));
    OK(cublasSdot(h, n, y, 1, x, 1, result));
    CHECK(cudaStreamEndCapture(stream, &graph) == cudaSuccess);
    CHECK(y[3] == 1); // not run yet
    cudaGraphExec_t exec;
    CHECK(cudaGraphInstantiate(&exec, graph, 0) == cudaSuccess);
    *alpha = 3;
    CHECK(cudaGraphLaunch(exec, stream) == cudaSuccess);
    CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
    CHECK(y[3] == 10 && *result == 3 * 285 + 45);
    *alpha = -1;
    CHECK(cudaGraphLaunch(exec, stream) == cudaSuccess);
    CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
    CHECK(y[3] == 7);

    // The host's: the value when captured
    OK(cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST));
    float host_alpha = 5;
    cudaGraph_t graph2;
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal) == cudaSuccess);
    OK(cublasSscal(h, n, &host_alpha, y, 1));
    CHECK(cudaStreamEndCapture(stream, &graph2) == cudaSuccess);
    host_alpha = 100;
    cudaGraphExec_t exec2;
    CHECK(cudaGraphInstantiate(&exec2, graph2, 0) == cudaSuccess);
    CHECK(cudaGraphLaunch(exec2, stream) == cudaSuccess);
    CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
    CHECK(y[3] == 35);

    // rotm's param with the host's pointers: its values when captured
    float px[2] = {1, 1}, py[2] = {1, 1};
    float param[5] = {-1, 2, 0, 0, 3};
    cudaGraph_t graph4;
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal) == cudaSuccess);
    OK(cublasSrotm(h, 2, px, 1, py, 1, param));
    CHECK(cudaStreamEndCapture(stream, &graph4) == cudaSuccess);
    param[1] = 100;
    cudaGraphExec_t exec4;
    CHECK(cudaGraphInstantiate(&exec4, graph4, 0) == cudaSuccess);
    CHECK(cudaGraphLaunch(exec4, stream) == cudaSuccess);
    CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
    CHECK(px[0] == 2 && py[1] == 3);
    cudaGraphExecDestroy(exec4);
    cudaGraphDestroy(graph4);

    // Copies in the stream's order, captured too
    float host[2 * n];
    for (int i = 0; i < 2 * n; ++i)
        host[i] = float(-i);
    cudaGraph_t graph3;
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal) == cudaSuccess);
    OK(cublasSetVectorAsync(n, sizeof(float), host, 2, y, 1, stream));
    CHECK(cublasSetVectorAsync(n, sizeof(float), host, 0, y, 1, stream) == CUBLAS_STATUS_INVALID_VALUE);
    CHECK(cudaStreamEndCapture(stream, &graph3) == cudaSuccess);
    cudaGraphExec_t exec3;
    CHECK(cudaGraphInstantiate(&exec3, graph3, 0) == cudaSuccess);
    CHECK(y[3] == 35); // not run yet
    CHECK(cudaGraphLaunch(exec3, stream) == cudaSuccess);
    CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
    CHECK(y[1] == -2 && y[9] == -18);

    OK(cublasSetStream(h, nullptr));
    cudaGraphExecDestroy(exec);
    cudaGraphExecDestroy(exec2);
    cudaGraphExecDestroy(exec3);
    cudaGraphDestroy(graph);
    cudaGraphDestroy(graph2);
    cudaGraphDestroy(graph3);
    cudaStreamDestroy(stream);
    cudaFree(x);
    cudaFree(y);
    cudaFree(alpha);
    cudaFree(result);
}

void copies()
{
    // Strided vectors, and matrices between leading dimensions
    const double v[6] = {1, 2, 3, 4, 5, 6};
    double w[9] = {};
    OK(cublasSetVector(3, sizeof(double), v, 2, w, 3));
    CHECK(w[0] == 1 && w[3] == 3 && w[6] == 5 && w[1] == 0);
    double back[3];
    OK(cublasGetVector(3, sizeof(double), w, 3, back, 1));
    CHECK(back[0] == 1 && back[1] == 3 && back[2] == 5);
    const int A[6] = {1, 2, 3, 4, 5, 6}; // 2 x 3, lda 2
    int B[12] = {};
    OK(cublasSetMatrix(2, 3, sizeof(int), A, 2, B, 4));
    CHECK(B[0] == 1 && B[1] == 2 && B[2] == 0 && B[4] == 3 && B[9] == 6);
    int C[6];
    OK(cublasGetMatrix(2, 3, sizeof(int), B, 4, C, 2));
    CHECK(std::memcmp(A, C, sizeof(A)) == 0);
    CHECK(cublasSetVector(3, sizeof(double), v, 0, w, 1) == CUBLAS_STATUS_INVALID_VALUE);
    CHECK(cublasSetMatrix(2, 3, sizeof(int), A, 1, B, 4) == CUBLAS_STATUS_INVALID_VALUE);
}

void management_and_errors(cublasHandle_t h)
{
    int version = 0;
    OK(cublasGetVersion(h, &version));
    CHECK(version == CUBLAS_VERSION);
    int major = -1;
    OK(cublasGetProperty(MAJOR_VERSION, &major));
    CHECK(major == CUBLAS_VER_MAJOR);
    CHECK(cublasGetCudartVersion() == CUDART_VERSION);
    CHECK(std::strcmp(cublasGetStatusName(CUBLAS_STATUS_INVALID_VALUE), "CUBLAS_STATUS_INVALID_VALUE") == 0);
    CHECK(std::strlen(cublasGetStatusString(CUBLAS_STATUS_SUCCESS)) > 0);
    cublasMath_t math;
    OK(cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH));
    OK(cublasGetMathMode(h, &math));
    CHECK(math == CUBLAS_TF32_TENSOR_OP_MATH);
    OK(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH));
    OK(cublasSetWorkspace(h, nullptr, 0));

    float a[4] = {1, 2, 3, 4}, b[4] = {1, 0, 0, 1}, c[4] = {};
    const float one = 1, zero = 0;
    // No handle
    CHECK(cublasSgemm(nullptr, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, 2, &one, a, 2, b, 2, &zero, c, 2) ==
          CUBLAS_STATUS_NOT_INITIALIZED);
    // Sizes and leading dimensions
    CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, 2, &one, a, 1, b, 2, &zero, c, 2) ==
          CUBLAS_STATUS_INVALID_VALUE);
    CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, -1, 2, 2, &one, a, 2, b, 2, &zero, c, 2) ==
          CUBLAS_STATUS_INVALID_VALUE);
    CHECK(cublasSgemv(h, CUBLAS_OP_N, 2, 2, &one, a, 2, b, 0, &zero, c, 1) == CUBLAS_STATUS_INVALID_VALUE);
    // Enumerations and scalars
    CHECK(cublasSgemm(h, cublasOperation_t(9), CUBLAS_OP_N, 2, 2, 2, &one, a, 2, b, 2, &zero, c, 2) ==
          CUBLAS_STATUS_INVALID_VALUE);
    CHECK(cublasStrsm(h, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_FULL, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, 2, 2, &one, a, 2, c,
                      2) == CUBLAS_STATUS_INVALID_VALUE);
    CHECK(cublasSaxpy(h, 2, nullptr, a, 1, c, 1) == CUBLAS_STATUS_INVALID_VALUE);
    CHECK(cublasSnrm2(h, 2, a, 1, nullptr) == CUBLAS_STATUS_INVALID_VALUE);
    // Nothing to do
    OK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, 0, 0, 0, &one, a, 1, b, 1, &zero, c, 1));
    CHECK(cublasDestroy(nullptr) == CUBLAS_STATUS_NOT_INITIALIZED);
    CHECK(cublasCreate(nullptr) == CUBLAS_STATUS_INVALID_VALUE);
}

} // namespace

int main()
{
    cublasHandle_t h;
    OK(cublasCreate(&h));
    every_routine<float>(h);
    every_routine<double>(h);
    every_routine<cf>(h);
    every_routine<cd>(h);
    mixed_precision(h);
    pointers_and_graphs(h);
    copies();
    management_and_errors(h);
    OK(cublasDestroy(h));
    if (failures != 0) {
        std::printf("cublas: %d failures\n", failures);
        return 1;
    }
    std::printf("cublas: all passed\n");
    return 0;
}
