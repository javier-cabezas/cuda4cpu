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

// cuFFT (cufft.h, cufftXt.h) against a direct DFT computed in long double:
// complex, real to complex and complex to real transforms in single and
// double precision, in one to three dimensions, of odd and even lengths,
// forward and inverse, in place (padded real arrays) and out of place, in
// batches, with the advanced layout (strides, distances and embeddings,
// batches interleaved), and through every way of making a plan. Also load
// and store callbacks (with offsets and shared memory), transforms captured
// in a graph, and the errors.

#include <cufft.h>
#include <cufftXt.h>

#include <cmath>
#include <complex>
#include <cstdio>
#include <numbers>
#include <random>
#include <type_traits>
#include <vector>

namespace {

using lc = std::complex<long double>;

int failures = 0;

void check(bool ok, const char *what, int line)
{
    if (!ok && ++failures <= 40)
        std::printf("line %d: %s\n", line, what);
}

#define CHECK(cond) check((cond), #cond, __LINE__)
#define OK(call) check((call) == CUFFT_SUCCESS, #call, __LINE__)

std::mt19937 generator(2024);

double uniform()
{
    return std::uniform_real_distribution<double>(-1, 1)(generator);
}

//! The DFT of a rank-dimensional complex array (slowest first), with
//! exp(sign 2 pi i k n / N), unnormalized
std::vector<lc> dft(const std::vector<lc> &x, const std::vector<int> &n, int sign)
{
    const int rank = int(n.size());
    std::size_t total = 1;
    for (int d : n)
        total *= d;
    std::vector<lc> out(total);
    std::vector<int> k(rank), j(rank);
    for (std::size_t a = 0; a < total; ++a) {
        std::size_t r = a;
        for (int d = rank - 1; d >= 0; --d) {
            k[d] = int(r % n[d]);
            r /= n[d];
        }
        lc sum = 0;
        for (std::size_t b = 0; b < total; ++b) {
            std::size_t q = b;
            long double phase = 0;
            for (int d = rank - 1; d >= 0; --d) {
                j[d] = int(q % n[d]);
                q /= n[d];
                phase += (long double)(k[d]) * j[d] / n[d];
            }
            phase *= sign * 2 * std::numbers::pi_v<long double>;
            sum += x[b] * lc(std::cos(phase), std::sin(phase));
        }
        out[a] = sum;
    }
    return out;
}

template <typename T>
long double tolerance()
{
    return std::is_same_v<T, float> ? 1e-5L : 1e-12L;
}

std::size_t product(const std::vector<int> &n)
{
    std::size_t p = 1;
    for (int d : n)
        p *= d;
    return p;
}

//! A contiguous array's index of a multi-index (slowest first) with the
//! last length replaced by last
std::size_t index_of(const std::vector<int> &idx, const std::vector<int> &n, int last)
{
    std::size_t i = 0;
    for (std::size_t d = 0; d < n.size(); ++d)
        i = i * (d + 1 == n.size() ? last : n[d]) + idx[d];
    return i;
}

std::vector<int> multi_index(std::size_t a, const std::vector<int> &n)
{
    std::vector<int> idx(n.size());
    for (int d = int(n.size()) - 1; d >= 0; --d) {
        idx[d] = int(a % n[d]);
        a /= n[d];
    }
    return idx;
}

template <typename T> using complex_t = std::conditional_t<std::is_same_v<T, float>, cufftComplex, cufftDoubleComplex>;

template <typename T>
cufftType c2c_type()
{
    return std::is_same_v<T, float> ? CUFFT_C2C : CUFFT_Z2Z;
}
template <typename T>
cufftType r2c_type()
{
    return std::is_same_v<T, float> ? CUFFT_R2C : CUFFT_D2Z;
}
template <typename T>
cufftType c2r_type()
{
    return std::is_same_v<T, float> ? CUFFT_C2R : CUFFT_Z2D;
}

template <typename T>
cufftResult exec_c2c(cufftHandle p, complex_t<T> *in, complex_t<T> *out, int dir)
{
    if constexpr (std::is_same_v<T, float>)
        return cufftExecC2C(p, in, out, dir);
    else
        return cufftExecZ2Z(p, in, out, dir);
}
template <typename T>
cufftResult exec_r2c(cufftHandle p, T *in, complex_t<T> *out)
{
    if constexpr (std::is_same_v<T, float>)
        return cufftExecR2C(p, in, out);
    else
        return cufftExecD2Z(p, in, out);
}
template <typename T>
cufftResult exec_c2r(cufftHandle p, complex_t<T> *in, T *out)
{
    if constexpr (std::is_same_v<T, float>)
        return cufftExecC2R(p, in, out);
    else
        return cufftExecZ2D(p, in, out);
}

//! A plan of these lengths and type: cufftPlan1d, 2d or 3d, or cufftPlanMany
template <typename T>
cufftHandle make_plan(const std::vector<int> &n, cufftType type, int batch, int how)
{
    cufftHandle p = 0;
    std::vector<int> dims = n;
    if (how == 0 && n.size() == 1)
        OK(cufftPlan1d(&p, n[0], type, batch));
    else if (how == 0 && n.size() == 2 && batch == 1)
        OK(cufftPlan2d(&p, n[0], n[1], type));
    else if (how == 0 && n.size() == 3 && batch == 1)
        OK(cufftPlan3d(&p, n[0], n[1], n[2], type));
    else if (how == 1) {
        OK(cufftCreate(&p));
        std::size_t work = 1;
        OK(cufftMakePlanMany(p, int(n.size()), dims.data(), nullptr, 1, 0, nullptr, 1, 0, type, batch, &work));
        CHECK(work == 0);
    } else
        OK(cufftPlanMany(&p, int(n.size()), dims.data(), nullptr, 1, 0, nullptr, 1, 0, type, batch));
    return p;
}

template <typename T>
void complex_transforms()
{
    for (const auto &n : {std::vector<int>{1}, std::vector<int>{16}, std::vector<int>{15}, std::vector<int>{97},
                          std::vector<int>{8, 6}, std::vector<int>{5, 7}, std::vector<int>{4, 5, 6}}) {
        for (int batch : {1, 3}) {
            for (int how : {0, 1, 2}) {
                for (int direction : {CUFFT_FORWARD, CUFFT_INVERSE}) {
                    for (bool in_place : {false, true}) {
                        const std::size_t total = product(n);
                        std::vector<complex_t<T>> in(total * batch), out(total * batch);
                        std::vector<std::vector<lc>> x(batch, std::vector<lc>(total));
                        for (int b = 0; b < batch; ++b)
                            for (std::size_t i = 0; i < total; ++i) {
                                x[b][i] = lc(uniform(), uniform());
                                in[b * total + i] = {T(x[b][i].real()), T(x[b][i].imag())};
                                x[b][i] = lc(in[b * total + i].x, in[b * total + i].y);
                            }
                        const cufftHandle p = make_plan<T>(n, c2c_type<T>(), batch, how);
                        complex_t<T> *o = in_place ? in.data() : out.data();
                        OK(exec_c2c<T>(p, in.data(), o, direction));
                        bool ok = true;
                        for (int b = 0; b < batch; ++b) {
                            const auto want = dft(x[b], n, direction);
                            for (std::size_t i = 0; i < total; ++i)
                                ok &= std::abs(lc(o[b * total + i].x, o[b * total + i].y) - want[i]) <=
                                      tolerance<T>() * (1 + total) * 2;
                        }
                        check(ok, "complex transform", __LINE__);
                        OK(cufftDestroy(p));
                    }
                }
            }
        }
    }
}

//! Real to complex, then complex to real: the R2C result is the DFT's first
//! n / 2 + 1 elements of the last dimension, and the C2R one total times
//! the signal; in place, the real array is padded
template <typename T>
void real_transforms()
{
    for (const auto &n : {std::vector<int>{1}, std::vector<int>{16}, std::vector<int>{15}, std::vector<int>{8, 6},
                          std::vector<int>{5, 7}, std::vector<int>{4, 5, 6}, std::vector<int>{3, 4, 7}}) {
        for (int batch : {1, 2}) {
            for (bool in_place : {false, true}) {
                const int last = n.back(), half = last / 2 + 1;
                const std::size_t total = product(n);
                const std::size_t complex_total = total / last * half;
                const std::size_t real_row = in_place ? 2 * half : last;
                const std::size_t real_total = total / last * real_row;
                std::vector<T> real(real_total * batch, T(1e9));   // padding garbage
                std::vector<lc> signal(total * batch);
                for (int b = 0; b < batch; ++b)
                    for (std::size_t a = 0; a < total; ++a) {
                        const T v = T(uniform());
                        real[b * real_total + index_of(multi_index(a, n), n, int(real_row))] = v;
                        signal[b * total + a] = v;
                    }
                std::vector<complex_t<T>> spectrum(complex_total * batch);
                complex_t<T> *s = in_place ? reinterpret_cast<complex_t<T> *>(real.data()) : spectrum.data();
                const cufftHandle forward = make_plan<T>(n, r2c_type<T>(), batch, batch == 1 ? 0 : 2);
                OK(exec_r2c<T>(forward, real.data(), s));
                bool ok = true;
                for (int b = 0; b < batch; ++b) {
                    const std::vector<lc> x(signal.begin() + b * total, signal.begin() + (b + 1) * total);
                    const auto want = dft(x, n, -1);
                    for (std::size_t a = 0; a < total; ++a) {
                        const auto idx = multi_index(a, n);
                        if (idx.back() >= half)
                            continue;
                        const complex_t<T> got = s[b * complex_total + index_of(idx, n, half)];
                        ok &= std::abs(lc(got.x, got.y) - want[a]) <= tolerance<T>() * (1 + total) * 2;
                    }
                }
                check(ok, "real to complex", __LINE__);
                // Back: total times the signal
                std::vector<T> back(real_total * batch);
                T *r = in_place ? real.data() : back.data();
                const cufftHandle inverse = make_plan<T>(n, c2r_type<T>(), batch, 1);
                OK(exec_c2r<T>(inverse, s, r));
                ok = true;
                for (int b = 0; b < batch; ++b)
                    for (std::size_t a = 0; a < total; ++a)
                        ok &= std::abs(r[b * real_total + index_of(multi_index(a, n), n, int(real_row))] -
                                       (long double)total * signal[b * total + a].real()) <=
                              tolerance<T>() * (1 + total) * 4;
                check(ok, "complex to real", __LINE__);
                OK(cufftDestroy(forward));
                OK(cufftDestroy(inverse));
            }
        }
    }
}

//! The advanced layout: batches interleaved (stride batch, distance 1), and
//! an embedding larger than the transform, in 2D
template <typename T>
void advanced_layout()
{
    // 1D, 4 transforms of 12 interleaved
    {
        int n[1] = {12};
        const int batch = 4;
        std::vector<complex_t<T>> data(12 * batch);
        std::vector<std::vector<lc>> x(batch, std::vector<lc>(12));
        for (int i = 0; i < 12; ++i)
            for (int b = 0; b < batch; ++b) {
                data[i * batch + b] = {T(uniform()), T(uniform())};
                x[b][i] = lc(data[i * batch + b].x, data[i * batch + b].y);
            }
        cufftHandle p;
        OK(cufftPlanMany(&p, 1, n, n, batch, 1, n, batch, 1, c2c_type<T>(), batch));
        OK(exec_c2c<T>(p, data.data(), data.data(), CUFFT_FORWARD));
        bool ok = true;
        for (int b = 0; b < batch; ++b) {
            const auto want = dft(x[b], {12}, -1);
            for (int i = 0; i < 12; ++i)
                ok &= std::abs(lc(data[i * batch + b].x, data[i * batch + b].y) - want[i]) <= tolerance<T>() * 30;
        }
        CHECK(ok);
        OK(cufftDestroy(p));
    }
    // 2D 4 x 6 in a 7 x 9 embedding, stride 2; out to a 4 x 6 one, stride 1,
    // through the 64-bit API
    {
        long long n[2] = {4, 6}, inembed[2] = {7, 9}, onembed[2] = {4, 6};
        const long long istride = 2, idist = 7 * 9 * 2 + 3, ostride = 1, odist = 24, batch = 2;
        std::vector<complex_t<T>> in(std::size_t(idist * batch)), out(std::size_t(odist * batch));
        std::vector<std::vector<lc>> x(batch, std::vector<lc>(24));
        for (int b = 0; b < batch; ++b)
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 6; ++j) {
                    complex_t<T> &e = in[std::size_t(b * idist + (i * inembed[1] + j) * istride)];
                    e = {T(uniform()), T(uniform())};
                    x[b][i * 6 + j] = lc(e.x, e.y);
                }
        cufftHandle p;
        OK(cufftCreate(&p));
        std::size_t work = 0;
        OK(cufftMakePlanMany64(p, 2, n, inembed, istride, idist, onembed, ostride, odist, c2c_type<T>(), batch, &work));
        OK(exec_c2c<T>(p, in.data(), out.data(), CUFFT_INVERSE));
        bool ok = true;
        for (int b = 0; b < batch; ++b) {
            const auto want = dft(x[b], {4, 6}, 1);
            for (int a = 0; a < 24; ++a)
                ok &= std::abs(lc(out[b * odist + a].x, out[b * odist + a].y) - want[a]) <= tolerance<T>() * 50;
        }
        CHECK(ok);
        OK(cufftDestroy(p));
    }
}

// Callbacks: each input element times 2 (and its offset recorded), and each
// output element stored to a second array, plus 1

struct callback_info {
    int calls = 0;
    std::size_t max_offset = 0;
    cufftDoubleComplex *copy = nullptr;
};

cufftDoubleComplex load_doubled(void *in, size_t offset, void *info, void *shared)
{
    auto *i = static_cast<callback_info *>(info);
    ++i->calls;
    i->max_offset = std::max(i->max_offset, offset);
    if (shared != nullptr)
        static_cast<char *>(shared)[0] = 1;
    const cufftDoubleComplex e = static_cast<cufftDoubleComplex *>(in)[offset];
    return {2 * e.x, 2 * e.y};
}

void store_copy(void *out, size_t offset, cufftDoubleComplex e, void *info, void *)
{
    static_cast<cufftDoubleComplex *>(out)[offset] = e;
    static_cast<callback_info *>(info)->copy[offset] = {e.x + 1, e.y};
}

__device__ cufftCallbackLoadZ load_pointer = load_doubled;

void callbacks()
{
    const int n = 10;
    std::vector<cufftDoubleComplex> data(n), copy(n);
    std::vector<lc> x(n);
    for (int i = 0; i < n; ++i) {
        data[i] = {uniform(), uniform()};
        x[i] = 2.0L * lc(data[i].x, data[i].y);
    }
    cufftHandle p;
    OK(cufftPlan1d(&p, n, CUFFT_Z2Z, 1));
    // The device function's address, as programs get it
    void *load = nullptr;
    CHECK(cudaMemcpyFromSymbol(&load, load_pointer, sizeof(load)) == cudaSuccess);
    void *store = reinterpret_cast<void *>(&store_copy);
    callback_info info{0, 0, copy.data()};
    void *info_ptr = &info;
    OK(cufftXtSetCallbackSharedSize(p, CUFFT_CB_LD_COMPLEX_DOUBLE, 16));
    OK(cufftXtSetCallback(p, &load, CUFFT_CB_LD_COMPLEX_DOUBLE, &info_ptr));
    OK(cufftXtSetCallback(p, &store, CUFFT_CB_ST_COMPLEX_DOUBLE, &info_ptr));
    // A callback of the other precision or kind is rejected
    CHECK(cufftXtSetCallback(p, &load, CUFFT_CB_LD_COMPLEX, &info_ptr) == CUFFT_INVALID_TYPE);
    OK(cufftExecZ2Z(p, data.data(), data.data(), CUFFT_FORWARD));
    CHECK(info.calls == n && info.max_offset == n - 1);
    const auto want = dft(x, {n}, -1);
    bool ok = true;
    for (int i = 0; i < n; ++i)
        ok &= std::abs(lc(data[i].x, data[i].y) - want[i]) < 1e-12L * 30 &&
              std::abs(lc(copy[i].x - 1, copy[i].y) - want[i]) < 1e-12L * 30;
    CHECK(ok);
    // Cleared: plain transforms again
    OK(cufftXtClearCallback(p, CUFFT_CB_LD_COMPLEX_DOUBLE));
    OK(cufftXtClearCallback(p, CUFFT_CB_ST_COMPLEX_DOUBLE));
    info.calls = 0;
    OK(cufftExecZ2Z(p, data.data(), data.data(), CUFFT_INVERSE));
    CHECK(info.calls == 0);
    OK(cufftDestroy(p));
}

//! A transform captured in a graph runs with the data the graph finds
void graphs()
{
    const int n = 8;
    float *real;
    cufftComplex *spectrum;
    CHECK(cudaMalloc(&real, n * sizeof(float)) == cudaSuccess);
    CHECK(cudaMalloc(&spectrum, (n / 2 + 1) * sizeof(cufftComplex)) == cudaSuccess);
    cufftHandle p;
    OK(cufftPlan1d(&p, n, CUFFT_R2C, 1));
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream) == cudaSuccess);
    OK(cufftSetStream(p, stream));
    cudaGraph_t graph;
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal) == cudaSuccess);
    OK(cufftExecR2C(p, real, spectrum));
    CHECK(cudaStreamEndCapture(stream, &graph) == cudaSuccess);
    cudaGraphExec_t exec;
    CHECK(cudaGraphInstantiate(&exec, graph, 0) == cudaSuccess);
    for (float value : {1.0f, 3.0f}) {
        for (int i = 0; i < n; ++i)
            real[i] = value;
        CHECK(cudaGraphLaunch(exec, stream) == cudaSuccess);
        CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
        CHECK(std::abs(spectrum[0].x - n * value) < 1e-4f && std::abs(spectrum[1].x) < 1e-4f);
    }
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    OK(cufftDestroy(p));
    cudaStreamDestroy(stream);
    cudaFree(real);
    cudaFree(spectrum);
}

void errors()
{
    cufftHandle p;
    CHECK(cufftPlan1d(&p, 0, CUFFT_C2C, 1) == CUFFT_INVALID_SIZE);
    CHECK(cufftPlan1d(&p, 8, CUFFT_C2C, 0) == CUFFT_INVALID_SIZE);
    CHECK(cufftPlan1d(&p, 8, cufftType(0x11), 1) == CUFFT_INVALID_TYPE);
    int n[4] = {2, 2, 2, 2};
    CHECK(cufftPlanMany(&p, 4, n, nullptr, 1, 0, nullptr, 1, 0, CUFFT_C2C, 1) == CUFFT_INVALID_VALUE);
    OK(cufftPlan1d(&p, 8, CUFFT_C2C, 1));
    cufftComplex c[8] = {};
    float r[8] = {};
    CHECK(cufftExecR2C(p, r, c) == CUFFT_INVALID_TYPE);
    CHECK(cufftExecZ2Z(p, nullptr, nullptr, CUFFT_FORWARD) == CUFFT_INVALID_TYPE);
    CHECK(cufftExecC2C(p, c, c, 0) == CUFFT_INVALID_VALUE);
    CHECK(cufftExecC2C(p, nullptr, c, CUFFT_FORWARD) == CUFFT_INVALID_VALUE);
    std::size_t size = 1;
    OK(cufftGetSize(p, &size));
    CHECK(size == 0);
    OK(cufftSetWorkArea(p, nullptr));
    OK(cufftSetAutoAllocation(p, 0));
    OK(cufftEstimate2d(4, 4, CUFFT_R2C, &size));
    OK(cufftDestroy(p));
    CHECK(cufftDestroy(p) == CUFFT_INVALID_PLAN);
    CHECK(cufftExecC2C(12345, c, c, CUFFT_FORWARD) == CUFFT_INVALID_PLAN);
    // A plan made without its sizes isn't one yet
    OK(cufftCreate(&p));
    CHECK(cufftExecC2C(p, c, c, CUFFT_FORWARD) == CUFFT_INVALID_PLAN);
    OK(cufftDestroy(p));
    int version = 0, major = -1;
    OK(cufftGetVersion(&version));
    OK(cufftGetProperty(MAJOR_VERSION, &major));
    CHECK(version == CUFFT_VERSION && major == CUFFT_VER_MAJOR);
}

} // namespace

int main()
{
    complex_transforms<float>();
    complex_transforms<double>();
    real_transforms<float>();
    real_transforms<double>();
    advanced_layout<float>();
    advanced_layout<double>();
    callbacks();
    graphs();
    errors();
    if (failures != 0) {
        std::printf("cufft: %d failures\n", failures);
        return 1;
    }
    std::printf("cufft: all passed\n");
    return 0;
}
