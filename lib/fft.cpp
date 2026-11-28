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

// cuFFT, hipFFT and rocFFT on the CPU. Every API's plan is a fft::plan: the
// kind (complex to complex, real to complex, complex to real), the
// precision, the lengths (slowest dimension first, as cuFFT's; rocFFT's are
// fastest first), the batch, and the input's and output's layouts (strides
// per dimension and a distance between transforms, in elements of each
// array, and an offset). cuFFT's basic layout is decided when the
// transform runs, since a real array is padded in place (n / 2 + 1 complex
// elements' worth on its last dimension) and contiguous out of place.
// execute() runs a batch on pocketfft, whose transforms take any strides:
// the batch is one more dimension, which isn't transformed. Callbacks load
// the input into a copy, or store the output from one, element by element,
// with the offsets of the program's arrays. Transforms run when executed,
// or as host nodes of a graph that their stream is being captured into
// (stream_tasks.hpp).

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <tuple>
#include <vector>

#include <omp.h>

#include "cufftXt.h"
#include "hipfft/hipfftXt.h"
#include "rocfft/rocfft.h"
#include "stream_tasks.hpp"

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#endif
// It runs on the OpenMP threads (POCKETFFT_OPENMP, cuda4cpu's change to
// it): with a pool of its own, a transform right after a kernel took 3 to 7
// times as long, while the OpenMP threads still spun. Its buffers come from
// posix_memalign.
#define POCKETFFT_OPENMP
#define POCKETFFT_USE_POSIX_MEMALIGN
#include "third_party/pocketfft/pocketfft_hdronly.h"
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

namespace cuda4cpu::detail::fft {

enum class kind { c2c, r2c, c2r };

enum class status {
    success, invalid_plan, alloc_failed, invalid_type, invalid_value, internal_error, exec_failed, invalid_size,
    not_supported, invalid_array_type, invalid_strides
};

//! An array's layout: strides per dimension (slowest first) and the
//! distance between transforms, in its elements, and an offset
struct layout {
    std::array<long long, 3> stride{};
    long long dist = 0;
    long long offset = 0;
};

struct callback {
    void *function = nullptr;
    void *data = nullptr;
    std::size_t shared = 0;
};

struct plan {
    bool made = false;
    bool dbl = false;
    kind k = kind::c2c;
    int rank = 0;
    std::array<long long, 3> n{1, 1, 1};   //!< The transform's lengths (the real array's for r2c and c2r)
    long long batch = 1;
    //! cuFFT's basic layout, decided when executed (in place or not);
    //! else in and out
    bool basic = true;
    layout in;
    layout out;
    double scale = 1;
    cudaStream_t stream = nullptr;
    callback load;
    callback store;
    std::size_t load_shared = 0;
    std::size_t store_shared = 0;
};

//! The complex array's lengths of a real transform: the last is n / 2 + 1
inline std::array<long long, 3> complex_lengths(const plan &p)
{
    std::array<long long, 3> c = p.n;
    c[p.rank - 1] = p.n[p.rank - 1] / 2 + 1;
    return c;
}

//! The lengths of the input or the output array
inline std::array<long long, 3> array_lengths(const plan &p, bool input)
{
    const bool complex_side = p.k == kind::c2c || (p.k == kind::r2c) != input;
    return complex_side && p.k != kind::c2c ? complex_lengths(p) : p.n;
}

//! A contiguous array of these lengths; padded, a real array whose last
//! dimension takes room for 2 (n / 2 + 1) reals, as real transforms in
//! place need
inline layout packed(const plan &p, std::array<long long, 3> lengths, bool padded)
{
    if (padded)
        lengths[p.rank - 1] = 2 * (p.n[p.rank - 1] / 2 + 1);
    layout l;
    long long s = 1;
    for (int d = p.rank - 1; d >= 0; --d) {
        l.stride[d] = s;
        s *= lengths[d];
    }
    l.dist = s;
    return l;
}

//! The basic layout's input and output, in place or not
inline std::pair<layout, layout> basic_layouts(const plan &p, bool in_place)
{
    switch (p.k) {
    case kind::c2c: return {packed(p, p.n, false), packed(p, p.n, false)};
    case kind::r2c: return {packed(p, p.n, in_place), packed(p, complex_lengths(p), false)};
    case kind::c2r: break;
    }
    return {packed(p, complex_lengths(p), false), packed(p, p.n, in_place)};
}

//! Calls f(offset) for every element of an array of these lengths and this
//! layout, in every transform of the batch
template <typename F>
void each_element(const plan &p, const std::array<long long, 3> &lengths, const layout &l, F &&f)
{
    const long long n0 = lengths[0];
    const long long n1 = p.rank > 1 ? lengths[1] : 1;
    const long long n2 = p.rank > 2 ? lengths[2] : 1;
    const long long s1 = p.rank > 1 ? l.stride[1] : 0;
    const long long s2 = p.rank > 2 ? l.stride[2] : 0;
    for (long long b = 0; b < p.batch; ++b)
        for (long long i = 0; i < n0; ++i)
            for (long long j = 0; j < n1; ++j)
                for (long long k = 0; k < n2; ++k)
                    f(std::size_t(l.offset + b * l.dist + i * l.stride[0] + j * s1 + k * s2));
}

//! The number of elements from the array's start to past its last one
inline std::size_t footprint(const plan &p, const std::array<long long, 3> &lengths, const layout &l)
{
    long long last = l.offset + (p.batch - 1) * l.dist;
    for (int d = 0; d < p.rank; ++d)
        last += (lengths[d] - 1) * l.stride[d];
    return std::size_t(last + 1);
}

//! Callbacks' types: the CUDA types the program's functions take
template <typename T> using complex_of = std::conditional_t<std::is_same_v<T, float>, float2, double2>;

template <typename E>
using load_function = E (*)(void *, std::size_t, void *, void *);
template <typename E>
using store_function = void (*)(void *, std::size_t, E, void *, void *);

//! Runs the plan's batch of transforms, forward or inverse, from in to out:
//! T is the precision, In and Out the arrays' elements (T or complex<T>)
template <typename T>
void execute_typed(const plan &p, void *in_ptr, void *out_ptr, bool forward)
{
    using C = std::complex<T>;
    const bool in_place = in_ptr == out_ptr; // cuFFT's basic layout: padded in place
    layout in = p.in, out = p.out;
    if (p.basic)
        std::tie(in, out) = basic_layouts(p, in_place);
    const bool real_in = p.k == kind::r2c, real_out = p.k == kind::c2r;
    const std::size_t in_size = real_in ? sizeof(T) : sizeof(C);
    const std::size_t out_size = real_out ? sizeof(T) : sizeof(C);
    const auto in_lengths = array_lengths(p, true);
    const auto out_lengths = array_lengths(p, false);

    // Load callbacks: the input loaded into a copy, with the same layout
    // Callbacks get the array at its offset, and offsets from there, as
    // rocFFT's kernels give them (cuFFT's arrays have no offsets); copies
    // start at the offset too
    std::vector<unsigned char> loaded;
    const void *source = static_cast<const unsigned char *>(in_ptr) + in.offset * in_size;
    layout in_rel = in, out_rel = out;
    in_rel.offset = out_rel.offset = 0;
    if (p.load.function != nullptr) {
        loaded.assign(footprint(p, in_lengths, in_rel) * in_size, 0);
        std::vector<unsigned char> shared(p.load.shared);
        void *const shared_ptr = shared.empty() ? nullptr : shared.data();
        void *const buffer = const_cast<void *>(source);
        void *data = p.load.data;
        if (real_in) {
            auto f = reinterpret_cast<load_function<T>>(p.load.function);
            T *to = reinterpret_cast<T *>(loaded.data());
            each_element(p, in_lengths, in_rel, [&](std::size_t o) { to[o] = f(buffer, o, data, shared_ptr); });
        } else {
            auto f = reinterpret_cast<load_function<complex_of<T>>>(p.load.function);
            C *to = reinterpret_cast<C *>(loaded.data());
            each_element(p, in_lengths, in_rel, [&](std::size_t o) {
                const complex_of<T> e = f(buffer, o, data, shared_ptr);
                to[o] = C(e.x, e.y);
            });
        }
        source = loaded.data();
    }
    // Store callbacks: the output into a copy, stored from it
    std::vector<unsigned char> stored;
    void *const out_buffer = static_cast<unsigned char *>(out_ptr) + out.offset * out_size;
    void *target = out_buffer;
    if (p.store.function != nullptr) {
        stored.assign(footprint(p, out_lengths, out_rel) * out_size, 0);
        target = stored.data();
    }

    // In place, pocketfft needs the same strides, and reads a line before
    // it writes it: complex arrays with the same layout. Anything else
    // (real transforms, other strides or offsets) transforms a copy of the
    // input
    std::vector<unsigned char> input_copy;
    if (in_ptr == out_ptr && p.load.function == nullptr && p.store.function == nullptr) {
        const bool same = p.k == kind::c2c && in.offset == out.offset && (p.batch == 1 || in.dist == out.dist) &&
                          std::equal(in.stride.begin(), in.stride.begin() + p.rank, out.stride.begin());
        if (!same) {
            input_copy.resize(footprint(p, in_lengths, in_rel) * in_size);
            std::memcpy(input_copy.data(), source, input_copy.size());
            source = input_copy.data();
        }
    }

    const std::size_t rank = std::size_t(p.rank);
    pocketfft::shape_t shape(rank + 1), axes(rank);
    pocketfft::stride_t stride_in(rank + 1), stride_out(rank + 1);
    // One transform: its distance is no stride (pocketfft compares them in
    // place)
    shape[0] = std::size_t(p.batch);
    stride_in[0] = p.batch == 1 ? 0 : std::ptrdiff_t(in.dist * (long long)in_size);
    stride_out[0] = p.batch == 1 ? 0 : std::ptrdiff_t(out.dist * (long long)out_size);
    for (std::size_t d = 0; d < rank; ++d) {
        shape[d + 1] = std::size_t(p.n[d]);
        stride_in[d + 1] = std::ptrdiff_t(in.stride[d] * (long long)in_size);
        stride_out[d + 1] = std::ptrdiff_t(out.stride[d] * (long long)out_size);
        axes[d] = d + 1;
    }
    long long elements = p.batch;
    for (int d = 0; d < p.rank; ++d)
        elements *= p.n[d];
    const std::size_t threads = elements > 32768 && !omp_in_parallel() ? std::size_t(omp_get_max_threads()) : 1;
    const T scale = T(p.scale);
    const auto *src = static_cast<const unsigned char *>(source);
    auto *dst = static_cast<unsigned char *>(target);
    switch (p.k) {
    case kind::c2c:
        pocketfft::c2c(shape, stride_in, stride_out, axes, forward, reinterpret_cast<const C *>(src),
                       reinterpret_cast<C *>(dst), scale, threads);
        break;
    case kind::r2c:
        pocketfft::r2c(shape, stride_in, stride_out, axes, true, reinterpret_cast<const T *>(src),
                       reinterpret_cast<C *>(dst), scale, threads);
        break;
    case kind::c2r:
        pocketfft::c2r(shape, stride_in, stride_out, axes, false, reinterpret_cast<const C *>(src),
                       reinterpret_cast<T *>(dst), scale, threads);
        break;
    }

    if (p.store.function != nullptr) {
        std::vector<unsigned char> shared(p.store.shared);
        void *const shared_ptr = shared.empty() ? nullptr : shared.data();
        void *data = p.store.data;
        if (real_out) {
            auto f = reinterpret_cast<store_function<T>>(p.store.function);
            const T *from = reinterpret_cast<const T *>(stored.data());
            each_element(p, out_lengths, out_rel, [&](std::size_t o) { f(out_buffer, o, from[o], data, shared_ptr); });
        } else {
            auto f = reinterpret_cast<store_function<complex_of<T>>>(p.store.function);
            const C *from = reinterpret_cast<const C *>(stored.data());
            each_element(p, out_lengths, out_rel, [&](std::size_t o) {
                f(out_buffer, o, complex_of<T>{from[o].real(), from[o].imag()}, data, shared_ptr);
            });
        }
    }
}

//! Runs the transforms in the plan's stream's order (a copy of the plan, as
//! it was: a captured transform keeps its parameters)
inline status execute(const plan &p, void *in, void *out, bool forward)
{
    if (!p.made)
        return status::invalid_plan;
    if (in == nullptr || out == nullptr)
        return status::invalid_value;
    auto result = std::make_shared<status>(status::success);
    const plan copy = p;
    const cudaError_t e = run_on_stream(p.stream, [copy, in, out, forward, result] {
        try {
            if (copy.dbl)
                execute_typed<double>(copy, in, out, forward);
            else
                execute_typed<float>(copy, in, out, forward);
        } catch (const std::bad_alloc &) {
            *result = status::alloc_failed;
        } catch (...) {
            *result = status::exec_failed;
        }
    });
    return e != cudaSuccess ? status::exec_failed : *result;
}

//! A plan's lengths, batch and layouts: in and out (cuFFT's inembed and
//! onembed, with strides and distances) or, if either is null, the basic
//! layout
inline status make(plan &made, int rank, const long long *n, const long long *inembed, long long istride,
                   long long idist, const long long *onembed, long long ostride, long long odist, int type,
                   long long batch)
{
    if (rank < 1 || rank > 3 || n == nullptr)
        return status::invalid_value;
    // A plan that fails leaves the handle's previous one as it was
    plan p = made;
    switch (type) {
    case CUFFT_C2C: p.k = kind::c2c; p.dbl = false; break;
    case CUFFT_Z2Z: p.k = kind::c2c; p.dbl = true; break;
    case CUFFT_R2C: p.k = kind::r2c; p.dbl = false; break;
    case CUFFT_D2Z: p.k = kind::r2c; p.dbl = true; break;
    case CUFFT_C2R: p.k = kind::c2r; p.dbl = false; break;
    case CUFFT_Z2D: p.k = kind::c2r; p.dbl = true; break;
    default: return status::invalid_type;
    }
    if (batch < 1)
        return status::invalid_size;
    for (int d = 0; d < rank; ++d)
        if (n[d] < 1)
            return status::invalid_size;
    p.rank = rank;
    p.n = {1, 1, 1};
    std::copy_n(n, rank, p.n.begin());
    p.batch = batch;
    // A null embedding is cuFFT's basic layout, for both arrays: the other
    // advanced parameters are ignored
    p.basic = inembed == nullptr || onembed == nullptr;
    if (!p.basic) {
        // A distance matters only between transforms; an embedding holds
        // the array's lengths
        if (istride < 1 || ostride < 1 || (batch > 1 && (idist < 1 || odist < 1)))
            return status::invalid_value;
        const auto in_lengths = array_lengths(p, true), out_lengths = array_lengths(p, false);
        for (int d = 1; d < rank; ++d)
            if (inembed[d] < in_lengths[d] || onembed[d] < out_lengths[d])
                return status::invalid_value;
        auto advanced = [&](const long long *embed, long long stride, long long dist) {
            layout l;
            l.stride[rank - 1] = stride;
            for (int d = rank - 2; d >= 0; --d)
                l.stride[d] = l.stride[d + 1] * embed[d + 1];
            l.dist = std::max(dist, 0LL);
            return l;
        };
        p.in = advanced(inembed, istride, idist);
        p.out = advanced(onembed, ostride, odist);
    }
    p.made = true;
    made = p;
    return status::success;
}

inline std::vector<long long> widen(int rank, const int *v)
{
    if (v == nullptr)
        return {};
    return std::vector<long long>(v, v + std::clamp(rank, 0, 3));
}

inline status make_int(plan &p, int rank, const int *n, const int *inembed, int istride, int idist,
                       const int *onembed, int ostride, int odist, int type, int batch)
{
    if (n == nullptr)
        return status::invalid_value;
    const auto ln = widen(rank, n), li = widen(rank, inembed), lo = widen(rank, onembed);
    return make(p, rank, ln.data(), inembed ? li.data() : nullptr, istride, idist, onembed ? lo.data() : nullptr,
                ostride, odist, type, batch);
}

//! Which precision and kind a type is, for the Exec functions' checks
inline bool matches(const plan &p, kind k, bool dbl)
{
    return p.k == k && p.dbl == dbl;
}

// cuFFT's handles are ints: an index into a table, never destroyed, since a
// static object's destructor may destroy a plan after main returns
std::mutex &handles_mutex = *new std::mutex;
std::vector<std::unique_ptr<plan>> &handles = *new std::vector<std::unique_ptr<plan>>;

inline plan *from_handle(int handle)
{
    std::lock_guard lock(handles_mutex);
    if (handle < 1 || std::size_t(handle) > handles.size())
        return nullptr;
    return handles[handle - 1].get();
}

inline int new_handle()
{
    std::lock_guard lock(handles_mutex);
    for (std::size_t i = 0; i < handles.size(); ++i)
        if (handles[i] == nullptr) {
            handles[i] = std::make_unique<plan>();
            return int(i + 1);
        }
    handles.push_back(std::make_unique<plan>());
    return int(handles.size());
}

inline bool destroy_handle(int handle)
{
    std::lock_guard lock(handles_mutex);
    if (handle < 1 || std::size_t(handle) > handles.size() || handles[handle - 1] == nullptr)
        return false;
    handles[handle - 1].reset();
    return true;
}

//! cuFFT's and hipFFT's results have the same values
inline int result_of(status s)
{
    switch (s) {
    case status::success: return CUFFT_SUCCESS;
    case status::invalid_plan: return CUFFT_INVALID_PLAN;
    case status::alloc_failed: return CUFFT_ALLOC_FAILED;
    case status::invalid_type: return CUFFT_INVALID_TYPE;
    case status::invalid_value:
    case status::invalid_array_type:
    case status::invalid_strides: return CUFFT_INVALID_VALUE;
    case status::exec_failed: return CUFFT_EXEC_FAILED;
    case status::invalid_size: return CUFFT_INVALID_SIZE;
    case status::not_supported: return CUFFT_NOT_SUPPORTED;
    case status::internal_error: break;
    }
    return CUFFT_INTERNAL_ERROR;
}

inline rocfft_status rocfft_of(status s)
{
    switch (s) {
    case status::success: return rocfft_status_success;
    case status::invalid_value:
    case status::invalid_type:
    case status::not_supported: return rocfft_status_invalid_arg_value;
    case status::invalid_size: return rocfft_status_invalid_dimensions;
    case status::invalid_array_type: return rocfft_status_invalid_array_type;
    case status::invalid_strides: return rocfft_status_invalid_strides;
    default: break;
    }
    return rocfft_status_failure;
}

//! A callback of a kind (cuFFT's and hipFFT's values): load or store, and
//! whether its type is the plan's
inline status set_callback(plan &p, void **routine, int type, void **info)
{
    if (!p.made)
        return status::invalid_plan;
    if (type < CUFFT_CB_LD_COMPLEX || type > CUFFT_CB_ST_REAL_DOUBLE)
        return status::invalid_value;
    const bool store = type >= CUFFT_CB_ST_COMPLEX;
    const bool dbl = type % 2 == 1;
    const bool real = (type % 4) >= 2;
    const bool real_side = store ? p.k == kind::c2r : p.k == kind::r2c;
    if (dbl != p.dbl || real != real_side)
        return status::invalid_type;
    callback &c = store ? p.store : p.load;
    c.function = routine != nullptr ? routine[0] : nullptr;
    c.data = info != nullptr ? info[0] : nullptr;
    c.shared = store ? p.store_shared : p.load_shared;
    return status::success;
}

inline status clear_callback(plan &p, int type)
{
    if (type < CUFFT_CB_LD_COMPLEX || type > CUFFT_CB_ST_REAL_DOUBLE)
        return status::invalid_value;
    (type >= CUFFT_CB_ST_COMPLEX ? p.store : p.load) = callback{};
    return status::success;
}

inline status set_shared_size(plan &p, int type, std::size_t size)
{
    if (type < CUFFT_CB_LD_COMPLEX || type > CUFFT_CB_ST_REAL_DOUBLE)
        return status::invalid_value;
    if (type >= CUFFT_CB_ST_COMPLEX)
        p.store.shared = p.store_shared = size;
    else
        p.load.shared = p.load_shared = size;
    return status::success;
}

} // namespace cuda4cpu::detail::fft

using namespace cuda4cpu::detail;

namespace {

// cuFFT's and hipFFT's functions differ only in their handles' and results'
// types: these take the plan

fft::status plan_many(fft::plan *p, int rank, int *n, int *inembed, int istride, int idist, int *onembed, int ostride,
                      int odist, int type, int batch, size_t *workSize)
{
    if (p == nullptr)
        return fft::status::invalid_plan;
    const fft::status s = fft::make_int(*p, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch);
    if (s == fft::status::success && workSize != nullptr)
        *workSize = 0;
    return s;
}

fft::status plan_many64(fft::plan *p, int rank, long long *n, long long *inembed, long long istride, long long idist,
                        long long *onembed, long long ostride, long long odist, int type, long long batch,
                        size_t *workSize)
{
    if (p == nullptr)
        return fft::status::invalid_plan;
    const fft::status s = fft::make(*p, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch);
    if (s == fft::status::success && workSize != nullptr)
        *workSize = 0;
    return s;
}

fft::status plan_nd(fft::plan *p, std::initializer_list<int> lengths, int type, int batch, size_t *workSize)
{
    std::vector<int> n(lengths);
    return plan_many(p, int(n.size()), n.data(), nullptr, 1, 0, nullptr, 1, 0, type, batch, workSize);
}

//! Checks a plan without one: the sizes it would take
fft::status estimate(int rank, int *n, int type, int batch, size_t *workSize)
{
    fft::plan p;
    const fft::status s = fft::make_int(p, rank, n, nullptr, 1, 0, nullptr, 1, 0, type, batch);
    if (s == fft::status::success && workSize != nullptr)
        *workSize = 0;
    return s;
}

fft::status work_size(const fft::plan *p, size_t *workSize)
{
    if (p == nullptr)
        return fft::status::invalid_plan;
    if (workSize != nullptr)
        *workSize = 0;
    return fft::status::success;
}

fft::status exec(fft::plan *p, void *in, void *out, fft::kind k, bool dbl, int direction)
{
    if (p == nullptr || !p->made)
        return fft::status::invalid_plan;
    if (!fft::matches(*p, k, dbl))
        return fft::status::invalid_type;
    if (direction != CUFFT_FORWARD && direction != CUFFT_INVERSE)
        return fft::status::invalid_value;
    return fft::execute(*p, in, out, direction == CUFFT_FORWARD);
}

cufftResult cufft(fft::status s)
{
    return cufftResult(fft::result_of(s));
}

hipfftResult hipfft(fft::status s)
{
    return hipfftResult(fft::result_of(s));
}

fft::plan *hip_plan(hipfftHandle h)
{
    return reinterpret_cast<fft::plan *>(h);
}

} // namespace

extern "C" {

// ----------------------------------------------------------------------------
// cuFFT

cufftResult cufftCreate(cufftHandle *handle)
{
    if (handle == nullptr)
        return CUFFT_INVALID_VALUE;
    *handle = fft::new_handle();
    return CUFFT_SUCCESS;
}

cufftResult cufftDestroy(cufftHandle plan)
{
    return fft::destroy_handle(plan) ? CUFFT_SUCCESS : CUFFT_INVALID_PLAN;
}

cufftResult cufftMakePlan1d(cufftHandle plan, int nx, cufftType type, int batch, size_t *workSize)
{
    return cufft(plan_nd(fft::from_handle(plan), {nx}, type, batch, workSize));
}

cufftResult cufftMakePlan2d(cufftHandle plan, int nx, int ny, cufftType type, size_t *workSize)
{
    return cufft(plan_nd(fft::from_handle(plan), {nx, ny}, type, 1, workSize));
}

cufftResult cufftMakePlan3d(cufftHandle plan, int nx, int ny, int nz, cufftType type, size_t *workSize)
{
    return cufft(plan_nd(fft::from_handle(plan), {nx, ny, nz}, type, 1, workSize));
}

cufftResult cufftMakePlanMany(cufftHandle plan, int rank, int *n, int *inembed, int istride, int idist,
                              int *onembed, int ostride, int odist, cufftType type, int batch, size_t *workSize)
{
    return cufft(plan_many(fft::from_handle(plan), rank, n, inembed, istride, idist, onembed, ostride, odist, type,
                           batch, workSize));
}

cufftResult cufftMakePlanMany64(cufftHandle plan, int rank, long long int *n, long long int *inembed,
                                long long int istride, long long int idist, long long int *onembed,
                                long long int ostride, long long int odist, cufftType type, long long int batch,
                                size_t *workSize)
{
    return cufft(plan_many64(fft::from_handle(plan), rank, n, inembed, istride, idist, onembed, ostride, odist, type,
                             batch, workSize));
}

} // extern "C"

namespace {
//! cufftPlan*: a handle and its plan, or no handle
template <typename F>
cufftResult new_plan(cufftHandle *plan, F &&make)
{
    if (plan == nullptr)
        return CUFFT_INVALID_VALUE;
    const cufftHandle h = fft::new_handle();
    const cufftResult r = make(h);
    if (r != CUFFT_SUCCESS) {
        fft::destroy_handle(h);
        return r;
    }
    *plan = h;
    return CUFFT_SUCCESS;
}
} // namespace

extern "C" {

cufftResult cufftPlan1d(cufftHandle *plan, int nx, cufftType type, int batch)
{
    return new_plan(plan, [&](cufftHandle h) { return cufftMakePlan1d(h, nx, type, batch, nullptr); });
}

cufftResult cufftPlan2d(cufftHandle *plan, int nx, int ny, cufftType type)
{
    return new_plan(plan, [&](cufftHandle h) { return cufftMakePlan2d(h, nx, ny, type, nullptr); });
}

cufftResult cufftPlan3d(cufftHandle *plan, int nx, int ny, int nz, cufftType type)
{
    return new_plan(plan, [&](cufftHandle h) { return cufftMakePlan3d(h, nx, ny, nz, type, nullptr); });
}

cufftResult cufftPlanMany(cufftHandle *plan, int rank, int *n, int *inembed, int istride, int idist, int *onembed,
                          int ostride, int odist, cufftType type, int batch)
{
    return new_plan(plan, [&](cufftHandle h) {
        return cufftMakePlanMany(h, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, nullptr);
    });
}

cufftResult cufftEstimate1d(int nx, cufftType type, int batch, size_t *workSize)
{
    return cufft(estimate(1, &nx, type, batch, workSize));
}

cufftResult cufftEstimate2d(int nx, int ny, cufftType type, size_t *workSize)
{
    int n[2] = {nx, ny};
    return cufft(estimate(2, n, type, 1, workSize));
}

cufftResult cufftEstimate3d(int nx, int ny, int nz, cufftType type, size_t *workSize)
{
    int n[3] = {nx, ny, nz};
    return cufft(estimate(3, n, type, 1, workSize));
}

cufftResult cufftEstimateMany(int rank, int *n, int *, int, int, int *, int, int, cufftType type, int batch,
                              size_t *workSize)
{
    return cufft(estimate(rank, n, type, batch, workSize));
}

cufftResult cufftGetSize1d(cufftHandle handle, int nx, cufftType type, int batch, size_t *workSize)
{
    fft::plan *p = fft::from_handle(handle);
    return p == nullptr ? CUFFT_INVALID_PLAN : cufftEstimate1d(nx, type, batch, workSize);
}

cufftResult cufftGetSize2d(cufftHandle handle, int nx, int ny, cufftType type, size_t *workSize)
{
    fft::plan *p = fft::from_handle(handle);
    return p == nullptr ? CUFFT_INVALID_PLAN : cufftEstimate2d(nx, ny, type, workSize);
}

cufftResult cufftGetSize3d(cufftHandle handle, int nx, int ny, int nz, cufftType type, size_t *workSize)
{
    fft::plan *p = fft::from_handle(handle);
    return p == nullptr ? CUFFT_INVALID_PLAN : cufftEstimate3d(nx, ny, nz, type, workSize);
}

cufftResult cufftGetSizeMany(cufftHandle handle, int rank, int *n, int *inembed, int istride, int idist,
                             int *onembed, int ostride, int odist, cufftType type, int batch, size_t *workArea)
{
    fft::plan *p = fft::from_handle(handle);
    return p == nullptr ? CUFFT_INVALID_PLAN
                        : cufftEstimateMany(rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch,
                                            workArea);
}

cufftResult cufftGetSizeMany64(cufftHandle plan, int rank, long long int *n, long long int *inembed,
                               long long int istride, long long int idist, long long int *onembed,
                               long long int ostride, long long int odist, cufftType type, long long int batch,
                               size_t *workSize)
{
    fft::plan *p = fft::from_handle(plan);
    if (p == nullptr)
        return CUFFT_INVALID_PLAN;
    fft::plan probe;
    return cufft(plan_many64(&probe, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize));
}

cufftResult cufftGetSize(cufftHandle handle, size_t *workSize)
{
    return cufft(work_size(fft::from_handle(handle), workSize));
}

cufftResult cufftSetWorkArea(cufftHandle plan, void *)
{
    return fft::from_handle(plan) == nullptr ? CUFFT_INVALID_PLAN : CUFFT_SUCCESS;
}

cufftResult cufftSetAutoAllocation(cufftHandle plan, int)
{
    return fft::from_handle(plan) == nullptr ? CUFFT_INVALID_PLAN : CUFFT_SUCCESS;
}

cufftResult cufftExecC2C(cufftHandle plan, cufftComplex *idata, cufftComplex *odata, int direction)
{
    return cufft(exec(fft::from_handle(plan), idata, odata, fft::kind::c2c, false, direction));
}

cufftResult cufftExecR2C(cufftHandle plan, cufftReal *idata, cufftComplex *odata)
{
    return cufft(exec(fft::from_handle(plan), idata, odata, fft::kind::r2c, false, CUFFT_FORWARD));
}

cufftResult cufftExecC2R(cufftHandle plan, cufftComplex *idata, cufftReal *odata)
{
    return cufft(exec(fft::from_handle(plan), idata, odata, fft::kind::c2r, false, CUFFT_INVERSE));
}

cufftResult cufftExecZ2Z(cufftHandle plan, cufftDoubleComplex *idata, cufftDoubleComplex *odata, int direction)
{
    return cufft(exec(fft::from_handle(plan), idata, odata, fft::kind::c2c, true, direction));
}

cufftResult cufftExecD2Z(cufftHandle plan, cufftDoubleReal *idata, cufftDoubleComplex *odata)
{
    return cufft(exec(fft::from_handle(plan), idata, odata, fft::kind::r2c, true, CUFFT_FORWARD));
}

cufftResult cufftExecZ2D(cufftHandle plan, cufftDoubleComplex *idata, cufftDoubleReal *odata)
{
    return cufft(exec(fft::from_handle(plan), idata, odata, fft::kind::c2r, true, CUFFT_INVERSE));
}

cufftResult cufftSetStream(cufftHandle plan, cudaStream_t stream)
{
    fft::plan *p = fft::from_handle(plan);
    if (p == nullptr)
        return CUFFT_INVALID_PLAN;
    p->stream = stream;
    return CUFFT_SUCCESS;
}

cufftResult cufftGetVersion(int *version)
{
    if (version == nullptr)
        return CUFFT_INVALID_VALUE;
    *version = CUFFT_VERSION;
    return CUFFT_SUCCESS;
}

cufftResult cufftGetProperty(libraryPropertyType type, int *value)
{
    if (value == nullptr)
        return CUFFT_INVALID_VALUE;
    switch (type) {
    case MAJOR_VERSION: *value = CUFFT_VER_MAJOR; return CUFFT_SUCCESS;
    case MINOR_VERSION: *value = CUFFT_VER_MINOR; return CUFFT_SUCCESS;
    case PATCH_LEVEL: *value = CUFFT_VER_PATCH; return CUFFT_SUCCESS;
    }
    return CUFFT_INVALID_TYPE;
}

cufftResult cufftXtSetCallback(cufftHandle plan, void **callback_routine, cufftXtCallbackType cbType,
                               void **caller_info)
{
    fft::plan *p = fft::from_handle(plan);
    return p == nullptr ? CUFFT_INVALID_PLAN : cufft(fft::set_callback(*p, callback_routine, cbType, caller_info));
}

cufftResult cufftXtClearCallback(cufftHandle plan, cufftXtCallbackType cbType)
{
    fft::plan *p = fft::from_handle(plan);
    return p == nullptr ? CUFFT_INVALID_PLAN : cufft(fft::clear_callback(*p, cbType));
}

cufftResult cufftXtSetCallbackSharedSize(cufftHandle plan, cufftXtCallbackType cbType, size_t sharedSize)
{
    fft::plan *p = fft::from_handle(plan);
    return p == nullptr ? CUFFT_INVALID_PLAN : cufft(fft::set_shared_size(*p, cbType, sharedSize));
}

// ----------------------------------------------------------------------------
// hipFFT

hipfftResult hipfftCreate(hipfftHandle *plan)
{
    if (plan == nullptr)
        return HIPFFT_INVALID_VALUE;
    *plan = reinterpret_cast<hipfftHandle>(new fft::plan);
    return HIPFFT_SUCCESS;
}

hipfftResult hipfftDestroy(hipfftHandle plan)
{
    if (plan == nullptr)
        return HIPFFT_INVALID_PLAN;
    delete hip_plan(plan);
    return HIPFFT_SUCCESS;
}

hipfftResult hipfftMakePlan1d(hipfftHandle plan, int nx, hipfftType type, int batch, size_t *workSize)
{
    return hipfft(plan_nd(hip_plan(plan), {nx}, type, batch, workSize));
}

hipfftResult hipfftMakePlan2d(hipfftHandle plan, int nx, int ny, hipfftType type, size_t *workSize)
{
    return hipfft(plan_nd(hip_plan(plan), {nx, ny}, type, 1, workSize));
}

hipfftResult hipfftMakePlan3d(hipfftHandle plan, int nx, int ny, int nz, hipfftType type, size_t *workSize)
{
    return hipfft(plan_nd(hip_plan(plan), {nx, ny, nz}, type, 1, workSize));
}

hipfftResult hipfftMakePlanMany(hipfftHandle plan, int rank, int *n, int *inembed, int istride, int idist,
                                int *onembed, int ostride, int odist, hipfftType type, int batch, size_t *workSize)
{
    return hipfft(
        plan_many(hip_plan(plan), rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize));
}

hipfftResult hipfftMakePlanMany64(hipfftHandle plan, int rank, long long int *n, long long int *inembed,
                                  long long int istride, long long int idist, long long int *onembed,
                                  long long int ostride, long long int odist, hipfftType type,
                                  long long int batch, size_t *workSize)
{
    return hipfft(plan_many64(hip_plan(plan), rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch,
                              workSize));
}

} // extern "C"

namespace {
template <typename F>
hipfftResult new_hip_plan(hipfftHandle *plan, F &&make)
{
    if (plan == nullptr)
        return HIPFFT_INVALID_VALUE;
    hipfftHandle h = reinterpret_cast<hipfftHandle>(new fft::plan);
    const hipfftResult r = make(h);
    if (r != HIPFFT_SUCCESS) {
        delete hip_plan(h);
        return r;
    }
    *plan = h;
    return HIPFFT_SUCCESS;
}
} // namespace

extern "C" {

hipfftResult hipfftPlan1d(hipfftHandle *plan, int nx, hipfftType type, int batch)
{
    return new_hip_plan(plan, [&](hipfftHandle h) { return hipfftMakePlan1d(h, nx, type, batch, nullptr); });
}

hipfftResult hipfftPlan2d(hipfftHandle *plan, int nx, int ny, hipfftType type)
{
    return new_hip_plan(plan, [&](hipfftHandle h) { return hipfftMakePlan2d(h, nx, ny, type, nullptr); });
}

hipfftResult hipfftPlan3d(hipfftHandle *plan, int nx, int ny, int nz, hipfftType type)
{
    return new_hip_plan(plan, [&](hipfftHandle h) { return hipfftMakePlan3d(h, nx, ny, nz, type, nullptr); });
}

hipfftResult hipfftPlanMany(hipfftHandle *plan, int rank, int *n, int *inembed, int istride, int idist,
                            int *onembed, int ostride, int odist, hipfftType type, int batch)
{
    return new_hip_plan(plan, [&](hipfftHandle h) {
        return hipfftMakePlanMany(h, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, nullptr);
    });
}

hipfftResult hipfftEstimate1d(int nx, hipfftType type, int batch, size_t *workSize)
{
    return hipfft(estimate(1, &nx, type, batch, workSize));
}

hipfftResult hipfftEstimate2d(int nx, int ny, hipfftType type, size_t *workSize)
{
    int n[2] = {nx, ny};
    return hipfft(estimate(2, n, type, 1, workSize));
}

hipfftResult hipfftEstimate3d(int nx, int ny, int nz, hipfftType type, size_t *workSize)
{
    int n[3] = {nx, ny, nz};
    return hipfft(estimate(3, n, type, 1, workSize));
}

hipfftResult hipfftEstimateMany(int rank, int *n, int *, int, int, int *, int, int, hipfftType type, int batch,
                                size_t *workSize)
{
    return hipfft(estimate(rank, n, type, batch, workSize));
}

hipfftResult hipfftGetSize1d(hipfftHandle plan, int nx, hipfftType type, int batch, size_t *workSize)
{
    return plan == nullptr ? HIPFFT_INVALID_PLAN : hipfftEstimate1d(nx, type, batch, workSize);
}

hipfftResult hipfftGetSize2d(hipfftHandle plan, int nx, int ny, hipfftType type, size_t *workSize)
{
    return plan == nullptr ? HIPFFT_INVALID_PLAN : hipfftEstimate2d(nx, ny, type, workSize);
}

hipfftResult hipfftGetSize3d(hipfftHandle plan, int nx, int ny, int nz, hipfftType type, size_t *workSize)
{
    return plan == nullptr ? HIPFFT_INVALID_PLAN : hipfftEstimate3d(nx, ny, nz, type, workSize);
}

hipfftResult hipfftGetSizeMany(hipfftHandle plan, int rank, int *n, int *inembed, int istride, int idist,
                               int *onembed, int ostride, int odist, hipfftType type, int batch, size_t *workSize)
{
    return plan == nullptr ? HIPFFT_INVALID_PLAN
                           : hipfftEstimateMany(rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch,
                                                workSize);
}

hipfftResult hipfftGetSizeMany64(hipfftHandle plan, int rank, long long int *n, long long int *inembed,
                                 long long int istride, long long int idist, long long int *onembed,
                                 long long int ostride, long long int odist, hipfftType type, long long int batch,
                                 size_t *workSize)
{
    if (plan == nullptr)
        return HIPFFT_INVALID_PLAN;
    fft::plan probe;
    return hipfft(plan_many64(&probe, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch,
                              workSize));
}

hipfftResult hipfftGetSize(hipfftHandle plan, size_t *workSize)
{
    return hipfft(work_size(hip_plan(plan), workSize));
}

hipfftResult hipfftSetAutoAllocation(hipfftHandle plan, int)
{
    return plan == nullptr ? HIPFFT_INVALID_PLAN : HIPFFT_SUCCESS;
}

hipfftResult hipfftSetWorkArea(hipfftHandle plan, void *)
{
    return plan == nullptr ? HIPFFT_INVALID_PLAN : HIPFFT_SUCCESS;
}

hipfftResult hipfftExecC2C(hipfftHandle plan, hipfftComplex *idata, hipfftComplex *odata, int direction)
{
    return hipfft(exec(hip_plan(plan), idata, odata, fft::kind::c2c, false, direction));
}

hipfftResult hipfftExecR2C(hipfftHandle plan, hipfftReal *idata, hipfftComplex *odata)
{
    return hipfft(exec(hip_plan(plan), idata, odata, fft::kind::r2c, false, HIPFFT_FORWARD));
}

hipfftResult hipfftExecC2R(hipfftHandle plan, hipfftComplex *idata, hipfftReal *odata)
{
    return hipfft(exec(hip_plan(plan), idata, odata, fft::kind::c2r, false, HIPFFT_BACKWARD));
}

hipfftResult hipfftExecZ2Z(hipfftHandle plan, hipfftDoubleComplex *idata, hipfftDoubleComplex *odata,
                           int direction)
{
    return hipfft(exec(hip_plan(plan), idata, odata, fft::kind::c2c, true, direction));
}

hipfftResult hipfftExecD2Z(hipfftHandle plan, hipfftDoubleReal *idata, hipfftDoubleComplex *odata)
{
    return hipfft(exec(hip_plan(plan), idata, odata, fft::kind::r2c, true, HIPFFT_FORWARD));
}

hipfftResult hipfftExecZ2D(hipfftHandle plan, hipfftDoubleComplex *idata, hipfftDoubleReal *odata)
{
    return hipfft(exec(hip_plan(plan), idata, odata, fft::kind::c2r, true, HIPFFT_BACKWARD));
}

hipfftResult hipfftSetStream(hipfftHandle plan, hipStream_t stream)
{
    if (plan == nullptr)
        return HIPFFT_INVALID_PLAN;
    hip_plan(plan)->stream = stream;
    return HIPFFT_SUCCESS;
}

hipfftResult hipfftGetVersion(int *version)
{
    if (version == nullptr)
        return HIPFFT_INVALID_VALUE;
    *version = HIPFFT_VERSION;
    return HIPFFT_SUCCESS;
}

hipfftResult hipfftGetProperty(hipfftLibraryPropertyType type, int *value)
{
    if (value == nullptr)
        return HIPFFT_INVALID_VALUE;
    switch (type) {
    case HIPFFT_MAJOR_VERSION: *value = hipfftVersionMajor; return HIPFFT_SUCCESS;
    case HIPFFT_MINOR_VERSION: *value = hipfftVersionMinor; return HIPFFT_SUCCESS;
    case HIPFFT_PATCH_LEVEL: *value = hipfftVersionPatch; return HIPFFT_SUCCESS;
    }
    return HIPFFT_INVALID_TYPE;
}

hipfftResult hipfftXtSetCallback(hipfftHandle plan, void **callbacks, hipfftXtCallbackType cbtype,
                                 void **callbackData)
{
    return plan == nullptr ? HIPFFT_INVALID_PLAN : hipfft(fft::set_callback(*hip_plan(plan), callbacks, cbtype, callbackData));
}

hipfftResult hipfftXtClearCallback(hipfftHandle plan, hipfftXtCallbackType cbtype)
{
    return plan == nullptr ? HIPFFT_INVALID_PLAN : hipfft(fft::clear_callback(*hip_plan(plan), cbtype));
}

hipfftResult hipfftXtSetCallbackSharedSize(hipfftHandle plan, hipfftXtCallbackType cbtype, size_t sharedSize)
{
    return plan == nullptr ? HIPFFT_INVALID_PLAN : hipfft(fft::set_shared_size(*hip_plan(plan), cbtype, sharedSize));
}

} // extern "C"

// ----------------------------------------------------------------------------
// rocFFT

struct rocfft_plan_description_t {
    rocfft_array_type in_type = rocfft_array_type_unset;
    rocfft_array_type out_type = rocfft_array_type_unset;
    std::size_t in_offset = 0;
    std::size_t out_offset = 0;
    std::vector<std::size_t> in_strides;   //!< Fastest dimension first, as rocFFT's
    std::vector<std::size_t> out_strides;
    std::size_t in_distance = 0;
    std::size_t out_distance = 0;
    double scale = 1;
};

struct rocfft_plan_t {
    fft::plan p;
    bool in_place = false;
    bool forward = true;
};

struct rocfft_execution_info_t {
    void *stream = nullptr;
    fft::callback load;
    fft::callback store;
};

namespace {

//! A rocFFT layout: the description's strides and distance (fastest first,
//! reversed into the plan's order), or the default's where it leaves them
fft::layout rocfft_layout(const fft::plan &p, bool input, const fft::layout &defaults,
                          const std::vector<std::size_t> &strides, std::size_t distance, std::size_t offset)
{
    fft::layout l = defaults;
    if (!strides.empty()) {
        for (int d = 0; d < p.rank; ++d)
            l.stride[p.rank - 1 - d] = (long long)strides[d];
        // The default distance: past the slowest dimension's last element
        l.dist = l.stride[0] * fft::array_lengths(p, input)[0];
    }
    if (distance != 0)
        l.dist = (long long)distance;
    l.offset = (long long)offset;
    return l;
}

} // namespace

extern "C" {

rocfft_status rocfft_setup(void)
{
    return rocfft_status_success;
}

rocfft_status rocfft_cleanup(void)
{
    return rocfft_status_success;
}

rocfft_status rocfft_get_version_string(char *buf, size_t len)
{
    static constexpr char version[] = "1.0.35-cuda4cpu";
    if (buf == nullptr || len < sizeof(version))
        return rocfft_status_invalid_arg_value;
    std::memcpy(buf, version, sizeof(version));
    return rocfft_status_success;
}

rocfft_status rocfft_plan_description_create(rocfft_plan_description *description)
{
    if (description == nullptr)
        return rocfft_status_invalid_arg_value;
    *description = new rocfft_plan_description_t;
    return rocfft_status_success;
}

rocfft_status rocfft_plan_description_destroy(rocfft_plan_description description)
{
    delete description;
    return rocfft_status_success;
}

rocfft_status rocfft_plan_description_set_scale_factor(rocfft_plan_description description, const double scale_factor)
{
    if (description == nullptr || !std::isfinite(scale_factor))
        return rocfft_status_invalid_arg_value;
    description->scale = scale_factor;
    return rocfft_status_success;
}

rocfft_status rocfft_plan_description_set_data_layout(rocfft_plan_description description,
                                                      const rocfft_array_type in_array_type,
                                                      const rocfft_array_type out_array_type, const size_t *in_offsets,
                                                      const size_t *out_offsets, const size_t in_strides_size,
                                                      const size_t *in_strides, const size_t in_distance,
                                                      const size_t out_strides_size, const size_t *out_strides,
                                                      const size_t out_distance)
{
    if (description == nullptr || in_strides_size > 3 || out_strides_size > 3 ||
        (in_strides_size != 0 && in_strides == nullptr) || (out_strides_size != 0 && out_strides == nullptr))
        return rocfft_status_invalid_arg_value;
    description->in_type = in_array_type;
    description->out_type = out_array_type;
    description->in_offset = in_offsets != nullptr ? in_offsets[0] : 0;
    description->out_offset = out_offsets != nullptr ? out_offsets[0] : 0;
    description->in_strides.assign(in_strides, in_strides + in_strides_size);
    description->out_strides.assign(out_strides, out_strides + out_strides_size);
    description->in_distance = in_distance;
    description->out_distance = out_distance;
    return rocfft_status_success;
}

rocfft_status rocfft_plan_create(rocfft_plan *plan, rocfft_result_placement placement,
                                 rocfft_transform_type transform_type, rocfft_precision precision, size_t dimensions,
                                 const size_t *lengths, size_t number_of_transforms,
                                 const rocfft_plan_description description)
{
    if (plan == nullptr || lengths == nullptr || number_of_transforms == 0)
        return rocfft_status_invalid_arg_value;
    if (dimensions < 1 || dimensions > 3)
        return rocfft_status_invalid_dimensions;
    if (precision != rocfft_precision_single && precision != rocfft_precision_double)
        return rocfft_status_invalid_arg_value;
    if (placement != rocfft_placement_inplace && placement != rocfft_placement_notinplace)
        return rocfft_status_invalid_arg_value;
    const bool dbl = precision == rocfft_precision_double;
    int type;
    rocfft_array_type in_type, out_type;
    switch (transform_type) {
    case rocfft_transform_type_complex_forward:
    case rocfft_transform_type_complex_inverse:
        type = dbl ? CUFFT_Z2Z : CUFFT_C2C;
        in_type = out_type = rocfft_array_type_complex_interleaved;
        break;
    case rocfft_transform_type_real_forward:
        type = dbl ? CUFFT_D2Z : CUFFT_R2C;
        in_type = rocfft_array_type_real;
        out_type = rocfft_array_type_hermitian_interleaved;
        break;
    case rocfft_transform_type_real_inverse:
        type = dbl ? CUFFT_Z2D : CUFFT_C2R;
        in_type = rocfft_array_type_hermitian_interleaved;
        out_type = rocfft_array_type_real;
        break;
    default: return rocfft_status_invalid_arg_value;
    }
    const rocfft_plan_description_t defaults;
    const rocfft_plan_description_t &d = description != nullptr ? *description : defaults;
    // Interleaved arrays of the transform's kinds
    if ((d.in_type != rocfft_array_type_unset && d.in_type != in_type) ||
        (d.out_type != rocfft_array_type_unset && d.out_type != out_type))
        return rocfft_status_invalid_array_type;
    if ((!d.in_strides.empty() && d.in_strides.size() != dimensions) ||
        (!d.out_strides.empty() && d.out_strides.size() != dimensions) ||
        std::count(d.in_strides.begin(), d.in_strides.end(), 0u) != 0 ||
        std::count(d.out_strides.begin(), d.out_strides.end(), 0u) != 0)
        return rocfft_status_invalid_strides;

    auto made = std::make_unique<rocfft_plan_t>();
    long long n[3];
    for (size_t i = 0; i < dimensions; ++i)
        n[dimensions - 1 - i] = (long long)lengths[i];
    const fft::status s =
        fft::make(made->p, int(dimensions), n, nullptr, 1, 0, nullptr, 1, 0, type, (long long)number_of_transforms);
    if (s != fft::status::success)
        return fft::rocfft_of(s);
    made->in_place = placement == rocfft_placement_inplace;
    made->forward = transform_type == rocfft_transform_type_complex_forward ||
                    transform_type == rocfft_transform_type_real_forward;
    const auto [in, out] = fft::basic_layouts(made->p, made->in_place);
    made->p.basic = false;
    made->p.in = rocfft_layout(made->p, true, in, d.in_strides, d.in_distance, d.in_offset);
    made->p.out = rocfft_layout(made->p, false, out, d.out_strides, d.out_distance, d.out_offset);
    made->p.scale = d.scale;
    *plan = made.release();
    return rocfft_status_success;
}

rocfft_status rocfft_plan_destroy(rocfft_plan plan)
{
    delete plan;
    return rocfft_status_success;
}

rocfft_status rocfft_plan_get_work_buffer_size(const rocfft_plan plan, size_t *size_in_bytes)
{
    if (plan == nullptr || size_in_bytes == nullptr)
        return rocfft_status_invalid_arg_value;
    *size_in_bytes = 0;
    return rocfft_status_success;
}

rocfft_status rocfft_plan_get_print(const rocfft_plan plan)
{
    if (plan == nullptr)
        return rocfft_status_invalid_arg_value;
    const fft::plan &p = plan->p;
    std::printf("rocfft plan (cuda4cpu): %d dimensions, lengths", p.rank);
    for (int d = p.rank - 1; d >= 0; --d)
        std::printf(" %lld", p.n[d]);
    std::printf(", %lld transforms, %s, %s\n", p.batch, p.dbl ? "double" : "single",
                plan->in_place ? "in place" : "not in place");
    return rocfft_status_success;
}

rocfft_status rocfft_execution_info_create(rocfft_execution_info *info)
{
    if (info == nullptr)
        return rocfft_status_invalid_arg_value;
    *info = new rocfft_execution_info_t;
    return rocfft_status_success;
}

rocfft_status rocfft_execution_info_destroy(rocfft_execution_info info)
{
    delete info;
    return rocfft_status_success;
}

rocfft_status rocfft_execution_info_set_work_buffer(rocfft_execution_info info, void *, const size_t)
{
    return info == nullptr ? rocfft_status_invalid_arg_value : rocfft_status_success;
}

rocfft_status rocfft_execution_info_set_stream(rocfft_execution_info info, void *stream)
{
    if (info == nullptr)
        return rocfft_status_invalid_arg_value;
    info->stream = stream;
    return rocfft_status_success;
}

rocfft_status rocfft_execution_info_set_load_callback(rocfft_execution_info info, void **cb_functions, void **cb_data,
                                                      size_t shared_mem_bytes)
{
    if (info == nullptr)
        return rocfft_status_invalid_arg_value;
    info->load = {cb_functions != nullptr ? cb_functions[0] : nullptr, cb_data != nullptr ? cb_data[0] : nullptr,
                  shared_mem_bytes};
    return rocfft_status_success;
}

rocfft_status rocfft_execution_info_set_store_callback(rocfft_execution_info info, void **cb_functions,
                                                       void **cb_data, size_t shared_mem_bytes)
{
    if (info == nullptr)
        return rocfft_status_invalid_arg_value;
    info->store = {cb_functions != nullptr ? cb_functions[0] : nullptr, cb_data != nullptr ? cb_data[0] : nullptr,
                   shared_mem_bytes};
    return rocfft_status_success;
}

rocfft_status rocfft_execute(const rocfft_plan plan, void *in_buffer[], void *out_buffer[], rocfft_execution_info info)
{
    if (plan == nullptr || in_buffer == nullptr || in_buffer[0] == nullptr)
        return rocfft_status_invalid_arg_value;
    void *in = in_buffer[0];
    void *out = plan->in_place ? in : (out_buffer != nullptr ? out_buffer[0] : nullptr);
    if (out == nullptr)
        return rocfft_status_invalid_arg_value;
    fft::plan p = plan->p;
    if (info != nullptr) {
        p.stream = static_cast<cudaStream_t>(info->stream);
        p.load = info->load;
        p.store = info->store;
    }
    return fft::rocfft_of(fft::execute(p, in, out, plan->forward));
}

} // extern "C"
