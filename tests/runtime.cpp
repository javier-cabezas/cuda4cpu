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


//
// Runtime API semantics: errors, launch configuration limits, memory
// management, symbols and intrinsics. Uses only cuda_runtime.h, like CUDA code.
//

#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <thread>
#include <vector>

#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

static unsigned errors = 0;

#define EXPECT(cond)                                                       \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("%s:%d: expected %s\n", __FILE__, __LINE__, #cond); \
            ++errors;                                                      \
        }                                                                  \
    } while (0)

__global__ void set_flag(int *flag)
{
    *flag = 1;
}

__constant__ float table[4];
__device__ int counter;

__global__ void read_table(float *out)
{
    out[threadIdx.x] = table[threadIdx.x];
    if (threadIdx.x == 0)
        counter += 1;
}

__global__ void intrinsics(float *out)
{
    out[0] = __expf(0.f);
    out[1] = __fdividef(1.f, 4.f);
    out[2] = __saturatef(2.f);
    out[3] = float(__popc(0xf0f0u));
}

static bool launch_rejected(dim3 grid, dim3 block)
{
    int flag = 0;
    cuda4cpu::launch(set_flag, grid, block).call(&flag);
    return flag == 0 && cudaGetLastError() == cudaErrorInvalidConfiguration;
}

//! f() computed in the rounding mode that fesetround sets: the reference for
//! the intrinsics with an explicit rounding mode
template <typename T, typename F>
static T with_rounding(int mode, F f)
{
    const int previous = std::fegetround();
    std::fesetround(mode);
    volatile T result = f();
    std::fesetround(previous);
    return result;
}

template <typename T>
static bool same_bits(T a, T b)
{
    return (std::isnan(a) && std::isnan(b)) || std::memcmp(&a, &b, sizeof(T)) == 0;
}

//! Checks every intrinsic with an explicit rounding mode against the same
//! operation under fesetround, on random and special operands, and returns
//! the number of results that differ
static unsigned rounding_mismatches()
{
    std::mt19937_64 random(2026);
    // Random bit patterns (NaN, infinity, subnormals and every exponent), and
    // values whose operations are usually inexact
    auto any_float = [&]() -> float {
        const uint64_t r = random();
        if (r % 3 == 0)
            return std::bit_cast<float>(uint32_t(r >> 32));
        return std::ldexp(float(int32_t(r >> 32)) / 2147483648.0f * 4.0f, int(r % 61) - 30);
    };
    auto any_double = [&]() -> double {
        const uint64_t r = random();
        if (r % 3 == 0)
            return std::bit_cast<double>(random());
        return std::ldexp(double(int64_t(random())) / 9223372036854775808.0 * 4.0, int(r % 61) - 30);
    };
    auto any_integer = [&] { const uint64_t r = random(); return r >> (r % 64); };

    const int modes[4] = {FE_TONEAREST, FE_TOWARDZERO, FE_UPWARD, FE_DOWNWARD};
    const char *mode_names[4] = {"rn", "rz", "ru", "rd"};
    unsigned mismatches = 0;
    auto report = [&](const char *name, int m, double x, double y, double z, double got, double want) {
        if (mismatches++ == 0)
            std::printf("runtime: %s_%s(%a, %a, %a) = %a, expected %a\n", name, mode_names[m], x, y, z, got, want);
    };

    // fns are the _rn, _rz, _ru and _rd variants; op computes the same in the current mode
    auto check1 = [&](const char *name, auto gen, auto op, auto... fns) {
        for (int i = 0; i < 20000; ++i) {
            const auto x = gen();
            using T = decltype(op(x));
            const T got[4] = {fns(x)...};
            for (int m = 0; m < 4; ++m) {
                volatile auto vx = x;
                const T want = with_rounding<T>(modes[m], [&] { return op(vx); });
                if (!same_bits(got[m], want))
                    report(name, m, double(x), 0, 0, double(got[m]), double(want));
            }
        }
    };
    auto check2 = [&](const char *name, auto gen, auto op, auto... fns) {
        for (int i = 0; i < 20000; ++i) {
            const auto x = gen(), y = gen();
            using T = decltype(op(x, y));
            const T got[4] = {fns(x, y)...};
            for (int m = 0; m < 4; ++m) {
                volatile auto vx = x, vy = y;
                const T want = with_rounding<T>(modes[m], [&] { return op(vx, vy); });
                if (!same_bits(got[m], want))
                    report(name, m, double(x), double(y), 0, double(got[m]), double(want));
            }
        }
    };
    auto check3 = [&](const char *name, auto gen, auto op, auto... fns) {
        for (int i = 0; i < 20000; ++i) {
            const auto x = gen(), y = gen(), z = gen();
            using T = decltype(op(x, y, z));
            const T got[4] = {fns(x, y, z)...};
            for (int m = 0; m < 4; ++m) {
                volatile auto vx = x, vy = y, vz = z;
                const T want = with_rounding<T>(modes[m], [&] { return op(vx, vy, vz); });
                if (!same_bits(got[m], want))
                    report(name, m, double(x), double(y), double(z), double(got[m]), double(want));
            }
        }
    };

#define VARIANTS(name) name##_rn, name##_rz, name##_ru, name##_rd
    check2("__fadd", any_float, [](float a, float b) { return a + b; }, VARIANTS(__fadd));
    check2("__fsub", any_float, [](float a, float b) { return a - b; }, VARIANTS(__fsub));
    check2("__fmul", any_float, [](float a, float b) { return a * b; }, VARIANTS(__fmul));
    check2("__fdiv", any_float, [](float a, float b) { return a / b; }, VARIANTS(__fdiv));
    check1("__frcp", any_float, [](float a) { return 1.0f / a; }, VARIANTS(__frcp));
    check1("__fsqrt", any_float, [](float a) { return std::sqrt(a); }, VARIANTS(__fsqrt));
    check3("__fmaf", any_float, [](float a, float b, float c) { return std::fma(a, b, c); }, VARIANTS(__fmaf));
    check2("__dadd", any_double, [](double a, double b) { return a + b; }, VARIANTS(__dadd));
    check2("__dsub", any_double, [](double a, double b) { return a - b; }, VARIANTS(__dsub));
    check2("__dmul", any_double, [](double a, double b) { return a * b; }, VARIANTS(__dmul));
    check2("__ddiv", any_double, [](double a, double b) { return a / b; }, VARIANTS(__ddiv));
    check1("__drcp", any_double, [](double a) { return 1.0 / a; }, VARIANTS(__drcp));
    check1("__dsqrt", any_double, [](double a) { return std::sqrt(a); }, VARIANTS(__dsqrt));
    check3("__fma", any_double, [](double a, double b, double c) { return std::fma(a, b, c); }, VARIANTS(__fma));
    check1("__int2float", [&] { return int(any_integer()); }, [](int a) { return float(a); }, VARIANTS(__int2float));
    check1("__uint2float", [&] { return unsigned(any_integer()); }, [](unsigned a) { return float(a); },
           VARIANTS(__uint2float));
    check1("__ll2float", [&] { return (long long)any_integer(); }, [](long long a) { return float(a); },
           VARIANTS(__ll2float));
    check1("__ull2float", any_integer, [](uint64_t a) { return float(a); }, VARIANTS(__ull2float));
    check1("__ll2double", [&] { return (long long)any_integer(); }, [](long long a) { return double(a); },
           VARIANTS(__ll2double));
    check1("__ull2double", any_integer, [](uint64_t a) { return double(a); }, VARIANTS(__ull2double));
    check1("__double2float", any_double, [](double a) { return float(a); }, VARIANTS(__double2float));
#undef VARIANTS
    return mismatches;
}

int main()
{
    // Last error: recorded by failing calls, peeked, then cleared
    EXPECT(cudaGetLastError() == cudaSuccess);
    EXPECT(cudaSetDevice(1) == cudaErrorInvalidDevice);
    EXPECT(cudaPeekAtLastError() == cudaErrorInvalidDevice);
    EXPECT(cudaGetLastError() == cudaErrorInvalidDevice);
    EXPECT(cudaGetLastError() == cudaSuccess);

    EXPECT(std::strcmp(cudaGetErrorName(cudaErrorInvalidValue), "cudaErrorInvalidValue") == 0);
    EXPECT(std::strcmp(cudaGetErrorString(cudaSuccess), "no error") == 0);
    EXPECT(std::strcmp(cudaGetErrorString(cudaErrorMemoryAllocation), "out of memory") == 0);
    EXPECT(std::strcmp(cudaGetErrorString(cudaError_t(12345)), "unrecognized error code") == 0);

    // Launch configurations outside CUDA's limits are rejected without running
    EXPECT(launch_rejected(1, 2048));
    EXPECT(launch_rejected(1, dim3(32, 64)));     // 2048 threads in total
    EXPECT(launch_rejected(1, dim3(1, 1, 65)));
    EXPECT(launch_rejected(dim3(1, 70000), 32));
    EXPECT(launch_rejected(0, 32));
    EXPECT(launch_rejected(1, 0));
    {
        int flag = 0;
        cuda4cpu::launch(set_flag, dim3(1, 65535), dim3(8, 8, 16)).call(&flag);
        EXPECT(flag == 1 && cudaGetLastError() == cudaSuccess);
    }

    // Allocation: 256-byte alignment, failures reported, size 0 allowed
    for (size_t size : {1, 100, 4096, 1000000}) {
        void *p = nullptr;
        EXPECT(cudaMalloc(&p, size) == cudaSuccess);
        EXPECT(reinterpret_cast<uintptr_t>(p) % 256 == 0);
        EXPECT(cudaFree(p) == cudaSuccess);
    }
    {
        void *p = reinterpret_cast<void *>(0x1);
        EXPECT(cudaMalloc(&p, size_t(1) << 60) == cudaErrorMemoryAllocation);
        EXPECT(p == reinterpret_cast<void *>(0x1));
        EXPECT(cudaGetLastError() == cudaErrorMemoryAllocation);
        // Rounding up to the alignment must not wrap around to a small size
        EXPECT(cudaMalloc(&p, std::numeric_limits<size_t>::max()) == cudaErrorMemoryAllocation);
        EXPECT(p == reinterpret_cast<void *>(0x1));
        cudaGetLastError();
        EXPECT(cudaMalloc(&p, 0) == cudaSuccess && p == nullptr);
    }

    // Typed overloads, managed and host memory
    {
        float *d = nullptr, *m = nullptr, *h = nullptr;
        EXPECT(cudaMalloc(&d, 16 * sizeof(float)) == cudaSuccess && d != nullptr);
        EXPECT(cudaMallocManaged(&m, 16 * sizeof(float)) == cudaSuccess && m != nullptr);
        EXPECT(cudaMallocHost(&h, 16 * sizeof(float)) == cudaSuccess && h != nullptr);
        cudaFreeHost(h);
        EXPECT(cudaHostAlloc(&h, 16 * sizeof(float), cudaHostAllocMapped) == cudaSuccess && h != nullptr);
        cudaFreeHost(h);
        cudaFree(d);
        cudaFree(m);
    }

    // Copies and memset
    {
        int src[4] = {1, 2, 3, 4}, dst[4] = {};
        EXPECT(cudaMemcpy(dst, src, sizeof(src), cudaMemcpyHostToHost) == cudaSuccess);
        EXPECT(std::memcmp(dst, src, sizeof(src)) == 0);
        EXPECT(cudaMemcpy(dst, src, sizeof(src), cudaMemcpyKind(7)) == cudaErrorInvalidMemcpyDirection);
        EXPECT(cudaGetLastError() == cudaErrorInvalidMemcpyDirection);
        EXPECT(cudaMemcpyAsync(dst, src, sizeof(src), cudaMemcpyDefault) == cudaSuccess);
        EXPECT(cudaMemset(dst, 0xff, sizeof(dst)) == cudaSuccess && dst[3] == -1);
        EXPECT(cudaMemsetAsync(dst, 0, sizeof(dst)) == cudaSuccess && dst[0] == 0);
    }

    // Large copies and memsets, which several OS threads share: unaligned,
    // overlapping (as memmove), and of pitched rows
    {
        const size_t n = size_t(8) << 20, pitch = 1024, width = 1000, height = n / pitch;
        unsigned char *a = nullptr, *b = nullptr;
        EXPECT(cudaMalloc(&a, n + 64) == cudaSuccess && cudaMalloc(&b, n + 64) == cudaSuccess);
        for (size_t i = 0; i < n + 64; ++i)
            a[i] = (unsigned char)(i * 7 + i / 4093);
        std::vector<unsigned char> ref(a, a + n + 64);

        b[0] = 1;
        EXPECT(cudaMemcpy(b + 3, a + 5, n, cudaMemcpyDefault) == cudaSuccess);
        EXPECT(b[0] == 1 && std::memcmp(b + 3, a + 5, n) == 0);

        std::memmove(ref.data() + 1, ref.data(), n);
        EXPECT(cudaMemcpy(a + 1, a, n, cudaMemcpyDefault) == cudaSuccess);
        EXPECT(std::memcmp(a, ref.data(), n + 64) == 0);

        const unsigned char after = b[n + 1];
        EXPECT(cudaMemset(b + 1, 0x5a, n) == cudaSuccess);
        EXPECT(b[0] == 1 && b[1] == 0x5a && b[n] == 0x5a && b[n + 1] == after);

        EXPECT(cudaMemcpy2D(b, pitch, a, pitch, width, height, cudaMemcpyDefault) == cudaSuccess);
        bool rows = true;
        for (size_t y = 0; y < height; ++y)
            rows = rows && std::memcmp(b + y * pitch, a + y * pitch, width) == 0 && b[y * pitch + width] == 0x5a;
        EXPECT(rows);

        EXPECT(cudaMemset2D(b, pitch, 0, width, height) == cudaSuccess);
        rows = true;
        for (size_t y = 0; y < height; ++y)
            rows = rows && b[y * pitch] == 0 && b[y * pitch + width - 1] == 0 && b[y * pitch + width] == 0x5a;
        EXPECT(rows);
        cudaFree(a);
        cudaFree(b);
    }

    // Symbols are passed by name
    {
        const float values[4] = {0.5f, 1.5f, 2.5f, 3.5f};
        EXPECT(cudaMemcpyToSymbol(table, values, sizeof(values)) == cudaSuccess);
        const float x = 9.f;
        EXPECT(cudaMemcpyToSymbol(table, &x, sizeof(float), 2 * sizeof(float)) == cudaSuccess);
        EXPECT(cudaMemcpyToSymbol(table, &x, sizeof(float), 4 * sizeof(float)) == cudaErrorInvalidValue);
        cudaGetLastError();

        const int zero = 0;
        EXPECT(cudaMemcpyToSymbol(counter, &zero, sizeof(int)) == cudaSuccess);

        float out[4] = {};
        cuda4cpu::launch(read_table, 1, 4).call(out);
        EXPECT(out[0] == 0.5f && out[1] == 1.5f && out[2] == 9.f && out[3] == 3.5f);

        int count = -1;
        EXPECT(cudaMemcpyFromSymbol(&count, counter, sizeof(int)) == cudaSuccess && count == 1);

        void *address = nullptr;
        size_t size = 0;
        EXPECT(cudaGetSymbolAddress(&address, table) == cudaSuccess && address == static_cast<void *>(table));
        EXPECT(cudaGetSymbolSize(&size, table) == cudaSuccess && size == sizeof(table));
    }

    // Events and streams with CUDA's default arguments
    {
        cudaEvent_t start, stop;
        cudaStream_t stream;
        EXPECT(cudaEventCreate(&start) == cudaSuccess && cudaEventCreate(&stop) == cudaSuccess);
        EXPECT(cudaStreamCreate(&stream) == cudaSuccess);
        EXPECT(cudaEventRecord(start) == cudaSuccess);
        EXPECT(cudaEventRecord(stop, stream) == cudaSuccess);
        EXPECT(cudaStreamWaitEvent(stream, stop) == cudaSuccess);
        EXPECT(cudaEventQuery(stop) == cudaSuccess);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaStreamDestroy(stream);
    }

    // Device queries
    {
        int count = 0, device = -1, value = 0;
        EXPECT(cudaGetDeviceCount(&count) == cudaSuccess && count == 1);
        EXPECT(cudaGetDevice(&device) == cudaSuccess && device == 0);
        EXPECT(cudaDeviceGetAttribute(&value, cudaDevAttrWarpSize, 0) == cudaSuccess && value == 32);
        EXPECT(cudaDeviceGetAttribute(&value, cudaDeviceAttr(9999), 0) == cudaErrorInvalidValue);
        cudaGetLastError();
        cudaDeviceProp prop;
        EXPECT(cudaGetDeviceProperties(&prop, 1) == cudaErrorInvalidDevice);
        cudaGetLastError();
        EXPECT(cudaGetDeviceProperties(&prop, 0) == cudaSuccess && prop.major == 6);
    }

    // Intrinsics, on the host and in a kernel
    {
        EXPECT(__popc(0xf0f0u) == 8 && __popcll(~0ull) == 64);
        EXPECT(__clz(1) == 31 && __clz(0) == 32 && __clzll(1) == 63);
        EXPECT(__ffs(0) == 0 && __ffs(8) == 4 && __ffsll(1ll << 40) == 41);
        EXPECT(__brev(1u) == 0x80000000u && __brev(0x12345678u) == 0x1e6a2c48u);
        EXPECT(__brevll(1ull) == 1ull << 63);
        EXPECT(__saturatef(-1.f) == 0.f && __saturatef(0.25f) == 0.25f && __saturatef(NAN) == 0.f);
        float s = 0.f, c = 0.f;
        __sincosf(0.f, &s, &c);
        EXPECT(s == 0.f && c == 1.f);
        EXPECT(__logf(1.f) == 0.f && __powf(2.f, 3.f) == 8.f && __exp10f(2.f) == 100.f);

        int ints[3] = {7, 8, 9};
        EXPECT(__ldg(ints + 1) == 8);

        float out[4] = {};
        cuda4cpu::launch(intrinsics, 1, 1).call(out);
        EXPECT(out[0] == 1.f && out[1] == 0.25f && out[2] == 1.f && out[3] == 8.f);
    }

    // Directed rounding and conversions, exact as in CUDA
    {
        const float one = 1.0f, tiny = 1e-8f;
        // Valgrind ignores the SSE rounding mode, so arithmetic always rounds to nearest under it.
        volatile float vone = one, vtiny = tiny;
        const bool honors_rounding_mode = with_rounding<float>(FE_UPWARD, [&] { return vone + vtiny; }) > 1.0f;
        if (honors_rounding_mode) {
            EXPECT(__fadd_rd(one, tiny) == 1.0f && __fadd_ru(one, tiny) > 1.0f && __fadd_rn(one, tiny) == 1.0f);
            EXPECT(__fadd_rz(-one, -tiny) == -1.0f && __fadd_rd(-one, -tiny) < -1.0f);
            EXPECT(__fdiv_rd(1.0f, 3.0f) < __fdiv_ru(1.0f, 3.0f));
            EXPECT(__dmul_rd(0.1, 3.0) < __dmul_ru(0.1, 3.0));
            EXPECT(__fsqrt_rd(2.0f) < __fsqrt_ru(2.0f));
            EXPECT(rounding_mismatches() == 0);
        } else {
            std::printf("runtime: the FPU ignores the rounding mode here (Valgrind?); skipping directed rounding checks\n");
        }
        EXPECT(__fadd_rn(one, tiny) == 1.0f && __dsqrt_rn(4.0) == 2.0);
        EXPECT(std::fegetround() == FE_TONEAREST);
        EXPECT(__float2int_rn(2.5f) == 2 && __float2int_rn(3.5f) == 4 && __float2int_rz(-2.7f) == -2);
        EXPECT(__float2int_ru(2.1f) == 3 && __float2int_rd(-2.1f) == -3);
        EXPECT(__float2int_rn(NAN) == 0 && __float2int_rn(1e20f) == std::numeric_limits<int>::max());
        EXPECT(__float2uint_rn(-5.0f) == 0u && __double2ll_rz(-1e30) == std::numeric_limits<long long>::min());
        EXPECT(__int2float_rd(16777217) == 16777216.0f && __int2float_ru(16777217) == 16777218.0f);
        EXPECT(__double2float_rd(0.1) < __double2float_ru(0.1));
        EXPECT(__int_as_float(0x3f800000) == 1.0f && __float_as_uint(-0.0f) == 0x80000000u);
        EXPECT(__longlong_as_double(__double_as_longlong(3.25)) == 3.25);
        EXPECT(__hiloint2double(__double2hiint(-7.5), __double2loint(-7.5)) == -7.5);
        EXPECT(__mul24(-3, 5) == -15 && __umul24(0x1000003u, 2u) == 6u);
        EXPECT(max(3, 7) == 7 && min(-1, 2u) == 2u && max(1.5f, 2.5) == 2.5 && min(4ll, -2ll) == -2ll);
    }

    // Scoped atomics behave like the plain ones
    {
        int counter = 0;
        atomicAdd_system(&counter, 2);
        atomicAdd_block(&counter, 3);
        EXPECT(counter == 5 && atomicMax_system(&counter, 9) == 5 && counter == 9);
    }

    // Mapped and registered host memory, stream-ordered allocation and pools
    {
        EXPECT(cudaSetDeviceFlags(cudaDeviceMapHost) == cudaSuccess);
        unsigned int flags = 0;
        EXPECT(cudaGetDeviceFlags(&flags) == cudaSuccess && flags == cudaDeviceMapHost);

        float *host = nullptr, *device = nullptr;
        EXPECT(cudaHostAlloc(&host, 64, cudaHostAllocMapped) == cudaSuccess);
        EXPECT(cudaHostGetDevicePointer(&device, host, 0) == cudaSuccess && device == host);
        cudaFreeHost(host);

        float local[4];
        EXPECT(cudaHostRegister(local, sizeof(local), cudaHostRegisterMapped) == cudaSuccess);
        EXPECT(cudaHostUnregister(local) == cudaSuccess);

        cudaStream_t stream;
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        void *p = nullptr;
        EXPECT(cudaMallocAsync(&p, 1000, stream) == cudaSuccess && p != nullptr);
        EXPECT(cudaFreeAsync(p, stream) == cudaSuccess);
        cudaMemPool_t pool = nullptr;
        unsigned long long threshold = ~0ull;
        EXPECT(cudaDeviceGetDefaultMemPool(&pool, 0) == cudaSuccess && pool != nullptr);
        EXPECT(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold) == cudaSuccess);
        int value = 0;
        EXPECT(cudaDeviceGetAttribute(&value, cudaDevAttrMemoryPoolsSupported, 0) == cudaSuccess && value == 1);

        cudaMemPoolProps props{};
        props.allocType = cudaMemAllocationTypePinned;
        props.location = {cudaMemLocationTypeDevice, 0};
        cudaMemPool_t created = nullptr;
        EXPECT(cudaMemPoolCreate(&created, &props) == cudaSuccess && created != nullptr && created != pool);
        EXPECT(cudaMallocFromPoolAsync(&p, 1000, created, stream) == cudaSuccess && p != nullptr);
        EXPECT(cudaFreeAsync(p, stream) == cudaSuccess);
        EXPECT(cudaMemPoolDestroy(created) == cudaSuccess);
        EXPECT(cudaMemPoolDestroy(pool) == cudaErrorInvalidValue);    // the default pool can't be destroyed
        cudaGetLastError();

        bool called = false;
        EXPECT(cudaLaunchHostFunc(stream, [](void *flag) { *static_cast<bool *>(flag) = true; }, &called) == cudaSuccess);
        EXPECT(called);
        cudaStreamDestroy(stream);
    }

    // The host's memory is the device's
    {
        size_t free = 0, total = 0;
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        EXPECT(cudaMemGetInfo(&free, &total) == cudaSuccess);
        EXPECT(total == prop.totalGlobalMem && free > 0 && free <= total);
        EXPECT(cudaMemGetInfo(nullptr, &total) == cudaErrorInvalidValue);
        cudaGetLastError();
    }

    // cudaStreamDefault is a null pointer constant, as in CUDA, so code passes
    // it where a stream goes. cudaStreamPerThread is each thread's own stream.
    {
        int flag = 0;
        cudaEvent_t event;
        cudaEventCreate(&event);
        EXPECT(cudaEventRecord(event, cudaStreamDefault) == cudaSuccess);
        EXPECT(cudaMemsetAsync(&flag, 0, sizeof(flag), cudaStreamDefault) == cudaSuccess);
        cuda4cpu::launch(set_flag, 1, 1, 0, cudaStreamDefault).call(&flag);
        EXPECT(flag == 1);
        EXPECT(cudaStreamSynchronize(cudaStreamLegacy) == cudaSuccess);
        cudaEventDestroy(event);

        cudaStream_t other = nullptr;
        std::thread([&] { other = cudaStreamPerThread; }).join();
        EXPECT(cudaStreamPerThread != nullptr && cudaStreamPerThread == cudaStreamPerThread);
        EXPECT(other != nullptr && other != cudaStreamPerThread);

        // Work issued to it is captured like any stream's
        cudaGraph_t graph = nullptr;
        EXPECT(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeGlobal) == cudaSuccess);
        flag = 0;
        cuda4cpu::launch(set_flag, 1, 1, 0, cudaStreamPerThread).call(&flag);
        EXPECT(cudaStreamEndCapture(cudaStreamPerThread, &graph) == cudaSuccess && flag == 0);
        size_t nodes = 0;
        EXPECT(cudaGraphGetNodes(graph, nullptr, &nodes) == cudaSuccess && nodes == 1);
        cudaGraphExec_t exec = nullptr;
        EXPECT(cudaGraphInstantiate(&exec, graph) == cudaSuccess);
        EXPECT(cudaGraphLaunch(exec, cudaStreamPerThread) == cudaSuccess && flag == 1);
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);

        EXPECT(cudaStreamDestroy(cudaStreamPerThread) == cudaErrorInvalidResourceHandle);
        cudaGetLastError();
    }

    // Device queries and configuration that have no effect on a CPU
    {
        int least = -1, greatest = -1, can_access = -1, overlap = -1;
        EXPECT(cudaDeviceGetStreamPriorityRange(&least, &greatest) == cudaSuccess && least == 0 && greatest == 0);
        EXPECT(cudaDeviceCanAccessPeer(&can_access, 0, 0) == cudaSuccess && can_access == 0);
        EXPECT(cudaDeviceGetAttribute(&overlap, cudaDevAttrGpuOverlap, 0) == cudaSuccess && overlap == 0);
        EXPECT(cudaFuncSetCacheConfig(set_flag, cudaFuncCachePreferL1) == cudaSuccess);
        EXPECT(cudaFuncSetAttribute(set_flag, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024) == cudaSuccess);
        cudaFuncAttributes attributes;
        EXPECT(cudaFuncGetAttributes(&attributes, set_flag) == cudaSuccess && attributes.maxThreadsPerBlock == 1024);
        EXPECT(cudaProfilerStart() == cudaSuccess && cudaProfilerStop() == cudaSuccess);
        EXPECT(CUDA_VERSION == CUDART_VERSION);
        EXPECT(std::strcmp(cudaGetErrorName(cudaErrorAssert), "cudaErrorAssert") == 0);
    }

    if (errors == 0)
        std::printf("runtime: PASS\n");
    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
