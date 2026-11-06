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
// Warp matrix functions (mma.h): every type, shape and layout of A and B, C
// and D in memory, against a host reference. The inputs are small integers,
// so that the products and sums are exact and the results must be equal;
// tf32 inputs carry low bits that the tensor cores ignore. Then fragments
// changed element by element (half's A, whose lanes hold elements twice),
// saturation, __float_to_tf32's rounding, CUDA's elements per lane, and the
// compute capability that CUDA4CPU_COMPUTE_CAPABILITY sets.
//

#include <mma.h>

#include <algorithm>
#include <bit>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

using namespace nvcuda;

static unsigned errors = 0;

#define EXPECT(cond)                                                       \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("%s:%d: expected %s\n", __FILE__, __LINE__, #cond); \
            ++errors;                                                      \
        }                                                                  \
    } while (0)

// CUDA's elements per lane, and element types
static_assert(wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>::num_elements == 16);
static_assert(wmma::fragment<wmma::matrix_b, 32, 8, 16, half, wmma::col_major>::num_elements == 16);
static_assert(wmma::fragment<wmma::accumulator, 16, 16, 16, float>::num_elements == 8);
static_assert(wmma::fragment<wmma::accumulator, 16, 16, 16, half>::num_elements == 8);
static_assert(wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major>::num_elements == 8);
static_assert(wmma::fragment<wmma::matrix_a, 32, 8, 16, unsigned char, wmma::row_major>::num_elements == 16);
static_assert(wmma::fragment<wmma::matrix_b, 32, 8, 16, unsigned char, wmma::col_major>::num_elements == 4);
static_assert(wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major>::num_elements == 4);
static_assert(wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major>::num_elements == 1);
static_assert(wmma::fragment<wmma::accumulator, 8, 8, 4, double>::num_elements == 2);
static_assert(std::is_same_v<wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major>::element_type,
                             wmma::precision::tf32>);
static_assert(std::is_same_v<wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32,
                                            wmma::row_major>::storage_element_type,
                             float>);

//! What memory holds for an element type
template <typename T>
using storage = std::conditional_t<std::is_same_v<T, wmma::precision::tf32>, float, T>;

template <typename T>
static long double value_of(T x)
{
    if constexpr (std::is_same_v<T, half>)
        return __half2float(x);
    else if constexpr (std::is_same_v<T, __nv_bfloat16>)
        return __bfloat162float(x);
    else
        return (long double)x;
}

//! Element (row, col) of a matrix whose leading dimension is ld
template <typename P>
static P &at(P *p, bool row_major, unsigned ld, int row, int col)
{
    return row_major ? p[size_t(row) * ld + size_t(col)] : p[size_t(col) * ld + size_t(row)];
}

//! Each warp of the block computes a tile of D = 2 (A B + C), over k_tiles
//! tiles of A and B. Each block has its own matrices, sizes[] apart.
template <typename T, int M, int N, int K, typename LA, typename LB, typename TC, typename TD>
__global__ void tiled(const storage<T> *a, unsigned lda, const storage<T> *b, unsigned ldb, const TC *c, unsigned ldc,
                      wmma::layout_t c_layout, TD *d, unsigned ldd, wmma::layout_t d_layout, int tiles_n, int k_tiles,
                      const size_t *sizes)
{
    constexpr bool a_rows = std::is_same_v<LA, wmma::row_major>, b_rows = std::is_same_v<LB, wmma::row_major>;
    const int warp = int(threadIdx.x / 32), tile_m = warp / tiles_n, tile_n = warp % tiles_n;
    a += blockIdx.x * sizes[0];
    b += blockIdx.x * sizes[1];
    c += blockIdx.x * sizes[2];
    d += blockIdx.x * sizes[3];

    wmma::fragment<wmma::accumulator, M, N, K, TC> acc;
    wmma::load_matrix_sync(acc, &at(c, c_layout == wmma::mem_row_major, ldc, tile_m * M, tile_n * N), ldc, c_layout);
    wmma::fragment<wmma::accumulator, M, N, K, TD> out;
    for (int t = 0; t < k_tiles; ++t) {
        wmma::fragment<wmma::matrix_a, M, N, K, T, LA> fa;
        wmma::fragment<wmma::matrix_b, M, N, K, T, LB> fb;
        wmma::load_matrix_sync(fa, &at(a, a_rows, lda, tile_m * M, t * K), lda);
        wmma::load_matrix_sync(fb, &at(b, b_rows, ldb, t * K, tile_n * N), ldb);
        if (t == 0)
            wmma::mma_sync(out, fa, fb, acc);
        else
            wmma::mma_sync(out, fa, fb, out);
    }
    for (int i = 0; i < out.num_elements; ++i)
        out.x[i] = out.x[i] + out.x[i];
    wmma::store_matrix_sync(&at(d, d_layout == wmma::mem_row_major, ldd, tile_m * M, tile_n * N), out, ldd, d_layout);
}

//! Small integers, which every type holds exactly, and tf32 inputs with low
//! bits that mma_sync must ignore
template <typename T>
static storage<T> input(std::mt19937 &random)
{
    if constexpr (std::is_same_v<T, unsigned char>)
        return storage<T>(std::uniform_int_distribution<int>(0, 255)(random));
    else if constexpr (std::is_same_v<T, signed char>)
        return storage<T>(std::uniform_int_distribution<int>(-128, 127)(random));
    else {
        const int n = std::uniform_int_distribution<int>(-4, 4)(random);
        if constexpr (std::is_same_v<T, wmma::precision::tf32>)
            return n == 0 ? 0.0f : float(n) + std::copysign(0x1p-12f, float(n));
        else
            return storage<T>(float(n));
    }
}

template <typename T>
static long double exact_input(storage<T> x)
{
    if constexpr (std::is_same_v<T, wmma::precision::tf32>)
        return std::trunc(x);
    else
        return value_of(x);
}

//! D = 2 (A B + C) for every memory layout of C and D, with tiles x tiles
//! warps per block, in each of blocks blocks
template <typename T, int M, int N, int K, typename LA, typename LB, typename TC, typename TD, int tiles = 2,
          int blocks = 1>
static void check(const char *name, std::mt19937 &random)
{
    constexpr int tiles_m = tiles, tiles_n = tiles, k_tiles = 2;
    constexpr int rows = tiles_m * M, cols = tiles_n * N, depth = k_tiles * K;
    constexpr bool a_rows = std::is_same_v<LA, wmma::row_major>, b_rows = std::is_same_v<LB, wmma::row_major>;
    // Every combination of the memory layouts of C and D, with leading
    // dimensions larger than the matrices
    for (int layouts = 0; layouts < 4; ++layouts) {
        const bool c_rows = layouts & 1, d_rows = layouts & 2;
        const unsigned lda = (a_rows ? depth : rows) + 8, ldb = (b_rows ? cols : depth) + 8;
        const unsigned ldc = (c_rows ? cols : rows) + 4, ldd = (d_rows ? cols : rows) + 4;
        const size_t sizes[4] = {size_t(lda) * (a_rows ? rows : depth), size_t(ldb) * (b_rows ? depth : cols),
                                 size_t(ldc) * (c_rows ? rows : cols), size_t(ldd) * (d_rows ? rows : cols)};
        std::vector<storage<T>> a(sizes[0] * blocks), b(sizes[1] * blocks);
        std::vector<TC> c(sizes[2] * blocks);
        std::vector<TD> d(sizes[3] * blocks, TD(0.0f));
        for (auto &x : a)
            x = input<T>(random);
        for (auto &x : b)
            x = input<T>(random);
        for (auto &x : c) {
            if constexpr (std::is_same_v<TC, int>)
                x = std::uniform_int_distribution<int>(-1000, 1000)(random);
            else
                x = TC(float(std::uniform_int_distribution<int>(-4, 4)(random)));
        }

        const auto c_layout = c_rows ? wmma::mem_row_major : wmma::mem_col_major;
        const auto d_layout = d_rows ? wmma::mem_row_major : wmma::mem_col_major;
        tiled<T, M, N, K, LA, LB, TC, TD><<<blocks, 32 * tiles_m * tiles_n>>>(
            a.data(), lda, b.data(), ldb, c.data(), ldc, c_layout, d.data(), ldd, d_layout, tiles_n, k_tiles, sizes);
        unsigned wrong = 0;
        for (int block = 0; block < blocks; ++block) {
            const storage<T> *pa = a.data() + block * sizes[0], *pb = b.data() + block * sizes[1];
            const TC *pc = c.data() + block * sizes[2];
            const TD *pd = d.data() + block * sizes[3];
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j) {
                    long double sum = value_of(at(pc, c_rows, ldc, i, j));
                    for (int k = 0; k < depth; ++k)
                        sum += exact_input<T>(at(pa, a_rows, lda, i, k)) * exact_input<T>(at(pb, b_rows, ldb, k, j));
                    wrong += value_of(at(pd, d_rows, ldd, i, j)) != 2 * sum;
                }
        }
        if (wrong != 0) {
            std::printf("%s, A %s, B %s, C %s, D %s: %u wrong elements\n", name, a_rows ? "row" : "col",
                        b_rows ? "row" : "col", c_rows ? "row" : "col", d_rows ? "row" : "col", wrong);
            ++errors;
        }
    }
}

//! Every layout of A and B
template <typename T, int M, int N, int K, typename TC, typename TD = TC>
static void check_layouts(const char *name, std::mt19937 &random)
{
    check<T, M, N, K, wmma::row_major, wmma::row_major, TC, TD>(name, random);
    check<T, M, N, K, wmma::row_major, wmma::col_major, TC, TD>(name, random);
    check<T, M, N, K, wmma::col_major, wmma::row_major, TC, TD>(name, random);
    check<T, M, N, K, wmma::col_major, wmma::col_major, TC, TD>(name, random);
}

//! D = (2 A)(3 B) + 1, with A and B changed element by element: every lane's
//! copy of an element must change the same way. A and B are packed (ld is
//! their rows' or columns' length), D is row-major.
template <int M, int N, int K, typename LA, typename LB>
__global__ void scaled(const half *a, const half *b, float *d)
{
    wmma::fragment<wmma::matrix_a, M, N, K, half, LA> fa;
    wmma::fragment<wmma::matrix_b, M, N, K, half, LB> fb;
    wmma::fragment<wmma::accumulator, M, N, K, float> acc;
    wmma::load_matrix_sync(fa, a, std::is_same_v<LA, wmma::row_major> ? K : M);
    wmma::load_matrix_sync(fb, b, std::is_same_v<LB, wmma::row_major> ? N : K);
    for (int i = 0; i < fa.num_elements; ++i)
        fa.x[i] = fa.x[i] * half(2.0f);
    for (int i = 0; i < fb.num_elements; ++i)
        fb.x[i] = fb.x[i] * half(3.0f);
    wmma::fill_fragment(acc, 1.0f);
    wmma::mma_sync(acc, fa, fb, acc);
    wmma::store_matrix_sync(d, acc, N, wmma::mem_row_major);
}

template <int M, int N, int K, typename LA, typename LB>
static void check_scaled(const char *name)
{
    constexpr bool a_rows = std::is_same_v<LA, wmma::row_major>, b_rows = std::is_same_v<LB, wmma::row_major>;
    std::vector<half> a(M * K), b(K * N);
    for (int i = 0; i < M * K; ++i)
        a[i] = half(float(i % 7 - 3));
    for (int i = 0; i < K * N; ++i)
        b[i] = half(float(i % 5 - 2));
    std::vector<float> d(M * N);
    scaled<M, N, K, LA, LB><<<1, 32>>>(a.data(), b.data(), d.data());
    unsigned wrong = 0;
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 1;
            for (int k = 0; k < K; ++k)
                sum += 6 * float(at(a.data(), a_rows, a_rows ? K : M, i, k)) * float(at(b.data(), b_rows, b_rows ? N : K, k, j));
            wrong += d[i * N + j] != sum;
        }
    if (wrong != 0) {
        std::printf("%s changed element by element: %u wrong elements\n", name, wrong);
        ++errors;
    }
}

//! Each lane reads, right after store_matrix_sync, an element that another
//! lane stored: on a GPU the warp's stores are one instruction
__global__ void store_then_read(float *out)
{
    __shared__ float tile[16 * 16];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, float(threadIdx.x / 32 + 1));
    wmma::store_matrix_sync(tile, acc, 16, wmma::mem_row_major);
    out[threadIdx.x] = tile[(threadIdx.x * 37 + 101) % 256];
}

//! One element of A and B each, so that D's first element is a * b + c:
//! with satf, and without
template <typename T, typename TC>
__global__ void single_product(storage<T> a, storage<T> b, TC c, bool satf, TC *d)
{
    constexpr int M = std::is_same_v<T, double> ? 8 : 16, N = M;
    constexpr int K = std::is_same_v<T, double> ? 4 : std::is_same_v<T, wmma::precision::tf32> ? 8 : 16;
    wmma::fragment<wmma::matrix_a, M, N, K, T, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, M, N, K, T, wmma::row_major> fb;
    wmma::fragment<wmma::accumulator, M, N, K, TC> acc;
    wmma::fill_fragment(fa, storage<T>(0));
    wmma::fill_fragment(fb, storage<T>(0));
    // Lane 0 holds A(0, 0) and B(0, 0), and for half lane 16 too. Elsewhere
    // lane 16 holds elements of row or column 0 that D(0, 0) doesn't use.
    if (threadIdx.x % 16 == 0) {
        fa.x[0] = a;
        fb.x[0] = b;
    }
    wmma::fill_fragment(acc, c);
    wmma::mma_sync(acc, fa, fb, acc, satf);
    wmma::store_matrix_sync(d, acc, N, wmma::mem_row_major);
}

template <typename T, typename TC>
static TC first_element(storage<T> a, storage<T> b, TC c, bool satf)
{
    std::vector<TC> d(16 * 16);
    single_product<T, TC><<<1, 32>>>(a, b, c, satf, d.data());
    return d[0];
}

int main()
{
    // The compute capability that CUDA4CPU_COMPUTE_CAPABILITY sets, as programs
    // check before using tensor cores; 6.0 without it, or with a wrong value
    {
        const char *setting = std::getenv("CUDA4CPU_COMPUTE_CAPABILITY");
        const bool raised = setting != nullptr && std::string(setting) == "7.5";
        cudaDeviceProp prop;
        int major = 0, minor = 0;
        EXPECT(cudaGetDeviceProperties(&prop, 0) == cudaSuccess);
        EXPECT(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0) == cudaSuccess);
        EXPECT(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0) == cudaSuccess);
        EXPECT(prop.major == (raised ? 7 : 6) && prop.minor == (raised ? 5 : 0));
        EXPECT(major == prop.major && minor == prop.minor);
    }

    std::mt19937 random(12345);

    check_layouts<half, 16, 16, 16, float>("half m16n16k16 float", random);
    check_layouts<half, 16, 16, 16, half>("half m16n16k16 half", random);
    check_layouts<half, 16, 16, 16, half, float>("half m16n16k16 half into float", random);
    check_layouts<half, 16, 16, 16, float, half>("half m16n16k16 float into half", random);
    check_layouts<half, 32, 8, 16, float>("half m32n8k16", random);
    check_layouts<half, 8, 32, 16, half>("half m8n32k16", random);
    check_layouts<__nv_bfloat16, 16, 16, 16, float>("bf16 m16n16k16", random);
    check_layouts<__nv_bfloat16, 32, 8, 16, float>("bf16 m32n8k16", random);
    check_layouts<__nv_bfloat16, 8, 32, 16, float>("bf16 m8n32k16", random);
    check_layouts<wmma::precision::tf32, 16, 16, 8, float>("tf32 m16n16k8", random);
    check_layouts<double, 8, 8, 4, double>("double m8n8k4", random);
    check_layouts<signed char, 16, 16, 16, int>("s8 m16n16k16", random);
    check_layouts<unsigned char, 32, 8, 16, int>("u8 m32n8k16", random);
    check_layouts<signed char, 8, 32, 16, int>("s8 m8n32k16", random);
    check_layouts<unsigned char, 16, 16, 16, int>("u8 m16n16k16", random);

    // 16 warps per block, in several blocks on several OS threads
    check<half, 16, 16, 16, wmma::row_major, wmma::col_major, float, float, 4, 8>("half, 16 warps in 8 blocks", random);
    check<signed char, 32, 8, 16, wmma::col_major, wmma::row_major, int, int, 4, 8>("s8, 16 warps in 8 blocks", random);

    // A and B changed element by element, in each shape and layout
    check_scaled<16, 16, 16, wmma::row_major, wmma::col_major>("half m16n16k16");
    check_scaled<32, 8, 16, wmma::col_major, wmma::row_major>("half m32n8k16");
    check_scaled<8, 32, 16, wmma::col_major, wmma::col_major>("half m8n32k16");

    // What store_matrix_sync stored, for the whole warp
    {
        std::vector<float> out(64);
        store_then_read<<<1, 32>>>(out.data());
        EXPECT(std::count(out.begin(), out.begin() + 32, 1.0f) == 32);
    }

    // Saturation: infinities become the largest finite value, NaN zero, and
    // integers stop at their limits instead of wrapping around
    {
        const half big = half(256.0f);
        EXPECT((float(first_element<half, half>(big, big, half(0.0f), false)) == INFINITY));
        EXPECT((float(first_element<half, half>(big, big, half(0.0f), true)) == 65504.0f));
        EXPECT((first_element<half, float>(big, big, -INFINITY, true) == std::numeric_limits<float>::lowest()));
        EXPECT((first_element<half, float>(big, big, -INFINITY, false) == -INFINITY));
        EXPECT((first_element<half, float>(big, big, NAN, true) == 0.0f));
        EXPECT((std::isnan(first_element<half, float>(big, big, NAN, false))));
        EXPECT((first_element<double, double>(1e300, 1e300, 0.0, true) == std::numeric_limits<double>::max()));
        EXPECT((first_element<signed char, int>(100, 100, INT_MAX - 5, false) == INT_MIN + 9994));
        EXPECT((first_element<signed char, int>(100, 100, INT_MAX - 5, true) == INT_MAX));
        EXPECT((first_element<signed char, int>(-128, 127, INT_MIN + 5, true) == INT_MIN));
        EXPECT((first_element<unsigned char, int>(255, 255, 0, false) == 65025));
    }

    // The conversion that mma_sync uses for half, against __half2float, for
    // every half
    {
        unsigned wrong = 0;
        for (uint32_t bits = 0; bits < 0x10000; ++bits) {
            const half h = __half_raw{uint16_t(bits)};
            const float fast = cuda4cpu::detail::wmma::half_to_float(h), reference = __half2float(h);
            wrong += std::isnan(reference) ? !std::isnan(fast)
                                           : std::bit_cast<uint32_t>(fast) != std::bit_cast<uint32_t>(reference);
        }
        EXPECT(wrong == 0);
    }

    // tf32: rounding to nearest with ties away from zero, and the low bits
    // that mma_sync ignores
    {
        const auto bits = [](float x) { return std::bit_cast<uint32_t>(x); };
        EXPECT(wmma::__float_to_tf32(1.0f + 0x1p-11f) == 1.0f + 0x1p-10f);
        EXPECT(wmma::__float_to_tf32(-(1.0f + 0x1p-11f)) == -(1.0f + 0x1p-10f));
        EXPECT(wmma::__float_to_tf32(1.0f + 0x1p-12f) == 1.0f);
        EXPECT(wmma::__float_to_tf32(1.0f + 0x1.8p-11f) == 1.0f + 0x1p-10f);
        EXPECT(std::isinf(wmma::__float_to_tf32(std::numeric_limits<float>::max())));
        EXPECT(std::isinf(wmma::__float_to_tf32(-INFINITY)) && std::isnan(wmma::__float_to_tf32(NAN)));
        EXPECT((bits(wmma::__float_to_tf32(3.14159265f)) & 0x1fffu) == 0);
        const float nan_with_low_bits = std::bit_cast<float>(0x7f800001u);
        EXPECT(std::isnan(wmma::__float_to_tf32(nan_with_low_bits)) &&
               (bits(wmma::__float_to_tf32(nan_with_low_bits)) & 0x1fffu) == 0);
        EXPECT((first_element<wmma::precision::tf32, float>(1.0f + 0x1p-11f, 3.0f, 0.0f, false)) == 3.0f);
    }

    if (errors == 0)
        std::printf("wmma: PASS\n");
    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
