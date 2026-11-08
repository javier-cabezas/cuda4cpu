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
// HIP's AMD platform (built with __HIP_PLATFORM_AMD__ and __gfx1101__) and
// AMD's GPU builtins: the platform's macros and device properties, dispatch
// IDs, every lane operation with lanes that returned (which aren't in EXEC)
// and in branches (each call has its own EXEC),
// rocPRIM's DPP scan and reduction, a spin-wait with s_sleep, and the
// integer, floating-point and atomic builtins against references written
// apart from amdgcn.h; and the rest of what rocPRIM uses: hip_bfloat16,
// the unsigned long atomics, Clang's __hip_atomic_* (ours under GCC),
// HIP_vector_type, hipStreamGetDevice and __amdgcn_processor__.
//

#include <hip/device_functions.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

#if !defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__) || !defined(__AMDGCN__) || \
    !defined(__GFX11__) || !defined(__gfx1101__) || defined(__gfx1100__) || __AMDGCN_WAVEFRONT_SIZE__ != 32
#error "the AMD platform's macros"
#endif

static unsigned errors = 0;

#define EXPECT(cond)                                                       \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("%s:%d: expected %s\n", __FILE__, __LINE__, #cond); \
            ++errors;                                                      \
        }                                                                  \
    } while (0)

static_assert(__builtin_amdgcn_wavefrontsize() == 32);
static_assert(std::is_same_v<HIP_vector_type<float, 4>, float4> && std::is_same_v<HIP_vector_type<signed char, 2>, char2> &&
              std::is_same_v<HIP_vector_type<unsigned long, 1>, ulong1>);
static_assert(__builtin_strcmp(__amdgcn_processor__, "gfx1101") == 0);

constexpr int warp = 32;

//! A value for each lane, distinct
static int value_of(int lane)
{
    return lane * 7 + 100;
}

// Lane operations. In each warp, the lanes from `exited` return first, so
// that they aren't in EXEC.
struct lane_results {
    int readfirstlane, readlane, writelane;
    unsigned exec_lo, ballot, inverse, mbcnt_id, mbcnt_mask;
    unsigned long long exec, ballot64;
    int bpermute, permute, swizzle_swap, swizzle_quad, swizzle_15;
    int dpp[40];
    int dpp_masked, dpp_bound, dpp8, permlane16, permlanex16, permlane64, shuffle;
    unsigned reduce_add, reduce_sub;
    int reduce_min, reduce_max, reduce_and, reduce_or, reduce_xor;
    float reduce_fadd, reduce_fmax;
};

//! DPP controls that RDNA3 has, and those of earlier GPUs that it doesn't
//! (wave shifts and rotations, row broadcasts), which are emulated too
constexpr int dpp_controls[] = {0xb1,  0x4e,  0x1b,  0x101, 0x103, 0x10f, 0x111, 0x114, 0x11f, 0x121,
                                0x124, 0x12f, 0x130, 0x134, 0x138, 0x13c, 0x140, 0x141, 0x142, 0x143,
                                0x150, 0x155, 0x15f, 0x160, 0x163, 0x16f};
constexpr int dpp_count = sizeof(dpp_controls) / sizeof(dpp_controls[0]);

__global__ void lane_ops(lane_results *out, int exited)
{
    const int lane = int(__lane_id()), v = value_of(lane);
    if (lane >= exited && lane < exited + 3)
        return;
    lane_results r{};
    r.readfirstlane = __builtin_amdgcn_readfirstlane(v);
    r.readlane = __builtin_amdgcn_readlane(v, 9);
    r.writelane = __builtin_amdgcn_writelane(v, 5, -1);
    r.exec = __builtin_amdgcn_read_exec();
    r.exec_lo = __builtin_amdgcn_read_exec_lo();
    r.ballot = __builtin_amdgcn_ballot_w32(lane % 3 == 0);
    r.ballot64 = __builtin_amdgcn_ballot_w64(lane % 3 == 0);
    r.inverse = __builtin_amdgcn_inverse_ballot_w32(0x0000ffffu);
    r.mbcnt_id = __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
    r.mbcnt_mask = __builtin_amdgcn_mbcnt_hi(0x55555555u, __builtin_amdgcn_mbcnt_lo(0x55555555u, 10u));
    r.bpermute = __builtin_amdgcn_ds_bpermute(4 * ((lane + 5) % warp), v);
    r.permute = __builtin_amdgcn_ds_permute(4 * ((lane + 3) % warp), v);
    r.swizzle_swap = __builtin_amdgcn_ds_swizzle(v, 1 << 10 | 0x1f);      // xor 1: swap pairs
    r.swizzle_quad = __builtin_amdgcn_ds_swizzle(v, 0x8000 | 0x1b);       // quads reversed
    r.swizzle_15 = __builtin_amdgcn_ds_swizzle(v, 15 << 5);               // lane 15 of each 32
    for (int i = 0; i < dpp_count; ++i)
        r.dpp[i] = __builtin_amdgcn_mov_dpp(v, dpp_controls[i], 0xf, 0xf, true);
    // Rows 0 and banks 1 and 3 written; out of bounds keeps the old value
    r.dpp_masked = __builtin_amdgcn_update_dpp(-7, v, 0x111, 0x2, 0x5, true);
    r.dpp_bound = __builtin_amdgcn_update_dpp(-7, v, 0x111, 0xf, 0xf, false);
    r.dpp8 = __builtin_amdgcn_mov_dpp8(v, 0 | 1 << 3 | 0 << 6 | 1 << 9 | 7 << 12 | 6 << 15 | 5 << 18 | 4 << 21);
    r.permlane16 = __builtin_amdgcn_permlane16(-7, v, 0x89abcdefu, 0x01234567u, false, false);
    r.permlanex16 = __builtin_amdgcn_permlanex16(-7, v, 0x76543210u, 0xfedcba98u, false, true);
    r.permlane64 = __builtin_amdgcn_permlane64(v);
    r.shuffle = __builtin_amdgcn_wave_shuffle(v, lane + 33);
    r.reduce_add = __builtin_amdgcn_wave_reduce_add_u32(unsigned(v), 0);
    r.reduce_sub = __builtin_amdgcn_wave_reduce_sub_u32(unsigned(v), 0);
    r.reduce_min = __builtin_amdgcn_wave_reduce_min_i32(v - 300, 0);
    r.reduce_max = __builtin_amdgcn_wave_reduce_max_i32(v, 0);
    r.reduce_and = __builtin_amdgcn_wave_reduce_and_b32(v | 0x100000, 0);
    r.reduce_or = __builtin_amdgcn_wave_reduce_or_b32(1 << lane, 0);
    r.reduce_xor = __builtin_amdgcn_wave_reduce_xor_b32(v, 0);
    r.reduce_fadd = __builtin_amdgcn_wave_reduce_fadd_f32(float(lane), 0);
    r.reduce_fmax = __builtin_amdgcn_wave_reduce_fmax_f32(lane == 4 ? NAN : float(lane), 0);
    out[blockIdx.x * blockDim.x + threadIdx.x] = r;
}

//! The source lane of DPP control ctrl, from the ISA's tables, or -1
static int dpp_reference(int lane, int ctrl)
{
    const int row = lane / 16 * 16, in_row = lane % 16;
    if (ctrl <= 0xff)
        return lane / 4 * 4 + ((ctrl >> (lane % 4 * 2)) & 3);
    const int kind = ctrl >> 4, n = ctrl & 15;
    switch (kind) {
    case 0x10: return in_row + n <= 15 ? lane + n : -1;               // row_shl
    case 0x11: return in_row - n >= 0 ? lane - n : -1;                // row_shr
    case 0x12: return row + (in_row + 16 - n) % 16;                   // row_ror
    case 0x15: return row + n;                                        // row_share
    case 0x16: return row + (in_row ^ n);                             // row_xmask
    }
    switch (ctrl) {
    case 0x130: return lane == warp - 1 ? -1 : lane + 1;
    case 0x134: return (lane + 1) % warp;
    case 0x138: return lane == 0 ? -1 : lane - 1;
    case 0x13c: return (lane + warp - 1) % warp;
    case 0x140: return row + 15 - in_row;
    case 0x141: return lane / 8 * 8 + 7 - lane % 8;
    case 0x142: return lane < 16 ? -1 : 15;
    case 0x143: return -1;
    }
    return -2;
}

static void check_lanes(int exited)
{
    std::vector<lane_results> out(2 * warp);
    lane_ops<<<1, 2 * warp>>>(out.data(), exited);
    const auto active = [&](int lane) { return lane < exited || lane >= exited + 3; };
    unsigned wrong = 0;
    unsigned exec = 0;
    for (int l = 0; l < warp; ++l)
        exec |= active(l) ? 1u << l : 0;
    const int first = std::countr_zero(exec);
    for (int w = 0; w < 2; ++w)
        for (int lane = 0; lane < warp; ++lane) {
            if (!active(lane))
                continue;
            const lane_results &r = out[w * warp + lane];
            const auto value = [&](int l) { return active(l) ? value_of(l) : 0; };
            wrong += r.readfirstlane != value_of(first);
            wrong += r.readlane != value(9);
            wrong += r.writelane != (lane == 5 ? value_of(first) : -1);
            wrong += r.exec != exec || r.exec_lo != exec;
            unsigned ballot = 0;
            for (int l = 0; l < warp; ++l)
                ballot |= active(l) && l % 3 == 0 ? 1u << l : 0;
            wrong += r.ballot != ballot || r.ballot64 != ballot;
            wrong += r.inverse != unsigned(lane < 16);
            wrong += r.mbcnt_id != unsigned(lane);
            wrong += r.mbcnt_mask != 10u + unsigned(std::popcount(0x55555555u & ((1u << lane) - 1)));
            wrong += r.bpermute != value((lane + 5) % warp);
            wrong += r.permute != value((lane - 3 + warp) % warp);
            wrong += r.swizzle_swap != value(lane ^ 1);
            wrong += r.swizzle_quad != value(lane / 4 * 4 + 3 - lane % 4);
            wrong += r.swizzle_15 != value(15);
            for (int i = 0; i < dpp_count; ++i) {
                const int from = dpp_reference(lane, dpp_controls[i]);
                const int expected = from == -2 ? -9999 : from < 0 ? 0 : value(from);
                if (r.dpp[i] != expected) {
                    std::printf("dpp 0x%x, lane %d (exited %d): got %d, expected %d\n", dpp_controls[i], lane, exited,
                                r.dpp[i], expected);
                    ++wrong;
                }
            }
            // row_shr:1 with row mask 0x2 (row 1) and bank mask 0x5 (banks 0 and 2)
            const bool written = lane >= 16 && (lane / 4 % 4 == 0 || lane / 4 % 4 == 2);
            wrong += r.dpp_masked != (!written ? -7 : lane % 16 == 0 ? 0 : value(lane - 1));
            const bool fetched = lane % 16 != 0 && active(lane - 1);
            wrong += r.dpp_bound != (fetched ? value_of(lane - 1) : -7);
            const int sel8[] = {0, 1, 0, 1, 7, 6, 5, 4};
            wrong += r.dpp8 != value(lane / 8 * 8 + sel8[lane % 8]);
            // permlane16 reverses each row; permlanex16 reads the other row
            const int rev = lane / 16 * 16 + 15 - lane % 16;
            wrong += r.permlane16 != (active(rev) ? value_of(rev) : -7);
            const int other = (lane ^ 16);
            wrong += r.permlanex16 != (active(other) ? value_of(other) : 0);
            wrong += r.permlane64 != value_of(lane);
            wrong += r.shuffle != value((lane + 1) % warp);
            unsigned add = 0, sub = 0;
            int mn = INT32_MAX, mx = INT32_MIN, a = -1, o = 0, x = 0;
            float fadd = -0.0f, fmax = -INFINITY;
            for (int l = 0; l < warp; ++l)
                if (active(l)) {
                    add += unsigned(value_of(l));
                    sub -= unsigned(value_of(l));
                    mn = std::min(mn, value_of(l) - 300);
                    mx = std::max(mx, value_of(l));
                    a &= value_of(l) | 0x100000;
                    o |= 1 << l;
                    x ^= value_of(l);
                    fadd += float(l);
                    if (l != 4)
                        fmax = std::max(fmax, float(l));
                }
            wrong += r.reduce_add != add || r.reduce_sub != sub || r.reduce_min != mn || r.reduce_max != mx;
            wrong += r.reduce_and != a || r.reduce_or != o || r.reduce_xor != x;
            wrong += r.reduce_fadd != fadd || r.reduce_fmax != fmax;
        }
    if (wrong != 0) {
        std::printf("lane operations with lanes %d to %d exited: %u wrong\n", exited, exited + 2, wrong);
        ++errors;
    }
}

// rocPRIM's inclusive scan and reduction of a 32-lane warp with DPP, as in
// warp_scan_dpp.hpp and warp_reduce_dpp.hpp: row shifts, then lane 15 of
// each 32 by ds_swizzle on GPUs without row broadcasts; quad permutations
// and row rotations
__global__ void rocprim_dpp(int *scan, int *reduce)
{
    const int lane = int(__lane_id()), row_lane = lane % 16;
    int output = lane + 1;
    int t = __builtin_amdgcn_mov_dpp(output, 0x111, 0xf, 0xf, false) + output;
    if (row_lane >= 1) output = t;
    t = __builtin_amdgcn_mov_dpp(output, 0x112, 0xf, 0xf, false) + output;
    if (row_lane >= 2) output = t;
    t = __builtin_amdgcn_mov_dpp(output, 0x114, 0xf, 0xf, false) + output;
    if (row_lane >= 4) output = t;
    t = __builtin_amdgcn_mov_dpp(output, 0x118, 0xf, 0xf, false) + output;
    if (row_lane >= 8) output = t;
    t = __builtin_amdgcn_ds_swizzle(output, 0x1e0) + output;
    if (lane % 32 >= 16) output = t;
    scan[threadIdx.x] = output;

    int r = lane + 1;
    r += __builtin_amdgcn_mov_dpp(r, 0xb1, 0xf, 0xf, false);
    r += __builtin_amdgcn_mov_dpp(r, 0x4e, 0xf, 0xf, false);
    r += __builtin_amdgcn_mov_dpp(r, 0x124, 0xf, 0xf, false);
    r += __builtin_amdgcn_mov_dpp(r, 0x128, 0xf, 0xf, false);
    r += __builtin_amdgcn_ds_swizzle(r, 0x1e0);
    reduce[threadIdx.x] = __shfl(r, warp - 1, warp);
}

//! Lanes in two branches, and in a branch that some lanes skip: each call's
//! EXEC is the lanes that reach it, as a GPU runs each branch on its own
__global__ void branches(int *out)
{
    const int lane = int(__lane_id());
    if (lane < 16)
        out[lane] = __builtin_amdgcn_readfirstlane(lane + 100);
    else
        out[lane] = __builtin_amdgcn_readfirstlane(lane + 200);
    if (lane % 3 == 0) {
        // Warp-aggregated atomic: the leader adds for the branch's lanes
        const unsigned mask = __builtin_amdgcn_ballot_w32(true);
        const int leader = std::countr_zero(mask);
        if (lane == leader)
            atomicAdd(&out[64], std::popcount(mask));
        out[32 + lane] = __builtin_amdgcn_wave_reduce_add_u32(1u, 0) == unsigned(std::popcount(mask));
    } else {
        out[32 + lane] = __builtin_amdgcn_wave_reduce_add_u32(1u, 0) == 32u - 11u;
    }
}

//! Lanes that skip a branch wait at the next call for the branch's lanes, as
//! on a GPU, where the branch runs first and the lanes reconverge; and a
//! wave_barrier in one branch waits for that branch's lanes only
__global__ void reconverge(unsigned *out)
{
    const unsigned lane = __lane_id();
    if (lane < 16)
        out[lane] = __builtin_amdgcn_read_exec_lo();
    out[32 + lane] = __builtin_amdgcn_read_exec_lo();
    out[64 + lane] = __activemask();
    if (lane % 2 == 0)
        __builtin_amdgcn_wave_barrier();
    else
        out[96 + lane] = __builtin_amdgcn_read_exec_lo();
}

//! Thread 0 waits for a flag that thread 40 sets, sleeping: on one OS
//! thread, the wait must let thread 40 run
__global__ void sleep_wait(int *out)
{
    __shared__ volatile int flag;
    if (threadIdx.x == 0)
        flag = 0;
    __builtin_amdgcn_s_barrier();
    if (threadIdx.x == 40) {
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
        flag = 1;
    }
    if (threadIdx.x == 0) {
        while (flag == 0)
            __builtin_amdgcn_s_sleep(1);
        out[0] = 1;
    }
}

__global__ void ids(unsigned *out)
{
    const unsigned i = ((blockIdx.y * gridDim.x + blockIdx.x) * blockDim.z * blockDim.y * blockDim.x) +
                       (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    unsigned *o = out + 12 * i;
    o[0] = __builtin_amdgcn_workitem_id_x() == threadIdx.x && __builtin_amdgcn_workitem_id_y() == threadIdx.y &&
           __builtin_amdgcn_workitem_id_z() == threadIdx.z;
    o[1] = __builtin_amdgcn_workgroup_id_x() == blockIdx.x && __builtin_amdgcn_workgroup_id_y() == blockIdx.y &&
           __builtin_amdgcn_workgroup_id_z() == blockIdx.z;
    o[2] = __builtin_amdgcn_workgroup_size_x();
    o[3] = __builtin_amdgcn_workgroup_size_y();
    o[4] = __builtin_amdgcn_workgroup_size_z();
    o[5] = __builtin_amdgcn_grid_size_x();
    o[6] = __builtin_amdgcn_grid_size_y();
    o[7] = __builtin_amdgcn_grid_size_z();
    // HIP's old way of reading the block size: the dispatch packet
    const auto *packet = static_cast<const uint16_t *>(__builtin_amdgcn_dispatch_ptr());
    o[8] = packet[2];
    o[9] = packet[1];
    o[10] = static_cast<const uint32_t *>(__builtin_amdgcn_dispatch_ptr())[3];   // grid_size_x
    o[11] = 1;
}

//! AMD's overloads for unsigned long, called as rocPRIM calls them, and
//! Clang's atomics for HIP
__global__ void long_atomics(unsigned long *sum, unsigned long *mx, unsigned long *bits, int *loaded)
{
    const unsigned long v = threadIdx.x + 1;
    ::atomicAdd(sum, v);
    ::atomicSub(sum, 1ul);
    atomicMax(mx, v << 40);
    atomicOr(bits, 1ul << (threadIdx.x % 64));
    __hip_atomic_fetch_add(loaded, 2, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __hip_atomic_fetch_max(loaded + 1, int(threadIdx.x), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
    int expected = 0;
    if (__hip_atomic_compare_exchange_strong(loaded + 2, &expected, 7, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE,
                                             __HIP_MEMORY_SCOPE_SYSTEM))
        __hip_atomic_store(loaded + 3, int(threadIdx.x) + 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    __hip_assert(__hip_atomic_load(loaded + 2, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) == 7);
}

__global__ void counters(unsigned *inc, unsigned *dec, float *sum, double *mn)
{
    __builtin_amdgcn_atomic_inc32(inc, 9u, __ATOMIC_RELAXED, "agent");
    __builtin_amdgcn_atomic_dec32(dec, 9u, __ATOMIC_RELAXED, "agent");
    __builtin_amdgcn_global_atomic_fadd_f32(sum, 0.5f);
    __builtin_amdgcn_global_atomic_fmin_f64(mn, double(threadIdx.x + blockIdx.x * blockDim.x) - 3.0);
}

int main()
{
    // The emulated GPU, as programs see it
    {
        hipDeviceProp_t prop;
        int major = 0, warp_size = 0;
        EXPECT(hipGetDeviceProperties(&prop, 0) == hipSuccess);
        EXPECT(std::strcmp(prop.gcnArchName, "gfx1101") == 0 && prop.major == 11 && prop.minor == 0);
        EXPECT(hipDeviceGetAttribute(&major, hipDeviceAttributeComputeCapabilityMajor, 0) == hipSuccess && major == 11);
        EXPECT(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0) == hipSuccess && warp_size == 32);
    }

    for (int exited : {29, 0, 14})
        check_lanes(exited);

    {
        std::vector<int> scan(2 * warp), reduce(2 * warp);
        rocprim_dpp<<<1, 2 * warp>>>(scan.data(), reduce.data());
        unsigned wrong = 0;
        for (int t = 0; t < 2 * warp; ++t) {
            const int l = t % warp;
            wrong += scan[t] != (l + 1) * (l + 2) / 2 || reduce[t] != warp * (warp + 1) / 2;
        }
        EXPECT(wrong == 0);
    }

    {
        std::vector<int> out(65);
        branches<<<1, warp>>>(out.data());
        unsigned wrong = 0;
        for (int l = 0; l < warp; ++l)
            wrong += out[l] != (l < 16 ? 100 : 216) || out[32 + l] != 1;
        EXPECT(wrong == 0 && out[64] == 11);
    }

    {
        std::vector<unsigned> out(128);
        reconverge<<<1, warp>>>(out.data());
        unsigned wrong = 0;
        for (int l = 0; l < warp; ++l)
            wrong += (l < 16 && out[l] != 0xffffu) || out[32 + l] != ~0u || out[64 + l] != ~0u ||
                     (l % 2 == 1 && out[96 + l] != 0xaaaaaaaau);
        EXPECT(wrong == 0);
    }

    {
        int done = 0;
        sleep_wait<<<1, 64>>>(&done);
        EXPECT(done == 1);
    }

    {
        const dim3 grid(2, 3), block(4, 2, 2);
        std::vector<unsigned> out(12 * 6 * 16);
        ids<<<grid, block>>>(out.data());
        unsigned wrong = 0;
        for (size_t i = 0; i < out.size(); i += 12) {
            const unsigned *o = &out[i];
            wrong += o[0] != 1 || o[1] != 1 || o[2] != 4 || o[3] != 2 || o[4] != 2 || o[5] != 8 || o[6] != 6 ||
                     o[7] != 2 || o[8] != 4 || o[9] != 3 || o[10] != 8 || o[11] != 1;
        }
        EXPECT(wrong == 0);
    }

    {
        unsigned inc = 0, dec = 0;
        float sum = 0;
        double mn = 1e9;
        counters<<<4, 64>>>(&inc, &dec, &sum, &mn);
        // 256 increments wrapping after 9 (0..9: 10 values), and decrements
        // from 0 wrapping to 9
        EXPECT(inc == 256 % 10 && dec == (10 - 256 % 10) % 10 && sum == 128.0f && mn == -3.0);
    }

    {
        unsigned long sum = 0, mx = 0, bits = 0;
        int loaded[4] = {};
        long_atomics<<<2, 64>>>(&sum, &mx, &bits, loaded);
        EXPECT(sum == 2 * 64 * 65 / 2 - 128 && mx == 64ul << 40 && bits == ~0ul);
        EXPECT(loaded[0] == 256 && loaded[1] == 63 && loaded[2] == 7 && loaded[3] >= 1);
    }

    {
        hipDevice_t device = -1;
        EXPECT(hipStreamGetDevice(nullptr, &device) == hipSuccess && device == 0 && hipGetStreamDeviceId(nullptr) == 0);
    }

    // hip_bfloat16: to nearest even, NaNs kept NaNs, truncation, arithmetic
    {
        EXPECT(float(hip_bfloat16(1.0f)) == 1.0f && hip_bfloat16(1.0f).data == 0x3f80);
        EXPECT(hip_bfloat16(1.0f + 0x1p-8f).data == 0x3f80);                // a tie: to even
        EXPECT(hip_bfloat16(1.0f + 0x1.8p-8f).data == 0x3f81);
        EXPECT(hip_bfloat16(1.0f + 0x1.8p-8f, hip_bfloat16::truncate).data == 0x3f80);
        const float nan_low = std::bit_cast<float>(0x7f800001u);
        EXPECT(std::isnan(hip_bfloat16(nan_low)) && std::isnan(hip_bfloat16(nan_low, hip_bfloat16::truncate)));
        EXPECT(std::isinf(hip_bfloat16(std::numeric_limits<float>::max())) && std::iszero(hip_bfloat16(-0.0f)));
        hip_bfloat16 x(2.0f);
        x += hip_bfloat16(0.5f);
        ++x;
        EXPECT(float(x) == 3.5f && float(-x) == -3.5f && x > hip_bfloat16(3.0f) && x * x == hip_bfloat16(12.25f));
        x = 0.25f;
        EXPECT(float(hip_bfloat16::round_to_bfloat16(0.25f)) == float(x));
    }

    // Integer and bit operations
    EXPECT(__builtin_amdgcn_perm(0x01020304u, 0x05060708u, 0x07060504u) == 0x01020304u);
    EXPECT(__builtin_amdgcn_perm(0x01020304u, 0x05060708u, 0x00010203u) == 0x08070605u);
    EXPECT(__builtin_amdgcn_perm(0x80008000u, 0x00800080u, 0x0c0d0b08u) == 0x00ffff00u);
    EXPECT(__builtin_amdgcn_perm(0x8000ffffu, 0x7fff0000u, 0x0a09080bu) == 0xff0000ffu);
    EXPECT(__builtin_amdgcn_alignbit(0x12345678u, 0x9abcdef0u, 8) == 0x789abcdeu);
    EXPECT(__builtin_amdgcn_alignbit(0x12345678u, 0x9abcdef0u, 40) == 0x789abcdeu);
    EXPECT(__builtin_amdgcn_alignbyte(0x12345678u, 0x9abcdef0u, 3) == 0x3456789au);
    EXPECT(__builtin_amdgcn_ubfe(0xf0f0abcdu, 8, 8) == 0xabu && __builtin_amdgcn_ubfe(0xffffffffu, 4, 0) == 0);
    EXPECT(__builtin_amdgcn_ubfe(0x80000000u, 28, 8) == 0x8u);
    EXPECT(__builtin_amdgcn_sbfe(int(0x0000f800u), 8, 8) == -8 && __builtin_amdgcn_sbfe(0x0700, 8, 4) == 7);
    EXPECT(__builtin_amdgcn_sbfe(int(0x80000000u), 28, 8) == -8);
    EXPECT(__builtin_amdgcn_lerp(0x01020304u, 0x03040506u, 0x00010001u) == 0x02030405u);
    EXPECT(__builtin_amdgcn_lerp(0x00ff00ffu, 0x00ff0000u, 0x00000001u) == 0x00ff0080u);
    EXPECT(__builtin_amdgcn_sad_u8(0x0a141e28u, 0x14141414u, 5) == 5 + 10 + 0 + 10 + 20);
    EXPECT(__builtin_amdgcn_sad_hi_u8(0x0a141e28u, 0x14141414u, 5) == (40u << 16) + 5);
    EXPECT(__builtin_amdgcn_sad_u16(0x00010100u, 0x00030050u, 1) == 1 + 2 + 0xb0);
    {
        // Dot products against byte and nibble arithmetic
        const int a = int(0x80ff7f02u), b = int(0x7f01ff03u);
        const int sbytes = -128 * 127 + -1 * 1 + 127 * -1 + 2 * 3;
        const int ubytes = 128 * 127 + 255 * 1 + 127 * 255 + 2 * 3;
        EXPECT(__builtin_amdgcn_sdot4(a, b, 10, false) == sbytes + 10);
        EXPECT(__builtin_amdgcn_udot4(unsigned(a), unsigned(b), 10u, false) == unsigned(ubytes + 10));
        EXPECT(__builtin_amdgcn_sudot4(true, a, false, b, 10, false) == -128 * 127 + -1 * 1 + 127 * 255 + 2 * 3 + 10);
        EXPECT(__builtin_amdgcn_sdot4(int(0x80808080u), int(0x80808080u), INT32_MAX, true) == INT32_MAX);
        EXPECT(__builtin_amdgcn_sdot4(int(0x80808080u), int(0x80808080u), INT32_MAX, false) ==
               int(uint32_t(INT32_MAX) + 4u * 16384u));
        const int na = int(0x8f7012ffu), nb = int(0x11f02a3bu);
        int snib = 0, unib = 0;
        for (int k = 0; k < 8; ++k) {
            const int x = (unsigned(na) >> (4 * k)) & 15, y = (unsigned(nb) >> (4 * k)) & 15;
            snib += (x >= 8 ? x - 16 : x) * (y >= 8 ? y - 16 : y);
            unib += x * y;
        }
        EXPECT(__builtin_amdgcn_sdot8(na, nb, 0, false) == snib);
        EXPECT(__builtin_amdgcn_udot8(unsigned(na), unsigned(nb), 0u, false) == unsigned(unib));
        EXPECT(__builtin_amdgcn_udot4(0xffffffffu, 0xffffffffu, 0xffffffffu, true) == 0xffffffffu);
    }
    EXPECT(__builtin_amdgcn_s_bitreplicate(0x80000005u) == 0xc000000000000033ull);
    EXPECT(__builtin_amdgcn_cvt_pk_u8_f32(254.6f, 1, 0x11223344u) == 0x1122ff44u);
    EXPECT(__builtin_amdgcn_cvt_pk_u8_f32(-3.0f, 3, 0x11223344u) == 0x00223344u);
    EXPECT(__builtin_amdgcn_cvt_pk_u8_f32(2.5f, 0, 0) == 2u && __builtin_amdgcn_cvt_pk_u8_f32(NAN, 0, 0xffu) == 0u);

    // Floating point
    EXPECT(__builtin_amdgcn_rcpf(4.0f) == 0.25f && __builtin_amdgcn_rcp(-0.5) == -2.0);
    EXPECT(__builtin_amdgcn_rsqf(16.0f) == 0.25f && __builtin_amdgcn_sqrtf(2.25f) == 1.5f);
    EXPECT(float(__builtin_amdgcn_rcph(_Float16(8.0f))) == 0.125f);
    EXPECT(std::fabs(__builtin_amdgcn_sinf(0.25f) - 1.0f) < 1e-6f && std::fabs(__builtin_amdgcn_cosf(0.5f) + 1.0f) < 1e-6f);
    EXPECT(std::fabs(__builtin_amdgcn_sinf(1e6f + 0.125f) - std::sqrt(0.5f)) < 1e-4f);
    EXPECT(std::isnan(__builtin_amdgcn_sinf(INFINITY)));
    EXPECT(std::signbit(__builtin_amdgcn_sinf(-0.0f)) && __builtin_amdgcn_sinf(0.0f) == 0.0f);
    EXPECT(__builtin_amdgcn_exp2f(-3.0f) == 0.125f && __builtin_amdgcn_logf(32.0f) == 5.0f);
    EXPECT(__builtin_amdgcn_ldexpf(3.0f, 4) == 48.0f && __builtin_amdgcn_ldexp(1.0, -1074) > 0.0);
    EXPECT(__builtin_amdgcn_frexp_mantf(48.0f) == 0.75f && __builtin_amdgcn_frexp_expf(48.0f) == 6);
    EXPECT(std::isinf(__builtin_amdgcn_frexp_mantf(INFINITY)) && __builtin_amdgcn_frexp_expf(NAN) == 0);
    EXPECT(__builtin_amdgcn_frexp_exph(_Float16(0.125f)) == -2);
    EXPECT(__builtin_amdgcn_fractf(2.75f) == 0.75f && __builtin_amdgcn_fractf(-0.25f) == 0.75f);
    EXPECT(__builtin_amdgcn_fractf(-1e-10f) == std::nextafter(1.0f, 0.0f));
    EXPECT(std::isnan(__builtin_amdgcn_fractf(INFINITY)) && __builtin_amdgcn_fract(-2.5) == 0.5);
    {
        // One value of each class, from bit 0 to 9: converting a signaling
        // NaN makes it quiet, so each type has its own
        using fl = std::numeric_limits<float>;
        using dl = std::numeric_limits<double>;
        const float floats[] = {fl::signaling_NaN(), fl::quiet_NaN(), -INFINITY, -1.0f, -fl::denorm_min(), -0.0f,
                                0.0f, fl::denorm_min(), 1.0f, INFINITY};
        const double doubles[] = {dl::signaling_NaN(), dl::quiet_NaN(), -INFINITY, -1.0, -dl::denorm_min(), -0.0,
                                  0.0, dl::denorm_min(), 1.0, INFINITY};
        const uint16_t halves[] = {0x7d00, 0x7e00, 0xfc00, 0xbc00, 0x8001, 0x8000, 0x0000, 0x0001, 0x3c00, 0x7c00};
        unsigned wrong = 0;
        for (int i = 0; i < 10; ++i)
            for (int bit = 0; bit < 10; ++bit) {
                wrong += __builtin_amdgcn_classf(floats[i], 1 << bit) != (bit == i);
                wrong += __builtin_amdgcn_class(doubles[i], 1 << bit) != (bit == i);
                wrong += __builtin_amdgcn_classh(std::bit_cast<_Float16>(halves[i]), 1 << bit) != (bit == i);
            }
        EXPECT(wrong == 0);
    }
    EXPECT(__builtin_amdgcn_fmed3f(3.0f, 1.0f, 2.0f) == 2.0f && __builtin_amdgcn_fmed3f(-1.0f, 5.0f, 9.0f) == 5.0f);
    EXPECT(__builtin_amdgcn_fmed3f(NAN, 4.0f, 2.0f) == 2.0f && float(__builtin_amdgcn_fmed3h(_Float16(1), _Float16(3), _Float16(2))) == 2.0f);

    if (errors == 0)
        std::printf("hip_amdgcn: PASS\n");
    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
