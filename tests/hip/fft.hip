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

// hipFFT and rocFFT, through their own names, against a direct DFT: hipFFT's
// plans, transforms and callbacks (from a __device__ variable, as programs
// get them), and rocFFT's descriptions (layouts with lengths and strides
// fastest dimension first, offsets, scale factors), placements, execution
// infos (streams, callbacks) and errors.

#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>
#include <hipfft/hipfftXt.h>
#include <rocfft/rocfft.h>

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <numbers>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char *what, int line)
{
    if (!ok) {
        ++failures;
        std::printf("line %d: %s\n", line, what);
    }
}

#define CHECK(cond) check((cond), #cond, __LINE__)
#define HIPFFT(call) check((call) == HIPFFT_SUCCESS, #call, __LINE__)
#define ROCFFT(call) check((call) == rocfft_status_success, #call, __LINE__)

using cd = std::complex<double>;

//! The 1D DFT, exp(sign 2 pi i k n / N)
std::vector<cd> dft(const std::vector<cd> &x, int sign)
{
    const std::size_t n = x.size();
    std::vector<cd> out(n);
    for (std::size_t k = 0; k < n; ++k)
        for (std::size_t j = 0; j < n; ++j)
            out[k] += x[j] * std::polar(1.0, sign * 2 * std::numbers::pi * double(k * j % n) / double(n));
    return out;
}

bool near(cd a, cd b)
{
    return std::abs(a - b) < 1e-9 * (1 + std::abs(b));
}

__device__ hipfftDoubleComplex load_negated(void *in, size_t offset, void *, void *)
{
    const hipfftDoubleComplex e = static_cast<hipfftDoubleComplex *>(in)[offset];
    return make_hipDoubleComplex(-e.x, -e.y);
}

__device__ auto load_pointer = load_negated;

void hipfft()
{
    // Z2Z, forward and back
    const int n = 12;
    std::vector<hipfftDoubleComplex> data(n);
    std::vector<cd> x(n);
    for (int i = 0; i < n; ++i) {
        data[i] = make_hipDoubleComplex(i % 5 - 2.0, i * 0.25);
        x[i] = {data[i].x, data[i].y};
    }
    hipfftHandle plan;
    HIPFFT(hipfftPlan1d(&plan, n, HIPFFT_Z2Z, 1));
    HIPFFT(hipfftExecZ2Z(plan, data.data(), data.data(), HIPFFT_FORWARD));
    const auto want = dft(x, -1);
    bool ok = true;
    for (int i = 0; i < n; ++i)
        ok &= near({data[i].x, data[i].y}, want[i]);
    CHECK(ok);
    HIPFFT(hipfftExecZ2Z(plan, data.data(), data.data(), HIPFFT_BACKWARD));
    CHECK(near({data[5].x, data[5].y}, double(n) * x[5]));

    // A load callback, from a __device__ variable
    void *callback = nullptr;
    CHECK(hipMemcpyFromSymbol(&callback, HIP_SYMBOL(load_pointer), sizeof(void *)) == hipSuccess);
    HIPFFT(hipfftXtSetCallback(plan, &callback, HIPFFT_CB_LD_COMPLEX_DOUBLE, nullptr));
    std::vector<hipfftDoubleComplex> in(n), out(n);
    for (int i = 0; i < n; ++i)
        in[i] = make_hipDoubleComplex(x[i].real(), x[i].imag());
    HIPFFT(hipfftExecZ2Z(plan, in.data(), out.data(), HIPFFT_FORWARD));
    CHECK(near({out[3].x, out[3].y}, -want[3]));
    HIPFFT(hipfftDestroy(plan));

    // R2C through a plan made in steps
    hipfftHandle r2c;
    HIPFFT(hipfftCreate(&r2c));
    size_t work = 1;
    HIPFFT(hipfftMakePlan1d(r2c, 8, HIPFFT_R2C, 1, &work));
    CHECK(work == 0);
    float real[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    hipfftComplex spectrum[5];
    HIPFFT(hipfftExecR2C(r2c, real, spectrum));
    CHECK(spectrum[0].x == 8 && std::abs(spectrum[1].x) < 1e-6f && std::abs(spectrum[4].y) < 1e-6f);
    CHECK(hipfftExecC2C(r2c, spectrum, spectrum, HIPFFT_FORWARD) == HIPFFT_INVALID_TYPE);
    HIPFFT(hipfftDestroy(r2c));

    int version = 0;
    HIPFFT(hipfftGetVersion(&version));
    CHECK(version == HIPFFT_VERSION);
    CHECK(hipfftPlan1d(&plan, -1, HIPFFT_C2C, 1) == HIPFFT_INVALID_SIZE);
}

double2 load_scaled(void *in, size_t offset, void *data, void *)
{
    const double2 e = static_cast<double2 *>(in)[offset];
    const double s = *static_cast<double *>(data);
    return {s * e.x, s * e.y};
}

void rocfft()
{
    ROCFFT(rocfft_setup());

    // A 2D complex transform, 3 x 4 (lengths fastest first: {4, 3}), not in
    // place, its input with a row stride of 5 and an offset of 2
    const size_t lengths[2] = {4, 3};
    const size_t in_strides[2] = {1, 5}, in_offset[1] = {2};
    std::vector<double2> in(2 + 15), out(12);
    std::vector<cd> x(12);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c) {
            in[2 + r * 5 + c] = {r + 0.5 * c, r - c * 0.25};
            x[r * 4 + c] = {in[2 + r * 5 + c].x, in[2 + r * 5 + c].y};
        }
    rocfft_plan_description description;
    ROCFFT(rocfft_plan_description_create(&description));
    ROCFFT(rocfft_plan_description_set_data_layout(description, rocfft_array_type_complex_interleaved,
                                                   rocfft_array_type_complex_interleaved, in_offset, nullptr, 2,
                                                   in_strides, 15, 0, nullptr, 0));
    ROCFFT(rocfft_plan_description_set_scale_factor(description, 0.5));
    rocfft_plan plan;
    ROCFFT(rocfft_plan_create(&plan, rocfft_placement_notinplace, rocfft_transform_type_complex_forward,
                              rocfft_precision_double, 2, lengths, 1, description));
    size_t work = 1;
    ROCFFT(rocfft_plan_get_work_buffer_size(plan, &work));
    CHECK(work == 0);
    void *ins[1] = {in.data()}, *outs[1] = {out.data()};
    ROCFFT(rocfft_execute(plan, ins, outs, nullptr));
    // The 2D DFT: rows, then columns
    bool ok = true;
    for (int k0 = 0; k0 < 3; ++k0)
        for (int k1 = 0; k1 < 4; ++k1) {
            cd sum = 0;
            for (int j0 = 0; j0 < 3; ++j0)
                for (int j1 = 0; j1 < 4; ++j1)
                    sum += x[j0 * 4 + j1] *
                           std::polar(1.0, -2 * std::numbers::pi * (double(k0 * j0) / 3 + double(k1 * j1) / 4));
            ok &= near({out[k0 * 4 + k1].x, out[k0 * 4 + k1].y}, 0.5 * sum);
        }
    CHECK(ok);
    ROCFFT(rocfft_plan_destroy(plan));
    ROCFFT(rocfft_plan_description_destroy(description));

    // Real forward in place (padded: 2 (n / 2 + 1) reals), then inverse, with
    // a stream and a load callback in the execution info
    const size_t n[1] = {10};
    std::vector<double> real(12, 0.0);
    for (int i = 0; i < 10; ++i)
        real[i] = std::sin(0.3 * i) + 1;
    const std::vector<double> signal(real.begin(), real.begin() + 10);
    rocfft_plan forward, inverse;
    ROCFFT(rocfft_plan_create(&forward, rocfft_placement_inplace, rocfft_transform_type_real_forward,
                              rocfft_precision_double, 1, n, 1, nullptr));
    ROCFFT(rocfft_plan_create(&inverse, rocfft_placement_notinplace, rocfft_transform_type_real_inverse,
                              rocfft_precision_double, 1, n, 1, nullptr));
    hipStream_t stream;
    CHECK(hipStreamCreate(&stream) == hipSuccess);
    rocfft_execution_info info;
    ROCFFT(rocfft_execution_info_create(&info));
    ROCFFT(rocfft_execution_info_set_stream(info, stream));
    void *buffers[1] = {real.data()};
    ROCFFT(rocfft_execute(forward, buffers, nullptr, info));
    double sum = 0;
    for (double v : signal)
        sum += v;
    CHECK(std::abs(real[0] - sum) < 1e-9 && std::abs(real[1]) < 1e-12);
    // Back, with the spectrum scaled by a callback: 2 * 10 times the signal
    double scale = 2;
    void *callback = reinterpret_cast<void *>(&load_scaled), *data = &scale;
    ROCFFT(rocfft_execution_info_set_load_callback(info, &callback, &data, 0));
    std::vector<double> back(10);
    void *outs2[1] = {back.data()};
    ROCFFT(rocfft_execute(inverse, buffers, outs2, info));
    CHECK(std::abs(back[7] - 20 * signal[7]) < 1e-9);
    ROCFFT(rocfft_execution_info_destroy(info));
    ROCFFT(rocfft_plan_destroy(forward));
    ROCFFT(rocfft_plan_destroy(inverse));
    CHECK(hipStreamDestroy(stream) == hipSuccess);

    // Errors: dimensions, array types, strides
    const size_t four[4] = {2, 2, 2, 2};
    CHECK(rocfft_plan_create(&plan, rocfft_placement_inplace, rocfft_transform_type_complex_forward,
                             rocfft_precision_single, 4, four, 1, nullptr) == rocfft_status_invalid_dimensions);
    ROCFFT(rocfft_plan_description_create(&description));
    ROCFFT(rocfft_plan_description_set_data_layout(description, rocfft_array_type_complex_planar,
                                                   rocfft_array_type_complex_planar, nullptr, nullptr, 0, nullptr, 0, 0,
                                                   nullptr, 0));
    CHECK(rocfft_plan_create(&plan, rocfft_placement_inplace, rocfft_transform_type_complex_forward,
                             rocfft_precision_single, 1, n, 1, description) == rocfft_status_invalid_array_type);
    ROCFFT(rocfft_plan_description_destroy(description));
    char version[32];
    ROCFFT(rocfft_get_version_string(version, sizeof(version)));
    CHECK(std::strncmp(version, "1.", 2) == 0);
    ROCFFT(rocfft_cleanup());
}

} // namespace

int main()
{
    hipfft();
    rocfft();
    if (failures != 0) {
        std::printf("hip_fft: %d failures\n", failures);
        return 1;
    }
    std::printf("hip_fft: all passed\n");
    return 0;
}
