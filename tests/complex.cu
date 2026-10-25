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
// Complex numbers (cuComplex.h), checked against long double arithmetic on
// random operands: additions and subtractions are exact in each precision,
// products, quotients, fused multiply-adds and absolute values within a few
// units in the last place. Then the cases CUDA's formulas are for (absolute
// values whose squares overflow or underflow, divisions by numbers whose
// squared magnitude overflows), the special values, and a kernel.
//

#include <cuComplex.h>

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

static unsigned errors = 0;

#define EXPECT(cond)                                                       \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("%s:%d: expected %s\n", __FILE__, __LINE__, #cond); \
            ++errors;                                                      \
        }                                                                  \
    } while (0)

static_assert(std::is_same_v<cuFloatComplex, float2> && std::is_same_v<cuDoubleComplex, double2>);
static_assert(std::is_same_v<cuComplex, cuFloatComplex> && sizeof(cuDoubleComplex) == 16);

//! Whether got is within units ulps of T's precision of the exact value,
//! relative to scale (the magnitude of the terms that were rounded)
template <typename T>
static bool close(T got, long double exact, long double scale, int units)
{
    const long double ulp = scale * std::numeric_limits<T>::epsilon();
    return std::fabs((long double)got - exact) <= units * ulp;
}

//! The checks for float or double: C is the complex type
template <typename T, typename C, typename Make, typename Add, typename Sub, typename Mul, typename Div, typename Abs,
          typename Fma>
static unsigned check_random(std::mt19937_64 &random, Make make, Add add, Sub sub, Mul mul, Div div, Abs abs, Fma fma)
{
    std::uniform_real_distribution<T> uniform(-1, 1);
    std::uniform_int_distribution<int> exponent(-20, 20);
    auto any = [&] { return std::ldexp(uniform(random), exponent(random)); };
    unsigned wrong = 0;
    for (int i = 0; i < 200000; ++i) {
        const C x = make(any(), any()), y = make(any(), any()), d = make(any(), any());
        using L = std::complex<long double>;
        const L lx(x.x, x.y), ly(y.x, y.y), ld(d.x, d.y);

        const C s = add(x, y), t = sub(x, y);
        wrong += s.x != T(x.x + y.x) || s.y != T(x.y + y.y) || t.x != T(x.x - y.x) || t.y != T(x.y - y.y);

        // Two products rounded, and their sum: within 2 ulps of the larger
        // product's magnitude
        const C p = mul(x, y);
        const L lp = lx * ly;
        const long double pscale = std::fabs(lx.real() * ly.real()) + std::fabs(lx.imag() * ly.imag()) +
                                   std::fabs(lx.real() * ly.imag()) + std::fabs(lx.imag() * ly.real());
        wrong += !close(p.x, lp.real(), pscale, 2) || !close(p.y, lp.imag(), pscale, 2);

        const C f = fma(x, y, d);
        const L lf = lx * ly + ld;
        wrong += !close(f.x, lf.real(), pscale + std::abs(ld), 3) || !close(f.y, lf.imag(), pscale + std::abs(ld), 3);

        // A quotient, after scaling: within a few ulps of |x / y|
        const C q = div(x, y);
        const L lq = lx / ly;
        wrong += !close(q.x, lq.real(), std::abs(lq), 6) || !close(q.y, lq.imag(), std::abs(lq), 6);

        const long double labs = std::hypot((long double)x.x, (long double)x.y);
        wrong += !close(abs(x), labs, labs, 2);
    }
    return wrong;
}

__global__ void dft(const float *input, cuFloatComplex *output, int n)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n)
        return;
    cuFloatComplex sum = make_cuFloatComplex(0.f, 0.f);
    for (int j = 0; j < n; ++j) {
        const float angle = -2.f * float(M_PI) * float(k * j % n) / float(n);
        sum = cuCfmaf(make_cuFloatComplex(input[j], 0.f), make_cuFloatComplex(cosf(angle), sinf(angle)), sum);
    }
    output[k] = sum;
}

__global__ void normalize(cuDoubleComplex *z, double *magnitude)
{
    const int i = threadIdx.x;
    magnitude[i] = cuCabs(z[i]);
    z[i] = cuCdiv(z[i], make_cuDoubleComplex(magnitude[i], 0));
}

int main()
{
    std::mt19937_64 random(20261025);

    EXPECT((check_random<float, cuFloatComplex>(random, make_cuFloatComplex, cuCaddf, cuCsubf, cuCmulf, cuCdivf,
                                                cuCabsf, cuCfmaf)) == 0);
    EXPECT((check_random<double, cuDoubleComplex>(random, make_cuDoubleComplex, cuCadd, cuCsub, cuCmul, cuCdiv,
                                                  cuCabs, cuCfma)) == 0);

    // Parts, conjugates and conversions
    {
        const cuComplex z = make_cuComplex(1.5f, -2.f);
        EXPECT(cuCrealf(z) == 1.5f && cuCimagf(z) == -2.f && cuCimagf(cuConjf(z)) == 2.f);
        const cuDoubleComplex w = cuComplexFloatToDouble(z);
        EXPECT(cuCreal(w) == 1.5 && cuCimag(cuConj(w)) == 2.0);
        const cuFloatComplex back = cuComplexDoubleToFloat(make_cuDoubleComplex(1.0 / 3, 0.1));
        EXPECT(back.x == float(1.0 / 3) && back.y == 0.1f);
        EXPECT(cuCabsf(make_cuFloatComplex(3.f, -4.f)) == 5.f && cuCabs(make_cuDoubleComplex(-5, 12)) == 13);
        const cuFloatComplex i = make_cuFloatComplex(0.f, 1.f), ii = cuCmulf(i, i);
        EXPECT(ii.x == -1.f && ii.y == 0.f);
        const cuDoubleComplex q = cuCdiv(make_cuDoubleComplex(1, 0), make_cuDoubleComplex(0, 2));
        EXPECT(q.x == 0 && q.y == -0.5);
    }

    // What CUDA's formulas avoid: squares that overflow or underflow
    {
        const float big = 2e38f, tiny = 3e-30f;
        EXPECT(std::isinf(big * big) && std::fabs(cuCabsf(make_cuFloatComplex(big, big)) - big * std::sqrt(2.f)) <= 1e32f);
        EXPECT(tiny * tiny == 0 && std::fabs(cuCabsf(make_cuFloatComplex(tiny, 4 * tiny / 3)) - 5 * tiny / 3) <= 1e-36f);
        const cuFloatComplex one = cuCdivf(make_cuFloatComplex(1e30f, 1e30f), make_cuFloatComplex(1e30f, 1e30f));
        EXPECT(std::fabs(one.x - 1.f) <= 1e-6f && std::fabs(one.y) <= 1e-6f);
        const cuDoubleComplex half = cuCdiv(make_cuDoubleComplex(1e300, 0), make_cuDoubleComplex(1e300, 1e300));
        EXPECT(std::fabs(half.x - 0.5) <= 1e-15 && std::fabs(half.y + 0.5) <= 1e-15);
        EXPECT(cuCabs(make_cuDoubleComplex(1e300, 1e300)) > 1.4e300 && !std::isinf(cuCabs(make_cuDoubleComplex(1e300, 1e300))));
    }

    // Special values: infinities and zeros
    {
        const float inf = std::numeric_limits<float>::infinity();
        EXPECT(std::isinf(cuCabsf(make_cuFloatComplex(inf, 1.f))) && std::isinf(cuCabs(make_cuDoubleComplex(1, -INFINITY))));
        EXPECT(cuCabsf(make_cuFloatComplex(0.f, 0.f)) == 0.f && cuCabsf(make_cuFloatComplex(-0.f, 7.f)) == 7.f);
        EXPECT(std::isnan(cuCdivf(make_cuFloatComplex(1.f, 1.f), make_cuFloatComplex(0.f, 0.f)).x));
    }

    // A discrete Fourier transform in a kernel, against long double
    {
        const int n = 64;
        std::vector<float> signal(n);
        for (int i = 0; i < n; ++i)
            signal[i] = std::sin(2.f * float(M_PI) * 5.f * float(i) / n) + 0.5f * std::cos(2.f * float(M_PI) * 9.f * float(i) / n);
        std::vector<cuFloatComplex> spectrum(n);
        dft<<<2, 32>>>(signal.data(), spectrum.data(), n);
        unsigned wrong = 0;
        for (int k = 0; k < n; ++k) {
            std::complex<long double> sum = 0;
            for (int j = 0; j < n; ++j)
                sum += (long double)signal[j] * std::polar(1.0L, -2 * 3.14159265358979323846264338327950288L * (k * j % n) / n);
            wrong += std::fabs(spectrum[k].x - sum.real()) > 1e-4 || std::fabs(spectrum[k].y - sum.imag()) > 1e-4;
        }
        EXPECT(wrong == 0);
        // The peaks: n / 2 at the sine's frequency (imaginary), n / 4 at the cosine's
        EXPECT(std::fabs(spectrum[5].y + n / 2) < 1e-3f && std::fabs(spectrum[9].x - n / 4) < 1e-3f);

        cuDoubleComplex z[3] = {make_cuDoubleComplex(3, 4), make_cuDoubleComplex(0, -2), make_cuDoubleComplex(-1, 1)};
        double magnitude[3];
        normalize<<<1, 3>>>(z, magnitude);
        EXPECT(magnitude[0] == 5 && magnitude[1] == 2 && std::fabs(magnitude[2] - std::sqrt(2.0)) < 1e-15);
        // Scaled by 1 / 5, which double doesn't hold exactly
        EXPECT(std::fabs(z[0].x - 0.6) < 1e-15 && std::fabs(z[0].y - 0.8) < 1e-15 && z[1].x == 0 && z[1].y == -1);
    }

    if (errors == 0)
        std::printf("complex: PASS\n");
    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
