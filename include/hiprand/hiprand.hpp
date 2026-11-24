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

// hipRAND's C++ API (hiprand_cpp): engines that own a generator of the C API
// (hiprand.h) and distributions that fill arrays from one, as in hipRAND.
#pragma once

#include <exception>
#include <string>
#include <type_traits>
#include <utility>

#include "hiprand.h"

namespace hiprand_cpp {

//! A failed call of the C API
class error : public std::exception {
public:
    using error_type = hiprandStatus_t;

    explicit error(error_type error) noexcept : error_(error), message_(to_string(error)) {}

    error_type error_code() const noexcept { return error_; }
    std::string error_string() const noexcept { return message_; }
    const char *what() const noexcept override { return message_.c_str(); }

    static std::string to_string(error_type error)
    {
        switch (error) {
        case HIPRAND_STATUS_SUCCESS:                   return "Success";
        case HIPRAND_STATUS_VERSION_MISMATCH:          return "Header file and linked library version do not match";
        case HIPRAND_STATUS_NOT_INITIALIZED:           return "Generator was not created using hiprandCreateGenerator";
        case HIPRAND_STATUS_ALLOCATION_FAILED:         return "Memory allocation failed during execution";
        case HIPRAND_STATUS_TYPE_ERROR:                return "Generator type is wrong";
        case HIPRAND_STATUS_OUT_OF_RANGE:              return "Argument out of range";
        case HIPRAND_STATUS_LENGTH_NOT_MULTIPLE:       return "Length requested is not a multiple of dimension";
        case HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED: return "GPU does not have double precision";
        case HIPRAND_STATUS_LAUNCH_FAILURE:            return "Kernel launch failure";
        case HIPRAND_STATUS_INTERNAL_ERROR:            return "Internal library error";
        default:                                       return "Unknown error";
        }
    }

private:
    error_type error_;
    std::string message_;
};

namespace detail {

inline void check(hiprandStatus_t status)
{
    if (status != HIPRAND_STATUS_SUCCESS)
        throw error(status);
}

//! An engine: a generator of the C API, of type Type
template <hiprandRngType_t Type, unsigned long long DefaultSeed, bool Quasi, typename Result = unsigned int>
class engine {
public:
    using result_type = Result;
    using seed_type = unsigned long long;
    using offset_type = unsigned long long;
    using dimensions_num_type = unsigned int;
    static constexpr seed_type default_seed = DefaultSeed;

    explicit engine(unsigned long long seed_or_dimensions = Quasi ? 1 : DefaultSeed, offset_type offset_value = 0)
    {
        check(hiprandCreateGenerator(&generator_, Type));
        try {
            if constexpr (Quasi)
                dimensions(dimensions_num_type(seed_or_dimensions));
            else
                seed(seed_or_dimensions);
            if (offset_value != 0)
                offset(offset_value);
        } catch (...) {
            hiprandDestroyGenerator(generator_);
            throw;
        }
    }
    engine(const engine &) = delete;
    engine &operator=(const engine &) = delete;
    engine(engine &&other) noexcept : generator_(other.generator_) { other.generator_ = nullptr; }
    engine &operator=(engine &&other) noexcept
    {
        std::swap(generator_, other.generator_);
        return *this;
    }
    ~engine() noexcept
    {
        if (generator_ != nullptr)
            hiprandDestroyGenerator(generator_);
    }

    void stream(hipStream_t value) { check(hiprandSetStream(generator_, value)); }
    void offset(offset_type value) { check(hiprandSetGeneratorOffset(generator_, value)); }
    void seed(seed_type value) requires (!Quasi) { check(hiprandSetPseudoRandomGeneratorSeed(generator_, value)); }
    void dimensions(dimensions_num_type value) requires Quasi
    {
        check(hiprandSetQuasiRandomGeneratorDimensions(generator_, value));
    }

    //! Fills output with size values
    void operator()(result_type *output, size_t size)
    {
        if constexpr (std::is_same_v<Result, unsigned long long>)
            check(hiprandGenerateLongLong(generator_, output, size));
        else
            check(hiprandGenerate(generator_, output, size));
    }

    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return ~result_type(0); }

    hiprandGenerator_t generator_ = nullptr;
};

}

template <unsigned long long DefaultSeed = 0>
using xorwow_engine = detail::engine<HIPRAND_RNG_PSEUDO_XORWOW, DefaultSeed, false>;
template <unsigned long long DefaultSeed = 0>
using philox4x32_10_engine = detail::engine<HIPRAND_RNG_PSEUDO_PHILOX4_32_10, DefaultSeed, false>;
template <unsigned long long DefaultSeed = 12345>
using mrg32k3a_engine = detail::engine<HIPRAND_RNG_PSEUDO_MRG32K3A, DefaultSeed, false>;
template <unsigned long long DefaultSeed = 0>
using mtgp32_engine = detail::engine<HIPRAND_RNG_PSEUDO_MTGP32, DefaultSeed, false>;
template <unsigned long long DefaultSeed = 0>
using mt19937_engine = detail::engine<HIPRAND_RNG_PSEUDO_MT19937, DefaultSeed, false>;
template <unsigned int DefaultNumDimensions = 1>
using sobol32_engine = detail::engine<HIPRAND_RNG_QUASI_SOBOL32, DefaultNumDimensions, true>;
template <unsigned int DefaultNumDimensions = 1>
using scrambled_sobol32_engine = detail::engine<HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32, DefaultNumDimensions, true>;
template <unsigned int DefaultNumDimensions = 1>
using sobol64_engine = detail::engine<HIPRAND_RNG_QUASI_SOBOL64, DefaultNumDimensions, true, unsigned long long>;
template <unsigned int DefaultNumDimensions = 1>
using scrambled_sobol64_engine =
    detail::engine<HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64, DefaultNumDimensions, true, unsigned long long>;

using xorwow = xorwow_engine<>;
using philox4x32_10 = philox4x32_10_engine<>;
using mrg32k3a = mrg32k3a_engine<>;
using mtgp32 = mtgp32_engine<>;
using mt19937 = mt19937_engine<>;
using sobol32 = sobol32_engine<>;
using scrambled_sobol32 = scrambled_sobol32_engine<>;
using sobol64 = sobol64_engine<>;
using scrambled_sobol64 = scrambled_sobol64_engine<>;
using default_random_engine = xorwow;

//! Unsigned integers over their whole range: 32 bits, or 64 from the 64-bit
//! Sobol' engines
template <typename IntType = unsigned int>
class uniform_int_distribution {
    static_assert(std::is_same_v<IntType, unsigned int> || std::is_same_v<IntType, unsigned long long>,
                  "cuda4cpu: hipRAND's uniform integers are unsigned int or unsigned long long");

public:
    using result_type = IntType;
    void reset() {}
    IntType min() const { return 0; }
    IntType max() const { return ~IntType(0); }

    template <typename Generator>
    void operator()(Generator &g, IntType *output, size_t size)
    {
        if constexpr (std::is_same_v<IntType, unsigned long long>)
            detail::check(hiprandGenerateLongLong(g.generator_, output, size));
        else
            detail::check(hiprandGenerate(g.generator_, output, size));
    }
};

//! Floats or doubles in (0, 1]
template <typename RealType = float>
class uniform_real_distribution {
public:
    using result_type = RealType;
    void reset() {}
    RealType min() const { return 0; }
    RealType max() const { return 1; }

    template <typename Generator>
    void operator()(Generator &g, RealType *output, size_t size)
    {
        if constexpr (std::is_same_v<RealType, double>)
            detail::check(hiprandGenerateUniformDouble(g.generator_, output, size));
        else
            detail::check(hiprandGenerateUniform(g.generator_, output, size));
    }
};

template <typename RealType = float>
class normal_distribution {
public:
    using result_type = RealType;

    explicit normal_distribution(RealType mean = 0, RealType stddev = 1) : mean_(mean), stddev_(stddev) {}
    void reset() {}
    RealType mean() const { return mean_; }
    RealType stddev() const { return stddev_; }

    template <typename Generator>
    void operator()(Generator &g, RealType *output, size_t size)
    {
        if constexpr (std::is_same_v<RealType, double>)
            detail::check(hiprandGenerateNormalDouble(g.generator_, output, size, mean_, stddev_));
        else
            detail::check(hiprandGenerateNormal(g.generator_, output, size, mean_, stddev_));
    }

private:
    RealType mean_, stddev_;
};

template <typename RealType = float>
class lognormal_distribution {
public:
    using result_type = RealType;

    explicit lognormal_distribution(RealType m = 0, RealType s = 1) : m_(m), s_(s) {}
    void reset() {}
    RealType m() const { return m_; }
    RealType s() const { return s_; }

    template <typename Generator>
    void operator()(Generator &g, RealType *output, size_t size)
    {
        if constexpr (std::is_same_v<RealType, double>)
            detail::check(hiprandGenerateLogNormalDouble(g.generator_, output, size, m_, s_));
        else
            detail::check(hiprandGenerateLogNormal(g.generator_, output, size, m_, s_));
    }

private:
    RealType m_, s_;
};

template <typename IntType = unsigned int>
class poisson_distribution {
public:
    using result_type = IntType;

    explicit poisson_distribution(double mean = 1.0) : mean_(mean) {}
    void reset() {}
    double mean() const { return mean_; }

    template <typename Generator>
    void operator()(Generator &g, IntType *output, size_t size)
    {
        detail::check(hiprandGeneratePoisson(g.generator_, output, size, mean_));
    }

private:
    double mean_;
};

inline int version()
{
    int value = 0;
    detail::check(hiprandGetVersion(&value));
    return value;
}

}
