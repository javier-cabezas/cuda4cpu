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

// HIP's vector types on its AMD platform, with HIP 7's interface:
// float4 is HIP_vector_type<float, 4>, a class template derived from
// HIP_vector_base<float, 4>, which holds x, y, z and w. Libraries extend
// them: rocWMMA specializes HIP_vector_base for its own element types and
// for 8 to 512 elements, reads them through get_native_vector() and
// Native_vec_, and builds them with make_vector_type<T, N>(). cuda/types.hpp
// defines the CUDA vector names as these on the AMD platform, and as CUDA's
// structs otherwise.
//
// The operations work element by element, through operator[], so that any
// contiguous storage of N elements, as a specialization's, works.
// Native_vec_ is Clang's ext_vector_type, or with GCC a struct of N
// elements; a 3-element vector's is always a struct, so that it keeps 3
// elements' size. As in HIP 7, a vector built from one value has it as x,
// and the other elements zero; make_vector_type<T, N>(value) broadcasts.

#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>

template <typename T, unsigned int n>
struct HIP_vector_base;
template <typename T, unsigned int rank>
struct HIP_vector_type;

namespace cuda4cpu::detail {

//! N elements, as Native_vec_ where Clang's vectors aren't available, or
//! for 3 elements
template <typename T, unsigned int N>
struct native_array {
    T d[N];

    constexpr T &operator[](unsigned int i) noexcept { return d[i]; }
    constexpr const T &operator[](unsigned int i) const noexcept { return d[i]; }
};

#if defined(__clang__)
template <typename T, unsigned int N>
struct native_vector {
    using type = T __attribute__((ext_vector_type(N)));
};
template <typename T>
struct native_vector<T, 3> {
    using type = native_array<T, 3>;
};
#else
template <typename T, unsigned int N>
struct native_vector {
    using type = native_array<T, N>;
};
#endif

}

//! The vector as its Native_vec_: the elements are the object's
template <typename T, unsigned int n>
inline typename HIP_vector_base<T, n>::Native_vec_ &get_native_vector(HIP_vector_base<T, n> &v) noexcept
{
    static_assert(sizeof(v) == sizeof(typename HIP_vector_base<T, n>::Native_vec_));
    return *reinterpret_cast<typename HIP_vector_base<T, n>::Native_vec_ *>(&v);
}

template <typename T, unsigned int n>
inline const typename HIP_vector_base<T, n>::Native_vec_ &get_native_vector(const HIP_vector_base<T, n> &v) noexcept
{
    static_assert(sizeof(v) == sizeof(typename HIP_vector_base<T, n>::Native_vec_));
    return *reinterpret_cast<const typename HIP_vector_base<T, n>::Native_vec_ *>(&v);
}

template <typename T>
struct HIP_vector_base<T, 1> {
    using Native_vec_ = typename cuda4cpu::detail::native_vector<T, 1>::type;
    using value_type = T;

    T x;

    HIP_vector_base() = default;
    explicit constexpr HIP_vector_base(T x_) noexcept : x(x_) {}
};

template <typename T>
struct alignas(2 * sizeof(T)) HIP_vector_base<T, 2> {
    using Native_vec_ = typename cuda4cpu::detail::native_vector<T, 2>::type;
    using value_type = T;

    T x, y;

    HIP_vector_base() = default;
    constexpr HIP_vector_base(T x_, T y_ = T()) noexcept : x(x_), y(y_) {}
};

template <typename T>
struct HIP_vector_base<T, 3> {
    using Native_vec_ = typename cuda4cpu::detail::native_vector<T, 3>::type;
    using value_type = T;

    T x, y, z;

    HIP_vector_base() = default;
    constexpr HIP_vector_base(T x_, T y_ = T(), T z_ = T()) noexcept : x(x_), y(y_), z(z_) {}
};

template <typename T>
struct alignas(4 * sizeof(T)) HIP_vector_base<T, 4> {
    using Native_vec_ = typename cuda4cpu::detail::native_vector<T, 4>::type;
    using value_type = T;

    T x, y, z, w;

    HIP_vector_base() = default;
    constexpr HIP_vector_base(T x_, T y_ = T(), T z_ = T(), T w_ = T()) noexcept : x(x_), y(y_), z(z_), w(w_) {}
};

template <typename T, std::size_t rank, std::size_t... indices>
constexpr HIP_vector_type<T, rank> make_vector_type_impl(T value, std::index_sequence<indices...>) noexcept
{
    return HIP_vector_type<T, rank>{((void)indices, value)...};
}

//! Every element value
template <typename T, unsigned int rank>
constexpr HIP_vector_type<T, rank> make_vector_type(T value) noexcept
{
    return make_vector_type_impl<T, rank>(value, std::make_index_sequence<rank>{});
}

template <typename T, unsigned int rank>
struct HIP_vector_type : public HIP_vector_base<T, rank> {
    using typename HIP_vector_base<T, rank>::Native_vec_;

    HIP_vector_type() = default;

    //! x, the other elements zero
    template <typename U, std::enable_if_t<std::is_convertible_v<U, T>> * = nullptr>
    explicit constexpr HIP_vector_type(U x_) noexcept : HIP_vector_base<T, rank>{static_cast<T>(x_)}
    {}

    template <typename... Us, std::enable_if_t<(rank > 1) && sizeof...(Us) == rank> * = nullptr>
    constexpr HIP_vector_type(Us... xs) noexcept : HIP_vector_base<T, rank>{static_cast<T>(xs)...}
    {}

    constexpr T &operator[](std::size_t i) noexcept { return reinterpret_cast<T *>(this)[i]; }
    constexpr const T &operator[](std::size_t i) const noexcept { return reinterpret_cast<const T *>(this)[i]; }

    constexpr HIP_vector_type &operator+=(const HIP_vector_type &v) noexcept
    {
        for (unsigned int i = 0; i < rank; ++i)
            (*this)[i] += v[i];
        return *this;
    }
    constexpr HIP_vector_type &operator-=(const HIP_vector_type &v) noexcept
    {
        for (unsigned int i = 0; i < rank; ++i)
            (*this)[i] -= v[i];
        return *this;
    }
    constexpr HIP_vector_type &operator*=(const HIP_vector_type &v) noexcept
    {
        for (unsigned int i = 0; i < rank; ++i)
            (*this)[i] *= v[i];
        return *this;
    }
    constexpr HIP_vector_type &operator/=(const HIP_vector_type &v) noexcept
    {
        for (unsigned int i = 0; i < rank; ++i)
            (*this)[i] /= v[i];
        return *this;
    }

    // Each element with a scalar
    template <typename U, std::enable_if_t<std::is_convertible_v<U, T>> * = nullptr>
    constexpr HIP_vector_type &operator+=(U v) noexcept
    {
        return *this += make_vector_type<T, rank>(static_cast<T>(v));
    }
    template <typename U, std::enable_if_t<std::is_convertible_v<U, T>> * = nullptr>
    constexpr HIP_vector_type &operator-=(U v) noexcept
    {
        return *this -= make_vector_type<T, rank>(static_cast<T>(v));
    }
    template <typename U, std::enable_if_t<std::is_convertible_v<U, T>> * = nullptr>
    constexpr HIP_vector_type &operator*=(U v) noexcept
    {
        return *this *= make_vector_type<T, rank>(static_cast<T>(v));
    }
    template <typename U, std::enable_if_t<std::is_convertible_v<U, T>> * = nullptr>
    constexpr HIP_vector_type &operator/=(U v) noexcept
    {
        return *this /= make_vector_type<T, rank>(static_cast<T>(v));
    }

    HIP_vector_type &operator++() noexcept { return *this += make_vector_type<T, rank>(T(1)); }
    HIP_vector_type &operator--() noexcept { return *this -= make_vector_type<T, rank>(T(1)); }
    HIP_vector_type operator++(int) noexcept
    {
        HIP_vector_type old(*this);
        ++*this;
        return old;
    }
    HIP_vector_type operator--(int) noexcept
    {
        HIP_vector_type old(*this);
        --*this;
        return old;
    }

    friend constexpr HIP_vector_type operator*(HIP_vector_type a, const HIP_vector_type &b) noexcept
    {
        return a *= b;
    }
    friend constexpr HIP_vector_type operator/(HIP_vector_type a, const HIP_vector_type &b) noexcept
    {
        return a /= b;
    }

    template <typename U = T, std::enable_if_t<std::is_signed_v<U> || std::is_floating_point_v<U> ||
                                               !std::is_arithmetic_v<U>> * = nullptr>
    constexpr HIP_vector_type operator-() const noexcept
    {
        HIP_vector_type r(*this);
        for (unsigned int i = 0; i < rank; ++i)
            r[i] = -r[i];
        return r;
    }

    template <typename U = T, std::enable_if_t<std::is_integral_v<U>> * = nullptr>
    constexpr HIP_vector_type operator~() const noexcept
    {
        HIP_vector_type r(*this);
        for (unsigned int i = 0; i < rank; ++i)
            r[i] = ~r[i];
        return r;
    }

#define CUDA4CPU_VECTOR_INTEGER_ASSIGN(op)                                          \
    template <typename U = T, std::enable_if_t<std::is_integral_v<U>> * = nullptr> \
    constexpr HIP_vector_type &operator op(const HIP_vector_type &v) noexcept      \
    {                                                                              \
        for (unsigned int i = 0; i < rank; ++i)                                    \
            (*this)[i] op v[i];                                                    \
        return *this;                                                              \
    }
    CUDA4CPU_VECTOR_INTEGER_ASSIGN(%=)
    CUDA4CPU_VECTOR_INTEGER_ASSIGN(^=)
    CUDA4CPU_VECTOR_INTEGER_ASSIGN(|=)
    CUDA4CPU_VECTOR_INTEGER_ASSIGN(&=)
    CUDA4CPU_VECTOR_INTEGER_ASSIGN(>>=)
    CUDA4CPU_VECTOR_INTEGER_ASSIGN(<<=)
#undef CUDA4CPU_VECTOR_INTEGER_ASSIGN
};

// The binary operators: of two vectors, or of a vector and a scalar, which
// every element gets
#define CUDA4CPU_VECTOR_BINARY(op, condition)                                                          \
    template <typename T, unsigned int n, std::enable_if_t<condition> * = nullptr>                     \
    constexpr HIP_vector_type<T, n> operator op(const HIP_vector_type<T, n> &a,                       \
                                                const HIP_vector_type<T, n> &b) noexcept               \
    {                                                                                                  \
        return HIP_vector_type<T, n>{a} op## = b;                                                      \
    }                                                                                                  \
    template <typename T, unsigned int n, typename U,                                                  \
              std::enable_if_t<(condition) && std::is_convertible_v<U, T>> * = nullptr>                \
    constexpr HIP_vector_type<T, n> operator op(const HIP_vector_type<T, n> &a, U b) noexcept         \
    {                                                                                                  \
        return HIP_vector_type<T, n>{a} op## = make_vector_type<T, n>(static_cast<T>(b));              \
    }                                                                                                  \
    template <typename T, unsigned int n, typename U,                                                  \
              std::enable_if_t<(condition) && std::is_convertible_v<U, T>> * = nullptr>                \
    constexpr HIP_vector_type<T, n> operator op(U a, const HIP_vector_type<T, n> &b) noexcept         \
    {                                                                                                  \
        return make_vector_type<T, n>(static_cast<T>(a)) op## = b;                                     \
    }
CUDA4CPU_VECTOR_BINARY(+, true)
CUDA4CPU_VECTOR_BINARY(-, true)
CUDA4CPU_VECTOR_BINARY(%, std::is_integral_v<T>)
CUDA4CPU_VECTOR_BINARY(^, std::is_integral_v<T>)
CUDA4CPU_VECTOR_BINARY(|, std::is_integral_v<T>)
CUDA4CPU_VECTOR_BINARY(&, std::is_integral_v<T>)
CUDA4CPU_VECTOR_BINARY(>>, std::is_integral_v<T>)
CUDA4CPU_VECTOR_BINARY(<<, std::is_integral_v<T>)
#undef CUDA4CPU_VECTOR_BINARY

// * and / of two vectors are the class's friends
template <typename T, unsigned int n, typename U, std::enable_if_t<std::is_convertible_v<U, T>> * = nullptr>
constexpr HIP_vector_type<T, n> operator*(const HIP_vector_type<T, n> &a, U b) noexcept
{
    return HIP_vector_type<T, n>{a} *= make_vector_type<T, n>(static_cast<T>(b));
}
template <typename T, unsigned int n, typename U, std::enable_if_t<std::is_convertible_v<U, T>> * = nullptr>
constexpr HIP_vector_type<T, n> operator*(U a, const HIP_vector_type<T, n> &b) noexcept
{
    return make_vector_type<T, n>(static_cast<T>(a)) *= b;
}
template <typename T, unsigned int n, typename U, std::enable_if_t<std::is_convertible_v<U, T>> * = nullptr>
constexpr HIP_vector_type<T, n> operator/(const HIP_vector_type<T, n> &a, U b) noexcept
{
    return HIP_vector_type<T, n>{a} /= make_vector_type<T, n>(static_cast<T>(b));
}
template <typename T, unsigned int n, typename U, std::enable_if_t<std::is_convertible_v<U, T>> * = nullptr>
constexpr HIP_vector_type<T, n> operator/(U a, const HIP_vector_type<T, n> &b) noexcept
{
    return make_vector_type<T, n>(static_cast<T>(a)) /= b;
}

//! Whether every element is equal
template <typename T, unsigned int n>
constexpr bool operator==(const HIP_vector_type<T, n> &a, const HIP_vector_type<T, n> &b) noexcept
{
    for (unsigned int i = 0; i < n; ++i)
        if (!(a[i] == b[i]))
            return false;
    return true;
}
template <typename T, unsigned int n, typename U, std::enable_if_t<std::is_convertible_v<U, T>> * = nullptr>
constexpr bool operator==(const HIP_vector_type<T, n> &a, U b) noexcept
{
    return a == make_vector_type<T, n>(static_cast<T>(b));
}
template <typename T, unsigned int n, typename U, std::enable_if_t<std::is_convertible_v<U, T>> * = nullptr>
constexpr bool operator==(U a, const HIP_vector_type<T, n> &b) noexcept
{
    return make_vector_type<T, n>(static_cast<T>(a)) == b;
}
template <typename T, unsigned int n>
constexpr bool operator!=(const HIP_vector_type<T, n> &a, const HIP_vector_type<T, n> &b) noexcept
{
    return !(a == b);
}
template <typename T, unsigned int n, typename U, std::enable_if_t<std::is_convertible_v<U, T>> * = nullptr>
constexpr bool operator!=(const HIP_vector_type<T, n> &a, U b) noexcept
{
    return !(a == b);
}
template <typename T, unsigned int n, typename U, std::enable_if_t<std::is_convertible_v<U, T>> * = nullptr>
constexpr bool operator!=(U a, const HIP_vector_type<T, n> &b) noexcept
{
    return !(a == b);
}

//! u as a vector of rankT elements of T: its first ones, then zeros
template <typename T, unsigned int rankT, typename U, unsigned int rankU,
          std::enable_if_t<rankT == 1 || rankT == 2 || rankT == 4> * = nullptr>
inline const HIP_vector_type<T, rankT> __hipMapVector(const HIP_vector_type<U, rankU> &u)
{
    HIP_vector_type<T, rankT> r{};
    for (unsigned int i = 0; i < rankT && i < rankU; ++i)
        r[i] = static_cast<T>(u[i]);
    return r;
}
