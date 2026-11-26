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

// Texture objects, as CUDA's: their descriptors (cudaChannelFormatDesc,
// cudaResourceDesc, cudaTextureDesc), cudaCreateTextureObject and the rest,
// and tex1Dfetch on linear memory. A texture object is a copy of its
// descriptors, which its handle points to. Textures of CUDA arrays (tex1D,
// tex2D, tex3D, filtering, addressing modes) aren't implemented yet: their
// objects fail with cudaErrorNotSupported.
#pragma once

#include <cstring>
#include <type_traits>

#include "memory.hpp"

namespace cuda4cpu {

inline namespace cuda_api {

enum cudaChannelFormatKind {
    cudaChannelFormatKindSigned   = 0,
    cudaChannelFormatKindUnsigned = 1,
    cudaChannelFormatKindFloat    = 2,
    cudaChannelFormatKindNone     = 3,
};

struct cudaChannelFormatDesc {
    int x, y, z, w;                 //!< Bits of each component
    cudaChannelFormatKind f;
};

struct cudaMipmappedArray;
using cudaMipmappedArray_t       = cudaMipmappedArray *;
using cudaMipmappedArray_const_t = const cudaMipmappedArray *;

enum cudaResourceType {
    cudaResourceTypeArray          = 0,
    cudaResourceTypeMipmappedArray = 1,
    cudaResourceTypeLinear         = 2,
    cudaResourceTypePitch2D        = 3,
};

struct cudaResourceDesc {
    cudaResourceType resType;
    union {
        struct {
            cudaArray_t array;
        } array;
        struct {
            cudaMipmappedArray_t mipmap;
        } mipmap;
        struct {
            void *devPtr;
            cudaChannelFormatDesc desc;
            size_t sizeInBytes;
        } linear;
        struct {
            void *devPtr;
            cudaChannelFormatDesc desc;
            size_t width;
            size_t height;
            size_t pitchInBytes;
        } pitch2D;
    } res;
    unsigned int flags;
};

enum cudaTextureAddressMode {
    cudaAddressModeWrap   = 0,
    cudaAddressModeClamp  = 1,
    cudaAddressModeMirror = 2,
    cudaAddressModeBorder = 3,
};

enum cudaTextureFilterMode {
    cudaFilterModePoint  = 0,
    cudaFilterModeLinear = 1,
};

enum cudaTextureReadMode {
    cudaReadModeElementType     = 0,
    cudaReadModeNormalizedFloat = 1,
};

struct cudaTextureDesc {
    cudaTextureAddressMode addressMode[3];
    cudaTextureFilterMode filterMode;
    cudaTextureReadMode readMode;
    int sRGB;
    float borderColor[4];
    int normalizedCoords;
    unsigned int maxAnisotropy;
    cudaTextureFilterMode mipmapFilterMode;
    float mipmapLevelBias;
    float minMipmapLevelClamp;
    float maxMipmapLevelClamp;
    int disableTrilinearOptimization;
    int seamlessCubemap;
};

enum cudaResourceViewFormat {
    cudaResViewFormatNone = 0,
};

struct cudaResourceViewDesc {
    cudaResourceViewFormat format;
    size_t width;
    size_t height;
    size_t depth;
    unsigned int firstMipmapLevel;
    unsigned int lastMipmapLevel;
    unsigned int firstLayer;
    unsigned int lastLayer;
};

using cudaTextureObject_t = unsigned long long;

}

namespace detail {

//! What a texture object's handle points to
struct texture_object {
    cudaResourceDesc resource;
    cudaTextureDesc texture;
};

inline const texture_object *texture_of(cudaTextureObject_t handle)
{
    return reinterpret_cast<const texture_object *>(handle);
}

//! The channels of T: a scalar, or one of the vector types
template <typename T>
constexpr cudaChannelFormatDesc channel_desc()
{
    auto scalar = [](auto value) -> cudaChannelFormatDesc {
        using S = decltype(value);
        const int bits = int(sizeof(S) * 8);
        if constexpr (std::is_floating_point_v<S>)
            return {bits, 0, 0, 0, cudaChannelFormatKindFloat};
        else if constexpr (std::is_integral_v<S> && std::is_signed_v<S>)
            return {bits, 0, 0, 0, cudaChannelFormatKindSigned};
        else if constexpr (std::is_integral_v<S>)
            return {bits, 0, 0, 0, cudaChannelFormatKindUnsigned};
        else
            return {0, 0, 0, 0, cudaChannelFormatKindNone};
    };
    if constexpr (std::is_arithmetic_v<T>) {
        return scalar(T{});
    } else if constexpr (requires(T t) { t.x; }) {
        cudaChannelFormatDesc d = scalar(T{}.x);
        const int bits = d.x;
        if constexpr (requires(T t) { t.y; })
            d.y = bits;
        if constexpr (requires(T t) { t.z; })
            d.z = bits;
        if constexpr (requires(T t) { t.w; })
            d.w = bits;
        return d;
    } else {
        return {0, 0, 0, 0, cudaChannelFormatKindNone};
    }
}

}

inline namespace cuda_api {

template <typename T>
inline cudaChannelFormatDesc cudaCreateChannelDesc()
{
    return detail::channel_desc<T>();
}

inline cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w, cudaChannelFormatKind f)
{
    return {x, y, z, w, f};
}

//! A texture object of linear or pitched memory: arrays aren't implemented
//! yet
inline cudaError_t cudaCreateTextureObject(cudaTextureObject_t *texture, const cudaResourceDesc *resource,
                                           const cudaTextureDesc *description,
                                           const cudaResourceViewDesc * /* view */)
{
    if (texture == nullptr || resource == nullptr || description == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    if (resource->resType != cudaResourceTypeLinear && resource->resType != cudaResourceTypePitch2D)
        return detail::record_error(cudaErrorNotSupported);
    const void *memory = resource->resType == cudaResourceTypeLinear ? resource->res.linear.devPtr
                                                                     : resource->res.pitch2D.devPtr;
    if (memory == nullptr)
        return detail::record_error(cudaErrorInvalidValue);
    *texture = reinterpret_cast<cudaTextureObject_t>(new detail::texture_object{*resource, *description});
    return cudaSuccess;
}

inline cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texture)
{
    delete detail::texture_of(texture);
    return cudaSuccess;
}

inline cudaError_t cudaGetTextureObjectResourceDesc(cudaResourceDesc *resource, cudaTextureObject_t texture)
{
    if (resource == nullptr || texture == 0)
        return detail::record_error(cudaErrorInvalidValue);
    *resource = detail::texture_of(texture)->resource;
    return cudaSuccess;
}

inline cudaError_t cudaGetTextureObjectTextureDesc(cudaTextureDesc *description, cudaTextureObject_t texture)
{
    if (description == nullptr || texture == 0)
        return detail::record_error(cudaErrorInvalidValue);
    *description = detail::texture_of(texture)->texture;
    return cudaSuccess;
}

//! Element x of a texture of linear memory, as T; 0 outside it, as on a GPU
//! (and 0 from a texture that isn't of linear memory). With
//! cudaReadModeNormalizedFloat, components of 8- and 16-bit integers become
//! floats in [0, 1] (unsigned) or [-1, 1] (signed).
template <typename T>
inline T tex1Dfetch(cudaTextureObject_t texture, int x)
{
    T value{};
    if (texture == 0)
        return value;
    const detail::texture_object &t = *detail::texture_of(texture);
    if (t.resource.resType != cudaResourceTypeLinear)
        return value;
    const auto &linear = t.resource.res.linear;
    const cudaChannelFormatDesc &c = linear.desc;
    const int widths[4] = {c.x, c.y, c.z, c.w};
    const size_t element = size_t(c.x + c.y + c.z + c.w) / 8;
    if (x < 0 || element == 0 || size_t(x) >= linear.sizeInBytes / element)
        return value;
    const unsigned char *at = static_cast<const unsigned char *>(linear.devPtr) + size_t(x) * element;
    bool normalize = t.texture.readMode == cudaReadModeNormalizedFloat && c.f != cudaChannelFormatKindFloat;
    for (int width : widths)
        normalize = normalize && (width == 0 || width == 8 || width == 16);
    if (!normalize) {
        std::memcpy(&value, at, std::min(sizeof(T), element));
        return value;
    }
    float out[4] = {};
    int components = 0;
    for (int width : widths) {
        if (width == 0)
            continue;
        const unsigned char *p = at;
        at += width / 8;
        const int i = components++;
        if (width == 8)
            out[i] = c.f == cudaChannelFormatKindSigned ? std::max(float(*reinterpret_cast<const signed char *>(p)) / 127.0f, -1.0f)
                                                        : float(*p) / 255.0f;
        else {
            short s;
            unsigned short u;
            std::memcpy(&s, p, 2);
            std::memcpy(&u, p, 2);
            out[i] = c.f == cudaChannelFormatKindSigned ? std::max(float(s) / 32767.0f, -1.0f) : float(u) / 65535.0f;
        }
    }
    std::memcpy(&value, out, std::min(sizeof(T), sizeof(out)));
    return value;
}

}

}
