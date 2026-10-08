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


// NVIDIA's NVTX, the annotations that profilers show. There is no profiler,
// so they do nothing; ranges still return their nesting level.

#pragma once

#include <cstdint>

using nvtxRangeId_t     = uint64_t;
using nvtxDomainHandle_t = struct nvtxDomainRegistration_st *;
using nvtxStringHandle_t = struct nvtxStringRegistration_st *;

enum nvtxColorType_t { NVTX_COLOR_UNKNOWN = 0, NVTX_COLOR_ARGB = 1 };
enum nvtxMessageType_t {
    NVTX_MESSAGE_UNKNOWN = 0, NVTX_MESSAGE_TYPE_ASCII = 1, NVTX_MESSAGE_TYPE_UNICODE = 2,
    NVTX_MESSAGE_TYPE_REGISTERED = 3
};
enum nvtxPayloadType_t {
    NVTX_PAYLOAD_UNKNOWN = 0, NVTX_PAYLOAD_TYPE_UNSIGNED_INT64 = 1, NVTX_PAYLOAD_TYPE_INT64 = 2,
    NVTX_PAYLOAD_TYPE_DOUBLE = 3, NVTX_PAYLOAD_TYPE_UNSIGNED_INT32 = 4, NVTX_PAYLOAD_TYPE_INT32 = 5,
    NVTX_PAYLOAD_TYPE_FLOAT = 6
};

#define NVTX_VERSION 3
#define NVTX_EVENT_ATTRIB_STRUCT_SIZE ((uint16_t)sizeof(nvtxEventAttributes_t))

union nvtxMessageValue_t {
    const char *ascii;
    const wchar_t *unicode;
    nvtxStringHandle_t registered;
};

struct nvtxEventAttributes_t {
    uint16_t version;
    uint16_t size;
    uint32_t category;
    int32_t colorType;
    uint32_t color;
    int32_t payloadType;
    int32_t reserved0;
    union {
        uint64_t ullValue;
        int64_t llValue;
        double dValue;
        uint32_t uiValue;
        int32_t iValue;
        float fValue;
    } payload;
    int32_t messageType;
    nvtxMessageValue_t message;
};

namespace cuda4cpu::detail {

inline thread_local int nvtx_depth = 0;

}

inline void nvtxMarkA(const char *) {}
inline void nvtxMarkW(const wchar_t *) {}
inline void nvtxMarkEx(const nvtxEventAttributes_t *) {}
inline nvtxRangeId_t nvtxRangeStartA(const char *) { return 0; }
inline nvtxRangeId_t nvtxRangeStartW(const wchar_t *) { return 0; }
inline nvtxRangeId_t nvtxRangeStartEx(const nvtxEventAttributes_t *) { return 0; }
inline void nvtxRangeEnd(nvtxRangeId_t) {}
inline int nvtxRangePushA(const char *) { return cuda4cpu::detail::nvtx_depth++; }
inline int nvtxRangePushW(const wchar_t *) { return cuda4cpu::detail::nvtx_depth++; }
inline int nvtxRangePushEx(const nvtxEventAttributes_t *) { return cuda4cpu::detail::nvtx_depth++; }
inline int nvtxRangePop() { return cuda4cpu::detail::nvtx_depth > 0 ? --cuda4cpu::detail::nvtx_depth : -1; }
inline void nvtxNameOsThreadA(uint32_t, const char *) {}
inline void nvtxNameCategoryA(uint32_t, const char *) {}
inline nvtxDomainHandle_t nvtxDomainCreateA(const char *) { return nullptr; }
inline void nvtxDomainDestroy(nvtxDomainHandle_t) {}
inline void nvtxDomainMarkEx(nvtxDomainHandle_t, const nvtxEventAttributes_t *) {}
inline int nvtxDomainRangePushEx(nvtxDomainHandle_t, const nvtxEventAttributes_t *a) { return nvtxRangePushEx(a); }
inline int nvtxDomainRangePop(nvtxDomainHandle_t) { return nvtxRangePop(); }
inline nvtxStringHandle_t nvtxDomainRegisterStringA(nvtxDomainHandle_t, const char *) { return nullptr; }
