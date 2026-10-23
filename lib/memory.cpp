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

// The registry of allocations, the unified memory hints, and HIP's virtual
// memory management.
//
// Every allocation (cudaMalloc, cudaMallocHost, cudaMallocManaged, ...) is
// recorded with its kind, so that cudaPointerGetAttributes can tell where a
// pointer points, and cudaFree can reject pointers that it didn't allocate.
// Registered host memory and the ranges mapped with hipMemMap are recorded
// the same way.
//
// Unified memory hints (cudaMemAdvise, cudaMemPrefetchAsync) change nothing,
// since all memory is host memory, but cudaMemRangeGetAttribute returns them:
// they are kept per range of pages, as disjoint ranges that updates split.
//
// Virtual memory management: physical memory (hipMemCreate) is a memfd, a
// reserved range (hipMemAddressReserve) is a PROT_NONE mapping, and hipMemMap
// maps the memfd over it. As on a GPU, mapped memory can't be accessed until
// hipMemSetAccess allows it: until then it faults.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string_view>
#include <tuple>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "cuda4cpu.hpp"

namespace cuda4cpu::detail {

//! Physical memory of HIP's virtual memory management. It is freed once its
//! handle is released and every mapping of it is unmapped.
struct vmm_allocation {
    int fd;
    size_t size;
    std::array<unsigned char, 64> properties;   //!< The hipMemAllocationProp it was created with
    unsigned refs;                              //!< The handle, retained handles and mappings
};

bool find_graph_allocation(const void *ptr, allocation_info &info);

namespace {

struct allocation {
    size_t size;
    memory_kind kind;
    unsigned flags;
    vmm_allocation *physical = nullptr;   //!< Of a mapped range
    cudaMemAccessFlags access = cudaMemAccessFlagsProtNone;
};

//! The unified memory hints of a page
struct hints {
    bool read_mostly = false;
    unsigned char accessed_by = 0;   //!< Bit 0: the device; bit 1: the host
    cudaMemLocation preferred{cudaMemLocationTypeInvalid, 0};
    cudaMemLocation prefetched{cudaMemLocationTypeInvalid, 0};
};

struct hint_range {
    uintptr_t end;
    hints value;
};

struct memory_state {
    std::mutex mutex;
    std::map<uintptr_t, allocation> allocations;   //!< By start address, disjoint
    std::map<uintptr_t, size_t> reservations;      //!< Ranges reserved by hipMemAddressReserve
    std::map<uintptr_t, hint_range> hint_ranges;   //!< Pages with hints, by start address, disjoint
};

//! Never destroyed: static objects may free memory after main returns
memory_state &state()
{
    static memory_state *s = new memory_state;
    return *s;
}

size_t page_size()
{
    static const size_t page = size_t(sysconf(_SC_PAGESIZE));
    return page;
}

//! The allocation that p points into, or end(). An allocation of 0 bytes
//! contains its start.
std::map<uintptr_t, allocation>::iterator containing(memory_state &s, uintptr_t p)
{
    auto it = s.allocations.upper_bound(p);
    if (it == s.allocations.begin())
        return s.allocations.end();
    --it;
    return p - it->first < std::max<size_t>(it->second.size, 1) ? it : s.allocations.end();
}

//! Whether an allocation overlaps [begin, end)
bool overlaps(memory_state &s, uintptr_t begin, uintptr_t end)
{
    auto it = s.allocations.lower_bound(begin);
    if (it != s.allocations.end() && it->first < end)
        return true;
    return it != s.allocations.begin() && std::prev(it)->first + std::prev(it)->second.size > begin;
}

//
// Hints, per range of pages
//

//! Splits the hint range that contains x, so that a range starts at x
void split(memory_state &s, uintptr_t x)
{
    auto it = s.hint_ranges.upper_bound(x);
    if (it == s.hint_ranges.begin())
        return;
    --it;
    if (it->first < x && x < it->second.end) {
        hint_range right = it->second;
        it->second.end = x;
        s.hint_ranges.emplace_hint(std::next(it), x, right);
    }
}

//! Calls update on the hints of every page of [begin, end)
template <typename F>
void update_hints(memory_state &s, uintptr_t begin, uintptr_t end, F update)
{
    split(s, begin);
    split(s, end);
    uintptr_t cur = begin;
    auto it = s.hint_ranges.lower_bound(begin);
    while (cur < end) {
        if (it == s.hint_ranges.end() || it->first > cur) {
            const uintptr_t gap_end = it == s.hint_ranges.end() ? end : std::min(it->first, end);
            it = s.hint_ranges.emplace_hint(it, cur, hint_range{gap_end, hints{}});
        }
        update(it->second.value);
        cur = it->second.end;
        ++it;
    }
}

//! Forgets the hints of [begin, end)
void erase_hints(memory_state &s, uintptr_t begin, uintptr_t end)
{
    if (s.hint_ranges.empty())
        return;
    split(s, begin);
    split(s, end);
    s.hint_ranges.erase(s.hint_ranges.lower_bound(begin), s.hint_ranges.lower_bound(end));
}

//! Calls visit with the hints of each range of pages in [begin, end)
template <typename F>
void visit_hints(memory_state &s, uintptr_t begin, uintptr_t end, F visit)
{
    uintptr_t cur = begin;
    auto it = s.hint_ranges.upper_bound(begin);
    if (it != s.hint_ranges.begin() && std::prev(it)->second.end > begin)
        --it;
    while (cur < end) {
        if (it == s.hint_ranges.end() || it->first > cur) {
            visit(hints{});
            cur = it == s.hint_ranges.end() ? end : it->first;
        } else {
            visit(it->second.value);
            cur = it->second.end;
            ++it;
        }
    }
}

//! The pages that contain [ptr, ptr + count)
std::pair<uintptr_t, uintptr_t> page_range(const void *ptr, size_t count)
{
    const uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
    const uintptr_t page = page_size();
    return {p / page * page, (p + count + page - 1) / page * page};
}

//! Checks a location of a hint. The host's locations have no id but a NUMA
//! node's, so they compare equal whatever id they come with.
cudaError_t check_location(cudaMemLocation &location)
{
    switch (location.type) {
    case cudaMemLocationTypeDevice:
        return location.id == 0 ? cudaSuccess : cudaErrorInvalidDevice;
    case cudaMemLocationTypeHost:
    case cudaMemLocationTypeHostNumaCurrent:
        location.id = 0;
        return cudaSuccess;
    case cudaMemLocationTypeHostNuma:
        return location.id >= 0 ? cudaSuccess : cudaErrorInvalidValue;
    default:
        return cudaErrorInvalidValue;
    }
}

bool same_location(const cudaMemLocation &a, const cudaMemLocation &b)
{
    return a.type == b.type && a.id == b.id;
}

//! The device number of a location, as CUDA's older attributes give it
int device_number(const cudaMemLocation &location)
{
    switch (location.type) {
    case cudaMemLocationTypeDevice:  return location.id;
    case cudaMemLocationTypeInvalid: return cudaInvalidDeviceId;
    default:                         return cudaCpuDeviceId;
    }
}

//! The location that every page of a range shares, or an invalid location
template <typename Get>
cudaMemLocation common_location(memory_state &s, uintptr_t begin, uintptr_t end, Get get)
{
    bool first = true, same = true;
    cudaMemLocation common{cudaMemLocationTypeInvalid, 0};
    visit_hints(s, begin, end, [&](const hints &h) {
        const cudaMemLocation l = get(h);
        if (first)
            common = l;
        else if (!same_location(common, l))
            same = false;
        first = false;
    });
    return same ? common : cudaMemLocation{cudaMemLocationTypeInvalid, 0};
}

//
// Virtual memory management
//

//! Drops a reference to physical memory (with the state locked)
void drop(vmm_allocation *a)
{
    if (--a->refs == 0) {
        close(a->fd);
        delete a;
    }
}

//! The mapping that contains p, or end()
std::map<uintptr_t, allocation>::iterator mapping_at(memory_state &s, uintptr_t p)
{
    auto it = containing(s, p);
    return it != s.allocations.end() && it->second.kind == memory_kind::mapped ? it : s.allocations.end();
}

//! Whether mappings cover [begin, end) without gaps
bool mapped(memory_state &s, uintptr_t begin, uintptr_t end)
{
    for (uintptr_t cur = begin; cur < end;) {
        auto it = mapping_at(s, cur);
        if (it == s.allocations.end())
            return false;
        cur = it->first + it->second.size;
    }
    return true;
}

//! Unmaps a mapping, which leaves its range reserved (with the state locked)
void unmap(memory_state &s, std::map<uintptr_t, allocation>::iterator it)
{
    mmap(reinterpret_cast<void *>(it->first), it->second.size, PROT_NONE,
         MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED, -1, 0);
    erase_hints(s, it->first, it->first + it->second.size);
    drop(it->second.physical);
    s.allocations.erase(it);
}

}

//
// The registry of allocations
//

void *allocate_aligned(size_t size, memory_kind kind, unsigned flags)
{
    // Rounding up to the alignment must not wrap around to a small size
    if (size > SIZE_MAX - allocation_alignment)
        return nullptr;

    void *ptr = std::aligned_alloc(allocation_alignment,
                                   (size + allocation_alignment - 1) / allocation_alignment * allocation_alignment);
    if (ptr == nullptr)
        return nullptr;

    // Memory that a program registered and then freed without unregistering
    // may come back: those registrations are stale, as are hints on it
    auto &s = state();
    const uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
    std::lock_guard lock(s.mutex);
    while (overlaps(s, p, p + size)) {
        auto it = s.allocations.lower_bound(p);
        if (it == s.allocations.end() || it->first >= p + size)
            --it;
        s.allocations.erase(it);
    }
    erase_hints(s, p, p + size);
    s.allocations.emplace(p, allocation{size, kind, flags});
    return ptr;
}

void register_symbol(const void *ptr, size_t size)
{
    auto &s = state();
    const uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
    std::lock_guard lock(s.mutex);
    s.allocations.try_emplace(p, allocation{size, memory_kind::symbol, 0});
}

namespace {

//! CUDA4CPU_MEMCPY_DIRECTION
enum class direction_check { warn, error, ignore };

direction_check direction_setting()
{
    static const direction_check setting = [] {
        const char *v = std::getenv("CUDA4CPU_MEMCPY_DIRECTION");
        if (v == nullptr || *v == '\0' || std::string_view(v) == "warn")
            return direction_check::warn;
        if (std::string_view(v) == "error")
            return direction_check::error;
        if (std::string_view(v) == "ignore")
            return direction_check::ignore;
        std::fprintf(stderr, "cuda4cpu: ignoring CUDA4CPU_MEMCPY_DIRECTION=%s: expected warn, error or ignore\n", v);
        return direction_check::warn;
    }();
    return setting;
}

//! Where a pointer points, for the direction check
enum class pointer_place { host, device, either };

pointer_place place(const void *ptr)
{
    allocation_info info;
    if (!find_allocation(ptr, info))
        return pointer_place::host;
    switch (info.kind) {
    case memory_kind::device:
    case memory_kind::mapped:
    case memory_kind::symbol:
        return pointer_place::device;
    default:
        return pointer_place::either;   // Pinned, registered and managed memory
    }
}

const char *place_name(pointer_place p)
{
    return p == pointer_place::host ? "host memory that CUDA didn't allocate" : "device memory";
}

const char *kind_name(cudaMemcpyKind kind)
{
    switch (kind) {
    case cudaMemcpyHostToHost:     return "cudaMemcpyHostToHost";
    case cudaMemcpyHostToDevice:   return "cudaMemcpyHostToDevice";
    case cudaMemcpyDeviceToHost:   return "cudaMemcpyDeviceToHost";
    default:                       return "cudaMemcpyDeviceToDevice";
    }
}

[[gnu::noinline, gnu::cold]] void report_copy_direction(cudaMemcpyKind kind, const char *side, pointer_place p)
{
    static std::mutex lock;
    static std::set<std::tuple<int, std::string_view, int>> reported;

    std::lock_guard guard(lock);
    if (!reported.emplace(int(kind), side, int(p)).second)
        return;
    std::fprintf(stderr,
                 "cuda4cpu: warning: a copy with %s has its %s in %s, which CUDA leaves undefined. "
                 "The copy goes ahead; CUDA4CPU_MEMCPY_DIRECTION=error makes it fail with "
                 "cudaErrorInvalidMemcpyDirection.\n",
                 kind_name(kind), side, place_name(p));
}

}

bool check_copy_direction(const void *dst, const void *src, cudaMemcpyKind kind)
{
    if (kind < cudaMemcpyHostToHost || kind > cudaMemcpyDefault)
        return false;
    const direction_check setting = direction_setting();
    if (kind == cudaMemcpyDefault || setting == direction_check::ignore)
        return true;

    const bool dst_device = kind == cudaMemcpyHostToDevice || kind == cudaMemcpyDeviceToDevice;
    const bool src_device = kind == cudaMemcpyDeviceToHost || kind == cudaMemcpyDeviceToDevice;
    const pointer_place dst_place = place(dst), src_place = place(src);
    // The place that a side mustn't be in
    const pointer_place dst_wrong = dst_device ? pointer_place::host : pointer_place::device;
    const pointer_place src_wrong = src_device ? pointer_place::host : pointer_place::device;
    if (dst_place != dst_wrong && src_place != src_wrong)
        return true;
    if (setting == direction_check::error)
        return false;
    if (dst_place == dst_wrong)
        report_copy_direction(kind, "destination", dst_place);
    if (src_place == src_wrong)
        report_copy_direction(kind, "source", src_place);
    return true;
}

bool release_allocation(void *ptr)
{
    auto &s = state();
    {
        std::lock_guard lock(s.mutex);
        auto it = s.allocations.find(reinterpret_cast<uintptr_t>(ptr));
        if (it == s.allocations.end() || it->second.kind == memory_kind::registered ||
            it->second.kind == memory_kind::mapped || it->second.kind == memory_kind::symbol)
            return false;
        s.allocations.erase(it);
    }
    std::free(ptr);
    return true;
}

bool find_allocation(const void *ptr, allocation_info &info)
{
    auto &s = state();
    {
        std::lock_guard lock(s.mutex);
        auto it = containing(s, reinterpret_cast<uintptr_t>(ptr));
        if (it != s.allocations.end()) {
            info = {reinterpret_cast<void *>(it->first), it->second.size, it->second.kind, it->second.flags};
            return true;
        }
    }
    return find_graph_allocation(ptr, info);
}

cudaError_t register_host(void *ptr, size_t size, unsigned flags)
{
    constexpr unsigned known = cudaHostRegisterPortable | cudaHostRegisterMapped | cudaHostRegisterIoMemory |
                               cudaHostRegisterReadOnly;
    if (ptr == nullptr || size == 0 || (flags & ~known) != 0)
        return record_error(cudaErrorInvalidValue);

    auto &s = state();
    const uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
    std::lock_guard lock(s.mutex);
    if (overlaps(s, p, p + size))
        return record_error(cudaErrorHostMemoryAlreadyRegistered);
    s.allocations.emplace(p, allocation{size, memory_kind::registered, flags});
    return cudaSuccess;
}

cudaError_t unregister_host(void *ptr)
{
    auto &s = state();
    std::lock_guard lock(s.mutex);
    auto it = s.allocations.find(reinterpret_cast<uintptr_t>(ptr));
    if (it == s.allocations.end() || it->second.kind != memory_kind::registered)
        return record_error(cudaErrorHostMemoryNotRegistered);
    s.allocations.erase(it);
    return cudaSuccess;
}

//
// Unified memory hints
//

cudaError_t advise(const void *ptr, size_t count, cudaMemoryAdvise advice, cudaMemLocation location)
{
    if (count == 0)
        return cudaSuccess;
    if (ptr == nullptr)
        return record_error(cudaErrorInvalidValue);

    switch (advice) {
    case cudaMemAdviseSetReadMostly:
    case cudaMemAdviseUnsetReadMostly:
        break;
    case cudaMemAdviseSetPreferredLocation:
    case cudaMemAdviseSetAccessedBy:
    case cudaMemAdviseUnsetAccessedBy:
        if (cudaError_t err = check_location(location))
            return record_error(err);
        if ((advice == cudaMemAdviseSetAccessedBy || advice == cudaMemAdviseUnsetAccessedBy) &&
            location.type != cudaMemLocationTypeDevice && location.type != cudaMemLocationTypeHost)
            return record_error(cudaErrorInvalidValue);
        break;
    case cudaMemAdviseUnsetPreferredLocation:
        break;
    default:
        return record_error(cudaErrorInvalidValue);
    }

    const unsigned char who = location.type == cudaMemLocationTypeDevice ? 1 : 2;
    auto &s = state();
    const auto [begin, end] = page_range(ptr, count);
    std::lock_guard lock(s.mutex);
    update_hints(s, begin, end, [&](hints &h) {
        switch (advice) {
        case cudaMemAdviseSetReadMostly:          h.read_mostly = true; break;
        case cudaMemAdviseUnsetReadMostly:        h.read_mostly = false; break;
        case cudaMemAdviseSetPreferredLocation:   h.preferred = location; break;
        case cudaMemAdviseUnsetPreferredLocation: h.preferred = {cudaMemLocationTypeInvalid, 0}; break;
        case cudaMemAdviseSetAccessedBy:          h.accessed_by |= who; break;
        case cudaMemAdviseUnsetAccessedBy:        h.accessed_by &= ~who; break;
        }
    });
    return cudaSuccess;
}

cudaError_t prefetch(const void *ptr, size_t count, cudaMemLocation location)
{
    if (count == 0)
        return cudaSuccess;
    if (ptr == nullptr)
        return record_error(cudaErrorInvalidValue);
    if (cudaError_t err = check_location(location))
        return record_error(err);

    auto &s = state();
    const auto [begin, end] = page_range(ptr, count);
    std::lock_guard lock(s.mutex);
    update_hints(s, begin, end, [&](hints &h) { h.prefetched = location; });
    return cudaSuccess;
}

cudaError_t range_attribute(void *data, size_t data_size, cudaMemRangeAttribute attribute, const void *ptr,
                            size_t count)
{
    if (data == nullptr || ptr == nullptr || count == 0)
        return record_error(cudaErrorInvalidValue);
    const bool array = attribute == cudaMemRangeAttributeAccessedBy;
    if (array ? data_size == 0 || data_size % sizeof(int) != 0 : data_size != sizeof(int))
        return record_error(cudaErrorInvalidValue);

    auto &s = state();
    const auto [begin, end] = page_range(ptr, count);
    std::lock_guard lock(s.mutex);
    const auto preferred = [](const hints &h) { return h.preferred; };
    const auto prefetched = [](const hints &h) { return h.prefetched; };
    int *out = static_cast<int *>(data);
    switch (attribute) {
    case cudaMemRangeAttributeReadMostly: {
        bool all = true;
        visit_hints(s, begin, end, [&](const hints &h) { all = all && h.read_mostly; });
        *out = all ? 1 : 0;
        break;
    }
    case cudaMemRangeAttributePreferredLocation:
        *out = device_number(common_location(s, begin, end, preferred));
        break;
    case cudaMemRangeAttributeLastPrefetchLocation:
        *out = device_number(common_location(s, begin, end, prefetched));
        break;
    case cudaMemRangeAttributePreferredLocationType:
        *out = common_location(s, begin, end, preferred).type;
        break;
    case cudaMemRangeAttributeLastPrefetchLocationType:
        *out = common_location(s, begin, end, prefetched).type;
        break;
    case cudaMemRangeAttributePreferredLocationId:
    case cudaMemRangeAttributeLastPrefetchLocationId: {
        const cudaMemLocation l = attribute == cudaMemRangeAttributePreferredLocationId
                                      ? common_location(s, begin, end, preferred)
                                      : common_location(s, begin, end, prefetched);
        const bool has_id = l.type == cudaMemLocationTypeDevice || l.type == cudaMemLocationTypeHostNuma;
        *out = has_id ? l.id : cudaInvalidDeviceId;
        break;
    }
    case cudaMemRangeAttributeAccessedBy: {
        // The devices with the hint on every page, then cudaInvalidDeviceId
        unsigned char all = 3;
        visit_hints(s, begin, end, [&](const hints &h) { all &= h.accessed_by; });
        const size_t n = data_size / sizeof(int);
        size_t i = 0;
        if ((all & 1) != 0 && i < n)
            out[i++] = 0;
        if ((all & 2) != 0 && i < n)
            out[i++] = cudaCpuDeviceId;
        while (i < n)
            out[i++] = cudaInvalidDeviceId;
        break;
    }
    default:
        return record_error(cudaErrorInvalidValue);
    }
    return cudaSuccess;
}

//
// Virtual memory management, for HIP's hipMem* functions
//

size_t vmm_granularity()
{
    return page_size();
}

cudaError_t vmm_create(vmm_allocation **handle, size_t size, const void *properties, size_t properties_size)
{
    if (handle == nullptr || size == 0 || size % page_size() != 0 ||
        properties_size > sizeof(vmm_allocation::properties))
        return record_error(cudaErrorInvalidValue);
    const int fd = memfd_create("cuda4cpu", MFD_CLOEXEC);
    if (fd < 0)
        return record_error(cudaErrorMemoryAllocation);
    if (ftruncate(fd, off_t(size)) != 0) {
        close(fd);
        return record_error(cudaErrorMemoryAllocation);
    }
    auto *a = new vmm_allocation{fd, size, {}, 1};
    std::memcpy(a->properties.data(), properties, properties_size);
    *handle = a;
    return cudaSuccess;
}

cudaError_t vmm_import_fd(vmm_allocation **handle, int fd, const void *properties, size_t properties_size)
{
    struct stat st;
    if (handle == nullptr || properties_size > sizeof(vmm_allocation::properties) || fstat(fd, &st) != 0 ||
        st.st_size <= 0)
        return record_error(cudaErrorInvalidValue);
    const int own = fcntl(fd, F_DUPFD_CLOEXEC, 0);
    if (own < 0)
        return record_error(cudaErrorInvalidValue);
    auto *a = new vmm_allocation{own, size_t(st.st_size), {}, 1};
    std::memcpy(a->properties.data(), properties, properties_size);
    *handle = a;
    return cudaSuccess;
}

cudaError_t vmm_export_fd(int *fd, vmm_allocation *handle)
{
    if (fd == nullptr || handle == nullptr)
        return record_error(cudaErrorInvalidValue);
    *fd = dup(handle->fd);
    return *fd < 0 ? record_error(cudaErrorInvalidValue) : cudaSuccess;
}

cudaError_t vmm_properties(void *properties, size_t properties_size, vmm_allocation *handle)
{
    if (properties == nullptr || handle == nullptr || properties_size > sizeof(vmm_allocation::properties))
        return record_error(cudaErrorInvalidValue);
    std::memcpy(properties, handle->properties.data(), properties_size);
    return cudaSuccess;
}

cudaError_t vmm_release(vmm_allocation *handle)
{
    if (handle == nullptr)
        return record_error(cudaErrorInvalidValue);
    auto &s = state();
    std::lock_guard lock(s.mutex);
    drop(handle);
    return cudaSuccess;
}

cudaError_t vmm_retain(vmm_allocation **handle, void *addr)
{
    if (handle == nullptr)
        return record_error(cudaErrorInvalidValue);
    auto &s = state();
    std::lock_guard lock(s.mutex);
    auto it = mapping_at(s, reinterpret_cast<uintptr_t>(addr));
    if (it == s.allocations.end())
        return record_error(cudaErrorInvalidValue);
    ++it->second.physical->refs;
    *handle = it->second.physical;
    return cudaSuccess;
}

cudaError_t vmm_reserve(void **ptr, size_t size, size_t alignment, void *addr)
{
    const size_t page = page_size();
    if (alignment == 0)
        alignment = page;
    if (ptr == nullptr || size == 0 || size % page != 0 || (alignment & (alignment - 1)) != 0 ||
        alignment % page != 0)
        return record_error(cudaErrorInvalidValue);

    // Reserve enough to align the start, then return the excess
    const size_t extra = alignment - page;
    void *raw = mmap(addr, size + extra, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (raw == MAP_FAILED)
        return record_error(cudaErrorMemoryAllocation);
    const uintptr_t start = (reinterpret_cast<uintptr_t>(raw) + alignment - 1) / alignment * alignment;
    if (start > reinterpret_cast<uintptr_t>(raw))
        munmap(raw, start - reinterpret_cast<uintptr_t>(raw));
    if (const size_t tail = reinterpret_cast<uintptr_t>(raw) + size + extra - (start + size); tail > 0)
        munmap(reinterpret_cast<void *>(start + size), tail);

    auto &s = state();
    std::lock_guard lock(s.mutex);
    s.reservations.emplace(start, size);
    *ptr = reinterpret_cast<void *>(start);
    return cudaSuccess;
}

cudaError_t vmm_free(void *ptr, size_t size)
{
    auto &s = state();
    const uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
    std::lock_guard lock(s.mutex);
    auto r = s.reservations.find(p);
    if (r == s.reservations.end() || r->second != size)
        return record_error(cudaErrorInvalidValue);

    // Mappings left in the range go with it
    for (auto it = s.allocations.lower_bound(p); it != s.allocations.end() && it->first < p + size;) {
        auto next = std::next(it);
        if (it->second.kind == memory_kind::mapped) {
            erase_hints(s, it->first, it->first + it->second.size);
            drop(it->second.physical);
            s.allocations.erase(it);
        }
        it = next;
    }
    munmap(ptr, size);
    s.reservations.erase(r);
    return cudaSuccess;
}

cudaError_t vmm_map(void *ptr, size_t size, size_t offset, vmm_allocation *handle)
{
    const size_t page = page_size();
    const uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
    if (handle == nullptr || size == 0 || p % page != 0 || size % page != 0 || offset % page != 0 ||
        offset > handle->size || size > handle->size - offset)
        return record_error(cudaErrorInvalidValue);

    auto &s = state();
    std::lock_guard lock(s.mutex);
    auto r = s.reservations.upper_bound(p);
    if (r == s.reservations.begin() || (--r, p + size > r->first + r->second) || overlaps(s, p, p + size))
        return record_error(cudaErrorInvalidValue);
    if (mmap(ptr, size, PROT_NONE, MAP_SHARED | MAP_FIXED, handle->fd, off_t(offset)) == MAP_FAILED)
        return record_error(cudaErrorMemoryAllocation);
    erase_hints(s, p, p + size);
    s.allocations.emplace(p, allocation{size, memory_kind::mapped, 0, handle});
    ++handle->refs;
    return cudaSuccess;
}

cudaError_t vmm_unmap(void *ptr, size_t size)
{
    auto &s = state();
    const uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
    std::lock_guard lock(s.mutex);
    // Whole mappings only: the range starts and ends where mappings do
    if (size == 0 || !mapped(s, p, p + size) || mapping_at(s, p)->first != p)
        return record_error(cudaErrorInvalidValue);
    auto last = mapping_at(s, p + size - 1);
    if (last->first + last->second.size != p + size)
        return record_error(cudaErrorInvalidValue);
    // Mappings that cover a range without gaps start where the previous ends
    for (uintptr_t cur = p; cur < p + size;) {
        auto it = s.allocations.find(cur);
        cur += it->second.size;
        unmap(s, it);
    }
    return cudaSuccess;
}

cudaError_t vmm_set_access(void *ptr, size_t size, cudaMemAccessFlags flags)
{
    int prot;
    switch (flags) {
    case cudaMemAccessFlagsProtNone:      prot = PROT_NONE; break;
    case cudaMemAccessFlagsProtRead:      prot = PROT_READ; break;
    case cudaMemAccessFlagsProtReadWrite: prot = PROT_READ | PROT_WRITE; break;
    default:                              return record_error(cudaErrorInvalidValue);
    }

    auto &s = state();
    const uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
    std::lock_guard lock(s.mutex);
    if (size == 0 || !mapped(s, p, p + size) || mprotect(ptr, size, prot) != 0)
        return record_error(cudaErrorInvalidValue);
    for (uintptr_t cur = p; cur < p + size;) {
        auto it = mapping_at(s, cur);
        it->second.access = flags;
        cur = it->first + it->second.size;
    }
    return cudaSuccess;
}

cudaError_t vmm_get_access(unsigned long long *flags, const void *ptr)
{
    if (flags == nullptr)
        return record_error(cudaErrorInvalidValue);
    auto &s = state();
    std::lock_guard lock(s.mutex);
    auto it = mapping_at(s, reinterpret_cast<uintptr_t>(ptr));
    if (it == s.allocations.end())
        return record_error(cudaErrorInvalidValue);
    *flags = it->second.access;
    return cudaSuccess;
}

}
