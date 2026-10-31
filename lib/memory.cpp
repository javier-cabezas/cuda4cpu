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
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include <cxxabi.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <fcntl.h>
#include <link.h>
#include <spawn.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include "cuda4cpu.hpp"

// AddressSanitizer's interface, called only when the program is built with
// it, so that the same build of the library works without: allocations are
// rounded up to allocation_alignment, and the rest is poisoned, so that a
// kernel that runs a few elements past the end is reported
extern "C" {
[[gnu::weak]] void __asan_poison_memory_region(void const volatile *addr, size_t size);
[[gnu::weak]] void __asan_unpoison_memory_region(void const volatile *addr, size_t size);
}

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

//! The return addresses of the calls that led to an allocation, innermost
//! first. Each stack is recorded once, however many allocations it made.
struct call_stack {
    std::vector<const void *> frames;
    auto operator<=>(const call_stack &) const = default;
};

//! With CUDA4CPU_LEAK_CHECK, the calls that led to this one; nullptr without
const call_stack *record_call_stack();

//! The size and call stack of each graph allocation that is allocated now
std::vector<std::pair<size_t, const call_stack *>> live_graph_allocations();

namespace {

struct allocation {
    size_t size;
    memory_kind kind;
    unsigned flags;
    vmm_allocation *physical = nullptr;   //!< Of a mapped range
    cudaMemAccessFlags access = cudaMemAccessFlagsProtNone;
    const call_stack *stack = nullptr;    //!< The calls that allocated it, with CUDA4CPU_LEAK_CHECK
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

namespace {

//! CUDA4CPU_INIT_MEMORY: the byte that new allocations are filled with, or -1
//! to leave them as the allocator returns them
int init_memory_setting()
{
    static const int setting = [] {
        const char *v = std::getenv("CUDA4CPU_INIT_MEMORY");
        if (v == nullptr || *v == '\0')
            return -1;
        char *end = nullptr;
        const unsigned long value = std::strtoul(v, &end, 0);
        if (*end != '\0' || value > 0xff) {
            std::fprintf(stderr, "cuda4cpu: ignoring CUDA4CPU_INIT_MEMORY=%s: expected a byte, such as 0xff\n", v);
            return -1;
        }
        return int(value);
    }();
    return setting;
}

//! CUDA4CPU_LEAK_CHECK: whether the allocations left at exit are reported
bool leak_check_setting()
{
    static const bool setting = [] {
        const char *v = std::getenv("CUDA4CPU_LEAK_CHECK");
        if (v == nullptr || *v == '\0' || std::string_view(v) == "ignore")
            return false;
        if (std::string_view(v) == "warn")
            return true;
        std::fprintf(stderr, "cuda4cpu: ignoring CUDA4CPU_LEAK_CHECK=%s: expected warn or ignore\n", v);
        return false;
    }();
    return setting;
}

}

void initialize_allocation(void *ptr, size_t size)
{
    if (const int byte = init_memory_setting(); byte >= 0)
        fill_rows(ptr, size, unsigned(byte), 1, size, 1);
}

const call_stack *record_call_stack()
{
    if (!leak_check_setting())
        return nullptr;
    void *frames[32];
    const int depth = backtrace(frames, int(std::size(frames)));
    // Never destroyed, like the allocations that point to them
    struct recorded_stacks {
        std::mutex mutex;
        std::set<call_stack> stacks;
    };
    static recorded_stacks &recorded = *new recorded_stacks;
    std::lock_guard lock(recorded.mutex);
    // Without this function's own frame
    return &*recorded.stacks.insert(call_stack{std::vector<const void *>(frames + std::min(depth, 1), frames + depth)})
                 .first;
}

//
// The registry of allocations
//

void *allocate_aligned(size_t size, memory_kind kind, unsigned flags)
{
    // Rounding up to the alignment must not wrap around to a small size
    if (size > SIZE_MAX - allocation_alignment)
        return nullptr;

    const size_t rounded = (size + allocation_alignment - 1) / allocation_alignment * allocation_alignment;
    void *ptr = std::aligned_alloc(allocation_alignment, rounded);
    if (ptr == nullptr)
        return nullptr;
    if (__asan_poison_memory_region != nullptr && rounded > size)
        __asan_poison_memory_region(static_cast<char *>(ptr) + size, rounded - size);
    initialize_allocation(ptr, size);
    const call_stack *stack = record_call_stack();

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
    s.allocations.emplace(p, allocation{size, kind, flags, nullptr, cudaMemAccessFlagsProtNone, stack});
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
    size_t size;
    {
        std::lock_guard lock(s.mutex);
        auto it = s.allocations.find(reinterpret_cast<uintptr_t>(ptr));
        if (it == s.allocations.end() || it->second.kind == memory_kind::registered ||
            it->second.kind == memory_kind::mapped || it->second.kind == memory_kind::symbol)
            return false;
        size = it->second.size;
        s.allocations.erase(it);
    }
    if (__asan_unpoison_memory_region != nullptr)
        __asan_unpoison_memory_region(ptr, (size + allocation_alignment - 1) / allocation_alignment * allocation_alignment);
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

namespace {

const char *memory_name(memory_kind kind)
{
    switch (kind) {
    case memory_kind::host:    return "pinned host memory";
    case memory_kind::managed: return "managed memory";
    default:                   return "device memory";
    }
}

//! A call, as the debug information or the symbol tables name it
struct source_frame {
    std::string function;   //!< Empty if unknown
    std::string file;       //!< Empty if unknown
    unsigned line = 0;
};

//! The object file (the program or a shared library) that an address is in,
//! and the address in that file
struct object_address {
    std::string object;   //!< Empty if no object has the address
    uintptr_t offset = 0;
    bool program = false;   //!< The object is the program, not a shared library
};

object_address object_of(const void *address)
{
    struct search {
        uintptr_t address;
        object_address found;
    } s{reinterpret_cast<uintptr_t>(address), {}};
    dl_iterate_phdr([](dl_phdr_info *info, size_t, void *data) {
        auto &s = *static_cast<search *>(data);
        for (int i = 0; i < info->dlpi_phnum; ++i) {
            const ElfW(Phdr) &h = info->dlpi_phdr[i];
            if (h.p_type == PT_LOAD && s.address - (info->dlpi_addr + h.p_vaddr) < h.p_memsz) {
                // The program is the only object without a name
                s.found = {info->dlpi_name, s.address - info->dlpi_addr, *info->dlpi_name == '\0'};
                return 1;
            }
        }
        return 0;
    }, &s);
    if (s.found.program) {
        char path[4096];
        const ssize_t n = readlink("/proc/self/exe", path, sizeof path);
        if (n > 0 && size_t(n) < sizeof path)
            s.found.object.assign(path, size_t(n));
    }
    return s.found;
}

//! What addr2line prints for offsets into an object file: each offset, then
//! the function and file:line of the call there, followed by those of the
//! calls it was inlined into. Empty if addr2line can't run.
std::string run_addr2line(const std::string &object, const std::vector<uintptr_t> &offsets)
{
    std::vector<std::string> args{"addr2line", "-a", "-f", "-C", "-i", "-e", object};
    for (uintptr_t offset : offsets) {
        char hex[32];
        std::snprintf(hex, sizeof hex, "%#zx", size_t(offset));
        args.emplace_back(hex);
    }
    std::vector<char *> argv;
    for (std::string &a : args)
        argv.push_back(a.data());
    argv.push_back(nullptr);

    int out[2];
    if (pipe2(out, O_CLOEXEC) != 0)
        return {};
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_adddup2(&actions, out[1], STDOUT_FILENO);
    posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, "/dev/null", O_WRONLY, 0);
    pid_t pid;
    const int failed = posix_spawnp(&pid, "addr2line", &actions, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    close(out[1]);
    std::string output;
    if (failed == 0) {
        char buffer[4096];
        for (;;) {
            const ssize_t n = read(out[0], buffer, sizeof buffer);
            if (n > 0)
                output.append(buffer, size_t(n));
            else if (n == 0 || errno != EINTR)
                break;
        }
        while (waitpid(pid, nullptr, 0) < 0 && errno == EINTR) {
        }
    }
    close(out[0]);
    return output;
}

//! The calls at each return address, innermost (inlined) first: from the
//! debug information through addr2line, or from the dynamic symbols
std::map<const void *, std::vector<source_frame>> symbolize(const std::set<const void *> &addresses)
{
    std::map<const void *, std::vector<source_frame>> frames;
    std::map<std::string, std::vector<std::pair<const void *, uintptr_t>>> by_object;
    for (const void *address : addresses) {
        // The call is just before the address it returns to
        const object_address at = object_of(static_cast<const char *>(address) - 1);
        if (!at.object.empty())
            by_object[at.object].emplace_back(address, at.offset);
    }
    // In batches, so that the command line stays well below its limit
    constexpr size_t batch = 1024;
    for (const auto &[object, located] : by_object) {
        for (size_t first = 0; first < located.size(); first += batch) {
            const size_t last = std::min(first + batch, located.size());
            std::vector<uintptr_t> offsets;
            for (size_t i = first; i < last; ++i)
                offsets.push_back(located[i].second);
            const std::string output = run_addr2line(object, offsets);
            std::vector<source_frame> *current = nullptr;
            size_t next = first;
            bool function_next = true;
            for (size_t begin = 0, end; begin < output.size(); begin = end + 1) {
                end = std::min(output.find('\n', begin), output.size());
                std::string_view line(output.data() + begin, end - begin);
                if (line.starts_with("0x")) {
                    current = next < last ? &frames[located[next++].first] : nullptr;
                    function_next = true;
                } else if (current != nullptr && function_next) {
                    current->push_back({line == "??" ? std::string() : std::string(line)});
                    function_next = false;
                } else if (current != nullptr) {
                    // file:line, maybe followed by " (discriminator N)"
                    line = line.substr(0, line.find(" ("));
                    const size_t colon = line.rfind(':');
                    if (colon != std::string_view::npos && line.substr(0, colon) != "??") {
                        current->back().file = line.substr(0, colon);
                        current->back().line = unsigned(std::strtoul(std::string(line.substr(colon + 1)).c_str(),
                                                                     nullptr, 10));
                    }
                    function_next = true;
                }
            }
        }
    }
    for (const void *address : addresses) {
        // addr2line's "??" for both: nothing is known
        std::vector<source_frame> &calls = frames[address];
        std::erase_if(calls, [](const source_frame &call) { return call.function.empty() && call.file.empty(); });
        if (!calls.empty())
            continue;
        Dl_info info;
        if (dladdr(static_cast<const char *>(address) - 1, &info) == 0 || info.dli_sname == nullptr)
            continue;
        int status = 0;
        char *demangled = abi::__cxa_demangle(info.dli_sname, nullptr, nullptr, &status);
        calls.assign(1, {demangled != nullptr ? demangled : info.dli_sname});
        std::free(demangled);
    }
    return frames;
}

//! Whether a file is one of cuda4cpu's headers that define allocation
//! functions. The debug information names inlined and static functions
//! without their namespace, so the file tells them apart. By the path's
//! end: the headers may have moved since, and -ffile-prefix-map changes
//! the start.
bool cuda4cpu_header(std::string_view file)
{
    for (std::string_view header : {"cuda/memory.hpp", "hip/hip_runtime_api.h"}) {
        if (file.ends_with(header) &&
            (file.size() == header.size() || file[file.size() - header.size() - 1] == '/'))
            return true;
    }
    return false;
}

//! Where the program made an allocation, for the report: "at file:line",
//! "in function" or "at object+offset", and whether it has the line
struct allocation_site {
    std::string where;
    bool line = false;
};

//! The innermost call that isn't cuda4cpu's own
allocation_site site_of(const call_stack *stack, const std::map<const void *, std::vector<source_frame>> &frames)
{
    if (stack == nullptr)
        return {"at an unknown place"};
    const object_address library = object_of(reinterpret_cast<const void *>(&record_call_stack));
    for (const void *address : stack->frames) {
        const std::vector<source_frame> &calls = frames.at(address);
        // In a function of the library, and in what it inlined
        if (!calls.empty() && calls.back().function.starts_with("cuda4cpu::"))
            continue;
        for (const source_frame &call : calls) {
            if (call.function.starts_with("cuda4cpu::") || cuda4cpu_header(call.file))
                continue;
            if (!call.file.empty() && call.line != 0)
                return {"at " + call.file + ":" + std::to_string(call.line), true};
            if (!call.function.empty())
                return {"in " + call.function};
        }
        if (calls.empty()) {
            // Nothing is known about it: skip cuda4cpu's shared library.
            // Linked into the program, the library's frames can't be told
            // apart from the program's.
            const object_address at = object_of(static_cast<const char *>(address) - 1);
            if (!library.program && at.object != library.object) {
                char offset[32];
                std::snprintf(offset, sizeof offset, "+%#zx", size_t(at.offset));
                return {"at " + at.object + offset};
            }
        }
    }
    return {"at an unknown place"};
}

//! Reports the allocations that the program never freed, by the call that
//! made them, largest first. A destructor of the library runs after the
//! program's static objects are destroyed, which may free memory, and its
//! priority puts it after the program's own destructor functions when the
//! library is linked statically: .fini_array runs them in reverse, those
//! with a priority after those without.
[[gnu::destructor(101)]] void report_leaks()
{
    if (!leak_check_setting())
        return;
    struct leak {
        size_t size;
        const call_stack *stack;
        const char *what;
    };
    std::vector<leak> left;
    {
        auto &s = state();
        std::lock_guard lock(s.mutex);
        for (const auto &[p, a] : s.allocations) {
            if (a.kind == memory_kind::device || a.kind == memory_kind::host || a.kind == memory_kind::managed)
                left.push_back({a.size, a.stack, memory_name(a.kind)});
        }
    }
    for (const auto &[size, stack] : live_graph_allocations())
        left.push_back({size, stack, "graph memory"});
    if (left.empty())
        return;

    std::set<const void *> addresses;
    for (const leak &l : left) {
        if (l.stack != nullptr)
            addresses.insert(l.stack->frames.begin(), l.stack->frames.end());
    }
    const auto frames = symbolize(addresses);

    struct site_leaks {
        std::string where;
        bool line;
        const char *what;
        size_t count = 0, bytes = 0;
    };
    std::vector<site_leaks> leaks;
    std::map<const call_stack *, allocation_site> sites;
    std::map<std::pair<std::string, const char *>, size_t> index;
    for (const leak &l : left) {
        auto site = sites.find(l.stack);
        if (site == sites.end())
            site = sites.emplace(l.stack, site_of(l.stack, frames)).first;
        auto [it, added] = index.try_emplace({site->second.where, l.what}, leaks.size());
        if (added)
            leaks.push_back({site->second.where, site->second.line, l.what});
        ++leaks[it->second].count;
        leaks[it->second].bytes += l.size;
    }
    std::ranges::stable_sort(leaks, std::greater{}, &site_leaks::bytes);

    constexpr size_t shown = 20;
    bool without_lines = false;
    for (size_t i = 0; i < std::min(leaks.size(), shown); ++i) {
        const site_leaks &l = leaks[i];
        without_lines |= !l.line;
        if (l.count == 1)
            std::fprintf(stderr, "cuda4cpu: warning: an allocation of %s (%zu bytes), made %s, was never freed.\n",
                         l.what, l.bytes, l.where.c_str());
        else
            std::fprintf(stderr,
                         "cuda4cpu: warning: %zu allocations of %s (%zu bytes in all), made %s, were never freed.\n",
                         l.count, l.what, l.bytes, l.where.c_str());
    }
    if (leaks.size() > shown) {
        size_t count = 0, bytes = 0;
        for (size_t i = shown; i < leaks.size(); ++i) {
            count += leaks[i].count;
            bytes += leaks[i].bytes;
        }
        std::fprintf(stderr, "cuda4cpu: warning: %zu more allocations (%zu bytes in all), made at %zu other places, "
                             "were never freed.\n", count, bytes, leaks.size() - shown);
    }
    if (without_lines)
        std::fprintf(stderr, "cuda4cpu: note: build the program with -g, and install addr2line (binutils), "
                             "to see the lines that made the allocations.\n");
}

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
