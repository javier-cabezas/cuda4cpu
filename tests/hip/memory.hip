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
// HIP's host APIs for memory and contexts, through HIP's names only: pointer
// attributes of every kind of allocation, memory range attributes after
// advice and prefetches, virtual memory management (physical memory mapped
// at two addresses, access that faults until it's set, sharing through a
// file descriptor), contexts, and launches between events (hip_ext.h).
//

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <sys/wait.h>
#include <unistd.h>

static unsigned errors = 0;

#define EXPECT(cond)                                                       \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("%s:%d: expected %s\n", __FILE__, __LINE__, #cond); \
            ++errors;                                                      \
        }                                                                  \
    } while (0)

__global__ void fill(int *out, int value, int n)
{
    const int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (i < n)
        out[i] = value + i;
}

//! Whether a child process that writes to p dies with SIGSEGV
static bool write_faults(int *p)
{
    std::fflush(stdout);
    const pid_t pid = fork();
    if (pid == 0) {
        std::signal(SIGSEGV, SIG_DFL);   // AddressSanitizer would report the fault and exit
        *static_cast<volatile int *>(p) = 1;
        _exit(0);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    return WIFSIGNALED(status) && WTERMSIG(status) == SIGSEGV;
}

static void pointer_attributes()
{
    int *device = nullptr, *managed = nullptr, *host = nullptr;
    EXPECT(hipMalloc(&device, 100 * sizeof(int)) == hipSuccess);
    EXPECT(hipMallocManaged(&managed, 100 * sizeof(int)) == hipSuccess);
    EXPECT(hipHostMalloc(&host, 100 * sizeof(int), hipHostMallocMapped) == hipSuccess);

    hipPointerAttribute_t a;
    EXPECT(hipPointerGetAttributes(&a, device + 10) == hipSuccess);
    EXPECT(a.type == hipMemoryTypeDevice && a.device == 0 && a.devicePointer == device + 10 &&
           a.hostPointer == nullptr && a.isManaged == 0);
    EXPECT(hipPointerGetAttributes(&a, managed) == hipSuccess);
    EXPECT(a.type == hipMemoryTypeManaged && a.isManaged == 1 && a.devicePointer == managed &&
           a.hostPointer == managed && a.allocationFlags == hipMemAttachGlobal);
    EXPECT(hipPointerGetAttributes(&a, host + 99) == hipSuccess);
    EXPECT(a.type == hipMemoryTypeHost && a.hostPointer == host + 99 && a.allocationFlags == hipHostMallocMapped);
    unsigned flags = 0;
    EXPECT(hipHostGetFlags(&flags, host) == hipSuccess && flags == hipHostMallocMapped);
    EXPECT(hipHostGetFlags(&flags, device) == hipErrorInvalidValue);

    // Allocation size and range, from any pointer into it
    size_t size = 0;
    void *base = nullptr;
    EXPECT(hipMemPtrGetInfo(device, &size) == hipSuccess && size == 100 * sizeof(int));
    EXPECT(hipMemGetAddressRange(&base, &size, device + 50) == hipSuccess && base == device &&
           size == 100 * sizeof(int));

    // Memory that no allocation has, registered memory, and memory after it's freed
    int local[64];
    EXPECT(hipPointerGetAttributes(&a, local) == hipSuccess);
    EXPECT(a.type == hipMemoryTypeUnregistered && a.device == hipInvalidDeviceId && a.devicePointer == nullptr);
    EXPECT(hipMemPtrGetInfo(local, &size) == hipErrorInvalidValue);
    EXPECT(hipHostRegister(local, sizeof(local), hipHostRegisterMapped) == hipSuccess);
    EXPECT(hipPointerGetAttributes(&a, local + 63) == hipSuccess && a.type == hipMemoryTypeHost);
    EXPECT(hipHostRegister(local + 8, 16, 0) == hipErrorHostMemoryAlreadyRegistered);
    EXPECT(hipHostUnregister(local + 1) == hipErrorHostMemoryNotRegistered);
    EXPECT(hipHostUnregister(local) == hipSuccess);
    EXPECT(hipPointerGetAttributes(&a, local) == hipSuccess && a.type == hipMemoryTypeUnregistered);

    EXPECT(hipFree(device) == hipSuccess);
    EXPECT(hipPointerGetAttributes(&a, device) == hipSuccess && a.type == hipMemoryTypeUnregistered);
    // Freeing twice, or what hipMalloc didn't allocate, fails
    EXPECT(hipFree(device) == hipErrorInvalidValue);
    EXPECT(hipFree(local) == hipErrorInvalidValue);
    EXPECT(hipFree(managed + 1) == hipErrorInvalidValue);
    EXPECT(hipFree(nullptr) == hipSuccess);
    EXPECT(hipFree(managed) == hipSuccess);
    EXPECT(hipHostFree(host) == hipSuccess);
    hipGetLastError();
}

static void range_attributes()
{
    // 16 pages, with hints on parts of them. Hints apply to whole pages, and
    // allocations are aligned to 256 bytes, so m starts at a page boundary.
    const size_t page = size_t(sysconf(_SC_PAGESIZE)), size = 16 * page;
    char *allocation = nullptr;
    EXPECT(hipMallocManaged(&allocation, size + page) == hipSuccess);
    char *m = reinterpret_cast<char *>((reinterpret_cast<uintptr_t>(allocation) + page - 1) / page * page);
    int value = -1;
    EXPECT(hipMemRangeGetAttribute(&value, sizeof(value), hipMemRangeAttributeReadMostly, m, size) == hipSuccess &&
           value == 0);
    EXPECT(hipMemRangeGetAttribute(&value, sizeof(value), hipMemRangeAttributePreferredLocation, m, size) ==
               hipSuccess &&
           value == hipInvalidDeviceId);

    // Read-mostly pages 0 to 7, then not page 3
    EXPECT(hipMemAdvise(m, 8 * page, hipMemAdviseSetReadMostly, 0) == hipSuccess);
    EXPECT(hipMemRangeGetAttribute(&value, sizeof(value), hipMemRangeAttributeReadMostly, m, 8 * page) ==
               hipSuccess &&
           value == 1);
    EXPECT(hipMemRangeGetAttribute(&value, sizeof(value), hipMemRangeAttributeReadMostly, m, 9 * page) ==
               hipSuccess &&
           value == 0);
    EXPECT(hipMemAdvise(m + 3 * page, 1, hipMemAdviseUnsetReadMostly, 0) == hipSuccess);
    EXPECT(hipMemRangeGetAttribute(&value, sizeof(value), hipMemRangeAttributeReadMostly, m, 3 * page) ==
               hipSuccess &&
           value == 1);
    EXPECT(hipMemRangeGetAttribute(&value, sizeof(value), hipMemRangeAttributeReadMostly, m + 3 * page + 5, 10) ==
               hipSuccess &&
           value == 0);
    EXPECT(hipMemRangeGetAttribute(&value, sizeof(value), hipMemRangeAttributeReadMostly, m + 4 * page, 4 * page) ==
               hipSuccess &&
           value == 1);

    // A preferred location over part of the range: the whole range has none
    EXPECT(hipMemAdvise(m, 4 * page, hipMemAdviseSetPreferredLocation, hipCpuDeviceId) == hipSuccess);
    EXPECT(hipMemRangeGetAttribute(&value, sizeof(value), hipMemRangeAttributePreferredLocation, m, 4 * page) ==
               hipSuccess &&
           value == hipCpuDeviceId);
    EXPECT(hipMemRangeGetAttribute(&value, sizeof(value), hipMemRangeAttributePreferredLocationType, m, page) ==
               hipSuccess &&
           value == hipMemLocationTypeHost);
    EXPECT(hipMemRangeGetAttribute(&value, sizeof(value), hipMemRangeAttributePreferredLocation, m, size) ==
               hipSuccess &&
           value == hipInvalidDeviceId);
    EXPECT(hipMemAdvise(m, size, hipMemAdviseSetPreferredLocation, 0) == hipSuccess);
    EXPECT(hipMemRangeGetAttribute(&value, sizeof(value), hipMemRangeAttributePreferredLocation, m, size) ==
               hipSuccess &&
           value == 0);
    EXPECT(hipMemRangeGetAttribute(&value, sizeof(value), hipMemRangeAttributePreferredLocationId, m, size) ==
               hipSuccess &&
           value == 0);
    EXPECT(hipMemAdvise(m, page, hipMemAdviseSetPreferredLocation, 1) == hipErrorInvalidDevice);

    // Accessed by the device everywhere, and by the host in pages 0 to 1
    EXPECT(hipMemAdvise(m, size, hipMemAdviseSetAccessedBy, 0) == hipSuccess);
    EXPECT(hipMemAdvise(m, 2 * page, hipMemAdviseSetAccessedBy, hipCpuDeviceId) == hipSuccess);
    int devices[3] = {7, 7, 7};
    EXPECT(hipMemRangeGetAttribute(devices, sizeof(devices), hipMemRangeAttributeAccessedBy, m, page) ==
               hipSuccess &&
           devices[0] == 0 && devices[1] == hipCpuDeviceId && devices[2] == hipInvalidDeviceId);
    EXPECT(hipMemRangeGetAttribute(devices, sizeof(devices), hipMemRangeAttributeAccessedBy, m, size) ==
               hipSuccess &&
           devices[0] == 0 && devices[1] == hipInvalidDeviceId);
    EXPECT(hipMemAdvise(m, size, hipMemAdviseUnsetAccessedBy, 0) == hipSuccess);
    EXPECT(hipMemRangeGetAttribute(devices, sizeof(devices), hipMemRangeAttributeAccessedBy, m, page) ==
               hipSuccess &&
           devices[0] == hipCpuDeviceId && devices[1] == hipInvalidDeviceId);

    // The last prefetch, several attributes at once, and invalid queries
    EXPECT(hipMemPrefetchAsync(m, size, 0, 0) == hipSuccess);
    EXPECT(hipMemPrefetchAsync(m + page, page, hipCpuDeviceId, 0) == hipSuccess);
    int last = 7, read_mostly = 7;
    void *data[] = {&last, &read_mostly};
    size_t sizes[] = {sizeof(last), sizeof(read_mostly)};
    hipMemRangeAttribute attributes[] = {hipMemRangeAttributeLastPrefetchLocation, hipMemRangeAttributeReadMostly};
    EXPECT(hipMemRangeGetAttributes(data, sizes, attributes, 2, m + page, page) == hipSuccess &&
           last == hipCpuDeviceId && read_mostly == 1);
    EXPECT(hipMemRangeGetAttributes(data, sizes, attributes, 2, m, 2 * page) == hipSuccess &&
           last == hipInvalidDeviceId && read_mostly == 1);
    EXPECT(hipMemRangeGetAttribute(&value, 2, hipMemRangeAttributeReadMostly, m, size) == hipErrorInvalidValue);
    EXPECT(hipMemRangeGetAttribute(&value, sizeof(value), hipMemRangeAttributeReadMostly, m, 0) ==
           hipErrorInvalidValue);

    // A new allocation where hints were (usually at the same address) doesn't
    // inherit them
    char *previous = allocation;
    EXPECT(hipFree(allocation) == hipSuccess);
    EXPECT(hipMallocManaged(&allocation, size + page) == hipSuccess);
    if (allocation == previous)
        EXPECT(hipMemRangeGetAttribute(&value, sizeof(value), hipMemRangeAttributeReadMostly, m + 4 * page,
                                       4 * page) == hipSuccess &&
               value == 0);
    EXPECT(hipFree(allocation) == hipSuccess);
    hipGetLastError();
}

static void virtual_memory()
{
    int supported = 0;
    EXPECT(hipDeviceGetAttribute(&supported, hipDeviceAttributeVirtualMemoryManagementSupported, 0) == hipSuccess &&
           supported == 1);

    hipMemAllocationProp prop = {};
    prop.type                = hipMemAllocationTypePinned;
    prop.requestedHandleType = hipMemHandleTypePosixFileDescriptor;
    prop.location            = {hipMemLocationTypeDevice, 0};
    size_t granularity = 0;
    EXPECT(hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum) == hipSuccess &&
           granularity > 0);
    const size_t size = 4 * granularity;

    hipMemGenericAllocationHandle_t handle;
    EXPECT(hipMemCreate(&handle, size + 1, &prop, 0) == hipErrorInvalidValue);
    EXPECT(hipMemCreate(&handle, size, &prop, 0) == hipSuccess);
    hipMemAllocationProp got = {};
    EXPECT(hipMemGetAllocationPropertiesFromHandle(&got, handle) == hipSuccess &&
           got.type == hipMemAllocationTypePinned && got.location.type == hipMemLocationTypeDevice);

    // The same physical memory at two addresses, the second one aligned
    void *a = nullptr, *b = nullptr;
    EXPECT(hipMemAddressReserve(&a, size, 0, nullptr, 0) == hipSuccess && a != nullptr);
    EXPECT(hipMemAddressReserve(&b, 2 * size, 16 * granularity, nullptr, 0) == hipSuccess &&
           reinterpret_cast<uintptr_t>(b) % (16 * granularity) == 0);
    EXPECT(hipMemMap(a, size, 0, handle, 0) == hipSuccess);
    EXPECT(hipMemMap(a, size, 0, handle, 0) == hipErrorInvalidValue);   // already mapped
    EXPECT(hipMemMap(static_cast<char *>(b) + size, size, 0, handle, 0) == hipSuccess);
    int *pa = static_cast<int *>(a), *pb = reinterpret_cast<int *>(static_cast<char *>(b) + size);

    // No access until it's set
    EXPECT(write_faults(pa));
    unsigned long long flags = 7;
    hipMemLocation device = {hipMemLocationTypeDevice, 0};
    EXPECT(hipMemGetAccess(&flags, &device, a) == hipSuccess && flags == hipMemAccessFlagsProtNone);
    hipMemAccessDesc access = {device, hipMemAccessFlagsProtReadWrite};
    EXPECT(hipMemSetAccess(a, size, &access, 1) == hipSuccess);
    EXPECT(hipMemSetAccess(b, size, &access, 1) == hipErrorInvalidValue);   // not mapped
    EXPECT(hipMemSetAccess(pb, size, &access, 1) == hipSuccess);
    EXPECT(hipMemGetAccess(&flags, &device, pa + 5) == hipSuccess && flags == hipMemAccessFlagsProtReadWrite);

    // A kernel writes through one address, and the host reads through the other
    const int n = int(size / sizeof(int));
    hipLaunchKernelGGL(fill, dim3((n + 255) / 256), dim3(256), 0, 0, pa, 100, n);
    EXPECT(hipDeviceSynchronize() == hipSuccess);
    bool same = true;
    for (int i = 0; i < n; ++i)
        same = same && pb[i] == 100 + i;
    EXPECT(same);
    hipPointerAttribute_t attributes;
    EXPECT(hipPointerGetAttributes(&attributes, pb + 3) == hipSuccess && attributes.type == hipMemoryTypeDevice);
    EXPECT(hipFree(a) == hipErrorInvalidValue);

    // Read-only access
    hipMemAccessDesc read = {device, hipMemAccessFlagsProtRead};
    EXPECT(hipMemSetAccess(pb, size, &read, 1) == hipSuccess);
    EXPECT(pb[1] == 101 && write_faults(pb));

    // A handle from a mapped address, and memory shared through a file descriptor
    hipMemGenericAllocationHandle_t retained, imported;
    EXPECT(hipMemRetainAllocationHandle(&retained, pa + 1) == hipSuccess && retained == handle);
    EXPECT(hipMemRelease(retained) == hipSuccess);
    int fd = -1;
    EXPECT(hipMemExportToShareableHandle(&fd, handle, hipMemHandleTypePosixFileDescriptor, 0) == hipSuccess &&
           fd >= 0);
    EXPECT(hipMemImportFromShareableHandle(&imported, reinterpret_cast<void *>(intptr_t(fd)),
                                           hipMemHandleTypePosixFileDescriptor) == hipSuccess);
    close(fd);

    // Unmapping whole mappings only; the address stays reserved for another one
    EXPECT(hipMemUnmap(a, granularity) == hipErrorInvalidValue);
    EXPECT(hipMemUnmap(a, size) == hipSuccess);
    EXPECT(hipMemRelease(handle) == hipSuccess);   // still mapped at pb
    EXPECT(pb[2] == 102);
    EXPECT(hipMemMap(a, size, 0, imported, 0) == hipSuccess);
    EXPECT(hipMemSetAccess(a, size, &access, 1) == hipSuccess);
    EXPECT(pa[3] == 103);
    pa[3] = 7;
    EXPECT(pb[3] == 7);

    EXPECT(hipMemUnmap(a, size) == hipSuccess);
    EXPECT(hipMemUnmap(pb, size) == hipSuccess);
    EXPECT(hipMemRelease(imported) == hipSuccess);
    EXPECT(hipMemAddressFree(a, size) == hipSuccess);
    EXPECT(hipMemAddressFree(b, size) == hipErrorInvalidValue);   // not the reserved size
    EXPECT(hipMemAddressFree(b, 2 * size) == hipSuccess);
    hipGetLastError();
}

static void contexts()
{
    hipCtx_t primary = nullptr, current = nullptr, created = nullptr, popped = nullptr;
    EXPECT(hipDevicePrimaryCtxRetain(&primary, 0) == hipSuccess && primary != nullptr);
    EXPECT(hipCtxGetCurrent(&current) == hipSuccess && current == primary);
    unsigned flags = 7;
    int active = 0;
    EXPECT(hipDevicePrimaryCtxGetState(0, &flags, &active) == hipSuccess && active == 1);

    EXPECT(hipCtxCreate(&created, hipDeviceScheduleYield, 0) == hipSuccess && created != primary);
    EXPECT(hipCtxGetCurrent(&current) == hipSuccess && current == created);
    EXPECT(hipCtxGetFlags(&flags) == hipSuccess && flags == hipDeviceScheduleYield);
    hipDevice_t device = -1;
    EXPECT(hipCtxGetDevice(&device) == hipSuccess && device == 0);
    EXPECT(hipCtxSynchronize() == hipSuccess);

    EXPECT(hipCtxPopCurrent(&popped) == hipSuccess && popped == created);
    EXPECT(hipCtxGetCurrent(&current) == hipSuccess && current == primary);
    EXPECT(hipCtxPopCurrent(&popped) == hipErrorInvalidContext);
    EXPECT(hipCtxPushCurrent(created) == hipSuccess);
    EXPECT(hipCtxSetCurrent(primary) == hipSuccess);
    EXPECT(hipCtxGetCurrent(&current) == hipSuccess && current == primary);
    EXPECT(hipCtxSetCurrent(created) == hipSuccess);
    EXPECT(hipCtxDestroy(created) == hipSuccess);
    EXPECT(hipCtxGetCurrent(&current) == hipSuccess && current == primary);
    EXPECT(hipCtxDestroy(primary) == hipErrorInvalidContext);
    EXPECT(hipCtxCreate(&created, 0, 1) == hipErrorInvalidDevice);
    EXPECT(hipDevicePrimaryCtxRelease(0) == hipSuccess);
    EXPECT(std::strcmp(hipGetErrorName(hipErrorInvalidContext), "hipErrorInvalidContext") == 0);
    hipGetLastError();
}

static void launches_between_events()
{
    const int n = 1000;
    int *out = nullptr;
    EXPECT(hipMalloc(&out, n * sizeof(int)) == hipSuccess);
    hipEvent_t start, stop;
    EXPECT(hipEventCreate(&start) == hipSuccess && hipEventCreate(&stop) == hipSuccess);

    hipExtLaunchKernelGGL(fill, dim3(4), dim3(256), 0, 0, start, stop, 0, out, 5, n);
    EXPECT(hipEventSynchronize(stop) == hipSuccess);
    float ms = -1;
    EXPECT(hipEventElapsedTime(&ms, start, stop) == hipSuccess && ms >= 0);
    EXPECT(out[0] == 5 && out[n - 1] == 5 + n - 1);

    // Without events, and through an argument array
    hipExtLaunchKernelGGL(fill, dim3(4), dim3(256), 0, 0, nullptr, nullptr, 0, out, 9, n);
    EXPECT(out[n - 1] == 9 + n - 1);
    int value = 20, count = n;
    void *args[] = {&out, &value, &count};
    EXPECT(hipExtLaunchKernel((void *)fill, dim3(4), dim3(256), args, 0, 0, start, stop,
                              0) == hipSuccess);
    EXPECT(hipEventSynchronize(stop) == hipSuccess && out[n - 1] == 20 + n - 1);

    hipEventDestroy(start);
    hipEventDestroy(stop);
    EXPECT(hipFree(out) == hipSuccess);
}

int main()
{
    pointer_attributes();
    range_attributes();
    virtual_memory();
    contexts();
    launches_between_events();

    if (errors == 0)
        std::printf("PASSED\n");
    return errors == 0 ? 0 : 1;
}
