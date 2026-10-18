# Supported CUDA features

**Kernel language**

- `__global__`, `__device__`, `__host__`, `__shared__`, `__constant__`,
  `__forceinline__`, `__launch_bounds__`
- `threadIdx`, `blockIdx`, `blockDim`, `gridDim`, `warpSize`
- `__syncthreads()`, and dynamic shared memory through `dynamic_shared<T>()`
- Warp functions: `__shfl_sync`, `__shfl_up_sync`, `__shfl_down_sync`,
  `__shfl_xor_sync` (any trivially copyable type, with `width`),
  `__ballot_sync`, `__any_sync`, `__all_sync`, `__match_any_sync`,
  `__match_all_sync`, `__activemask`, `__syncwarp`
- Cooperative groups (`cooperative_groups.h`):
  - `thread_block` and `grid_group`
  - `thread_block_tile<N>` for N up to 1024. Tiles of up to 32 threads also
    have shuffles, votes and matches.
  - `coalesced_group` (`coalesced_threads()`)
  - `tiled_partition`, both static and dynamic, `labeled_partition` and
    `binary_partition`
  - `reduce`, `inclusive_scan` and `exclusive_scan`, with `plus`, `less`,
    `greater`, `bit_and`, `bit_or` and `bit_xor`
  - `grid.sync()` in kernels launched with `cudaLaunchCooperativeKernel`
- Atomics: `atomicAdd`, `atomicSub`, `atomicExch`, `atomicMin`, `atomicMax`,
  `atomicInc`, `atomicDec`, `atomicCAS`, `atomicAnd`, `atomicOr`, `atomicXor`
  for the same types as CUDA, and `__threadfence`, `__threadfence_block`,
  `__threadfence_system`
- `dim3`, the built-in vector types (`float4`, `int2`, `uchar3`, ...) and their
  `make_<type>` functions
- Half precision (`cuda_fp16.h`): `__half` and `__half2` (`half`, `half2`),
  with CUDA's layout, implicit conversions and operators, and their
  intrinsics:
  - conversions to and from float, double and integers in every rounding
    mode, bit reinterpretation, and `half2` packing (`__floats2half2_rn`,
    `__low2float`, `__halves2half2`, ...)
  - arithmetic (`__hadd`, `__hfma`, `__hmul2`, `__h2div`, ... and the `_sat`
    and `_relu` forms), comparisons (ordered, unordered, `__hb*2` and
    `_mask`), `__hmax`/`__hmin`, `__hisnan`, `__hisinf`
  - math (`hsqrt`, `hrcp`, `hexp`, `hsin`, `hfloor`, ... and their `h2`
    forms), shuffles, `__ldg`, and `atomicAdd` on `__half` and `__half2`

  Arithmetic (including `__hfma`), `hsqrt`, `hrcp` and conversions are
  correctly rounded, subnormals included, and conversions honor their
  `_rn`/`_rz`/`_rd`/`_ru` suffix. The other math functions round their
  double precision result, so they are at least as accurate as CUDA's.
- Math: the C++ standard library, plus `rsqrtf`, `rsqrt` and the fast
  intrinsics `__expf`, `__exp10f`, `__logf`, `__log2f`, `__log10f`, `__sinf`,
  `__cosf`, `__tanf`, `__sincosf`, `__powf`, `__fdividef` and `__saturatef`.
  They are computed at full precision.
- Integer intrinsics: `__popc`, `__popcll`, `__clz`, `__clzll`, `__ffs`,
  `__ffsll`, `__brev`, `__brevll`, `__mul24`, `__umul24`, and `__ldg`
- `min` and `max` for integer and floating-point arguments, as in CUDA
- Arithmetic with explicit rounding (`__fadd_rd`, `__dmul_ru`, `__fsqrt_rz`,
  ... for float and double), conversions with explicit rounding
  (`__float2int_rn`, `__double2float_rd`, `__int2float_ru`, ...) and bit
  reinterpretation (`__int_as_float`, `__double_as_longlong`, ...). Directed
  rounding is exact: the operation runs with the FPU's rounding mode switched.
- Scoped atomics (`atomicAdd_block`, `atomicCAS_system`, ...), which behave
  like the device-scope ones
- `__grid_constant__`, `__align__`, `__managed__`
- `clock()`, `clock64()` and `__nanosleep()`. The clocks count at the
  device's `clockRate`, so that kernels that wait for a number of cycles wait
  that long.

**Runtime API**

| Area | Functions |
|---|---|
| Errors | `cudaGetLastError`, `cudaPeekAtLastError`, `cudaGetErrorName`, `cudaGetErrorString`, and `cudaError_t` with CUDA's codes |
| Device | `cudaGetDeviceCount`, `cudaSetDevice`, `cudaGetDevice`, `cudaSetDeviceFlags`, `cudaGetDeviceFlags`, `cudaGetDeviceProperties`, `cudaDeviceGetAttribute`, `cudaDriverGetVersion`, `cudaRuntimeGetVersion`, `cudaDeviceSynchronize`, `cudaDeviceReset`, `cudaDeviceSetLimit`, `cudaDeviceGetLimit` (`cudaLimitStackSize` only), `cudaDeviceGetStreamPriorityRange`, `cudaDeviceCanAccessPeer`, and the deprecated `cudaThreadSynchronize` and `cudaThreadExit` |
| Memory | `cudaMalloc`, `cudaMallocHost`, `cudaHostAlloc`, `cudaMallocManaged` (with typed overloads, as in CUDA), `cudaFree`, `cudaFreeHost` (both reject pointers that they didn't allocate), `cudaPointerGetAttributes`, `cudaMemGetInfo` (the host's available and physical memory), `cudaMemcpy`, `cudaMemcpyAsync`, `cudaMemset`, `cudaMemsetAsync`, `cudaMemcpyToSymbol`, `cudaMemcpyFromSymbol`, `cudaGetSymbolAddress`, `cudaGetSymbolSize` |
| 2D and 3D memory | `cudaMallocPitch`, `cudaMalloc3D`, `cudaMemcpy2D`, `cudaMemcpy2DAsync`, `cudaMemcpy3D`, `cudaMemcpy3DAsync`, `cudaMemset2D`, `cudaMemset2DAsync`, `cudaMemset3D`, `cudaMemcpyToSymbolAsync`, `cudaMemcpyFromSymbolAsync`; CUDA arrays aren't implemented |
| More memory | Stream-ordered allocation (`cudaMallocAsync`, `cudaFreeAsync`, `cudaMallocFromPoolAsync`, the default memory pool, `cudaMemPoolCreate`, `cudaMemPoolDestroy`), mapped and registered host memory (`cudaHostGetDevicePointer`, `cudaHostRegister`, `cudaHostUnregister`, `cudaHostGetFlags`), and unified memory hints (`cudaMemPrefetchAsync`, `cudaMemAdvise`, `cudaStreamAttachMemAsync`, and `cudaMemRangeGetAttribute`/`cudaMemRangeGetAttributes`, which return them). All memory is host memory, so the hints and pool settings have no effect, but the hints are recorded per page. |
| Streams | `cudaStreamCreate`, `cudaStreamCreateWithFlags`, `cudaStreamCreateWithPriority`, `cudaStreamDestroy`, `cudaStreamGetFlags`, `cudaStreamGetPriority`, `cudaStreamQuery`, `cudaStreamSynchronize`, `cudaStreamWaitEvent`, `cudaStreamAddCallback`, `cudaLaunchHostFunc`, and the default streams `cudaStreamLegacy` and `cudaStreamPerThread` (one per OS thread) |
| Configuration hints, accepted and ignored | `cudaFuncSetAttribute`, `cudaFuncGetAttributes`, `cudaFuncSetCacheConfig`, `cudaDeviceSetCacheConfig`, `cudaDeviceSetSharedMemConfig`, `cudaStreamSetAttribute` (L2 access policy), `cudaCtxResetPersistingL2Cache`, `cudaProfilerStart`, `cudaProfilerStop` |
| Headers | `cuda_runtime.h`, `cuda_runtime_api.h`, `cuda.h` (`CUDA_VERSION` and the runtime API; the driver API isn't implemented), `cuda_profiler_api.h`, `vector_types.h`, `vector_functions.h`, `device_launch_parameters.h`, `cooperative_groups.h`, `cooperative_groups/reduce.h`, `cooperative_groups/scan.h`, `cuda_fp16.h`, `nvtx3/nvToolsExt.h` and `nvToolsExt.h` (NVTX annotations, which do nothing) |
| Graphs | Explicit graphs with every node type except external semaphores: kernel, memcpy, memset, host, child graph, empty, event record and wait, memory allocation and free, and conditional nodes (if/else, while, switch, set by `cudaGraphSetConditional`), including `cudaGraphAddNode`. Dependencies, queries, `cudaGraphClone`, `cudaGraphDebugDotPrint`, instantiation, launch, `cudaGraphExecUpdate` and the `cudaGraphExec*NodeSetParams` functions, and the graph memory attributes. |
| Stream capture | `cudaStreamBeginCapture`, `cudaStreamBeginCaptureToGraph`, `cudaStreamEndCapture`, `cudaStreamIsCapturing`, `cudaStreamGetCaptureInfo`, `cudaStreamUpdateCaptureDependencies`, and fork and join through events. Kernel launches, async copies and memsets, `cudaLaunchHostFunc`, `cudaMallocAsync` and `cudaFreeAsync` are captured. |
| Launch | `cudaLaunchKernel`, `cudaLaunchCooperativeKernel`, `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, `cudaOccupancyMaxPotentialBlockSize`, `cudaOccupancyAvailableDynamicSMemPerBlock` |
| Events | `cudaEventCreate`, `cudaEventCreateWithFlags`, `cudaEventDestroy`, `cudaEventRecord`, `cudaEventQuery`, `cudaEventSynchronize`, `cudaEventElapsedTime` |
