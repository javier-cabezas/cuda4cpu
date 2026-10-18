# HIP

cuda4cpu compiles HIP code the way HIP's NVIDIA platform does: on top of the
CUDA runtime API, with 32-lane warps. HIP sources define
`__HIP_PLATFORM_NVIDIA__` and `__HIPCC__`, so code that tells the platforms
apart takes its NVIDIA branch. That branch uses CUDA features cuda4cpu
implements, where the AMD branch may assume 64-lane wavefronts or call AMD GPU
builtins.

- **Headers:** `hip/hip_runtime.h`, `hip/hip_runtime_api.h`,
  `hip/hip_cooperative_groups.h`, `hip/hip_fp16.h`, `hip/hip_vector_types.h`,
  `hip/hip_version.h` (HIP 7.2) and `hip/hip_common.h`.
- **Runtime API:** every CUDA function, type and constant that cuda4cpu
  implements has its HIP name. `hipMalloc`, `hipStream_t` and
  `hipMemcpyHostToDevice` are macros for the CUDA names, as many HIP names
  are on the NVIDIA platform. So they keep CUDA's overloads and default
  arguments, and compiler errors mention the CUDA names.
- **HIP's own additions:**
  - `hipDeviceProp_t` (with `gcnArchName`, the CPU's architecture, and `arch`)
  - `hipDeviceGetAttribute` with HIP's attributes
  - HIP's error names (`hipErrorOutOfMemory`, ...) in `hipGetErrorName`
  - `hipHostMalloc` and its flags, `hipExtMallocWithFlags`
  - the device and memory functions of HIP's driver API: `hipInit`,
    `hipDeviceGet`, `hipDeviceGetName`, `hipDeviceComputeCapability`,
    `hipDeviceTotalMem`, `hipMemcpyHtoD`/`DtoH`/`DtoD` (and `Async`),
    `hipMemsetD8`/`D16`/`D32` (and `Async`)
  - `hipMemcpyWithStream`, `hipChooseDevice`, `HIP_SYMBOL`
- **Kernel language:** CUDA's, plus:
  - launches with `hipLaunchKernelGGL`, `HIP_KERNEL_NAME`, and the function
    template form with `HIP_TEMPLATE_KERNEL_LAUNCH`
  - `HIP_DYNAMIC_SHARED`
  - `hipThreadIdx_x` and the other `hip*Idx_*`/`hip*Dim_*` names
  - the warp functions without a mask (`__shfl`, `__shfl_up`, `__shfl_down`,
    `__shfl_xor`, `__ballot`, `__any`, `__all`) and `__lane_id`
  - `__bitextract_u32`/`u64`, `__bitinsert_u32`/`u64`
  - `wall_clock64` and `__clock64`
  - `unsafeAtomicAdd`, `safeAtomicAdd`, `atomicAddNoRet`
  - the `__HIP_ARCH_HAS_*` feature macros
- **Not implemented:** what CUDA lacks here too (textures, bfloat16, ...),
  plus 8-bit floats (`hip/hip_fp8.h`), `hip/hip_complex.h`, HIPRTC, the module API (`hipModuleLoad`,
  ...), contexts, virtual memory management, `hipPointerGetAttributes` and
  `hipMemRangeGetAttribute`.

As in clang's HIP mode, HIP sources get the kernel language implicitly
(`cuda_runtime.h`), and include `hip/hip_runtime.h` themselves. So macros
that configure it, such as `HIP_TEMPLATE_KERNEL_LAUNCH`, work when defined
before the `#include`. `hipLaunchKernelGGL` is a macro, as in HIP. Like a
`<<<...>>>` launch, it deduces template arguments and applies default
arguments. A kernel name whose template arguments contain commas goes in
`HIP_KERNEL_NAME(...)`.
