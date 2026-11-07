# HIP

cuda4cpu compiles HIP code the way HIP's NVIDIA platform does: on top of the
CUDA runtime API, with 32-lane warps. HIP sources define
`__HIP_PLATFORM_NVIDIA__` and `__HIPCC__`, so code that tells the platforms
apart takes its NVIDIA branch. That branch uses CUDA features cuda4cpu
implements, where the AMD branch may assume 64-lane wavefronts or call AMD GPU
builtins. Code written only for AMD GPUs can compile on
[HIP's AMD platform](#hips-amd-platform) instead, emulating an RDNA3 GPU.

HIP projects build with `cuda4cpu-hipcc` in place of hipcc, and the toolkit
directory (`<prefix>/lib/cuda4cpu/toolkit`) is laid out like a ROCm
installation, with `bin/hipcc`, `bin/hipconfig`, `libamdhip64` and HIP's
CMake package, for Makefiles and `find_package(hip)`. CMake's HIP language
works too, with HIP's NVIDIA platform and the toolkit's toolchain file. See
[Using cuda4cpu in your project](usage.md#using-cuda4cpu-in-your-project).

- **Headers:** `hip/hip_runtime.h`, `hip/hip_runtime_api.h`, `hip/hip_ext.h`,
  `hip/hip_cooperative_groups.h`, `hip/hip_fp16.h`, `hip/hip_bf16.h`
  (`__hip_bfloat16`, `__hip_bfloat162` and the `HIPRT_*_BF16` constants, on
  CUDA's bfloat16), `hip/hip_fp8.h` (see below), `hip/hip_complex.h`
  (`hipFloatComplex`, `hipCaddf`, ... on CUDA's `cuComplex.h`, plus
  `hipCsqabsf` and `hipCsqabs`), `hip/hip_vector_types.h`,
  `hip/hip_version.h` (HIP 7.2) and `hip/hip_common.h`.
- **Runtime API:** every CUDA function, type and constant that cuda4cpu
  implements has its HIP name. `hipMalloc`, `hipStream_t` and
  `hipMemcpyHostToDevice` are macros for the CUDA names, as many HIP names
  are on the NVIDIA platform. So they keep CUDA's overloads and default
  arguments, and compiler errors mention the CUDA names.
- **HIP's own additions:**
  - `hipDeviceProp_t` (with `gcnArchName`, the CPU's architecture, and
    `arch`), and `hipGetDevicePropertiesR0600`, the name HIP's headers give
    `hipGetDeviceProperties`
  - `hipDeviceGetAttribute` with HIP's attributes
  - HIP's error names (`hipErrorOutOfMemory`, ...) in `hipGetErrorName`, and
    `hipCUDAErrorTohipError`/`hipErrorToCudaError` (the codes are the same)
  - `hipHostMalloc` and its flags, `hipHostFree`, `hipExtMallocWithFlags`
  - pointers: `hipPointerGetAttributes` (with `isManaged` and
    `allocationFlags`), `hipMemPtrGetInfo` and `hipMemGetAddressRange`. Like
    CUDA's `cudaPointerGetAttributes`, they know every allocation, registered
    host memory and mapped virtual memory, from any pointer into it.
  - virtual memory management: `hipMemCreate`, `hipMemRelease`,
    `hipMemAddressReserve`, `hipMemAddressFree`, `hipMemMap`, `hipMemUnmap`,
    `hipMemSetAccess`, `hipMemGetAccess`, `hipMemGetAllocationGranularity`,
    `hipMemGetAllocationPropertiesFromHandle`, `hipMemRetainAllocationHandle`,
    and sharing through a file descriptor (`hipMemExportToShareableHandle`,
    `hipMemImportFromShareableHandle`). Physical memory is a `memfd`, so the
    same memory can be mapped at several addresses, and as on a GPU, mapped
    memory faults until `hipMemSetAccess` allows access.
  - contexts: `hipCtxCreate`, `hipCtxDestroy`, `hipCtxPushCurrent`,
    `hipCtxPopCurrent`, `hipCtxSetCurrent`, `hipCtxGetCurrent`,
    `hipCtxGetDevice`, `hipCtxGetFlags`, `hipCtxSynchronize`, the cache and
    shared memory configuration, and the primary context
    (`hipDevicePrimaryCtxRetain`, `Release`, `GetState`, `SetFlags`, `Reset`).
    There is one device, so contexts share everything.
  - the device and memory functions of HIP's driver API: `hipInit`,
    `hipDeviceGet`, `hipDeviceGetName`, `hipDeviceComputeCapability`,
    `hipDeviceTotalMem`, `hipDeviceGetUuid`, `hipMemcpyHtoD`/`DtoH`/`DtoD`
    (and `Async`), `hipMemsetD8`/`D16`/`D32` (and `Async`)
  - launches between two events (`hip/hip_ext.h`): `hipExtLaunchKernelGGL`
    and `hipExtLaunchKernel`
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
  - 8-bit floats (`hip/hip_fp8.h`). HIP has them only on AMD GPUs, so this
    is AMD's API: the OCP formats (`__HIP_E4M3`, `__HIP_E5M2`, CUDA's) and
    AMD's FNUZ ones (`__HIP_E4M3_FNUZ`, `__HIP_E5M2_FNUZ`: no infinity, no
    negative zero, NaN at 0x80), the `__hip_cvt_*` conversions (E8M0 too,
    with `hipRoundMode`), and the classes `__hip_fp8_e4m3`,
    `__hip_fp8x2_e4m3`, `__hip_fp8x4_e4m3` and their E5M2 and FNUZ
    counterparts, which convert implicitly and saturate. Values that don't
    fit follow AMD's rules, which differ from CUDA's: anything beyond the
    largest finite value overflows, even if it would round down to it, and
    an infinity stays infinite in E5M2 and becomes NaN in the other formats,
    even when saturating.
- **Not implemented:** what CUDA lacks here too (textures, ...), and what
  needs
  device code compiled or loaded at run time: HIPRTC and the module API
  (`hipModuleLoad`, ...). Also `hipPointerGetAttribute` and
  `hipDrvPointerGetAttributes`, the driver API's attribute-by-attribute
  form.

As in clang's HIP mode, HIP sources get the kernel language implicitly
(`cuda_runtime.h`), and include `hip/hip_runtime.h` themselves. So macros
that configure it, such as `HIP_TEMPLATE_KERNEL_LAUNCH`, work when defined
before the `#include`. `hipLaunchKernelGGL` is a macro, as in HIP. Like a
`<<<...>>>` launch, it deduces template arguments and applies default
arguments. A kernel name whose template arguments contain commas goes in
`HIP_KERNEL_NAME(...)`.

## HIP's AMD platform

Code written only for AMD GPUs takes branches under `__HIP_PLATFORM_AMD__`
and the GPU's macros (`__gfx1100__`, `__GFX11__`), and calls AMD's GPU
builtins (`__builtin_amdgcn_*`). `cuda4cpu-hipcc --cuda4cpu-platform=amd`
(or `CUDA4CPU_HIP_PLATFORM=amd`, for any driver and HIP source) compiles it
as for an RDNA3 GPU, whose wavefronts have 32 lanes, like cuda4cpu's warps:

```sh
cuda4cpu-hipcc --cuda4cpu-platform=amd --offload-arch=gfx1101 -o app app.hip
```

- **The GPU:** `--offload-arch` names it among RDNA3's: `gfx1100` (the
  default), `gfx1101`, `gfx1102`, `gfx1103`, `gfx1150`, `gfx1151`,
  `gfx1152` and `gfx1153`. Without the switch, `--offload-arch` is ignored,
  as before. Wave64 (`-mwavefrontsize64`) and other GPUs are rejected.
- **Macros:** `__HIP_PLATFORM_AMD__`, `__AMDGCN__`, `__AMDGPU__`, the
  GPU's (`__gfx1101__`, `__GFX11__`) and `__AMDGCN_WAVEFRONT_SIZE__` (32),
  instead of `__HIP_PLATFORM_NVIDIA__`. Builds that don't go through the
  drivers can define `__HIP_PLATFORM_AMD__` and a `__gfx11xx__` macro, and
  `hip/hip_common.h` adds the rest. `__HIP_DEVICE_COMPILE__` isn't defined,
  since host and device code compile together: define it
  (`-D__HIP_DEVICE_COMPILE__=1`) to compile code under it, such as
  rocPRIM's DPP paths, but code under `#ifndef __HIP_DEVICE_COMPILE__` then
  drops out.
- **Device properties:** `gcnArchName` is the GPU's name, and `major` and
  `minor` its gfx version (11 and 0 for `gfx1101`).
- **Builtins** (`hip/amdgcn.h`, which `hip/hip_runtime.h` includes):
  - dispatch IDs and sizes: `workitem_id_*`, `workgroup_id_*`,
    `workgroup_size_*`, `grid_size_*`, `wavefrontsize`, and
    `dispatch_ptr`, HSA's dispatch packet;
  - lane operations: `readfirstlane`, `readlane`, `writelane`,
    `read_exec*`, `ballot_w32`/`w64`, `inverse_ballot_*`, `mbcnt_lo`/`hi`,
    `ds_bpermute`, `ds_permute`, `ds_swizzle` (bit masks and quad
    permutations), `mov_dpp`, `update_dpp` (every DPP control, with row
    and bank masks and `bound_ctrl`), `mov_dpp8`, `permlane16`,
    `permlanex16`, `permlane64`, `wave_shuffle` and `wave_reduce_*`;
  - synchronization and timing: `s_barrier`, `wave_barrier`, `fence`,
    `s_sleep` (which lets the block's other threads run, so spin-waits
    work), `s_memtime`, `s_memrealtime`, and hints that do nothing
    (`s_waitcnt`, `sched_barrier`, `sched_group_barrier`, `iglp_opt`,
    `s_setprio`, cache flushes);
  - integers and bits: `perm`, `alignbit`, `alignbyte`, `ubfe`, `sbfe`,
    `lerp`, `sad_u8`, `sad_hi_u8`, `sad_u16`, `sdot4`, `sdot8`, `udot4`,
    `udot8`, `sudot4`, `sudot8`, `s_bitreplicate`, `cvt_pk_u8_f32`;
  - floating point: `rcp`, `rsq`, `sqrt`, `sin` and `cos` of turns,
    `exp2`, `log` (base 2), `ldexp`, `frexp_mant`, `frexp_exp`, `fract`,
    `class`, `fmed3`, in their double, float and half forms where AMD has
    them;
  - atomics: `atomic_inc`, `atomic_dec`, and the floating-point adds,
    minimums and maximums on global, flat and LDS memory.

  Lane operations exchange the values of the lanes that take part (EXEC: not
  those that returned, or that wait at a barrier in another branch). Reading
  a lane outside EXEC gives zero, or keeps the old value where the
  instruction does, where a GPU may give a stale one. The semantics come from
  AMD's ISA manuals and LLVM's AMDGPU backend, without a GPU to compare
  with: rocPRIM's DPP scan and reduction give the right results. They are
  inline functions, so `__has_builtin(__builtin_amdgcn_...)` is false, and
  code that checks it takes its other path. Builtins of other AMD GPUs
  compile too, unchecked.
- **Not yet:** builtins with Clang's vector types (WMMA, packed
  conversions, `fdot2`), fp8/fp4/fp6 conversions, buffer resources,
  `ds_swizzle`'s FFT and rotate modes, `is_shared`/`is_private`, RDNA4,
  and CDNA's 64-lane wavefronts and MFMA: see the
  [roadmap](../ROADMAP.md#amd-gpu-builtins). AMD runs a wavefront's lanes
  in lockstep, and cuda4cpu doesn't: code that relies on it without
  `__syncwarp()` or `wave_barrier()` races, as it may under CUDA.
