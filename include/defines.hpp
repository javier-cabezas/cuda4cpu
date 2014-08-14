#ifndef CUDA4CPU_DEFINES_HPP_
#define CUDA4CPU_DEFINES_HPP_

//
// CUDA keyword overrides
//
#ifdef __global__
#undef __global__
#endif

#ifdef __device__
#undef __device__
#endif

#ifdef __host__
#undef __host__
#endif

#ifdef __shared__
#undef __shared__
#endif

#ifdef __constant__
#undef __constant__
#endif

#define __global__
#define __shared__ static thread_local
#define __syncthreads cuda4cpu::thread_block::syncthreads

#define threadIdx cuda4cpu::thread_block::get_thread()
#define blockIdx  cuda4cpu::thread_block::get_block()

#define blockDim cuda4cpu::thread_block::get_block_dim()
#define gridDim  cuda4cpu::thread_block::get_grid_dim()

#endif // CUDA4CPU_DEFINES_HPP_
