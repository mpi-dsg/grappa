/**
 *  Copyright (c) 2024 by Data Systems Group, MPI-SWS
 */
#ifndef CSRC_BASE_GLAUNCHER_CUH_
#define CSRC_BASE_GLAUNCHER_CUH_

#include <cuda_runtime.h>

#include <algorithm>  // for std::min
#include <stdexcept>  // for std::runtime_errpr
#include <utility>    // for std::forward

#include "csrc/base/gerror.h"

#define HOST
#define HOST_INLINE inline
#ifdef __CUDACC__
#define DEVICE __device__
#define DEVICE_HOST __device__ __host__
#define DEVICE_HOST_INLINE __device__ __host__ __forceinline__
#define DEVICE_INLINE __device__ __forceinline__
#else
#define DEVICE
#define DEVICE_HOST
#define DEVICE_HOST_INLINE inline
#define DEVICE_INLINE
#endif

// volta
#define NUM_SM 80
#define MAX_BLOCK_PER_SM 32
#define MAX_BLOCK_SIZE 1024

#define MAX_GRID_SIZE (NUM_SM * MAX_BLOCK_PER_SM)

#define TID_1D (threadIdx.x + blockIdx.x * blockDim.x)
#define N_TID_1D (gridDim.x * blockDim.x)

#define LANE_KERNEL_1D_LOOP(i, n) \
  for (int32_t i = TID_1D, step = N_TID_1D; i < (n); i += step)

namespace t10n {

template <typename L, typename... Args>
__global__ void cuda_kernel(L lambda, Args... args) {
  lambda(args...);
}

HOST_INLINE void fit_grid_block_size(int* block_num, int* block_size,
                                     size_t work_size) {
  *block_size = MAX_BLOCK_SIZE;
  *block_num = std::min(
      MAX_GRID_SIZE,
      static_cast<int>((work_size + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE));
}

template <typename L, typename... Args>
HOST_INLINE void LaunchKernel(size_t len, L lambda, Args&&... args) {
  int grid_size, block_size;

  fit_grid_block_size(&grid_size, &block_size, len);
  cuda_kernel<<<grid_size, block_size, 0, 0>>>(lambda,
                                               std::forward<Args>(args)...);
}

}  // namespace t10n

#endif  // CSRC_BASE_GLAUNCHER_CUH_
