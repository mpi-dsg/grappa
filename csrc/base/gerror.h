/**
 *  Copyright (c) 2024 by Data Systems Group, MPI-SWS
 */
#ifndef CSRC_BASE_GERROR_H_
#define CSRC_BASE_GERROR_H_

#include <cuda_runtime.h>

#include <stdexcept>  // for std::runtime_errpr

// CUDA API error checking
#define CUDA_CHECK(err)                                                  \
  do {                                                                   \
    cudaError_t err_ = (err);                                            \
    if (err_ != cudaSuccess) {                                           \
      std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__); \
      throw std::runtime_error("CUDA error");                            \
    }                                                                    \
  } while (0)

#endif  // CSRC_BASE_GERROR_H_
