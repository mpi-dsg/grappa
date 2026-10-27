/**
 *  Copyright (c) 2024 by Data Systems Group, MPI-SWS
 */
#ifndef CSRC_BASE_GALLOC_H_
#define CSRC_BASE_GALLOC_H_

#include <climits>  // for UINT_MAX

#include "csrc/base/gerror.h"

namespace t10n {

template <class _Tp>
class galloc {
 public:
  typedef _Tp* pointer;
  typedef const _Tp* const_pointer;
  typedef _Tp& reference;
  typedef const _Tp& const_reference;
  typedef size_t size_type;

  static _Tp* allocate(size_type n, const void* = 0) {
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, sizeof(_Tp) * n));
    return static_cast<_Tp*>(ptr);
  }

  static void deallocate(pointer p, size_type n) { CUDA_CHECK(cudaFree(p)); }

  size_type max_size() const noexcept {
    return static_cast<size_type>(UINT_MAX / sizeof(_Tp));
  }
};

}  // namespace t10n

#endif  // CSRC_BASE_GALLOC_H_
